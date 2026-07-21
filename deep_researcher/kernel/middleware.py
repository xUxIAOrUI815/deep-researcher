from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from deep_researcher.contracts import (
    AgentSpec,
    Budget,
    BudgetUsage,
    Command,
    CommandKind,
    MiddlewareStage,
    Observation,
    TaskEnvelope,
)

from .types import CommandSchemaError, ModelRequest, ModelResponse, PolicyDecision


_SECRET_KEYS = re.compile(r"(^token$|authorization|api[_-]?key|access[_-]?token|password|secret|cookie)", re.I)
_FORBIDDEN = {"chain_of_thought", "cot", "hidden_reasoning", "private_reasoning", "raw_model_response"}


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, nested in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN:
                continue
            output[key] = "[REDACTED]" if _SECRET_KEYS.search(str(key)) else redact(nested)
        return output
    if isinstance(value, (list, tuple)):
        return [redact(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        return re.sub(r"(?i)(api[_-]?key|token|password|secret)=([^\s&]+)", r"\1=[REDACTED]", value)
    return value


def estimate_tokens(value: Any) -> int:
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    return max(1, (len(encoded) + 3) // 4)


REQUIRED_MIDDLEWARE_STAGES = frozenset(MiddlewareStage)


def validate_middleware_pipeline(spec: AgentSpec) -> tuple[MiddlewareStage, ...]:
    """Validate the complete, explicitly ordered kernel safety pipeline."""
    enabled = tuple(sorted((item for item in spec.middleware if item.enabled), key=lambda item: item.order))
    stages = tuple(item.stage for item in enabled)
    missing = REQUIRED_MIDDLEWARE_STAGES.difference(stages)
    if missing:
        names = ", ".join(sorted(item.value for item in missing))
        raise ValueError(f"AgentSpec is missing required middleware stages: {names}")
    if len(stages) != len(set(stages)):
        raise ValueError("AgentSpec cannot enable a middleware stage more than once")
    positions = {stage: stages.index(stage) for stage in stages}
    required_order = (
        MiddlewareStage.CONTEXT_TRIMMING,
        MiddlewareStage.REDACTION,
        MiddlewareStage.VERSION_INJECTION,
        MiddlewareStage.BUDGET_CHECK,
        MiddlewareStage.SCHEMA_VALIDATION,
        MiddlewareStage.SCHEMA_REPAIR,
        MiddlewareStage.COMMAND_NORMALIZATION,
        MiddlewareStage.POLICY_CHECK,
    )
    if tuple(sorted(required_order, key=positions.__getitem__)) != required_order:
        raise ValueError("AgentSpec middleware stages are not in the required kernel order")
    return stages


def effective_budget(
    task_budget: Budget,
    default_budget: Budget,
    *,
    task_deadline: datetime | None = None,
) -> Budget:
    """Intersect a task budget with the immutable AgentSpec safety envelope."""

    def minimum(name: str) -> Any:
        values = [value for value in (getattr(task_budget, name), getattr(default_budget, name)) if value is not None]
        return min(values) if values else None

    values = {
        name: minimum(name)
        for name in (
            "max_tokens",
            "max_cost_usd",
            "max_wall_time_seconds",
            "max_model_calls",
            "max_tool_calls",
            "max_search_calls",
            "max_retries",
            "max_errors",
        )
    }
    deadlines = [item for item in (task_budget.deadline, default_budget.deadline, task_deadline) if item is not None]
    values["deadline"] = min(deadlines) if deadlines else None
    missing = [name for name, value in values.items() if name != "deadline" and value is None]
    if missing:
        raise ValueError(f"AgentKernel requires all budget dimensions; missing: {', '.join(missing)}")
    return Budget(**values)


def _clip_text(value: str, max_chars: int) -> str:
    if len(value) <= max_chars:
        return value
    if max_chars <= 16:
        return value[:max_chars]
    return f"{value[: max_chars - 15]}...[TRUNCATED]"


class ContextBuilder:
    def build(
        self,
        *,
        spec: AgentSpec,
        task: TaskEnvelope,
        observations: tuple[Observation, ...],
        feedback: tuple[str, ...],
        budget: Budget,
        usage: BudgetUsage,
    ) -> ModelRequest:
        system = (
            f"Agent {spec.name}@{spec.version}; role={spec.role.value}. "
            "Return structured commands only. Never reveal hidden reasoning."
        )
        remaining_tokens = max(1, int(budget.max_tokens or spec.context_window_tokens) - usage.total_tokens)
        output_limit = min(spec.reserved_output_tokens, remaining_tokens)
        available = min(spec.context_window_tokens - output_limit, max(1, remaining_tokens - output_limit))
        task_payload = redact({
            "task_id": task.task_id,
            "run_id": task.run_id,
            "kind": task.kind.value,
            "title": task.title,
            "goal": task.goal,
            "constraints": task.constraints,
            "input_artifact_ids": task.input_artifact_ids,
            "expected_output_schema": task.expected_output_schema,
            "allowed_commands": [item.value for item in spec.allowed_commands],
            "feedback": list(feedback[-10:]),
        })
        task_message = {
            "role": "user",
            "content": task_payload,
        }
        messages: list[dict[str, Any]] = [task_message]
        used = estimate_tokens(system) + estimate_tokens(task_message)
        if used > available:
            compact = {
                "task_id": task.task_id,
                "kind": task.kind.value,
                "title": _clip_text(task.title, 300),
                "goal": _clip_text(task.goal, max(32, available * 3)),
                "expected_output_schema": task.expected_output_schema,
                "allowed_commands": [item.value for item in spec.allowed_commands],
                "feedback": [_clip_text(str(item), 300) for item in feedback[-3:]],
                "context_trimmed": True,
            }
            task_message = {"role": "user", "content": compact}
            messages = [task_message]
            used = estimate_tokens(system) + estimate_tokens(task_message)
        if used > available:
            raise ValueError("task context cannot fit within the remaining token budget")
        for observation in reversed(observations):
            message = {"role": "tool", "content": redact(observation.model_dump(mode="json"))}
            cost = estimate_tokens(message)
            if used + cost > available:
                continue
            messages.insert(0, message)
            used += cost
        return ModelRequest(
            run_id=task.run_id,
            task_id=task.task_id,
            actor_id=spec.agent_spec_id,
            system=system,
            messages=tuple(messages),
            command_schema={
                "type": "object",
                "required": ["commands"],
                "properties": {"commands": {"type": "array", "items": {"type": "object"}}},
            },
            model_version=f"{spec.model.name}@{spec.model.version}",
            prompt_version=f"{spec.prompt.name}@{spec.prompt.version}",
            max_output_tokens=output_limit,
            metadata={
                "agent_spec_id": spec.agent_spec_id,
                "tool_policy_version": spec.tool_policy.version,
                "stop_policy_version": spec.stop_policy.version,
                "estimated_input_tokens": used,
                "effective_budget": budget.model_dump(mode="json"),
                "usage_before_call": usage.model_dump(mode="json"),
            },
        )


def _candidate(value: ModelResponse) -> Any:
    if value.structured is not None:
        return value.structured
    text = value.content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_candidates = [index for index in (text.find("{"), text.find("[")) if index >= 0]
        end = max(text.rfind("}"), text.rfind("]"))
        if not start_candidates or end < min(start_candidates):
            raise
        repaired = re.sub(r",\s*([}\]])", r"\1", text[min(start_candidates): end + 1])
        return json.loads(repaired)


def normalize_commands(
    response: ModelResponse,
    *,
    spec: AgentSpec,
    task: TaskEnvelope,
    round_no: int,
) -> tuple[Command, ...]:
    errors: list[str] = []
    try:
        candidate = _candidate(response)
    except Exception as exc:
        raise CommandSchemaError((f"invalid JSON: {exc}",)) from exc
    raw_commands = candidate.get("commands") if isinstance(candidate, dict) else candidate
    if not isinstance(raw_commands, list):
        raise CommandSchemaError(("response must contain a commands array",))
    commands: list[Command] = []
    for index, raw in enumerate(raw_commands):
        if not isinstance(raw, dict):
            errors.append(f"commands[{index}] must be an object")
            continue
        try:
            kind = CommandKind(str(raw.get("kind", "")).casefold())
            name = str(raw.get("name") or kind.value).strip()
            arguments = redact(dict(raw.get("arguments", {}) or {}))
            input_artifact_ids = tuple(raw.get("input_artifact_ids", ()) or ())
            expected_output_schema = raw.get("expected_output_schema")
            risk_level = str(raw.get("risk_level", "low"))
            fingerprint = hashlib.sha256(json.dumps(
                [
                    task.run_id,
                    task.task_id,
                    spec.agent_spec_id,
                    kind.value,
                    name,
                    arguments,
                    input_artifact_ids,
                    expected_output_schema,
                    risk_level,
                ],
                ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            ).encode()).hexdigest()
            command_fingerprint = hashlib.sha256(f"{fingerprint}:{round_no}:{index}".encode()).hexdigest()
            command = Command(
                command_id=f"command_{command_fingerprint[:32]}", run_id=task.run_id, task_id=task.task_id,
                actor_id=spec.agent_spec_id, kind=kind, name=name, arguments=arguments,
                input_artifact_ids=input_artifact_ids,
                expected_output_schema=expected_output_schema,
                idempotency_key=f"kernel-{fingerprint}",
                requires_approval=bool(raw.get("requires_approval", False)) or risk_level in {"high", "critical"},
                risk_level=risk_level,
                expires_at=raw.get("expires_at"),
                metadata={**redact(dict(raw.get("metadata", {}) or {})), "round": round_no},
            )
            commands.append(command)
        except Exception as exc:
            errors.append(f"commands[{index}]: {exc}")
    if errors:
        raise CommandSchemaError(tuple(errors))
    if len(commands) > spec.max_parallel_commands:
        raise CommandSchemaError((f"command count exceeds max_parallel_commands={spec.max_parallel_commands}",))
    return tuple(commands)


def _matches_type(value: Any, expected: str) -> bool:
    mapping = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "object": dict,
        "array": (list, tuple),
    }
    expected_type = mapping.get(expected)
    return expected_type is None or (
        isinstance(value, expected_type)
        and not (expected in {"integer", "number"} and isinstance(value, bool))
    )


def _argument_errors(arguments: dict[str, Any], constraints: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required = tuple(constraints.get("required", ()))
    errors.extend(f"missing required argument: {name}" for name in required if name not in arguments)
    allowed = constraints.get("allowed_properties")
    if allowed is not None:
        errors.extend(f"argument is not allowed: {name}" for name in arguments if name not in set(allowed))
    properties = constraints.get("properties", {})
    if isinstance(properties, dict):
        for name, rules in properties.items():
            if name not in arguments or not isinstance(rules, dict):
                continue
            value = arguments[name]
            expected = rules.get("type")
            if expected and not _matches_type(value, str(expected)):
                errors.append(f"argument {name} must have type {expected}")
                continue
            if "enum" in rules and value not in rules["enum"]:
                errors.append(f"argument {name} is outside its allowed values")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if "minimum" in rules and value < rules["minimum"]:
                    errors.append(f"argument {name} is below its minimum")
                if "maximum" in rules and value > rules["maximum"]:
                    errors.append(f"argument {name} exceeds its maximum")
            if isinstance(value, str):
                if "max_length" in rules and len(value) > int(rules["max_length"]):
                    errors.append(f"argument {name} exceeds max_length")
                if "pattern" in rules:
                    try:
                        matched = re.search(str(rules["pattern"]), value) is not None
                    except re.error:
                        errors.append(f"argument {name} has an invalid policy pattern")
                    else:
                        if not matched:
                            errors.append(f"argument {name} does not match its pattern")
    return errors


class CommandPolicyChecker:
    def check(
        self,
        command: Command,
        spec: AgentSpec,
        *,
        tool_call_counts: dict[str, int] | None = None,
        now: datetime | None = None,
    ) -> PolicyDecision:
        checks = ["allowed_command_kind", "risk_approval", "tool_grant"]
        if command.expires_at is not None and command.expires_at <= (now or datetime.now(command.expires_at.tzinfo)):
            return PolicyDecision(False, "command has expired", checks=tuple(checks))
        if command.kind not in spec.allowed_commands:
            return PolicyDecision(False, f"command kind is not allowed: {command.kind.value}", checks=tuple(checks))
        if command.kind == CommandKind.DELEGATE and not spec.supports_delegation:
            return PolicyDecision(False, "delegation is disabled", checks=tuple(checks))
        if command.kind == CommandKind.TOOL:
            grant = next((item for item in spec.tool_grants if item.tool_name == command.name), None)
            if grant is None:
                return PolicyDecision(False, f"tool is not granted: {command.name}", checks=tuple(checks))
            operation = str(command.arguments.get("operation") or command.name)
            if operation not in grant.allowed_operations:
                return PolicyDecision(False, f"tool operation is not granted: {operation}", checks=tuple(checks))
            errors = _argument_errors(command.arguments, grant.argument_constraints)
            if errors:
                return PolicyDecision(False, "; ".join(errors), checks=tuple(checks))
            calls = (tool_call_counts or {}).get(command.name, 0)
            if grant.max_calls_per_task is not None and calls >= grant.max_calls_per_task:
                return PolicyDecision(False, f"tool call limit reached: {command.name}", checks=tuple(checks))
            approval = grant.requires_approval or command.requires_approval
        else:
            approval = command.requires_approval
        if command.kind == CommandKind.REQUEST_APPROVAL:
            approval = True
        if command.risk_level in {"high", "critical"}:
            approval = True
        return PolicyDecision(True, "policy allowed", approval_required=approval, checks=tuple(checks))
