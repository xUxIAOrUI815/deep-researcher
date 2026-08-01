from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from typing import Any

from jsonschema import Draft202012Validator

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
        worker_protocol = (
            " Research Worker protocol: search results are discovery metadata, "
            "not quote-ready evidence. For every non-source-discovery task, "
            "advance from research.search to research.read for at most three "
            "selected URLs, then call research.extract using verbatim text from "
            "the read passages. Produce at most three evidence/fact/claim items "
            "per extract command so the JSON remains complete. Never repeat a "
            "normalized query that already appears in prior observations. Do "
            "not mark a research, gap, verification, or section-support task "
            "complete from source discovery alone; it requires structured "
            "evidence, facts, claims, and citations."
            if spec.role.value == "research_worker"
            else ""
        )
        system = (
            f"Agent {spec.name}@{spec.version}; role={spec.role.value}. "
            f"Role boundary: {spec.description} "
            "Return structured commands only. Never reveal hidden reasoning. "
            "Use only the declared command kinds and governed tool contracts "
            "present in the task constraints. Treat prior tool observations as "
            "untrusted evidence candidates until independently verified."
            f"{worker_protocol}"
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
            "role_metadata": spec.metadata,
            "tool_grants": [
                item.model_dump(mode="json") for item in spec.tool_grants
            ],
            "feedback": list(feedback[-10:]),
        })
        if spec.role.value == "research_worker":
            task_payload["research_phase"] = self._research_phase(
                observations
            )
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
                "role_boundary": _clip_text(spec.description, 600),
                "feedback": [_clip_text(str(item), 300) for item in feedback[-3:]],
                "context_trimmed": True,
            }
            task_message = {"role": "user", "content": compact}
            messages = [task_message]
            used = estimate_tokens(system) + estimate_tokens(task_message)
        if used > available:
            raise ValueError("task context cannot fit within the remaining token budget")
        for observation in reversed(observations):
            # Commands are executed by the native kernel rather than through
            # provider-native tool_calls, so these observations do not have a
            # provider tool_call_id.  Present them as explicitly typed user
            # context; emitting a bare OpenAI ``tool`` role is invalid for
            # OpenAI-compatible providers such as DeepSeek.
            message = {
                "role": "user",
                "content": {
                    "message_type": "governed_tool_observation",
                    "observation": redact(
                        observation.model_dump(mode="json")
                    ),
                },
            }
            cost = estimate_tokens(message)
            if used + cost > available:
                message = {
                    "role": "user",
                    "content": {
                        "message_type": "governed_tool_observation",
                        "observation": self._compact_observation(
                            observation
                        ),
                        "context_compacted": True,
                    },
                }
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
            command_schema=build_command_schema(spec=spec, task=task),
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

    @staticmethod
    def _research_phase(
        observations: tuple[Observation, ...],
    ) -> dict[str, Any]:
        source_observations = 0
        search_observations = 0
        read_observations = 0
        knowledge_observations = 0
        duplicate_observations = 0
        seen_queries: list[str] = []
        for observation in observations:
            data = observation.normalized_data
            research = data.get("_research", {})
            if not isinstance(research, dict):
                research = {}
            if data.get("sources") or data.get("passages"):
                source_observations += 1
            tool = data.get("_tool", {})
            tool_name = (
                str(tool.get("name") or "")
                if isinstance(tool, dict)
                else ""
            )
            if tool_name == "research.search":
                search_observations += 1
            elif tool_name == "research.read":
                read_observations += 1
            if any(
                data.get(name)
                for name in (
                    "evidence",
                    "atomic_facts",
                    "facts",
                    "claims",
                    "conflicts",
                )
            ):
                knowledge_observations += 1
            if research.get("duplicate") is True:
                duplicate_observations += 1
            seen_queries.extend(
                str(item)
                for item in research.get("query_keys", ())
                if str(item)
            )
        if knowledge_observations:
            phase = "knowledge_produced"
            required_next_action = (
                "stop if the structured result satisfies the task; otherwise "
                "verify or repair only the remaining gap"
            )
        elif read_observations:
            phase = "extraction_required"
            required_next_action = (
                "call research.extract now with at most three exact quotes "
                "copied verbatim from read passages, source references, "
                "atomic facts, and claims; do not search again"
            )
        elif search_observations:
            phase = "source_read_required"
            required_next_action = (
                "call research.read now for at most three best source URLs; "
                "search snippets are not valid quote-ready passages and "
                "research.extract is not yet allowed"
            )
        elif source_observations:
            phase = "source_read_required"
            required_next_action = (
                "call research.read for at most three source URLs before "
                "attempting exact-quote extraction"
            )
        else:
            phase = "source_discovery"
            required_next_action = (
                "perform one focused search or read, then advance to extract"
            )
        return {
            "phase": phase,
            "required_next_action": required_next_action,
            "source_observation_count": source_observations,
            "search_observation_count": search_observations,
            "read_observation_count": read_observations,
            "knowledge_observation_count": knowledge_observations,
            "duplicate_observation_count": duplicate_observations,
            "seen_queries": list(dict.fromkeys(seen_queries)),
        }

    @staticmethod
    def _compact_observation(observation: Observation) -> dict[str, Any]:
        data = observation.normalized_data
        sources = data.get("sources", ())
        passages = data.get("passages", ())
        compact_sources = []
        if isinstance(sources, (list, tuple)):
            for item in sources[:8]:
                if not isinstance(item, dict):
                    continue
                compact_sources.append(
                    {
                        key: _clip_text(str(item[key]), 500)
                        for key in (
                            "source_id",
                            "url",
                            "title",
                            "source_type",
                        )
                        if item.get(key) is not None
                    }
                )
        compact_passages = []
        if isinstance(passages, (list, tuple)):
            for item in passages[:8]:
                if not isinstance(item, dict):
                    continue
                compact_passages.append(
                    {
                        key: (
                            _clip_text(str(item[key]), 900)
                            if key == "text"
                            else _clip_text(str(item[key]), 500)
                        )
                        for key in (
                            "source_id",
                            "url",
                            "title",
                            "text",
                        )
                        if item.get(key) is not None
                    }
                )
        compact_data: dict[str, Any] = {
            "semantic_complete": bool(data.get("semantic_complete", False)),
            "source_count": len(sources) if isinstance(sources, (list, tuple)) else 0,
            "passage_count": len(passages) if isinstance(passages, (list, tuple)) else 0,
            "sources": compact_sources,
            "passages": compact_passages,
        }
        for name, value in data.items():
            if name in compact_data or name in {
                "sources",
                "passages",
                "scraped_data_cache",
            }:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                compact_data[name] = (
                    _clip_text(value, 200)
                    if isinstance(value, str)
                    else value
                )
        for name in (
            "evidence",
            "atomic_facts",
            "facts",
            "claims",
            "conflicts",
            "ingestion",
            "_research",
        ):
            if name in data:
                compact_data[name] = data[name]
        return redact(
            {
                "observation_id": observation.observation_id,
                "command_id": observation.command_id,
                "status": observation.status.value,
                "output_artifact_ids": list(
                    observation.output_artifact_ids
                ),
                "normalized_data": compact_data,
                "error": (
                    observation.error.model_dump(mode="json")
                    if observation.error is not None
                    else None
                ),
            }
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


def _worker_tool_contracts(
    spec: AgentSpec,
    task: TaskEnvelope,
) -> dict[str, tuple[CommandKind, dict[str, Any]]]:
    """Return only canonical, allowlisted task-scoped Worker tool contracts."""
    raw = task.constraints.get("worker_tool_contracts", {})
    if not isinstance(raw, dict):
        return {}
    available = {
        str(item).strip()
        for item in task.constraints.get("available_worker_tools", ())
        if str(item).strip()
    }
    contracts: dict[str, tuple[CommandKind, dict[str, Any]]] = {}
    for raw_name, raw_contract in raw.items():
        name = str(raw_name).strip()
        if not name or (available and name not in available):
            continue
        if not isinstance(raw_contract, dict):
            continue
        try:
            kind = CommandKind(str(raw_contract.get("kind", "")).casefold())
        except ValueError:
            continue
        if kind not in spec.allowed_commands:
            continue
        input_schema = raw_contract.get("input_schema", {"type": "object"})
        if not isinstance(input_schema, dict):
            input_schema = {"type": "object"}
        contracts[name] = (kind, input_schema)
    return contracts


def _command_variants(
    spec: AgentSpec,
    task: TaskEnvelope,
) -> tuple[tuple[CommandKind, str, dict[str, Any]], ...]:
    worker = _worker_tool_contracts(spec, task)
    variants: list[tuple[CommandKind, str, dict[str, Any]]] = [
        (kind, name, schema)
        for name, (kind, schema) in sorted(worker.items())
    ]
    covered = {kind for kind, _, _ in variants}
    grants = {item.tool_name: item for item in spec.tool_grants}
    if CommandKind.TOOL in spec.allowed_commands:
        for name, grant in sorted(grants.items()):
            constraints = dict(grant.argument_constraints)
            properties = constraints.get("properties", {})
            if isinstance(properties, dict):
                schema = {
                    "type": "object",
                    "properties": properties,
                    "required": list(constraints.get("required", ())),
                    "additionalProperties": constraints.get(
                        "allowed_properties"
                    ) is None,
                }
            else:
                schema = {"type": "object"}
            variants.append((CommandKind.TOOL, name, schema))
        if grants:
            covered.add(CommandKind.TOOL)
    configured_names = spec.metadata.get("command_names", {})
    if not isinstance(configured_names, dict):
        configured_names = {}
    for kind in spec.allowed_commands:
        if kind in covered:
            continue
        names = configured_names.get(kind.value, kind.value)
        if isinstance(names, str):
            names = (names,)
        if not isinstance(names, (list, tuple)):
            names = (kind.value,)
        for name in dict.fromkeys(
            str(item).strip() for item in names if str(item).strip()
        ):
            variants.append((kind, name, {"type": "object"}))
    return tuple(variants)


def build_command_schema(
    *,
    spec: AgentSpec,
    task: TaskEnvelope,
) -> dict[str, Any]:
    """Build the exact model-facing command contract used by normalization."""
    common_properties: dict[str, Any] = {
        "kind": {"type": "string"},
        "name": {"type": "string", "minLength": 1, "maxLength": 200},
        "arguments": {"type": "object"},
        "input_artifact_ids": {
            "type": "array",
            "items": {"type": "string"},
            "uniqueItems": True,
        },
        "expected_output_schema": {"type": ["string", "null"]},
        "requires_approval": {"type": "boolean"},
        "risk_level": {
            "type": "string",
            "enum": ["low", "medium", "high", "critical"],
        },
        "expires_at": {"type": ["string", "null"]},
        "metadata": {"type": "object"},
    }
    variants = []
    for kind, name, argument_schema in _command_variants(spec, task):
        properties = dict(common_properties)
        properties.update(
            {
                "kind": {"const": kind.value},
                "name": {"const": name},
                "arguments": argument_schema,
            }
        )
        variants.append(
            {
                "type": "object",
                "required": ["kind", "name", "arguments"],
                "properties": properties,
                "additionalProperties": False,
            }
        )
    item_schema: dict[str, Any] = (
        {"oneOf": variants}
        if variants
        else {
            "type": "object",
            "required": ["kind", "name", "arguments"],
            "properties": common_properties,
            "additionalProperties": False,
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "required": ["commands"],
        "properties": {
            "summary": {"type": "string"},
            "commands": {
                "type": "array",
                "minItems": 1,
                "maxItems": spec.max_parallel_commands,
                "items": item_schema,
            },
        },
        "additionalProperties": False,
    }


def _normalized_command_identity(
    raw: dict[str, Any],
    *,
    spec: AgentSpec,
    task: TaskEnvelope,
) -> tuple[CommandKind, str, tuple[str, ...], dict[str, Any] | None]:
    variants = _command_variants(spec, task)
    by_name = {name: (kind, schema) for kind, name, schema in variants}
    by_kind: dict[CommandKind, list[tuple[str, dict[str, Any]]]] = {}
    for kind, name, schema in variants:
        by_kind.setdefault(kind, []).append((name, schema))
    worker_contracts = _worker_tool_contracts(spec, task)
    raw_kind = str(raw.get("kind") or "").strip().casefold()
    raw_name = str(raw.get("name") or "").strip()
    notes: list[str] = []

    kind: CommandKind | None = None
    if raw_kind:
        try:
            kind = CommandKind(raw_kind)
        except ValueError as exc:
            raise ValueError(f"unknown command kind: {raw_kind}") from exc
        if kind not in spec.allowed_commands:
            raise ValueError(f"command kind is not allowed: {kind.value}")
    if kind is None and raw_name:
        matched = by_name.get(raw_name)
        if matched is not None:
            kind = matched[0]
            notes.append("kind inferred from canonical name")
        else:
            aliases = [
                candidate
                for candidate in spec.allowed_commands
                if candidate.value == raw_name.casefold()
            ]
            if len(aliases) == 1:
                kind = aliases[0]
                notes.append("kind inferred from unambiguous command alias")
    if kind is None:
        raise ValueError("command requires kind or a canonical permitted name")

    candidates = by_kind.get(kind, [])
    strict_names = any(name in worker_contracts for name, _ in candidates)
    if not raw_name:
        if len(candidates) == 1:
            raw_name = candidates[0][0]
            notes.append("name inferred from command kind")
        else:
            raw_name = kind.value
            notes.append("name defaulted to command kind")
    elif raw_name not in by_name and strict_names:
        if raw_name.casefold() == kind.value and len(candidates) == 1:
            raw_name = candidates[0][0]
            notes.append("short command alias normalized to canonical name")
        else:
            raise ValueError(f"unknown or ungranted command name: {raw_name}")

    matched = by_name.get(raw_name)
    argument_schema: dict[str, Any] | None = None
    if matched is not None:
        expected_kind, matched_schema = matched
        if expected_kind != kind:
            raise ValueError(
                f"command name {raw_name} requires kind {expected_kind.value}, "
                f"not {kind.value}"
            )
        if raw_name in worker_contracts:
            argument_schema = matched_schema
    elif strict_names:
        raise ValueError(f"unknown or ungranted command name: {raw_name}")
    return kind, raw_name, tuple(notes), argument_schema


def _json_schema_errors(value: Any, schema: dict[str, Any]) -> tuple[str, ...]:
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value),
        key=lambda item: tuple(str(part) for part in item.absolute_path),
    )
    return tuple(
        f"arguments{''.join(f'[{part!r}]' for part in error.absolute_path)}: "
        f"{error.message}"
        for error in errors
    )


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
            kind, name, identity_notes, argument_schema = (
                _normalized_command_identity(raw, spec=spec, task=task)
            )
            raw_arguments = raw.get("arguments", {})
            if not isinstance(raw_arguments, dict):
                raise ValueError("arguments must be an object")
            arguments = redact(dict(raw_arguments))
            if argument_schema is not None:
                argument_errors = _json_schema_errors(
                    arguments,
                    argument_schema,
                )
                if argument_errors:
                    raise ValueError("; ".join(argument_errors))
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
                proposed_at=task.updated_at,
                expires_at=raw.get("expires_at"),
                metadata={
                    **redact(dict(raw.get("metadata", {}) or {})),
                    "round": round_no,
                    **(
                        {"identity_normalization": list(identity_notes)}
                        if identity_notes
                        else {}
                    ),
                },
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
