from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from deep_researcher.contracts import Command, CommandKind

from .gateway import redact_gateway_value
from .registry import ToolRegistry


class FunctionCallNormalizationError(ValueError):
    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class FunctionCallNormalizer:
    """Normalize common provider function-call envelopes into Command."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    def normalize(
        self,
        payload: Any,
        *,
        run_id: str,
        task_id: str,
        actor_id: str,
        provider: str = "generic",
    ) -> tuple[Command, ...]:
        candidates = self._candidates(payload)
        if not candidates:
            raise FunctionCallNormalizationError(("function-calling response contains no tool calls",))
        commands: list[Command] = []
        errors: list[str] = []
        for index, candidate in enumerate(candidates):
            try:
                name, arguments, provider_call_id = self._extract(candidate)
                definition, _ = self.registry.resolve(name)
                semantic = json.dumps(
                    [run_id, task_id, actor_id, definition.identity, arguments],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                idempotency = hashlib.sha256(semantic.encode()).hexdigest()
                command_seed = hashlib.sha256(f"{semantic}:{provider_call_id}:{index}".encode()).hexdigest()
                commands.append(
                    Command(
                        command_id=f"command_{command_seed[:32]}",
                        run_id=run_id,
                        task_id=task_id,
                        actor_id=actor_id,
                        kind=CommandKind.TOOL,
                        name=definition.name,
                        arguments=redact_gateway_value(arguments),
                        expected_output_schema="ToolExecutionResult@1",
                        idempotency_key=f"function-{idempotency}",
                        requires_approval=definition.requires_approval,
                        risk_level=definition.risk_level.value,
                        metadata={
                            "protocol": "function_calling",
                            "provider": provider,
                            "provider_call_id": provider_call_id,
                            "tool_version": definition.version,
                        },
                    )
                )
            except Exception as exc:
                errors.append(f"tool_calls[{index}]: {exc}")
        if errors:
            raise FunctionCallNormalizationError(tuple(errors))
        return tuple(commands)

    def schemas(self, *, active_only: bool = True) -> tuple[dict[str, Any], ...]:
        return tuple(
            {
                "type": "function",
                "function": {
                    "name": definition.name,
                    "description": definition.description,
                    "parameters": definition.input_schema,
                    "strict": True,
                    "x-tool-version": definition.version,
                    "x-risk-level": definition.risk_level.value,
                    "x-requires-approval": definition.requires_approval,
                },
            }
            for definition in self.registry.definitions(active_only=active_only)
        )

    @staticmethod
    def _candidates(payload: Any) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if not isinstance(payload, dict):
            return []
        if isinstance(payload.get("tool_calls"), list):
            return [item for item in payload["tool_calls"] if isinstance(item, dict)]
        choices = payload.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict) and isinstance(message.get("tool_calls"), list):
                return [item for item in message["tool_calls"] if isinstance(item, dict)]
        if payload.get("type") in {"function", "tool_use"} or "function" in payload or "name" in payload:
            return [payload]
        return []

    @staticmethod
    def _extract(candidate: dict[str, Any]) -> tuple[str, dict[str, Any], str]:
        provider_call_id = str(candidate.get("id") or candidate.get("call_id") or "provider-call")
        function = candidate.get("function") if isinstance(candidate.get("function"), dict) else candidate
        name = str(function.get("name") or candidate.get("name") or "").strip()
        if not name:
            raise ValueError("function name is required")
        arguments: Any = function.get("arguments")
        if arguments is None:
            arguments = candidate.get("input", {})
        if isinstance(arguments, str):
            text = arguments.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
            try:
                arguments = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"function arguments are invalid JSON: {exc}") from exc
        if not isinstance(arguments, dict):
            raise ValueError("function arguments must be an object")
        return name, arguments, provider_call_id
