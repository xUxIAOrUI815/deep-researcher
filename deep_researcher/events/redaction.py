from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any

from deep_researcher.contracts import EventType, RunEvent


_HIDDEN_REASONING_KEYS = frozenset({"chain_of_thought", "cot", "hidden_reasoning", "private_reasoning", "reasoning_content"})
_DEFAULT_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "cookie",
        "credential",
        "password",
        "refresh_token",
        "secret",
        "session_cookie",
        "token",
    }
)
_SECRET_PATTERNS = (
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"\b(?:sk|pk|rk)-[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)([?&](?:api[_-]?key|token|secret)=)[^&\s]+"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|secret|password)=[^&\s]+"),
)


@dataclass(frozen=True)
class RedactionPolicy:
    replacement: str = "[REDACTED]"
    sensitive_keys: frozenset[str] = field(default_factory=lambda: _DEFAULT_SENSITIVE_KEYS)

    def redact_text(self, value: str) -> str:
        redacted = value
        for pattern in _SECRET_PATTERNS:
            if pattern.pattern.startswith("(?i)([?&]"):
                redacted = pattern.sub(lambda match: f"{match.group(1)}{self.replacement}", redacted)
            else:
                redacted = pattern.sub(self.replacement, redacted)
        return redacted

    def redact(self, value: Any) -> Any:
        if isinstance(value, Enum):
            return value
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            result: dict[str, Any] = {}
            for raw_key, nested in value.items():
                key = str(raw_key)
                normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
                if normalized in _HIDDEN_REASONING_KEYS:
                    continue
                if normalized in self.sensitive_keys or any(marker in normalized for marker in ("password", "secret", "api_key", "access_token")):
                    result[key] = self.replacement
                else:
                    result[key] = self.redact(nested)
            return result
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return type(value)(self.redact(item) for item in value)
        return value

    def sanitize_decision_payload(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        allowed = {
            "observation_summary",
            "selected_command_ids",
            "alternatives_considered",
            "policy_checks",
            "outcome_artifact_ids",
            "stop_reason",
            "message",
            "metadata",
        }
        selected = {key: value for key, value in payload.items() if key in allowed}
        metadata = selected.get("metadata")
        if isinstance(metadata, Mapping):
            selected["metadata"] = {
                key: value
                for key, value in metadata.items()
                if not any(
                    marker in str(key).strip().lower().replace("-", "_")
                    for marker in ("raw_model", "model_response", "completion", "prompt", "transcript", "reasoning")
                )
            }
        return self.redact(selected)

    def redact_event(self, event: RunEvent) -> RunEvent:
        values = event.model_dump(mode="python")
        payload = values.get("payload", {})
        values["payload"] = (
            self.sanitize_decision_payload(payload)
            if event.event_type == EventType.DECISION_RECORDED
            else self.redact(payload)
        )
        if values.get("error"):
            values["error"] = self.redact(values["error"])
        return RunEvent.model_validate(values)
