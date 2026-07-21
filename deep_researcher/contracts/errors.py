from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier


class ErrorCategory(str, Enum):
    USER_INPUT = "user_input"
    POLICY_DENIED = "policy_denied"
    APPROVAL_REQUIRED = "approval_required"
    BUDGET_EXHAUSTED = "budget_exhausted"
    TRANSIENT_PROVIDER = "transient_provider"
    PERMANENT_PROVIDER = "permanent_provider"
    PROTOCOL = "protocol"
    SCHEMA_VALIDATION = "schema_validation"
    VERIFICATION = "verification"
    STORAGE = "storage"
    CONFLICT = "conflict"
    CANCELLED = "cancelled"
    INTERNAL = "internal"


class ErrorRecord(ContractModel):
    error_id: str = Field(default_factory=lambda: new_id("error"))
    category: ErrorCategory
    code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2000)
    retryable: bool = False
    fatal: bool = False
    attempt: int = Field(default=1, ge=1)
    actor_id: str | None = None
    task_id: str | None = None
    command_id: str | None = None
    detail_artifact_id: str | None = None
    occurred_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("error_id", "actor_id", "task_id", "command_id", "detail_artifact_id")
    @classmethod
    def _valid_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("occurred_at")
    @classmethod
    def _valid_occurred_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        return value
