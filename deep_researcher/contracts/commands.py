from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier
from .budgets import BudgetUsage
from .errors import ErrorRecord


class CommandKind(str, Enum):
    SEARCH = "search"
    READ = "read"
    EXTRACT = "extract"
    DELEGATE = "delegate"
    COMPARE = "compare"
    VERIFY_SOURCE = "verify_source"
    SYNTHESIZE = "synthesize"
    REVIEW = "review"
    TOOL = "tool"
    REQUEST_APPROVAL = "request_approval"
    STOP = "stop"


class CommandStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTING = "executing"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ObservationStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class Command(ContractModel):
    command_id: str = Field(default_factory=lambda: new_id("command"))
    run_id: str
    task_id: str
    actor_id: str
    kind: CommandKind
    name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    expected_output_schema: str | None = Field(default=None, max_length=255)
    idempotency_key: str = Field(min_length=8, max_length=255)
    requires_approval: bool = False
    risk_level: str = Field(default="low", pattern=r"^(low|medium|high|critical)$")
    proposed_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("command_id", "run_id", "task_id", "actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("input_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("input artifacts must be unique")
        return value

    @model_validator(mode="after")
    def _approval_for_risk(self) -> "Command":
        if self.risk_level in {"high", "critical"} and not self.requires_approval:
            raise ValueError("high-risk commands require approval")
        if self.expires_at is not None and self.expires_at <= self.proposed_at:
            raise ValueError("command expiry must be after proposal time")
        return self


class Observation(ContractModel):
    observation_id: str = Field(default_factory=lambda: new_id("observation"))
    command_id: str
    run_id: str
    task_id: str
    actor_id: str
    status: ObservationStatus
    output_artifact_ids: tuple[str, ...] = ()
    normalized_data: dict[str, Any] = Field(default_factory=dict)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    error: ErrorRecord | None = None
    attempt: int = Field(default=1, ge=1)
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator("observation_id", "command_id", "run_id", "task_id", "actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("output_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("output artifacts must be unique")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "Observation":
        failed = self.status in {
            ObservationStatus.FAILED,
            ObservationStatus.TIMEOUT,
            ObservationStatus.CANCELLED,
            ObservationStatus.REJECTED,
        }
        if failed and self.error is None:
            raise ValueError("non-success observation requires an error")
        if self.status == ObservationStatus.SUCCEEDED and self.error is not None:
            raise ValueError("successful observation cannot include an error")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self


class AgentDecisionSummary(ContractModel):
    decision_id: str = Field(default_factory=lambda: new_id("decision"))
    run_id: str
    task_id: str
    actor_id: str
    observation_summary: str = Field(min_length=1, max_length=4000)
    selected_command_ids: tuple[str, ...]
    alternatives_considered: tuple[str, ...] = ()
    policy_checks: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("decision_id", "run_id", "task_id", "actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("selected_command_ids")
    @classmethod
    def _command_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("decision must select at least one command")
        for item in value:
            validate_identifier(item)
        return value
