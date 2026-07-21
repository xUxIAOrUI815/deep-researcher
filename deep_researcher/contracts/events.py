from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier
from .budgets import BudgetUsage
from .errors import ErrorRecord
from .versioning import ComponentVersionSet


class EventType(str, Enum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    SPAN_STARTED = "span_started"
    SPAN_COMPLETED = "span_completed"
    SPAN_FAILED = "span_failed"
    TASK_CREATED = "task_created"
    TASK_STATE_CHANGED = "task_state_changed"
    COMMAND_PROPOSED = "command_proposed"
    COMMAND_APPROVED = "command_approved"
    COMMAND_REJECTED = "command_rejected"
    TOOL_STARTED = "tool_started"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    MODEL_STARTED = "model_started"
    MODEL_COMPLETED = "model_completed"
    MODEL_FAILED = "model_failed"
    ARTIFACT_CREATED = "artifact_created"
    EVIDENCE_CHANGED = "evidence_changed"
    VERIFICATION_COMPLETED = "verification_completed"
    REPORT_CHANGED = "report_changed"
    BUDGET_CHANGED = "budget_changed"
    RETRY_SCHEDULED = "retry_scheduled"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_RESOLVED = "approval_resolved"
    POLICY_DECIDED = "policy_decided"
    DECISION_RECORDED = "decision_recorded"
    CHECKPOINT_CREATED = "checkpoint_created"
    CUSTOM = "custom"


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class SpanKind(str, Enum):
    RUN = "run"
    AGENT = "agent"
    TASK = "task"
    MODEL = "model"
    TOOL = "tool"
    VERIFICATION = "verification"
    STORAGE = "storage"
    EVALUATION = "evaluation"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING = "waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TERMINAL_TYPES = {
    EventType.RUN_STARTED: RunStatus.RUNNING,
    EventType.RUN_COMPLETED: RunStatus.SUCCEEDED,
    EventType.RUN_FAILED: RunStatus.FAILED,
    EventType.RUN_CANCELLED: RunStatus.CANCELLED,
}


class RunEvent(ContractModel):
    """Append-only fact describing an externally observable runtime change."""

    event_id: str = Field(default_factory=lambda: new_id("event"))
    sequence_no: int = Field(ge=1)
    event_type: EventType
    level: EventLevel = EventLevel.INFO
    status: RunStatus
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    span_kind: SpanKind
    correlation_id: str
    causation_event_id: str | None = None
    run_id: str
    thread_id: str
    task_id: str | None = None
    actor_id: str
    producer_id: str
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    state_artifact_id: str | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    attempt: int = Field(default=1, ge=1)
    error: ErrorRecord | None = None
    component_versions: ComponentVersionSet
    occurred_at: datetime = Field(default_factory=utc_now)
    recorded_at: datetime = Field(default_factory=utc_now)
    payload: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "event_id",
        "trace_id",
        "span_id",
        "parent_span_id",
        "correlation_id",
        "causation_event_id",
        "run_id",
        "thread_id",
        "task_id",
        "actor_id",
        "producer_id",
        "state_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("input_artifact_ids", "output_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("artifact references must be unique")
        return value

    @field_validator("occurred_at", "recorded_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "RunEvent":
        if self.parent_span_id == self.span_id:
            raise ValueError("a span cannot parent itself")
        expected = _TERMINAL_TYPES.get(self.event_type)
        if expected is not None and self.status != expected:
            raise ValueError(f"{self.event_type.value} requires {expected.value} status")
        if self.status == RunStatus.FAILED and self.error is None:
            raise ValueError("failure events require an error record")
        if self.status == RunStatus.SUCCEEDED and self.error is not None:
            raise ValueError("successful events cannot include an error")
        if self.recorded_at < self.occurred_at:
            raise ValueError("recorded_at cannot precede occurred_at")
        return self
