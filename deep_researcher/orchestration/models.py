from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    Budget,
    BudgetUsage,
    ContractModel,
    TaskEnvelope,
    TaskStatus,
    utc_now,
)
from deep_researcher.contracts._base import validate_identifier


class RunControlStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class ApprovalStatus(str, Enum):
    REQUESTED = "requested"
    APPROVED = "approved"
    REJECTED = "rejected"


class SchedulerEventType(str, Enum):
    RUN_CREATED = "run_created"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    RUN_CANCELLED = "run_cancelled"
    RUN_COMPLETED = "run_completed"
    TASK_CREATED = "task_created"
    TASK_SPLIT = "task_split"
    TASK_MERGED = "task_merged"
    TASK_READY = "task_ready"
    TASK_CLAIMED = "task_claimed"
    TASK_HEARTBEAT = "task_heartbeat"
    TASK_DEFERRED = "task_deferred"
    TASK_PRUNED = "task_pruned"
    TASK_CANCELLED = "task_cancelled"
    TASK_RETRIED = "task_retried"
    TASK_FAILED = "task_failed"
    TASK_COMPLETED = "task_completed"
    TASK_PAUSED = "task_paused"
    TASK_RESUMED = "task_resumed"
    TASK_EDITED = "task_edited"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_APPROVED = "approval_approved"
    APPROVAL_REJECTED = "approval_rejected"
    LEASE_RECOVERED = "lease_recovered"
    BUDGET_UPDATED = "budget_updated"
    PROJECTION_REBUILT = "projection_rebuilt"


class SchedulerEvent(ContractModel):
    event_id: str
    mutation_id: str
    fingerprint: str = Field(min_length=64, max_length=64)
    run_id: str
    sequence_no: int = Field(ge=1)
    event_type: SchedulerEventType
    task_id: str | None = None
    actor_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("event_id", "mutation_id", "run_id", "task_id", "actor_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class ApprovalRecord(ContractModel):
    approval_id: str
    task_id: str
    requested_by: str
    reason: str = Field(min_length=1, max_length=2000)
    status: ApprovalStatus = ApprovalStatus.REQUESTED
    requested_at: datetime = Field(default_factory=utc_now)
    resolved_by: str | None = None
    resolution_note: str | None = Field(default=None, max_length=2000)
    resolved_at: datetime | None = None

    @field_validator("approval_id", "task_id", "requested_by", "resolved_by")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _resolution_consistent(self) -> "ApprovalRecord":
        resolved = self.status in {ApprovalStatus.APPROVED, ApprovalStatus.REJECTED}
        if resolved != (self.resolved_at is not None and self.resolved_by is not None):
            raise ValueError("resolved approval requires resolver and timestamp")
        return self


class RunControl(ContractModel):
    run_id: str
    status: RunControlStatus = RunControlStatus.ACTIVE
    max_concurrency: int = Field(default=4, gt=0, le=1024)
    projection_revision: int = Field(default=0, ge=0)
    cancellation_reason: str | None = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return validate_identifier(value)


class TaskRecord(ContractModel):
    envelope: TaskEnvelope
    revision: int = Field(default=1, ge=1)
    available_at: datetime = Field(default_factory=utc_now)
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    result_id: str | None = None
    output_artifact_ids: tuple[str, ...] = ()
    error_ref: str | None = None
    merged_into_task_id: str | None = None
    approval: ApprovalRecord | None = None
    defer_reason: str | None = Field(default=None, max_length=2000)
    pause_reason: str | None = Field(default=None, max_length=2000)
    paused_by_run: bool = False
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("available_at", "lease_expires_at", "last_heartbeat_at", "updated_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("scheduler timestamps must be timezone-aware")
        return value

    @field_validator("lease_owner", "result_id", "output_artifact_ids", "error_ref", "merged_into_task_id")
    @classmethod
    def _references(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("artifact references must be unique")
            return value
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "TaskRecord":
        if (self.lease_owner is None) != (self.lease_expires_at is None):
            raise ValueError("task lease owner and expiry must be set together")
        if self.lease_owner is not None and self.envelope.status != TaskStatus.RUNNING:
            raise ValueError("only running tasks may hold a lease")
        if self.envelope.status == TaskStatus.WAITING_APPROVAL:
            if self.approval is None or self.approval.status != ApprovalStatus.REQUESTED:
                raise ValueError("waiting-approval task requires a pending approval")
        if self.envelope.status == TaskStatus.MERGED and self.merged_into_task_id is None:
            raise ValueError("merged task requires a target task")
        return self

    @property
    def task_id(self) -> str:
        return self.envelope.task_id

    @property
    def run_id(self) -> str:
        return self.envelope.run_id


class TaskLease(ContractModel):
    task: TaskEnvelope
    worker_id: str
    lease_expires_at: datetime
    projection_revision: int = Field(ge=1)

    @field_validator("worker_id")
    @classmethod
    def _worker_id(cls, value: str) -> str:
        return validate_identifier(value)


class SchedulerSnapshot(ContractModel):
    control: RunControl
    tasks: tuple[TaskRecord, ...]
    captured_at: datetime = Field(default_factory=utc_now)

    @property
    def by_id(self) -> dict[str, TaskRecord]:
        return {record.task_id: record for record in self.tasks}


class SchedulerTaskQuery(ContractModel):
    run_id: str
    after_task_id: str | None = None
    statuses: tuple[TaskStatus, ...] = ()
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("run_id", "after_task_id")
    @classmethod
    def _task_query_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class SchedulerTaskPage(ContractModel):
    items: tuple[TaskRecord, ...]
    next_after_task_id: str | None = None

    @field_validator("next_after_task_id")
    @classmethod
    def _task_cursor(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class SchedulerEventQuery(ContractModel):
    run_id: str
    after_sequence: int = Field(default=0, ge=0)
    event_types: tuple[SchedulerEventType, ...] = ()
    task_id: str | None = None
    limit: int = Field(default=100, ge=1, le=1000)

    @field_validator("run_id", "task_id")
    @classmethod
    def _event_query_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class SchedulerEventPage(ContractModel):
    items: tuple[SchedulerEvent, ...]
    next_after_sequence: int | None = Field(default=None, ge=1)


class ThinRuntimeState(ContractModel):
    run_id: str
    projection_revision: int = Field(ge=0)
    run_status: RunControlStatus
    active_task_ids: tuple[str, ...] = ()
    ready_task_ids: tuple[str, ...] = ()
    waiting_approval_task_ids: tuple[str, ...] = ()
    budget_usage: BudgetUsage = Field(default_factory=BudgetUsage)
    artifact_ids: tuple[str, ...] = ()
    error_ref: str | None = None
    final_report_artifact_id: str | None = None
    task_counts: dict[str, int] = Field(default_factory=dict)

    @field_validator("run_id", "error_ref", "final_report_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("active_task_ids", "ready_task_ids", "waiting_approval_task_ids", "artifact_ids")
    @classmethod
    def _id_lists(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(value) != len(set(value)):
            raise ValueError("thin-state references must be unique")
        return value


class TaskEdit(ContractModel):
    title: str | None = Field(default=None, min_length=1, max_length=300)
    goal: str | None = Field(default=None, min_length=1, max_length=4000)
    constraints: dict[str, Any] | None = None
    dependency_task_ids: tuple[str, ...] | None = None
    input_artifact_ids: tuple[str, ...] | None = None
    expected_output_schema: str | None = Field(default=None, min_length=1, max_length=255)
    budget: Budget | None = None
    priority: float | None = Field(default=None, ge=0.0, le=1.0)
    deadline: datetime | None = None
    clear_deadline: bool = False
    assigned_actor_id: str | None = None
    clear_assigned_actor: bool = False
    tags: tuple[str, ...] | None = None

    @model_validator(mode="after")
    def _not_empty(self) -> "TaskEdit":
        meaningful = {
            name
            for name in self.model_fields_set
            if name not in {"clear_deadline", "clear_assigned_actor"}
        }
        if not meaningful and not self.clear_deadline and not self.clear_assigned_actor:
            raise ValueError("task edit requires at least one field")
        return self


class TaskCompletion(ContractModel):
    result_id: str
    output_artifact_ids: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @field_validator("result_id")
    @classmethod
    def _result_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("output_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(value) != len(set(value)):
            raise ValueError("completion artifacts must be unique")
        return value


class RecoveryReport(ContractModel):
    run_id: str
    recovered_task_ids: tuple[str, ...] = ()
    failed_task_ids: tuple[str, ...] = ()
    promoted_task_ids: tuple[str, ...] = ()
    projection_revision: int = Field(ge=0)

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("recovered_task_ids", "failed_task_ids", "promoted_task_ids")
    @classmethod
    def _task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return value


def thin_state_from_snapshot(
    snapshot: SchedulerSnapshot,
    *,
    final_report_artifact_id: str | None = None,
) -> ThinRuntimeState:
    active: list[str] = []
    ready: list[str] = []
    waiting: list[str] = []
    artifacts: list[str] = []
    counts: dict[str, int] = {}
    usage = BudgetUsage()
    error_ref: str | None = None
    for record in snapshot.tasks:
        status = record.envelope.status.value
        counts[status] = counts.get(status, 0) + 1
        if record.envelope.status == TaskStatus.RUNNING:
            active.append(record.task_id)
        elif record.envelope.status == TaskStatus.READY:
            ready.append(record.task_id)
        elif record.envelope.status == TaskStatus.WAITING_APPROVAL:
            waiting.append(record.task_id)
        artifacts.extend(record.output_artifact_ids)
        usage = usage.plus(
            input_tokens=record.budget_usage.input_tokens,
            output_tokens=record.budget_usage.output_tokens,
            cost_usd=record.budget_usage.cost_usd,
            wall_time_seconds=record.budget_usage.wall_time_seconds,
            model_calls=record.budget_usage.model_calls,
            tool_calls=record.budget_usage.tool_calls,
            search_calls=record.budget_usage.search_calls,
            retries=record.budget_usage.retries,
            errors=record.budget_usage.errors,
        )
        error_ref = record.error_ref or error_ref
    return ThinRuntimeState(
        run_id=snapshot.control.run_id,
        projection_revision=snapshot.control.projection_revision,
        run_status=snapshot.control.status,
        active_task_ids=tuple(sorted(active)),
        ready_task_ids=tuple(sorted(ready)),
        waiting_approval_task_ids=tuple(sorted(waiting)),
        budget_usage=usage,
        artifact_ids=tuple(dict.fromkeys(artifacts)),
        error_ref=error_ref,
        final_report_artifact_id=final_report_artifact_id,
        task_counts=counts,
    )
