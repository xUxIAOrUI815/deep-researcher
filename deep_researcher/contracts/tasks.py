from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier
from .budgets import Budget, BudgetUsage
from .errors import ErrorRecord


class TaskKind(str, Enum):
    ROOT = "root"
    SOURCE_DISCOVERY = "source_discovery"
    RESEARCH = "research"
    GAP = "gap"
    CONFLICT = "conflict"
    VERIFICATION = "verification"
    SECTION_SUPPORT = "section_support"
    SYNTHESIS = "synthesis"
    REVIEW = "review"
    REPAIR = "repair"


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    DEFERRED = "deferred"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PRUNED = "pruned"
    MERGED = "merged"


_ALLOWED_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING: frozenset({TaskStatus.READY, TaskStatus.DEFERRED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}),
    TaskStatus.READY: frozenset({TaskStatus.RUNNING, TaskStatus.DEFERRED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}),
    TaskStatus.RUNNING: frozenset({TaskStatus.PAUSED, TaskStatus.WAITING_APPROVAL, TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.PAUSED: frozenset({TaskStatus.READY, TaskStatus.CANCELLED}),
    TaskStatus.WAITING_APPROVAL: frozenset({TaskStatus.READY, TaskStatus.CANCELLED, TaskStatus.FAILED}),
    TaskStatus.DEFERRED: frozenset({TaskStatus.PENDING, TaskStatus.CANCELLED, TaskStatus.PRUNED}),
    TaskStatus.FAILED: frozenset({TaskStatus.PENDING, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED: frozenset(),
    TaskStatus.CANCELLED: frozenset(),
    TaskStatus.PRUNED: frozenset(),
    TaskStatus.MERGED: frozenset(),
}


class TaskEnvelope(ContractModel):
    task_id: str = Field(default_factory=lambda: new_id("task"))
    run_id: str
    parent_task_id: str | None = None
    dependency_task_ids: tuple[str, ...] = ()
    kind: TaskKind
    status: TaskStatus = TaskStatus.PENDING
    title: str = Field(min_length=1, max_length=300)
    goal: str = Field(min_length=1, max_length=4000)
    constraints: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    expected_output_schema: str = Field(min_length=1, max_length=255)
    budget: Budget
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline: datetime | None = None
    attempt: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    created_by: str
    assigned_actor_id: str | None = None
    tags: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("task_id", "run_id", "parent_task_id", "created_by", "assigned_actor_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("dependency_task_ids", "input_artifact_ids")
    @classmethod
    def _id_tuples(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("identifier list must not contain duplicates")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "TaskEnvelope":
        if self.task_id == self.parent_task_id or self.task_id in self.dependency_task_ids:
            raise ValueError("task cannot depend on itself")
        if self.parent_task_id is not None and self.parent_task_id in self.dependency_task_ids:
            raise ValueError("parent task must not also be a dependency")
        if self.attempt > self.max_attempts:
            raise ValueError("attempt cannot exceed max_attempts")
        if self.deadline is not None and (self.deadline.tzinfo is None or self.deadline.utcoffset() is None):
            raise ValueError("deadline must be timezone-aware")
        return self

    def transition(self, target: TaskStatus, *, at: datetime | None = None) -> "TaskEnvelope":
        allowed = _ALLOWED_TRANSITIONS[self.status]
        if target not in allowed:
            raise ValueError(f"invalid task transition: {self.status.value} -> {target.value}")
        updates: dict[str, Any] = {"status": target, "updated_at": at or utc_now()}
        if target == TaskStatus.RUNNING:
            updates["attempt"] = self.attempt + 1
        return self.model_copy(update=updates)


class TaskResultStatus(str, Enum):
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"
    DEFERRED = "deferred"
    REJECTED = "rejected"


class TaskResult(ContractModel):
    result_id: str = Field(default_factory=lambda: new_id("result"))
    task_id: str
    run_id: str
    actor_id: str
    status: TaskResultStatus
    output_artifact_ids: tuple[str, ...] = ()
    summary: str = Field(min_length=1, max_length=4000)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: ErrorRecord | None = None
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator("result_id", "task_id", "run_id", "actor_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("output_artifact_ids")
    @classmethod
    def _output_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("output artifacts must be unique")
        return value

    @model_validator(mode="after")
    def _result_consistency(self) -> "TaskResult":
        failed = self.status in {TaskResultStatus.FAILED, TaskResultStatus.CANCELLED, TaskResultStatus.REJECTED}
        if failed and self.error is None:
            raise ValueError("failed, cancelled, or rejected results require an error")
        if self.status == TaskResultStatus.SUCCEEDED and self.error is not None:
            raise ValueError("successful result cannot contain an error")
        if self.completed_at < self.started_at:
            raise ValueError("completed_at cannot precede started_at")
        return self
