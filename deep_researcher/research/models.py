from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    Budget,
    BudgetUsage,
    ContractModel,
    TaskKind,
    TaskResultStatus,
    utc_now,
)
from deep_researcher.contracts._base import (
    new_id,
    validate_identifier,
)


_RESEARCH_TASK_KINDS = {
    TaskKind.SOURCE_DISCOVERY,
    TaskKind.RESEARCH,
    TaskKind.GAP,
    TaskKind.CONFLICT,
    TaskKind.VERIFICATION,
    TaskKind.SECTION_SUPPORT,
    TaskKind.REPAIR,
}


class SupervisorPlanAction(str, Enum):
    DECOMPOSE = "decompose"
    REPLAN = "replan"
    REQUEST_APPROVAL = "request_approval"
    CONVERGED = "converged"
    STOP = "stop"


class ConvergenceAction(str, Enum):
    CONTINUE = "continue"
    REPLAN = "replan"
    AWAIT_APPROVAL = "await_approval"
    COMPLETE = "complete"
    COMPLETE_WITH_GAPS = "complete_with_gaps"
    STOP_LOW_GAIN = "stop_low_gain"
    STOP_BUDGET = "stop_budget"
    STOP_MAX_CYCLES = "stop_max_cycles"
    CANCEL = "cancel"


class DedupKind(str, Enum):
    QUERY = "query"
    SOURCE = "source"
    TASK = "task"
    KNOWLEDGE = "knowledge"
    ARTIFACT = "artifact"


class DedupStatus(str, Enum):
    CLAIMED = "claimed"
    COMPLETED = "completed"
    FAILED = "failed"


class DedupDecisionKind(str, Enum):
    NEW = "new"
    OWNED_REPLAY = "owned_replay"
    DUPLICATE_IN_PROGRESS = "duplicate_in_progress"
    DUPLICATE_COMPLETED = "duplicate_completed"
    RETRY = "retry"


class SupervisorTaskProposal(ContractModel):
    proposal_key: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    kind: TaskKind
    title: str = Field(min_length=1, max_length=300)
    goal: str = Field(min_length=1, max_length=4000)
    constraints: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    expected_output_schema: str = Field(min_length=1, max_length=255)
    budget: Budget
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    deadline: datetime | None = None
    dependency_keys: tuple[str, ...] = ()
    max_attempts: int = Field(default=3, ge=1, le=20)
    assigned_actor_id: str | None = None
    tags: tuple[str, ...] = ()

    @field_validator("input_artifact_ids")
    @classmethod
    def _artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(value) != len(set(value)):
            raise ValueError("task proposal input artifacts must be unique")
        return value

    @field_validator("assigned_actor_id")
    @classmethod
    def _actor(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("dependency_keys", "tags")
    @classmethod
    def _unique_strings(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in value if item.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("task proposal string lists must be unique")
        return normalized

    @field_validator("deadline")
    @classmethod
    def _deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and (
            value.tzinfo is None or value.utcoffset() is None
        ):
            raise ValueError("task proposal deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _research_only(self) -> "SupervisorTaskProposal":
        if self.kind not in _RESEARCH_TASK_KINDS:
            raise ValueError(
                f"Supervisor may schedule research work only, not {self.kind.value}"
            )
        if self.proposal_key in self.dependency_keys:
            raise ValueError("task proposal cannot depend on itself")
        return self


class SupervisorPlan(ContractModel):
    plan_id: str = Field(default_factory=lambda: new_id("plan"))
    run_id: str
    root_task_id: str
    cycle: int = Field(ge=0)
    action: SupervisorPlanAction
    objective: str = Field(min_length=1, max_length=4000)
    tasks: tuple[SupervisorTaskProposal, ...] = ()
    decision_summary: str = Field(min_length=1, max_length=4000)
    approval_reason: str | None = Field(default=None, max_length=2000)
    stop_reason: str | None = Field(default=None, max_length=2000)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("plan_id", "run_id", "root_task_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _consistent(self) -> "SupervisorPlan":
        task_actions = {
            SupervisorPlanAction.DECOMPOSE,
            SupervisorPlanAction.REPLAN,
        }
        if self.action in task_actions and not self.tasks:
            raise ValueError("decomposition and replanning require tasks")
        if self.action not in task_actions and self.tasks:
            raise ValueError("non-planning supervisor actions cannot contain tasks")
        if self.action == SupervisorPlanAction.REQUEST_APPROVAL:
            if not self.approval_reason:
                raise ValueError("approval action requires a reason")
        elif self.approval_reason is not None:
            raise ValueError("approval_reason is valid only for approval actions")
        if self.action in {
            SupervisorPlanAction.CONVERGED,
            SupervisorPlanAction.STOP,
        }:
            if not self.stop_reason:
                raise ValueError("terminal supervisor action requires a stop reason")
        elif self.stop_reason is not None:
            raise ValueError("stop_reason is valid only for terminal actions")
        keys = [item.proposal_key for item in self.tasks]
        if len(keys) != len(set(keys)):
            raise ValueError("supervisor task proposal keys must be unique")
        known = set(keys)
        for item in self.tasks:
            unknown = set(item.dependency_keys) - known
            if unknown:
                raise ValueError(
                    f"task {item.proposal_key} has unknown dependencies: "
                    f"{sorted(unknown)}"
                )
        self._validate_acyclic()
        return self

    def _validate_acyclic(self) -> None:
        dependencies = {
            item.proposal_key: set(item.dependency_keys)
            for item in self.tasks
        }
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key: str) -> None:
            if key in visiting:
                raise ValueError("supervisor plan task dependencies contain a cycle")
            if key in visited:
                return
            visiting.add(key)
            for dependency in dependencies.get(key, ()):
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in dependencies:
            visit(key)


class TaskReservation(ContractModel):
    run_id: str
    task_id: str
    canonical_task_id: str
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    inserted: bool

    @field_validator("run_id", "task_id", "canonical_task_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class DedupDecision(ContractModel):
    run_id: str
    kind: DedupKind
    key_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    normalized_key: str = Field(min_length=1, max_length=4000)
    owner_task_id: str
    requesting_task_id: str
    decision: DedupDecisionKind
    artifact_ids: tuple[str, ...] = ()
    lease_expires_at: datetime | None = None

    @field_validator("run_id", "owner_task_id", "requesting_task_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class InformationGainAssessment(ContractModel):
    task_id: str
    command_id: str
    score: float = Field(ge=0.0, le=1.0)
    new_artifact_ids: tuple[str, ...] = ()
    new_source_keys: tuple[str, ...] = ()
    new_query_keys: tuple[str, ...] = ()
    new_knowledge_keys: tuple[str, ...] = ()
    duplicate: bool = False
    decision_summary: str = Field(min_length=1, max_length=2000)

    @field_validator("task_id", "command_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("new_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class ResearchWorkerResult(ContractModel):
    worker_result_id: str = Field(default_factory=lambda: new_id("worker_result"))
    run_id: str
    task_id: str
    worker_id: str
    task_result_id: str
    task_attempt: int = Field(ge=1)
    status: TaskResultStatus
    output_artifact_ids: tuple[str, ...] = ()
    query_keys: tuple[str, ...] = ()
    source_keys: tuple[str, ...] = ()
    knowledge_keys: tuple[str, ...] = ()
    information_gain: float = Field(ge=0.0)
    command_count: int = Field(ge=0)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    summary: str = Field(min_length=1, max_length=4000)
    retry_scheduled: bool = False
    error_ref: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "worker_result_id",
        "run_id",
        "task_id",
        "worker_id",
        "task_result_id",
        "error_ref",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("output_artifact_ids")
    @classmethod
    def _artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))

    @field_validator("query_keys", "source_keys", "knowledge_keys")
    @classmethod
    def _keys(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item for item in value if item))


class MergedResearchResult(ContractModel):
    merge_id: str = Field(default_factory=lambda: new_id("research_merge"))
    run_id: str
    worker_result_ids: tuple[str, ...]
    task_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    query_keys: tuple[str, ...]
    source_keys: tuple[str, ...]
    knowledge_keys: tuple[str, ...]
    total_information_gain: float = Field(ge=0.0)
    merge_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("merge_id", "run_id", "merge_artifact_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("worker_result_ids", "task_ids", "artifact_ids")
    @classmethod
    def _ids_list(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class ConvergencePolicy(ContractModel):
    required_section_ids: tuple[str, ...] = ()
    minimum_section_coverage: float = Field(default=0.85, ge=0.0, le=1.0)
    minimum_citation_coverage: float = Field(default=1.0, ge=0.0, le=1.0)
    low_gain_threshold: float = Field(default=0.03, ge=0.0, le=1.0)
    max_low_gain_cycles: int = Field(default=2, ge=1, le=100)
    max_cycles: int = Field(default=12, ge=1, le=1000)
    run_budget: Budget
    minimum_replan_token_reserve: int = Field(
        default=72_000,
        ge=0,
        le=10_000_000,
    )
    estimated_tokens_per_planned_task: int = Field(
        default=48_000,
        ge=1_000,
        le=1_000_000,
    )
    max_gap_replan_cycles: int = Field(default=3, ge=0, le=100)
    minimum_reportable_verified_claims: int = Field(default=1, ge=1, le=1000)
    stop_on_any_severe_conflict: bool = True
    require_no_high_impact_blockers: bool = True

    @field_validator("required_section_ids")
    @classmethod
    def _sections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class ConvergenceSnapshot(ContractModel):
    run_id: str
    cycle: int = Field(ge=0)
    required_section_ids: tuple[str, ...]
    complete_section_ids: tuple[str, ...]
    coverage_gap_section_ids: tuple[str, ...]
    blocked_high_impact_claim_ids: tuple[str, ...]
    severe_conflict_ids: tuple[str, ...]
    verified_claim_count: int = Field(default=0, ge=0)
    verified_citation_count: int = Field(default=0, ge=0)
    active_task_ids: tuple[str, ...]
    pending_task_ids: tuple[str, ...]
    runnable_task_ids: tuple[str, ...] = ()
    dependency_blocked_task_ids: tuple[str, ...] = ()
    failed_task_ids: tuple[str, ...]
    waiting_approval_task_ids: tuple[str, ...]
    low_gain_cycles: int = Field(ge=0)
    latest_information_gain: float = Field(ge=0.0)
    budget_usage: BudgetUsage
    budget_exhausted: bool
    cancelled: bool
    captured_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "run_id",
        "required_section_ids",
        "complete_section_ids",
        "coverage_gap_section_ids",
        "blocked_high_impact_claim_ids",
        "severe_conflict_ids",
        "active_task_ids",
        "pending_task_ids",
        "runnable_task_ids",
        "dependency_blocked_task_ids",
        "failed_task_ids",
        "waiting_approval_task_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class ConvergenceDecision(ContractModel):
    decision_id: str = Field(default_factory=lambda: new_id("convergence"))
    run_id: str
    cycle: int = Field(ge=0)
    action: ConvergenceAction
    reasons: tuple[str, ...]
    snapshot: ConvergenceSnapshot
    decision_artifact_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("decision_id", "run_id", "decision_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("reasons")
    @classmethod
    def _reasons(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(item.strip() for item in value if item.strip()))
        if not normalized:
            raise ValueError("convergence decision requires at least one reason")
        return normalized


class ResearchRunOutcome(ContractModel):
    run_id: str
    action: ConvergenceAction
    cycles: int = Field(ge=0)
    decision_artifact_id: str
    merged_result_artifact_ids: tuple[str, ...] = ()
    completed_task_ids: tuple[str, ...] = ()
    failed_task_ids: tuple[str, ...] = ()
    error_refs: tuple[str, ...] = ()
    waiting_approval_task_ids: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "run_id",
        "decision_artifact_id",
        "merged_result_artifact_ids",
        "completed_task_ids",
        "failed_task_ids",
        "error_refs",
        "waiting_approval_task_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)
