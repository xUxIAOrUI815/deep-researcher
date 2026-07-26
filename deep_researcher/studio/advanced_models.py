from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    AgentSpec,
    BudgetUsage,
    ComponentKind,
    ComponentVersionSet,
    ContractModel,
    ErrorRecord,
    TaskEnvelope,
    new_id,
    utc_now,
)
from deep_researcher.contracts._base import validate_identifier


class ReplayMode(str, Enum):
    SAVED_TOOL_RESULTS = "saved_tool_results"
    LIVE_ENVIRONMENT = "live_environment"


class ReplayRequestStatus(str, Enum):
    WAITING_APPROVAL = "waiting_approval"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReplayAttemptStatus(str, Enum):
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABANDONED = "abandoned"


class ReplayJournalKind(str, Enum):
    CREATED = "created"
    APPROVAL_GRANTED = "approval_granted"
    QUEUED = "queued"
    ATTEMPT_STARTED = "attempt_started"
    ATTEMPT_WAITING_APPROVAL = "attempt_waiting_approval"
    ATTEMPT_SUCCEEDED = "attempt_succeeded"
    ATTEMPT_FAILED = "attempt_failed"
    ATTEMPT_ABANDONED = "attempt_abandoned"


class ReplayModelExchange(ContractModel):
    """One sealed model response consumed by a network-free replay."""

    operation: str = Field(pattern=r"^(complete|repair)$")
    model_version: str
    prompt_version: str
    response: dict[str, Any]
    request_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )


class ReplayToolExchange(ContractModel):
    """One sealed tool observation, keyed by semantic command content."""

    command_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    command_kind: str
    tool_name: str = Field(min_length=1, max_length=200)
    arguments: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    observation: dict[str, Any]
    side_effecting: bool = False
    requires_approval: bool = False

    @field_validator("input_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class ReplayVerificationExchange(ContractModel):
    command_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    feedback: dict[str, Any]


class ReplayCapsule(ContractModel):
    """Immutable execution material required to rerun an eligible span."""

    capsule_id: str = Field(default_factory=lambda: new_id("replay_capsule"))
    source_run_id: str
    source_span_id: str
    source_event_ids: tuple[str, ...]
    source_event_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    task: TaskEnvelope
    agent_spec: AgentSpec
    source_component_versions: ComponentVersionSet
    model_exchanges: tuple[ReplayModelExchange, ...]
    tool_exchanges: tuple[ReplayToolExchange, ...] = ()
    verification_exchanges: tuple[ReplayVerificationExchange, ...]
    dataset_sample_artifact_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "capsule_id",
        "source_run_id",
        "source_span_id",
        "dataset_sample_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("source_event_ids")
    @classmethod
    def _event_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a replay capsule requires source events")
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("replay capsule source events must be unique")
        return value

    @model_validator(mode="after")
    def _source_identity(self) -> "ReplayCapsule":
        if self.task.run_id != self.source_run_id:
            raise ValueError("replay capsule task belongs to another run")
        if not self.model_exchanges:
            raise ValueError(
                "network-free replay requires sealed model exchanges"
            )
        if not self.verification_exchanges:
            raise ValueError(
                "network-free replay requires sealed verification exchanges"
            )
        return self


class ReplayEligibility(ContractModel):
    source_run_id: str
    source_span_id: str
    eligible: bool
    terminal_event_id: str | None = None
    terminal_failed: bool = False
    capsule_artifact_id: str | None = None
    reasons: tuple[str, ...] = ()
    supported_modes: tuple[ReplayMode, ...] = ()

    @field_validator(
        "source_run_id",
        "source_span_id",
        "terminal_event_id",
        "capsule_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class ReplayRequest(ContractModel):
    replay_request_id: str = Field(
        default_factory=lambda: new_id("replay_request")
    )
    source_run_id: str
    source_span_id: str
    source_terminal_event_id: str
    capsule_artifact_id: str
    mode: ReplayMode
    selected_component_versions: ComponentVersionSet
    restart_failed_span: bool = False
    requested_by: str
    reason: str = Field(min_length=1, max_length=4000)
    dataset_sample_artifact_id: str | None = None
    source_event_count: int = Field(gt=0)
    source_event_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    required_approval_fingerprints: tuple[str, ...] = ()
    environment_label: str = Field(min_length=1, max_length=300)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "replay_request_id",
        "source_run_id",
        "source_span_id",
        "source_terminal_event_id",
        "capsule_artifact_id",
        "requested_by",
        "dataset_sample_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("required_approval_fingerprints")
    @classmethod
    def _fingerprints(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(value))
        if any(
            len(item) != 64
            or any(char not in "0123456789abcdef" for char in item)
            for item in normalized
        ):
            raise ValueError("approval fingerprints must be SHA-256 hex")
        return normalized

    @model_validator(mode="after")
    def _mode_label(self) -> "ReplayRequest":
        if (
            self.mode == ReplayMode.SAVED_TOOL_RESULTS
            and self.environment_label != "sealed-network-free"
        ):
            raise ValueError(
                "saved-tool-result replay must use sealed-network-free label"
            )
        if (
            self.mode == ReplayMode.LIVE_ENVIRONMENT
            and self.environment_label == "sealed-network-free"
        ):
            raise ValueError("live replay requires an explicit live label")
        return self


class ReplayApprovalGrant(ContractModel):
    approval_id: str = Field(
        default_factory=lambda: new_id("replay_approval")
    )
    replay_request_id: str
    command_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    approved_by: str
    reason: str = Field(min_length=1, max_length=2000)
    granted_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "approval_id",
        "replay_request_id",
        "approved_by",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class ReplayAttempt(ContractModel):
    attempt_id: str = Field(default_factory=lambda: new_id("replay_attempt"))
    replay_request_id: str
    attempt_no: int = Field(ge=1)
    target_run_id: str
    target_thread_id: str
    target_trace_id: str
    status: ReplayAttemptStatus
    started_at: datetime
    completed_at: datetime | None = None
    result_artifact_id: str | None = None
    error: ErrorRecord | None = None
    pending_approval_fingerprints: tuple[str, ...] = ()

    @field_validator(
        "attempt_id",
        "replay_request_id",
        "target_run_id",
        "target_thread_id",
        "target_trace_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _status_fields(self) -> "ReplayAttempt":
        terminal = self.status != ReplayAttemptStatus.RUNNING
        if terminal != (self.completed_at is not None):
            raise ValueError(
                "only terminal replay attempts have a completion time"
            )
        failed = self.status in {
            ReplayAttemptStatus.FAILED,
            ReplayAttemptStatus.ABANDONED,
        }
        if failed != (self.error is not None):
            raise ValueError(
                "failed/abandoned replay attempts require an error only"
            )
        if (
            self.status == ReplayAttemptStatus.WAITING_APPROVAL
            and not self.pending_approval_fingerprints
        ):
            raise ValueError(
                "approval-waiting attempt requires pending fingerprints"
            )
        if (
            self.status == ReplayAttemptStatus.SUCCEEDED
            and self.result_artifact_id is None
        ):
            raise ValueError("successful replay attempt requires a result")
        return self


class ReplayExecutionOutcome(ContractModel):
    replay_request_id: str
    attempt_id: str
    target_run_id: str
    status: ReplayAttemptStatus
    output_artifact_ids: tuple[str, ...] = ()
    result_artifact_id: str | None = None
    network_calls: int = Field(ge=0)
    network_accounting_complete: bool = True
    environment_label: str
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    pending_approval_fingerprints: tuple[str, ...] = ()
    error: ErrorRecord | None = None
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "replay_request_id",
        "attempt_id",
        "target_run_id",
        "result_artifact_id",
        "output_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _mode_consistency(self) -> "ReplayExecutionOutcome":
        if (
            self.environment_label == "sealed-network-free"
            and self.network_calls != 0
        ):
            raise ValueError("sealed replay cannot report network calls")
        if (
            self.status == ReplayAttemptStatus.SUCCEEDED
            and self.result_artifact_id is None
        ):
            raise ValueError("successful replay outcome requires a result")
        if self.status == ReplayAttemptStatus.FAILED and self.error is None:
            raise ValueError("failed replay outcome requires an error")
        if (
            self.status == ReplayAttemptStatus.WAITING_APPROVAL
            and not self.pending_approval_fingerprints
        ):
            raise ValueError(
                "approval-waiting replay requires pending fingerprints"
            )
        return self


class ReplayJournalEntry(ContractModel):
    journal_event_id: str = Field(
        default_factory=lambda: new_id("replay_journal")
    )
    replay_request_id: str
    sequence: int = Field(ge=1)
    kind: ReplayJournalKind
    payload: dict[str, Any]
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("journal_event_id", "replay_request_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class ReplayRecord(ContractModel):
    request: ReplayRequest
    status: ReplayRequestStatus
    revision: int = Field(ge=1)
    recovery_count: int = Field(ge=0)
    approvals: tuple[ReplayApprovalGrant, ...] = ()
    attempts: tuple[ReplayAttempt, ...] = ()
    latest_outcome: ReplayExecutionOutcome | None = None


class ReplayRecordPage(ContractModel):
    items: tuple[ReplayRecord, ...]
    next_cursor: str | None = None


class ComparisonValue(ContractModel):
    left: float
    right: float
    delta: float


class GraphComparison(ContractModel):
    left_node_count: int = Field(ge=0)
    right_node_count: int = Field(ge=0)
    added_node_ids: tuple[str, ...] = ()
    removed_node_ids: tuple[str, ...] = ()
    changed_node_ids: tuple[str, ...] = ()
    left_edge_count: int = Field(ge=0)
    right_edge_count: int = Field(ge=0)
    added_edge_ids: tuple[str, ...] = ()
    removed_edge_ids: tuple[str, ...] = ()


class ComponentComparison(ContractModel):
    component_kind: str
    component_name: str
    left_version_id: str | None = None
    right_version_id: str | None = None
    changed: bool


class StudioABComparison(ContractModel):
    comparison_id: str = Field(
        default_factory=lambda: new_id("studio_comparison")
    )
    left_run_id: str
    right_run_id: str
    left_span_id: str | None = None
    right_span_id: str | None = None
    dataset_sample_artifact_id: str
    left_event_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    right_event_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    run_status: dict[str, str]
    span_summary: GraphComparison
    task_graph: GraphComparison
    evidence_graph: GraphComparison
    components: tuple[ComponentComparison, ...]
    metrics: dict[str, ComparisonValue]
    convergence: dict[str, Any]
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)
    publishes_versions: bool = False

    @field_validator(
        "comparison_id",
        "left_run_id",
        "right_run_id",
        "left_span_id",
        "right_span_id",
        "dataset_sample_artifact_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _distinct_runs(self) -> "StudioABComparison":
        if self.left_run_id == self.right_run_id:
            raise ValueError("A/B comparison requires distinct runs")
        if (self.left_span_id is None) != (self.right_span_id is None):
            raise ValueError("span comparison requires both span IDs")
        if self.publishes_versions:
            raise ValueError("Studio A/B comparison cannot publish versions")
        return self


class ComponentDiffLine(ContractModel):
    operation: str = Field(pattern=r"^(equal|insert|delete|replace)$")
    left_start: int = Field(ge=0)
    left_end: int = Field(ge=0)
    right_start: int = Field(ge=0)
    right_end: int = Field(ge=0)
    left_lines: tuple[str, ...] = ()
    right_lines: tuple[str, ...] = ()


class StudioComponentDiff(ContractModel):
    diff_id: str = Field(default_factory=lambda: new_id("component_diff"))
    component_kind: ComponentKind
    component_name: str
    left_version_id: str
    right_version_id: str
    left_artifact_id: str
    right_artifact_id: str
    left_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    right_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    changes: tuple[ComponentDiffLine, ...]
    unified_diff: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "diff_id",
        "left_version_id",
        "right_version_id",
        "left_artifact_id",
        "right_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _supported_kind(self) -> "StudioComponentDiff":
        if self.component_kind not in {
            ComponentKind.PROMPT,
            ComponentKind.SKILL,
            ComponentKind.TOOL_POLICY,
            ComponentKind.STOP_POLICY,
            ComponentKind.VERIFICATION_POLICY,
        }:
            raise ValueError(
                "component diff supports Prompt, Skill, and Policy only"
            )
        return self


class StudioBadcase(ContractModel):
    badcase_id: str = Field(default_factory=lambda: new_id("badcase"))
    source_run_id: str
    source_span_id: str
    source_event_ids: tuple[str, ...]
    input_artifact_ids: tuple[str, ...]
    component_versions: ComponentVersionSet
    component_version_ids: tuple[str, ...]
    evaluation_ids: tuple[str, ...]
    evaluation_artifact_ids: tuple[str, ...]
    dataset_sample_artifact_id: str
    human_note: str = Field(min_length=1, max_length=8000)
    created_by: str
    artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)
    triggers_change: bool = False

    @field_validator(
        "badcase_id",
        "source_run_id",
        "source_span_id",
        "dataset_sample_artifact_id",
        "created_by",
        "artifact_id",
        "source_event_ids",
        "input_artifact_ids",
        "component_version_ids",
        "evaluation_ids",
        "evaluation_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            if not value:
                raise ValueError("badcase provenance lists cannot be empty")
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _no_automatic_change(self) -> "StudioBadcase":
        if self.triggers_change:
            raise ValueError("badcase creation cannot trigger changes")
        return self
