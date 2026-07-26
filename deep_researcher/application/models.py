from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from deep_researcher.contracts import utc_now
from deep_researcher.contracts._base import validate_identifier


class ApplicationRunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ResearchCreateRequest(BaseModel):
    query: str = Field(min_length=1, max_length=8000)
    instructions: str = Field(default="", max_length=16000)
    depth: Literal["quick", "standard", "deep"] = "standard"

    model_config = {"strict": True}

    @field_validator("query")
    @classmethod
    def _query(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("query cannot be blank")
        return normalized


class ResearchCreateResponse(BaseModel):
    research_id: str
    thread_id: str
    session_id: str
    status: str
    console_url: str
    report_url: str

    model_config = {"strict": True}


class RunApprovalRequest(BaseModel):
    approved_by: str = Field(min_length=1, max_length=200)
    note: str = Field(min_length=1, max_length=2000)

    model_config = {"strict": True}

    @field_validator("approved_by")
    @classmethod
    def _approved_by(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("note")
    @classmethod
    def _note(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("approval note cannot be blank")
        return normalized


class RunCancellationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)

    model_config = {"strict": True}

    @field_validator("reason")
    @classmethod
    def _reason(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("cancellation reason cannot be blank")
        return normalized


class ApplicationRunRecord(BaseModel):
    research_id: str
    thread_id: str
    session_id: str
    run_id: str
    trace_id: str
    root_task_id: str
    report_id: str
    query: str
    instructions: str = ""
    depth: Literal["quick", "standard", "deep"] = "standard"
    status: ApplicationRunStatus = ApplicationRunStatus.QUEUED
    current_stage: str = "queued"
    report_artifact_id: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    resumed: bool = False
    revision: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class StudioReplayCreateRequest(BaseModel):
    mode: Literal["saved_tool_results", "live_environment"]
    selected_component_versions: dict[str, Any]
    requested_by: str
    reason: str
    restart_failed_span: bool = False
    environment_label: str | None = None


class StudioReplayApprovalRequest(BaseModel):
    command_fingerprint: str
    approved_by: str
    reason: str


class StudioABComparisonRequest(BaseModel):
    left_run_id: str
    right_run_id: str
    dataset_sample_artifact_id: str
    left_span_id: str | None = None
    right_span_id: str | None = None


class StudioBadcaseCreateRequest(BaseModel):
    source_run_id: str
    source_span_id: str
    dataset_sample_artifact_id: str
    evaluation_ids: list[str]
    evaluation_artifact_ids: list[str]
    human_note: str
    created_by: str
    additional_input_artifact_ids: list[str] = Field(default_factory=list)


class TimelineEventSummary(BaseModel):
    event_id: str = ""
    event_type: str
    timestamp: str
    level: str = "info"
    message: str = ""
    node_name: str | None = None
    agent_name: str | None = None
    task_id: str | None = None
    section_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    sequence_no: int = 0
    run_id: str = ""
    trace_id: str = ""
    span_id: str = ""
    parent_span_id: str | None = None
    span_kind: str = ""
    actor_id: str = ""
    status: str = ""
    input_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    state_artifact_id: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float = 0.0
    attempt: int = 1
    error: dict[str, Any] | None = None
    component_versions: dict[str, Any] = Field(default_factory=dict)
    permissions: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class KnowledgeSummary(BaseModel):
    source_count: int = 0
    claim_count: int = 0
    fact_count: int = 0
    evidence_count: int = 0
    conflict_count: int = 0
    open_gap_count: int = 0
    section_pack_count: int = 0

    model_config = {"strict": True}


class ConsoleRunListItem(BaseModel):
    schema_version: Literal["ConsoleRunListItem@2"] = "ConsoleRunListItem@2"
    research_id: str
    run_id: str
    session_id: str
    query: str
    depth: Literal["quick", "standard", "deep"]
    status: str
    current_stage: str
    current_round: int = 0
    has_report: bool = False
    resumed: bool = False
    created_at: str
    updated_at: str
    console_url: str
    report_url: str

    model_config = {"strict": True}


class ConsoleIdentityView(BaseModel):
    research_id: str
    thread_id: str
    session_id: str
    run_id: str
    trace_id: str
    root_task_id: str
    report_id: str
    query: str
    instructions: str = ""
    depth: Literal["quick", "standard", "deep"]
    created_at: str
    updated_at: str
    resumed: bool = False
    has_report: bool = False

    model_config = {"strict": True}


class ConsoleProgressStep(BaseModel):
    step_id: Literal[
        "queued",
        "research",
        "verification",
        "synthesis",
        "review",
        "complete",
    ]
    label: str
    status: Literal[
        "waiting",
        "active",
        "completed",
        "blocked",
        "failed",
        "cancelled",
    ]
    role_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleRoleView(BaseModel):
    role_id: Literal[
        "research_supervisor",
        "research_worker_pool",
        "evidence_verifier",
        "synthesis_writer",
        "report_reviewer",
    ]
    label: str
    status: Literal[
        "waiting",
        "active",
        "completed",
        "blocked",
        "failed",
        "cancelled",
    ]
    task_id: str | None = None
    target: str = ""
    last_event_sequence: int = 0
    last_event_type: str | None = None
    last_output_summary: str = ""

    model_config = {"strict": True}


class ConsoleRuntimeView(BaseModel):
    status: str
    current_stage: str
    current_round: int = 0
    elapsed_seconds: float = 0.0
    active_role_id: str | None = None
    active_task_id: str | None = None
    decision: str | None = None
    decision_reasons: tuple[str, ...] = ()
    error_code: str | None = None
    error_message: str | None = None
    progress: tuple[ConsoleProgressStep, ...] = ()
    roles: tuple[ConsoleRoleView, ...] = ()

    model_config = {"strict": True}


class ConsoleApprovalView(BaseModel):
    approval_id: str
    task_id: str
    task_title: str
    requested_by: str
    reason: str
    status: str
    requested_at: str
    resolved_by: str | None = None
    resolution_note: str | None = None
    resolved_at: str | None = None

    model_config = {"strict": True}


class ConsoleActionsView(BaseModel):
    terminal: bool
    can_approve: bool
    can_cancel: bool
    waiting_approval_task_ids: tuple[str, ...] = ()
    approvals: tuple[ConsoleApprovalView, ...] = ()

    model_config = {"strict": True}


class ConsoleTaskView(BaseModel):
    task_id: str
    parent_task_id: str | None = None
    dependency_task_ids: tuple[str, ...] = ()
    depth: int = Field(default=0, ge=0)
    kind: str
    status: str
    title: str
    goal: str
    constraints: dict[str, Any] = Field(default_factory=dict)
    input_artifact_ids: tuple[str, ...] = ()
    expected_output_schema: str
    priority: float
    deadline: str | None = None
    attempt: int = 0
    max_attempts: int = 1
    created_by: str
    assigned_actor_id: str | None = None
    tags: tuple[str, ...] = ()
    created_at: str
    updated_at: str
    budget: dict[str, Any] = Field(default_factory=dict)
    budget_usage: dict[str, Any] = Field(default_factory=dict)
    lease_owner: str | None = None
    lease_expires_at: str | None = None
    result_id: str | None = None
    output_artifact_ids: tuple[str, ...] = ()
    error_ref: str | None = None
    merged_into_task_id: str | None = None
    defer_reason: str | None = None
    pause_reason: str | None = None
    approval: ConsoleApprovalView | None = None

    model_config = {"strict": True}


class ConsoleSchedulerView(BaseModel):
    status: str
    projection_revision: int = 0
    max_concurrency: int = 0
    cancellation_reason: str | None = None
    task_counts: dict[str, int] = Field(default_factory=dict)
    active_task_ids: tuple[str, ...] = ()
    ready_task_ids: tuple[str, ...] = ()
    waiting_approval_task_ids: tuple[str, ...] = ()
    tasks: tuple[ConsoleTaskView, ...] = ()

    model_config = {"strict": True}


class ConsoleCoverageView(BaseModel):
    required_section_ids: tuple[str, ...] = ()
    complete_section_ids: tuple[str, ...] = ()
    gap_section_ids: tuple[str, ...] = ()
    blocked_high_impact_claim_ids: tuple[str, ...] = ()
    severe_conflict_ids: tuple[str, ...] = ()
    required_count: int = 0
    complete_count: int = 0
    completion_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    ready_for_reporting: bool = False

    model_config = {"strict": True}


class ConsoleSectionView(BaseModel):
    section_id: str
    parent_section_id: str | None = None
    title: str
    goal: str
    order: int
    status: str
    coverage_status: str
    coverage_score: float
    citation_score: float
    claim_ids: tuple[str, ...] = ()
    required_claim_ids: tuple[str, ...] = ()
    unsupported_claim_ids: tuple[str, ...] = ()
    conflicted_claim_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleGapView(BaseModel):
    section_id: str
    section_title: str
    coverage_status: str
    coverage_score: float
    citation_score: float
    unsupported_claim_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleConflictView(BaseModel):
    conflict_id: str
    summary: str
    status: str
    severity: str
    high_impact: bool = False
    claim_ids: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()
    resolution: str | None = None
    resolution_kind: str | None = None
    resolution_evidence_ids: tuple[str, ...] = ()
    updated_at: str

    model_config = {"strict": True}


class ConsoleSourceView(BaseModel):
    source_id: str
    canonical_url: str
    title: str | None = None
    publisher: str | None = None
    source_type: str
    source_level: str
    status: str
    authority_score: float
    published_at: str | None = None
    discovered_at: str
    task_id: str | None = None

    model_config = {"strict": True}


class ConsoleCitationView(BaseModel):
    citation_id: str
    claim_id: str
    evidence_id: str
    source_id: str
    source_title: str
    canonical_url: str
    publisher: str | None = None
    locator: str
    quote: str

    model_config = {"strict": True}


class ConsoleVerifiedClaimView(BaseModel):
    claim_id: str
    statement: str
    importance: float
    high_impact: bool = False
    citation_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleEvidencePacketView(BaseModel):
    packet_id: str
    artifact_id: str | None = None
    report_id: str
    created_at: str
    verified_claims: tuple[ConsoleVerifiedClaimView, ...] = ()
    citations: tuple[ConsoleCitationView, ...] = ()
    section_count: int = 0
    gap_count: int = 0
    conflict_count: int = 0

    model_config = {"strict": True}


class ConsoleEvidenceView(BaseModel):
    knowledge: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
    coverage: ConsoleCoverageView = Field(default_factory=ConsoleCoverageView)
    sections: tuple[ConsoleSectionView, ...] = ()
    gaps: tuple[ConsoleGapView, ...] = ()
    conflicts: tuple[ConsoleConflictView, ...] = ()
    packets: tuple[ConsoleEvidencePacketView, ...] = ()
    sources: tuple[ConsoleSourceView, ...] = ()

    model_config = {"strict": True}


class ConsoleReviewScoreView(BaseModel):
    dimension: str
    score: float
    rationale: str

    model_config = {"strict": True}


class ConsoleReviewFindingView(BaseModel):
    finding_id: str
    dimension: str
    severity: str
    message: str
    section_id: str | None = None
    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleRepairActionView(BaseModel):
    action_id: str
    kind: str
    reason: str
    section_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()

    model_config = {"strict": True}


class ConsoleReviewView(BaseModel):
    review_id: str
    revision_id: str
    decision: str
    decision_summary: str
    scores: tuple[ConsoleReviewScoreView, ...] = ()
    findings: tuple[ConsoleReviewFindingView, ...] = ()
    repair_actions: tuple[ConsoleRepairActionView, ...] = ()
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    model_config = {"strict": True}


class ConsoleRevisionView(BaseModel):
    revision_id: str
    revision: int
    title: str
    parent_revision_id: str | None = None
    report_artifact_id: str
    draft_artifact_id: str
    citation_map_artifact_id: str
    evidence_packet_artifact_id: str
    statement_count: int = 0
    citation_count: int = 0
    usage: dict[str, Any] = Field(default_factory=dict)
    created_at: str

    model_config = {"strict": True}


class ConsoleReportOutcomeView(BaseModel):
    status: str
    revisions: int
    final_revision_id: str | None = None
    final_report_artifact_id: str | None = None
    citation_map_artifact_id: str | None = None
    final_review_id: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    summary: str
    completed_at: str

    model_config = {"strict": True}


class ConsoleReportingView(BaseModel):
    report_id: str
    title: str
    artifact_ready: bool = False
    revision_count: int = 0
    latest_revision: ConsoleRevisionView | None = None
    latest_review: ConsoleReviewView | None = None
    outcome: ConsoleReportOutcomeView | None = None
    outline: tuple[ConsoleSectionView, ...] = ()

    model_config = {"strict": True}


class ConsoleNavigationView(BaseModel):
    console_url: str
    report_url: str
    studio_url: str
    trace_export_json_url: str
    trace_export_ndjson_url: str

    model_config = {"strict": True}


class ConsoleWorkspaceResponse(BaseModel):
    schema_version: Literal["ConsoleWorkspace@2"] = "ConsoleWorkspace@2"
    identity: ConsoleIdentityView
    runtime: ConsoleRuntimeView
    actions: ConsoleActionsView
    scheduler: ConsoleSchedulerView
    evidence: ConsoleEvidenceView
    reporting: ConsoleReportingView
    timeline: tuple[TimelineEventSummary, ...] = ()
    navigation: ConsoleNavigationView

    model_config = {"strict": True}


class ReportWorkspaceResponse(BaseModel):
    schema_version: Literal["ReportWorkspace@2"] = "ReportWorkspace@2"
    identity: ConsoleIdentityView
    runtime: ConsoleRuntimeView
    evidence: ConsoleEvidenceView
    reporting: ConsoleReportingView
    markdown: str = ""
    navigation: ConsoleNavigationView

    model_config = {"strict": True}


class DebugViewResponse(BaseModel):
    research_id: str
    session_id: str
    status: str
    state_summary: dict[str, Any] = Field(default_factory=dict)
    context_summary: dict[str, Any] = Field(default_factory=dict)
    trace: list[TimelineEventSummary] = Field(default_factory=list)
    raw_state: dict[str, Any] = Field(default_factory=dict)
    snapshot_summary: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}
