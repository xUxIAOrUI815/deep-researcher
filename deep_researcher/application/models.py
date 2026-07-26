from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from deep_researcher.contracts import utc_now


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


class RunCancellationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


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


class ActiveAgentSummary(BaseModel):
    name: str = "idle"
    status: str = "idle"
    target: str = ""
    last_output_summary: str = ""

    model_config = {"strict": True}


class ContextPanelSummary(BaseModel):
    planner: dict[str, Any] = Field(default_factory=dict)
    researcher: dict[str, Any] = Field(default_factory=dict)
    writer: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class ConsoleRunSummary(BaseModel):
    research_id: str
    thread_id: str
    session_id: str
    query: str
    status: str
    current_stage: str
    current_round: int = 0
    elapsed_seconds: float = 0.0
    resumed: bool = False
    has_report: bool = False
    root_task_id: str | None = None
    active_task_id: str | None = None
    planner_state: dict[str, Any] = Field(default_factory=dict)
    report_outline: dict[str, Any] = Field(default_factory=dict)
    task_tree: dict[str, Any] = Field(default_factory=dict)
    timeline: list[TimelineEventSummary] = Field(default_factory=list)
    knowledge_summary: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
    latest_coverage_snapshot: dict[str, Any] | None = None
    open_gaps: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[dict[str, Any]] = Field(default_factory=list)
    section_packs: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[dict[str, Any]] = Field(default_factory=list)
    active_agent: ActiveAgentSummary = Field(default_factory=ActiveAgentSummary)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)
    run_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class ReportViewResponse(BaseModel):
    research_id: str
    session_id: str
    query: str
    status: str
    title: str = ""
    markdown: str = ""
    outline: dict[str, Any] = Field(default_factory=dict)
    report: dict[str, Any] = Field(default_factory=dict)
    knowledge_summary: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
    latest_coverage_snapshot: dict[str, Any] | None = None
    open_gaps: list[dict[str, Any]] = Field(default_factory=list)
    section_packs: list[dict[str, Any]] = Field(default_factory=list)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)

    model_config = {"strict": True}


class DebugViewResponse(BaseModel):
    research_id: str
    session_id: str
    status: str
    state_summary: dict[str, Any] = Field(default_factory=dict)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)
    trace: list[TimelineEventSummary] = Field(default_factory=list)
    raw_state: dict[str, Any] = Field(default_factory=dict)
    snapshot_summary: dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}
