from typing import Dict, List, Optional, Literal, Any, TypedDict
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import uuid

from schemas.task_tree import TaskNode, TaskTreePatch


class PlannerAction(str, Enum):
    CONTINUE_RESEARCH = "continue_research"
    START_WRITING = "start_writing"
    STOP = "stop"


class SourceLevel(str, Enum):
    S = "S"
    A = "A"
    B = "B"
    C = "C"


class RunMetadata(BaseModel):
    research_id: str
    thread_id: str
    run_id: str
    trace_id: str
    session_id: Optional[str] = None
    graph_version: str = "graph.v1"
    prompt_version: str = "prompt.v1"
    root_query: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class PlannerState(BaseModel):
    action: Optional[Literal["continue_research", "start_writing", "stop"]] = None
    rationale: str = ""
    convergence_summary: str = ""
    writing_constraints: Dict[str, Any] = Field(default_factory=dict)
    task_updates: List[TaskTreePatch] = Field(default_factory=list)
    new_task_ids: List[str] = Field(default_factory=list)
    active_task_id: Optional[str] = None
    next_task_id: Optional[str] = None
    stop_reason: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True, "use_enum_values": True}


class AtomicFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_url: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    task_id: Optional[str] = None
    snippet: str = ""
    source_level: Literal["S", "A", "B", "C"] = SourceLevel.C.value
    verified_count: int = 0
    is_conflict: bool = False
    conflict_with: List[str] = Field(default_factory=list)
    confidence_reason: Optional[str] = None

    model_config = {"strict": True, "use_enum_values": True}


class Claim(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    fact_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    task_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source_id: str = ""
    source_url: str = ""
    quote: str = ""
    summary: str = ""
    fact_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    task_id: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class ConflictRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    fact_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    description: str = ""
    severity: str = "medium"
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class TokenUsage(BaseModel):
    planning_tokens: int = 0
    research_tokens: int = 0
    distillation_tokens: int = 0
    writing_tokens: int = 0
    total_tokens: int = 0

    model_config = {"strict": True}


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float = 0.0
    raw_content: Optional[str] = None

    model_config = {"strict": True}


class ScrapedData(BaseModel):
    url: str
    markdown: str
    title: str = ""
    fetch_method: Literal["jina", "playwright"] = "jina"
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

    model_config = {"strict": True}


class ResearcherOutputs(BaseModel):
    task_id: Optional[str] = None
    queries: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    passages: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    search_result_ids: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)
    scraped_data_ids: List[str] = Field(default_factory=list)
    search_results_cache: List[SearchResult] = Field(default_factory=list)
    scraped_data_cache: List[ScrapedData] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class DistillerOutputs(BaseModel):
    task_id: Optional[str] = None
    clean_passages: List[Dict[str, Any]] = Field(default_factory=list)
    atomic_facts: List[AtomicFact] = Field(default_factory=list)
    claims: List[Claim] = Field(default_factory=list)
    evidence: List[Evidence] = Field(default_factory=list)
    conflicts: List[ConflictRecord] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    conflict_ids: List[str] = Field(default_factory=list)
    knowledge_refs: Dict[str, Any] = Field(default_factory=dict)
    section_evidence_packs: List[Dict[str, Any]] = Field(default_factory=list)
    compression_summary: str = ""
    unresolved_gaps: List[str] = Field(default_factory=list)
    suggested_followups: List[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class KnowledgeRefs(BaseModel):
    collection_name: str = ""
    fact_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    conflict_ids: List[str] = Field(default_factory=list)
    source_ids: List[str] = Field(default_factory=list)

    model_config = {"strict": True}


class ReportSection(BaseModel):
    section_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str
    goal: str = ""
    order: int = 0

    model_config = {"strict": True}


class ReportOutline(BaseModel):
    title: str = ""
    sections: List[ReportSection] = Field(default_factory=list)
    rationale: str = ""
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class SectionGoal(BaseModel):
    section_id: str
    goal: str
    required_claim_types: List[str] = Field(default_factory=list)
    priority: float = 0.0

    model_config = {"strict": True}


class SectionEvidencePack(BaseModel):
    pack_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    section_id: str
    goal: str = ""
    claim_ids: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)
    conflict_ids: List[str] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    notes: str = ""
    created_by: str = "distiller"
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class FinalReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    markdown: str = ""
    section_ids: List[str] = Field(default_factory=list)
    evidence_pack_ids: List[str] = Field(default_factory=list)
    citation_map: Dict[str, List[str]] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class ResearchGraphState(TypedDict, total=False):
    user_query: str
    normalized_query: str
    run_metadata: dict
    planner_state: dict
    task_tree: dict
    root_task_id: str | None
    active_task_id: str | None
    current_focus: str | None
    researcher_outputs: dict
    distiller_outputs: dict
    knowledge_refs: dict
    report_outline: dict
    section_goals: list
    section_evidence_packs: list
    final_report: dict | str | None
    token_usage: dict
    state_events: list
    error_state: dict | None
    completed_tasks: list
    failed_tasks: list
    fact_pool: list
    atomic_facts: list
    messages: list
    raw_scraped_data: list
    search_results: list


class ResearchState(BaseModel):
    user_query: str = ""
    normalized_query: str = ""
    run_metadata: Optional[RunMetadata] = None
    planner_state: PlannerState = Field(default_factory=PlannerState)
    task_tree: Dict[str, TaskNode] = Field(default_factory=dict)
    fact_pool: List[str] = Field(default_factory=list)
    atomic_facts: List[AtomicFact] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    current_focus: Optional[str] = None
    active_task_id: Optional[str] = None
    root_task_id: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    raw_scraped_data: List[ScrapedData] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)
    researcher_outputs: ResearcherOutputs = Field(default_factory=ResearcherOutputs)
    distiller_outputs: DistillerOutputs = Field(default_factory=DistillerOutputs)
    knowledge_refs: KnowledgeRefs = Field(default_factory=KnowledgeRefs)
    report_outline: ReportOutline = Field(default_factory=ReportOutline)
    section_goals: List[SectionGoal] = Field(default_factory=list)
    section_evidence_packs: List[SectionEvidencePack] = Field(default_factory=list)
    final_report: Optional[FinalReport] = None
    state_events: List[Dict[str, Any]] = Field(default_factory=list)
    error_state: Optional[Dict[str, Any]] = None

    model_config = {"strict": True}


def compute_priority(relevance: float, depth: int, depth_penalty: float = 0.3) -> float:
    depth_factor = max(0.0, 1.0 - (depth * depth_penalty))
    return relevance * 0.7 + depth_factor * 0.3
