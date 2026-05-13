from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ResearchSessionRecord(BaseModel):
    research_id: str
    session_id: str
    root_query: str = ""
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    last_active_at: datetime = Field(default_factory=datetime.now)
    current_round: int = 0
    current_active_task_id: Optional[str] = None
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionSourceRecord(BaseModel):
    source_id: str
    research_id: str
    url: str
    title: str = ""
    domain: str = ""
    source_type: str = "web"
    authority_score: float = 0.0
    freshness_score: float = 0.0
    task_id: Optional[str] = None
    first_seen_round: int = 0
    last_seen_round: int = 0
    first_seen_at: datetime = Field(default_factory=datetime.now)
    last_seen_at: datetime = Field(default_factory=datetime.now)
    is_active: bool = True
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionClaimRecord(BaseModel):
    claim_id: str
    research_id: str
    task_id: Optional[str] = None
    section_id: str = ""
    canonical_text: str
    raw_text: str = ""
    confidence: float = 0.0
    status: str = "active"
    source_count: int = 0
    evidence_count: int = 0
    fact_count: int = 0
    first_seen_round: int = 0
    last_seen_round: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    dedup_key: str = ""
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionFactRecord(BaseModel):
    fact_id: str
    research_id: str
    task_id: Optional[str] = None
    section_id: str = ""
    canonical_text: str
    raw_text: str = ""
    snippet: str = ""
    confidence: float = 0.0
    verified_count: int = 0
    source_count: int = 0
    status: str = "active"
    dedup_key: str = ""
    first_seen_round: int = 0
    last_seen_round: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionEvidenceRecord(BaseModel):
    evidence_id: str
    research_id: str
    task_id: Optional[str] = None
    section_id: str = ""
    source_id: str = ""
    quote_text: str = ""
    summary_text: str = ""
    quality_score: float = 0.0
    confidence: float = 0.0
    status: str = "active"
    dedup_key: str = ""
    first_seen_round: int = 0
    last_seen_round: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionConflictRecord(BaseModel):
    conflict_id: str
    research_id: str
    task_id: Optional[str] = None
    section_id: str = ""
    conflict_type: str = "semantic_conflict"
    description: str = ""
    severity: str = "medium"
    status: str = "active"
    claim_count: int = 0
    evidence_count: int = 0
    first_seen_round: int = 0
    last_seen_round: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    dedup_key: str = ""
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionSectionPackRecord(BaseModel):
    pack_id: str
    research_id: str
    section_id: str
    section_title: str = ""
    goal: str = ""
    coverage_score: float = 0.0
    status: str = "active"
    claim_count: int = 0
    fact_count: int = 0
    evidence_count: int = 0
    conflict_count: int = 0
    notes: str = ""
    first_seen_round: int = 0
    last_updated_round: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionCoverageSnapshotRecord(BaseModel):
    snapshot_id: str
    research_id: str
    round_no: int
    avg_section_coverage: float = 0.0
    evidence_density: float = 0.0
    conflict_pressure: float = 0.0
    sufficiency_level: str = "insufficient"
    completed_section_count: int = 0
    partial_section_count: int = 0
    uncovered_section_count: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    raw_summary_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionUnresolvedGapRecord(BaseModel):
    gap_id: str
    research_id: str
    round_no: int
    task_id: Optional[str] = None
    section_id: str = ""
    gap_text: str
    gap_type: str = "coverage_gap"
    severity: str = "medium"
    status: str = "open"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionNoveltySnapshotRecord(BaseModel):
    snapshot_id: str
    research_id: str
    round_no: int
    new_fact_count: int = 0
    merged_fact_count: int = 0
    new_source_count: int = 0
    new_claim_count: int = 0
    new_evidence_count: int = 0
    novelty_ratio: float = 0.0
    novelty_level: str = "low"
    created_at: datetime = Field(default_factory=datetime.now)
    metadata_json: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class SessionSnapshot(BaseModel):
    session: Optional[ResearchSessionRecord] = None
    knowledge_refs: Dict[str, Any] = Field(default_factory=dict)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    section_evidence_packs: List[Dict[str, Any]] = Field(default_factory=list)
    latest_coverage_snapshot: Optional[Dict[str, Any]] = None
    open_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    latest_novelty_snapshot: Optional[Dict[str, Any]] = None
    stats: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}
