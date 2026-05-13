from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class SessionRetrievalQuery(BaseModel):
    section_id: Optional[str] = None
    task_id: Optional[str] = None
    source_id: Optional[str] = None
    source_url: Optional[str] = None
    claim_id: Optional[str] = None
    gap_text: Optional[str] = None
    topic: Optional[str] = None
    include_sources: bool = True
    include_claims: bool = True
    include_facts: bool = True
    include_evidence: bool = True
    include_conflicts: bool = True
    include_gaps: bool = True
    include_section_packs: bool = True
    sort_by: Literal["relevance", "authority", "recency", "coverage", "confidence"] = "relevance"
    sort_desc: bool = True
    limit_per_type: int = 10
    deduplicate: bool = True
    semantic_query: Optional[str] = None
    semantic_weight: float = Field(default=0.0, ge=0.0, le=1.0)

    model_config = {"strict": True}


class SessionRetrievalResult(BaseModel):
    research_id: str
    session_id: str = ""
    query: SessionRetrievalQuery
    section_packs: List[Dict[str, Any]] = Field(default_factory=list)
    claims: List[Dict[str, Any]] = Field(default_factory=list)
    facts: List[Dict[str, Any]] = Field(default_factory=list)
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    latest_coverage_snapshot: Optional[Dict[str, Any]] = None
    latest_novelty_snapshot: Optional[Dict[str, Any]] = None
    source_registry: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    retrieval_meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class PlannerContext(BaseModel):
    research_id: str
    session_id: str = ""
    coverage_summary: Dict[str, Any] = Field(default_factory=dict)
    latest_coverage_snapshot: Optional[Dict[str, Any]] = None
    latest_novelty_snapshot: Optional[Dict[str, Any]] = None
    section_readiness: List[Dict[str, Any]] = Field(default_factory=list)
    section_packs: List[Dict[str, Any]] = Field(default_factory=list)
    unresolved_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    conflict_hotspots: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_claims: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_evidence: List[Dict[str, Any]] = Field(default_factory=list)
    writing_ready_sections: List[str] = Field(default_factory=list)
    retrieval_meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class ResearcherContext(BaseModel):
    research_id: str
    session_id: str = ""
    already_seen_source_ids: List[str] = Field(default_factory=list)
    already_seen_source_urls: List[str] = Field(default_factory=list)
    source_registry: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    unresolved_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_claims: List[Dict[str, Any]] = Field(default_factory=list)
    relevant_facts: List[Dict[str, Any]] = Field(default_factory=list)
    focus_sections: List[str] = Field(default_factory=list)
    authority_gaps: List[str] = Field(default_factory=list)
    search_dedup_hints: Dict[str, Any] = Field(default_factory=dict)
    retrieval_meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class WriterContext(BaseModel):
    research_id: str
    session_id: str = ""
    context_source: Literal["session", "fallback", "mixed"] = "fallback"
    section_evidence_packs: List[Dict[str, Any]] = Field(default_factory=list)
    section_contexts: List[Dict[str, Any]] = Field(default_factory=list)
    retrieval_meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}
