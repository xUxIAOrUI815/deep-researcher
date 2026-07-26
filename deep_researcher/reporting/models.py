from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    Budget,
    BudgetUsage,
    ConflictSeverity,
    ConflictStatus,
    ContractModel,
    utc_now,
)
from deep_researcher.contracts._base import new_id, validate_identifier


class StatementCertainty(str, Enum):
    DEFINITIVE = "definitive"
    QUALIFIED = "qualified"
    CONFLICTED = "conflicted"


class ReviewDimension(str, Enum):
    COMPLETENESS = "completeness"
    SUPPORT = "support"
    CITATION = "citation"
    CONFLICTS = "conflicts"
    INSTRUCTION_FOLLOWING = "instruction_following"
    DEPTH = "depth"
    ORGANIZATION = "organization"
    READABILITY = "readability"


class ReviewActionKind(str, Enum):
    TARGETED_RESEARCH = "targeted_research"
    CITATION_REPAIR = "citation_repair"
    LOCAL_REWRITE = "local_rewrite"
    STRUCTURAL_REWRITE = "structural_rewrite"
    ACCEPT = "accept"
    REJECT = "reject"


class FindingSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class LoopStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    BUDGET_EXHAUSTED = "budget_exhausted"
    REVISION_EXHAUSTED = "revision_exhausted"
    APPROVAL_REQUIRED = "approval_required"
    CANCELLED = "cancelled"


class VerifiedCitationPacket(ContractModel):
    citation_id: str
    claim_id: str
    evidence_id: str
    passage_id: str
    snapshot_id: str
    source_id: str
    source_title: str
    canonical_url: str
    publisher: str | None = None
    locator: str
    quote: str

    @field_validator(
        "citation_id",
        "claim_id",
        "evidence_id",
        "passage_id",
        "snapshot_id",
        "source_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class VerifiedClaimPacket(ContractModel):
    claim_id: str
    statement: str = Field(min_length=1, max_length=8000)
    importance: float = Field(ge=0.0, le=1.0)
    high_impact: bool = False
    citation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]

    @field_validator("claim_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("citation_ids", "source_ids")
    @classmethod
    def _refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _has_support(self) -> "VerifiedClaimPacket":
        if not self.citation_ids or not self.source_ids:
            raise ValueError("verified Writer claims require citations and sources")
        return self


class EvidenceGapPacket(ContractModel):
    claim_id: str
    section_id: str
    statement: str
    reason: str
    high_impact: bool = False

    @field_validator("claim_id", "section_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class ConflictPacket(ContractModel):
    conflict_id: str
    claim_ids: tuple[str, ...]
    summary: str
    status: ConflictStatus
    severity: ConflictSeverity
    high_impact: bool
    resolution: str | None = None

    @field_validator("conflict_id", "claim_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class WriterSectionPacket(ContractModel):
    section_id: str
    title: str
    goal: str
    order: int = Field(ge=0)
    required_claim_ids: tuple[str, ...]
    verified_claim_ids: tuple[str, ...]
    gap_claim_ids: tuple[str, ...]
    conflict_ids: tuple[str, ...]

    @field_validator(
        "section_id",
        "required_claim_ids",
        "verified_claim_ids",
        "gap_claim_ids",
        "conflict_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class WriterEvidencePacket(ContractModel):
    packet_id: str = Field(default_factory=lambda: new_id("writer_packet"))
    run_id: str
    report_id: str
    research_question: str
    sections: tuple[WriterSectionPacket, ...]
    claims: tuple[VerifiedClaimPacket, ...]
    citations: tuple[VerifiedCitationPacket, ...]
    gaps: tuple[EvidenceGapPacket, ...] = ()
    conflicts: tuple[ConflictPacket, ...] = ()
    packet_artifact_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("packet_id", "run_id", "report_id", "packet_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "WriterEvidencePacket":
        claim_ids = [item.claim_id for item in self.claims]
        citation_ids = [item.citation_id for item in self.citations]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Writer packet claims must be unique")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("Writer packet citations must be unique")
        known_claims = set(claim_ids)
        known_citations = set(citation_ids)
        for claim in self.claims:
            if not set(claim.citation_ids).issubset(known_citations):
                raise ValueError("Writer claim references an unknown citation")
        for citation in self.citations:
            if citation.claim_id not in known_claims:
                raise ValueError("Writer citation references an unknown claim")
        return self


class DraftStatement(ContractModel):
    statement_id: str
    text: str = Field(min_length=1, max_length=8000)
    claim_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]
    certainty: StatementCertainty = StatementCertainty.DEFINITIVE

    @field_validator("statement_id", "claim_ids", "citation_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _supported(self) -> "DraftStatement":
        if not self.claim_ids or not self.citation_ids:
            raise ValueError("a factual draft statement requires claims and citations")
        return self


class GapDisclosure(ContractModel):
    claim_id: str
    text: str = Field(min_length=1, max_length=4000)

    @field_validator("claim_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)


class ConflictDisclosure(ContractModel):
    conflict_id: str
    text: str = Field(min_length=1, max_length=5000)
    citation_ids: tuple[str, ...] = ()

    @field_validator("conflict_id", "citation_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class SectionDraftProposal(ContractModel):
    section_id: str
    title: str
    statements: tuple[DraftStatement, ...]
    gap_disclosures: tuple[GapDisclosure, ...] = ()
    conflict_disclosures: tuple[ConflictDisclosure, ...] = ()

    @field_validator("section_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)


class WriterDraftProposal(ContractModel):
    run_id: str
    report_id: str
    revision: int = Field(ge=1)
    title: str = Field(min_length=1, max_length=1000)
    sections: tuple[SectionDraftProposal, ...]
    decision_summary: str = Field(min_length=1, max_length=2000)

    @field_validator("run_id", "report_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _sections_unique(self) -> "WriterDraftProposal":
        ids = [item.section_id for item in self.sections]
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("Writer draft sections must be non-empty and unique")
        return self


class CitationMapEntry(ContractModel):
    marker: str = Field(pattern=r"^\[[1-9][0-9]*\]$")
    citation_id: str
    claim_id: str
    evidence_id: str
    source_id: str
    canonical_url: str
    locator: str
    quote: str

    @field_validator("citation_id", "claim_id", "evidence_id", "source_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class CitationMap(ContractModel):
    citation_map_id: str = Field(default_factory=lambda: new_id("citation_map"))
    run_id: str
    report_id: str
    revision: int = Field(ge=1)
    entries: tuple[CitationMapEntry, ...]
    artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("citation_map_id", "run_id", "report_id", "artifact_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _unique_contiguous_entries(self) -> "CitationMap":
        markers = [item.marker for item in self.entries]
        citation_ids = [item.citation_id for item in self.entries]
        expected = [f"[{index}]" for index in range(1, len(markers) + 1)]
        if markers != expected:
            raise ValueError("citation map markers must be contiguous and ordered")
        if len(citation_ids) != len(set(citation_ids)):
            raise ValueError("citation map citation IDs must be unique")
        return self


class ReportRevision(ContractModel):
    revision_id: str = Field(default_factory=lambda: new_id("report_revision"))
    run_id: str
    report_id: str
    revision: int = Field(ge=1)
    title: str
    markdown: str = Field(min_length=1)
    section_ids: tuple[str, ...]
    statement_ids: tuple[str, ...]
    report_artifact_id: str
    draft_artifact_id: str
    citation_map_artifact_id: str
    evidence_packet_artifact_id: str
    parent_revision_id: str | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "revision_id",
        "run_id",
        "report_id",
        "report_artifact_id",
        "draft_artifact_id",
        "citation_map_artifact_id",
        "evidence_packet_artifact_id",
        "parent_revision_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _unique_report_references(self) -> "ReportRevision":
        if not self.section_ids or len(self.section_ids) != len(
            set(self.section_ids)
        ):
            raise ValueError("report revision section IDs must be non-empty and unique")
        if len(self.statement_ids) != len(set(self.statement_ids)):
            raise ValueError("report revision statement IDs must be unique")
        return self


class RubricScore(ContractModel):
    dimension: ReviewDimension
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1, max_length=2000)


class ReviewFinding(ContractModel):
    finding_id: str = Field(default_factory=lambda: new_id("finding"))
    dimension: ReviewDimension
    severity: FindingSeverity
    message: str = Field(min_length=1, max_length=3000)
    section_id: str | None = None
    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()

    @field_validator("finding_id", "section_id", "claim_ids", "citation_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value) if value is not None else None


class ReportRepairAction(ContractModel):
    action_id: str = Field(default_factory=lambda: new_id("repair_action"))
    kind: ReviewActionKind
    reason: str = Field(min_length=1, max_length=3000)
    section_ids: tuple[str, ...] = ()
    claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("action_id", "section_ids", "claim_ids", "citation_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _repair_only(self) -> "ReportRepairAction":
        if self.kind in {ReviewActionKind.ACCEPT, ReviewActionKind.REJECT}:
            raise ValueError("accept/reject are Reviewer decisions, not repair actions")
        return self


class ReviewerDecision(ContractModel):
    review_id: str = Field(default_factory=lambda: new_id("report_review"))
    run_id: str
    report_id: str
    revision_id: str
    decision: ReviewActionKind
    scores: tuple[RubricScore, ...]
    findings: tuple[ReviewFinding, ...] = ()
    repair_actions: tuple[ReportRepairAction, ...] = ()
    decision_summary: str = Field(min_length=1, max_length=3000)
    review_artifact_id: str | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "review_id",
        "run_id",
        "report_id",
        "revision_id",
        "review_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "ReviewerDecision":
        dimensions = [item.dimension for item in self.scores]
        if set(dimensions) != set(ReviewDimension) or len(dimensions) != len(set(dimensions)):
            raise ValueError("Reviewer must score every rubric dimension exactly once")
        terminal = self.decision in {
            ReviewActionKind.ACCEPT,
            ReviewActionKind.REJECT,
        }
        if terminal and self.repair_actions:
            raise ValueError("terminal review decisions cannot include repairs")
        if not terminal and not self.repair_actions:
            raise ValueError("repair decisions require structured repair actions")
        if not terminal and self.decision not in {
            item.kind for item in self.repair_actions
        }:
            raise ValueError("primary review decision must appear in repair actions")
        return self


class ReportLoopPolicy(ContractModel):
    max_revisions: int = Field(default=6, ge=1, le=100)
    max_targeted_research_rounds: int = Field(default=2, ge=0, le=20)
    minimum_score: float = Field(default=0.8, ge=0.0, le=1.0)
    minimum_support_score: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_citation_score: float = Field(default=1.0, ge=0.0, le=1.0)
    minimum_sources_per_statement: int = Field(default=1, ge=1, le=20)
    high_impact_minimum_sources: int = Field(default=2, ge=2, le=20)
    run_budget: Budget


class ReportLoopOutcome(ContractModel):
    run_id: str
    report_id: str
    status: LoopStatus
    revisions: int = Field(ge=0)
    final_revision_id: str | None = None
    final_report_artifact_id: str | None = None
    citation_map_artifact_id: str | None = None
    final_review_id: str | None = None
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    summary: str
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "run_id",
        "report_id",
        "final_revision_id",
        "final_report_artifact_id",
        "citation_map_artifact_id",
        "final_review_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _terminal_references(self) -> "ReportLoopOutcome":
        if self.status == LoopStatus.ACCEPTED:
            required = (
                self.final_revision_id,
                self.final_report_artifact_id,
                self.citation_map_artifact_id,
                self.final_review_id,
            )
            if any(item is None for item in required):
                raise ValueError(
                    "accepted report outcomes require final revision, report, "
                    "citation map, and review references"
                )
        return self
