from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Protocol

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    BudgetUsage,
    ClaimStatus,
    ContractModel,
    EvidenceRelation,
    VerificationResult,
    utc_now,
)
from deep_researcher.contracts._base import validate_identifier


class SemanticLabel(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class KnowledgeTier(str, Enum):
    CANDIDATE = "candidate"
    VERIFIED = "verified"


class VerificationPolicy(ContractModel):
    policy_version_id: str
    support_threshold: float = Field(default=0.72, gt=0.0, le=1.0)
    partial_support_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.72, gt=0.0, le=1.0)
    minimum_authority: float = Field(default=0.45, ge=0.0, le=1.0)
    high_impact_minimum_authority: float = Field(default=0.7, ge=0.0, le=1.0)
    minimum_independent_sources: int = Field(default=1, ge=1, le=20)
    high_impact_minimum_independent_sources: int = Field(default=2, ge=2, le=20)
    freshness_days: int = Field(default=730, ge=1, le=36500)
    verification_threshold: float = Field(default=0.75, gt=0.0, le=1.0)
    section_coverage_threshold: float = Field(default=0.85, gt=0.0, le=1.0)
    citation_coverage_threshold: float = Field(default=1.0, gt=0.0, le=1.0)
    high_impact_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    max_repair_rounds: int = Field(default=3, ge=0, le=20)
    max_repairs_per_verification: int = Field(default=8, ge=1, le=100)
    max_quotes_per_evidence: int = Field(default=8, ge=1, le=100)

    @field_validator("policy_version_id")
    @classmethod
    def _policy_version_id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _threshold_order(self) -> "VerificationPolicy":
        if self.partial_support_threshold >= self.support_threshold:
            raise ValueError(
                "partial support threshold must be below support threshold"
            )
        if self.high_impact_minimum_authority < self.minimum_authority:
            raise ValueError(
                "high-impact authority cannot be weaker than the normal threshold"
            )
        if (
            self.high_impact_minimum_independent_sources
            < self.minimum_independent_sources
        ):
            raise ValueError(
                "high-impact source independence cannot be weaker than the normal threshold"
            )
        return self


class SemanticJudgment(ContractModel):
    label: SemanticLabel
    score: float = Field(ge=0.0, le=1.0)
    overreach_fragments: tuple[str, ...] = ()
    contradiction_fragments: tuple[str, ...] = ()
    decision_summary: str = Field(min_length=1, max_length=2000)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)

    @field_validator("overreach_fragments", "contradiction_fragments")
    @classmethod
    def _bounded_fragments(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            dict.fromkeys(item.strip()[:300] for item in value if item.strip())
        )
        if len(normalized) > 50:
            raise ValueError("semantic judgment contains too many fragments")
        return normalized


class SemanticVerificationAdapter(Protocol):
    async def judge(
        self,
        *,
        run_id: str,
        task_id: str | None,
        subject_id: str,
        statement: str,
        evidence_id: str,
        relation: EvidenceRelation,
        passages: tuple[str, ...],
    ) -> SemanticJudgment: ...


class EvidenceAssessment(ContractModel):
    evidence_id: str
    relation: EvidenceRelation
    grounded: bool
    passage_integrity: bool
    source_ids: tuple[str, ...] = ()
    snapshot_ids: tuple[str, ...] = ()
    source_authority: float = Field(default=0.0, ge=0.0, le=1.0)
    source_fresh: bool = True
    semantic: SemanticJudgment
    citation_ids: tuple[str, ...] = ()
    issues: tuple[str, ...] = ()

    @field_validator("evidence_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("source_ids", "snapshot_ids", "citation_ids")
    @classmethod
    def _reference_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))


class ClaimVerificationSummary(ContractModel):
    claim_id: str
    status: ClaimStatus
    support_score: float = Field(ge=0.0, le=1.0)
    contradiction_score: float = Field(ge=0.0, le=1.0)
    independent_source_count: int = Field(ge=0)
    verified_evidence_ids: tuple[str, ...] = ()
    verified_citation_ids: tuple[str, ...] = ()
    stale_source_ids: tuple[str, ...] = ()
    high_impact_blocked: bool = False
    verification_result: VerificationResult

    @field_validator(
        "claim_id",
        "verified_evidence_ids",
        "verified_citation_ids",
        "stale_source_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class SectionCoverageAssessment(ContractModel):
    section_id: str
    result_artifact_id: str
    required_claim_ids: tuple[str, ...]
    supported_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    conflicted_claim_ids: tuple[str, ...]
    stale_claim_ids: tuple[str, ...]
    uncited_claim_ids: tuple[str, ...]
    coverage_score: float = Field(ge=0.0, le=1.0)
    citation_score: float = Field(ge=0.0, le=1.0)
    blocked: bool
    assessed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "section_id",
        "result_artifact_id",
        "required_claim_ids",
        "supported_claim_ids",
        "unsupported_claim_ids",
        "conflicted_claim_ids",
        "stale_claim_ids",
        "uncited_claim_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class RunVerificationSummary(ContractModel):
    run_id: str
    claim_results: tuple[ClaimVerificationSummary, ...]
    section_results: tuple[SectionCoverageAssessment, ...]
    verification_artifact_ids: tuple[str, ...]
    blocked_high_impact_claim_ids: tuple[str, ...]
    open_severe_conflict_ids: tuple[str, ...]
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "run_id",
        "verification_artifact_ids",
        "blocked_high_impact_claim_ids",
        "open_severe_conflict_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)
