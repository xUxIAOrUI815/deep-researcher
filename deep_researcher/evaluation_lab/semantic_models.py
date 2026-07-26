from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    BudgetUsage,
    ClaimStatus,
    ContractModel,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    utc_now,
)
from deep_researcher.contracts._base import (
    new_id,
    validate_identifier,
)

from .models import EvaluationSnapshot


class JudgeDimension(str, Enum):
    CLAIM_SUPPORT = "claim_support"
    COMPLETENESS = "completeness"
    DEPTH = "depth"
    INSTRUCTION_FOLLOWING = "instruction_following"
    ORGANIZATION = "organization"
    READABILITY = "readability"
    SAFETY = "safety"


class SemanticClaimObservation(ContractModel):
    claim_id: str
    status: ClaimStatus
    support_score: float = Field(ge=0.0, le=1.0)
    factual: bool = True
    cited: bool
    high_impact: bool = False

    @field_validator("claim_id")
    @classmethod
    def _claim_id(cls, value: str) -> str:
        return validate_identifier(value)


class RetrievalResultObservation(ContractModel):
    source_id: str
    rank: int = Field(ge=1)
    relevant: bool
    authority_score: float = Field(ge=0.0, le=1.0)
    fresh: bool
    domain: str = Field(min_length=1, max_length=255)
    source_type: str = Field(min_length=1, max_length=100)

    @field_validator("source_id")
    @classmethod
    def _source_id(cls, value: str) -> str:
        return validate_identifier(value)


class RetrievalCaseObservation(ContractModel):
    case_id: str
    relevant_source_ids: tuple[str, ...]
    results: tuple[RetrievalResultObservation, ...]

    @field_validator("case_id", "relevant_source_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("relevant source IDs must be unique")
            if not value:
                raise ValueError("retrieval cases require relevance labels")
            return value
        return validate_identifier(value)

    @model_validator(mode="after")
    def _unique_results(self) -> "RetrievalCaseObservation":
        source_ids = [item.source_id for item in self.results]
        ranks = [item.rank for item in self.results]
        if len(source_ids) != len(set(source_ids)):
            raise ValueError("retrieval result source IDs must be unique")
        if len(ranks) != len(set(ranks)):
            raise ValueError("retrieval result ranks must be unique")
        return self


class ReportRequirementObservation(ContractModel):
    required_topics: tuple[str, ...]
    covered_topics: tuple[str, ...]
    required_instructions: tuple[str, ...]
    satisfied_instructions: tuple[str, ...]
    word_count: int = Field(ge=0)
    section_count: int = Field(ge=0)
    target_word_count: int = Field(default=1, ge=1)
    target_section_count: int = Field(default=1, ge=1)

    @field_validator(
        "required_topics",
        "covered_topics",
        "required_instructions",
        "satisfied_instructions",
    )
    @classmethod
    def _normalized(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(
            item.strip() for item in value if item.strip()
        )
        if len(normalized) != len(set(item.casefold() for item in normalized)):
            raise ValueError("report requirement entries must be unique")
        return normalized

    @model_validator(mode="after")
    def _subsets(self) -> "ReportRequirementObservation":
        required_topics = {
            item.casefold() for item in self.required_topics
        }
        covered_topics = {
            item.casefold() for item in self.covered_topics
        }
        if not covered_topics <= required_topics:
            raise ValueError("covered topics must be required topics")
        required_instructions = {
            item.casefold() for item in self.required_instructions
        }
        satisfied = {
            item.casefold() for item in self.satisfied_instructions
        }
        if not satisfied <= required_instructions:
            raise ValueError(
                "satisfied instructions must be required instructions"
            )
        return self


class BlindJudgeCandidate(ContractModel):
    candidate_id: str
    report_markdown: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    evidence_summary: str = Field(min_length=1)
    requirement_summary: str = Field(min_length=1)

    @field_validator("candidate_id")
    @classmethod
    def _candidate_id(cls, value: str) -> str:
        return validate_identifier(value)


class BlindJudgeRequest(ContractModel):
    panel_id: str
    blind_label: str = Field(pattern=r"^candidate_[a-f0-9]{12}$")
    report_markdown: str
    instruction: str
    evidence_summary: str
    requirement_summary: str
    rubric_dimensions: tuple[JudgeDimension, ...]
    randomization_seed: int = Field(ge=0)

    @field_validator("panel_id")
    @classmethod
    def _panel_id(cls, value: str) -> str:
        return validate_identifier(value)


class JudgeBallot(ContractModel):
    ballot_id: str = Field(default_factory=lambda: new_id("judge_ballot"))
    panel_id: str
    blind_label: str = Field(pattern=r"^candidate_[a-f0-9]{12}$")
    judge_version_id: str
    rubric_version_id: str
    scores: dict[JudgeDimension, float]
    passed: bool
    violations: tuple[str, ...] = ()
    rationale_summary: str = Field(min_length=1, max_length=2000)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "ballot_id",
        "panel_id",
        "judge_version_id",
        "rubric_version_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _all_scores(self) -> "JudgeBallot":
        if set(self.scores) != set(JudgeDimension):
            raise ValueError("judge ballots require every rubric dimension")
        if any(value < 0.0 or value > 1.0 for value in self.scores.values()):
            raise ValueError("judge scores must be within [0, 1]")
        return self


class JudgeDisagreementRecord(ContractModel):
    disagreement_id: str = Field(
        default_factory=lambda: new_id("judge_disagreement")
    )
    candidate_id: str
    dimension: JudgeDimension
    judge_scores: dict[str, float]
    score_range: float = Field(ge=0.0, le=1.0)
    population_variance: float = Field(ge=0.0)
    mixed_pass_votes: bool

    @field_validator("disagreement_id", "candidate_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class JudgeCandidateConsensus(ContractModel):
    candidate_id: str
    blind_labels: tuple[str, ...]
    ballot_ids: tuple[str, ...]
    consensus_scores: dict[JudgeDimension, float]
    pass_votes: int = Field(ge=0)
    total_votes: int = Field(gt=0)
    passed: bool

    @field_validator("candidate_id", "ballot_ids")
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return value
        return validate_identifier(value)

    @model_validator(mode="after")
    def _vote_consistency(self) -> "JudgeCandidateConsensus":
        if self.pass_votes > self.total_votes:
            raise ValueError("pass votes exceed total votes")
        if self.passed != (self.pass_votes > self.total_votes / 2):
            raise ValueError("consensus pass state disagrees with majority")
        if set(self.consensus_scores) != set(JudgeDimension):
            raise ValueError("consensus requires every rubric dimension")
        return self


class JudgePanelResult(ContractModel):
    panel_result_id: str = Field(
        default_factory=lambda: new_id("judge_panel_result")
    )
    panel_id: str
    judge_version_ids: tuple[str, ...]
    rubric_version_id: str
    candidate_order_by_judge: dict[str, tuple[str, ...]]
    blind_label_map: dict[str, str]
    ballots: tuple[JudgeBallot, ...]
    candidates: tuple[JudgeCandidateConsensus, ...]
    disagreements: tuple[JudgeDisagreementRecord, ...]
    randomization_seed: int = Field(ge=0)
    result_artifact_id: str
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "panel_result_id",
        "panel_id",
        "judge_version_ids",
        "rubric_version_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("judge versions must be unique")
            return value
        return validate_identifier(value)

    @model_validator(mode="after")
    def _panel_consistency(self) -> "JudgePanelResult":
        if len(self.judge_version_ids) < 3:
            raise ValueError("multi-judge panels require at least three judges")
        if len(self.judge_version_ids) % 2 == 0:
            raise ValueError("multi-judge panels require an odd judge count")
        if set(self.candidate_order_by_judge) != set(
            self.judge_version_ids
        ):
            raise ValueError("candidate order is missing a judge")
        candidate_ids = {item.candidate_id for item in self.candidates}
        if set(self.blind_label_map.values()) != candidate_ids:
            raise ValueError("blind label mapping and candidates differ")
        return self


class HumanRating(ContractModel):
    rating_id: str = Field(default_factory=lambda: new_id("human_rating"))
    candidate_id: str
    reviewer_id: str
    scores: dict[JudgeDimension, float]
    passed: bool
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("rating_id", "candidate_id", "reviewer_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _scores(self) -> "HumanRating":
        if set(self.scores) != set(JudgeDimension):
            raise ValueError("human ratings require every rubric dimension")
        if any(value < 0.0 or value > 1.0 for value in self.scores.values()):
            raise ValueError("human scores must be within [0, 1]")
        return self


class JudgeCalibrationRecord(ContractModel):
    calibration_id: str = Field(
        default_factory=lambda: new_id("judge_calibration")
    )
    judge_version_ids: tuple[str, ...]
    rubric_version_id: str
    panel_result_ids: tuple[str, ...]
    human_rating_ids: tuple[str, ...]
    sample_count: int = Field(ge=3)
    pearson_by_dimension: dict[JudgeDimension, float]
    spearman_by_dimension: dict[JudgeDimension, float]
    mean_absolute_error_by_dimension: dict[JudgeDimension, float]
    overall_human_correlation: float = Field(ge=-1.0, le=1.0)
    pass_agreement_rate: float = Field(ge=0.0, le=1.0)
    minimum_correlation: float = Field(ge=-1.0, le=1.0)
    minimum_pass_agreement: float = Field(ge=0.0, le=1.0)
    accepted: bool
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "calibration_id",
        "judge_version_ids",
        "rubric_version_id",
        "panel_result_ids",
        "human_rating_ids",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _decision(self) -> "JudgeCalibrationRecord":
        dimensions = set(JudgeDimension)
        if (
            set(self.pearson_by_dimension) != dimensions
            or set(self.spearman_by_dimension) != dimensions
            or set(self.mean_absolute_error_by_dimension) != dimensions
        ):
            raise ValueError("calibration metrics require every dimension")
        expected = (
            self.overall_human_correlation >= self.minimum_correlation
            and self.pass_agreement_rate >= self.minimum_pass_agreement
        )
        if self.accepted != expected:
            raise ValueError("calibration acceptance disagrees with thresholds")
        return self


class SemanticEvaluationInput(ContractModel):
    subject_version_id: str
    bundle_id: str
    dataset_id: str
    dataset_split: DatasetSplit
    dataset_purpose: DatasetPurpose
    dataset_access_record_id: str
    snapshot: EvaluationSnapshot
    claims: tuple[SemanticClaimObservation, ...]
    retrieval_cases: tuple[RetrievalCaseObservation, ...]
    report_requirements: ReportRequirementObservation
    judge_panel_result_id: str
    judge_candidate_id: str
    calibration_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "subject_version_id",
        "bundle_id",
        "dataset_id",
        "dataset_access_record_id",
        "judge_panel_result_id",
        "judge_candidate_id",
        "calibration_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _unique_claims(self) -> "SemanticEvaluationInput":
        claim_ids = [item.claim_id for item in self.claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("semantic claim observations must be unique")
        case_ids = [item.case_id for item in self.retrieval_cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("retrieval cases must be unique")
        return self


class SemanticEvaluationResult(ContractModel):
    semantic_evaluation_id: str = Field(
        default_factory=lambda: new_id("semantic_evaluation")
    )
    subject_version_id: str
    bundle_id: str
    dataset_id: str
    dataset_split: DatasetSplit
    dataset_purpose: DatasetPurpose
    dataset_access_record_id: str
    snapshot_id: str
    deterministic_report_id: str
    judge_panel_result_id: str
    calibration_id: str
    metrics: tuple[EvaluationMetric, ...]
    aggregate_score: float = Field(ge=0.0, le=1.0)
    passed: bool
    failed_metric_names: tuple[str, ...]
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "semantic_evaluation_id",
        "subject_version_id",
        "bundle_id",
        "dataset_id",
        "dataset_access_record_id",
        "snapshot_id",
        "deterministic_report_id",
        "judge_panel_result_id",
        "calibration_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _metrics(self) -> "SemanticEvaluationResult":
        names = [item.name for item in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("semantic metric names must be unique")
        if not self.passed and not self.failed_metric_names:
            raise ValueError("failed semantic evaluation requires failed metrics")
        if self.passed and self.failed_metric_names:
            raise ValueError("passed semantic evaluation has failed metrics")
        return self
