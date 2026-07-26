from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    ContractModel,
    EvaluationMetric,
    utc_now,
)
from deep_researcher.contracts._base import (
    new_id,
    validate_identifier,
)


class GateStage(str, Enum):
    SELECTION = "selection"
    FINAL_PROMOTION = "final_promotion"
    POST_RELEASE = "post_release"


class GateOutcome(str, Enum):
    ADVANCE_TO_FINAL = "advance_to_final"
    PROMOTE = "promote"
    REJECT = "reject"
    KEEP = "keep"
    ROLLBACK = "rollback"


class ReleaseGatePolicy(ContractModel):
    policy_id: str = Field(default_factory=lambda: new_id("release_policy"))
    required_improvements: dict[str, float]
    non_regression_tolerances: dict[str, float]
    maximum_cost_usd: float = Field(gt=0.0)
    maximum_cost_increase_fraction: float = Field(ge=0.0)
    maximum_variances: dict[str, float]
    safety_minimums: dict[str, float]
    protocol_minimum: float = Field(ge=0.0, le=1.0)
    minimum_human_correlation: float = Field(ge=-1.0, le=1.0)
    minimum_human_pass_agreement: float = Field(ge=0.0, le=1.0)
    require_semantic_pass: bool = True
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("policy_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _policy(self) -> "ReleaseGatePolicy":
        if not self.required_improvements:
            raise ValueError("release policy requires improvement metrics")
        if not self.non_regression_tolerances:
            raise ValueError("release policy requires non-regression metrics")
        mappings = (
            self.required_improvements,
            self.non_regression_tolerances,
            self.maximum_variances,
            self.safety_minimums,
        )
        if any(
            not name or value < 0
            for mapping in mappings
            for name, value in mapping.items()
        ):
            raise ValueError("release policy metric bounds must be nonnegative")
        return self


class GateEvaluationEvidence(ContractModel):
    candidate_version_id: str
    baseline_version_id: str
    rollback_target_version_id: str | None = None
    baseline_metrics: tuple[EvaluationMetric, ...]
    candidate_metrics: tuple[EvaluationMetric, ...]
    candidate_variances: dict[str, float]
    semantic_evaluation_ids: tuple[str, ...]
    dataset_access_record_ids: tuple[str, ...]
    calibration_id: str
    evaluation_artifact_ids: tuple[str, ...]

    @field_validator(
        "candidate_version_id",
        "baseline_version_id",
        "rollback_target_version_id",
        "semantic_evaluation_ids",
        "dataset_access_record_ids",
        "calibration_id",
        "evaluation_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("gate evidence IDs must be unique")
            return value
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _metrics(self) -> "GateEvaluationEvidence":
        for label, values in (
            ("baseline", self.baseline_metrics),
            ("candidate", self.candidate_metrics),
        ):
            names = [item.name for item in values]
            if len(names) != len(set(names)):
                raise ValueError(f"{label} gate metrics must be unique")
        if not self.semantic_evaluation_ids:
            raise ValueError("gate evidence requires semantic evaluations")
        if not self.dataset_access_record_ids:
            raise ValueError("gate evidence requires audited dataset access")
        if not self.evaluation_artifact_ids:
            raise ValueError("gate evidence requires evaluation artifacts")
        if any(value < 0 for value in self.candidate_variances.values()):
            raise ValueError("gate variances cannot be negative")
        return self


class GateCheck(ContractModel):
    check_name: str = Field(min_length=1, max_length=300)
    category: str = Field(min_length=1, max_length=100)
    passed: bool
    observed: float | str | bool
    required: float | str | bool
    detail: str = Field(min_length=1, max_length=2000)


class ReleaseGateDecision(ContractModel):
    gate_decision_id: str = Field(
        default_factory=lambda: new_id("gate_decision")
    )
    policy_id: str
    stage: GateStage
    outcome: GateOutcome
    candidate_version_id: str
    baseline_version_id: str
    rollback_target_version_id: str | None = None
    checks: tuple[GateCheck, ...]
    passed: bool
    failed_check_names: tuple[str, ...]
    semantic_evaluation_ids: tuple[str, ...]
    dataset_access_record_ids: tuple[str, ...]
    calibration_id: str
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "gate_decision_id",
        "policy_id",
        "candidate_version_id",
        "baseline_version_id",
        "rollback_target_version_id",
        "semantic_evaluation_ids",
        "dataset_access_record_ids",
        "calibration_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return value
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _decision(self) -> "ReleaseGateDecision":
        failed = tuple(
            item.check_name for item in self.checks if not item.passed
        )
        if failed != self.failed_check_names:
            raise ValueError("gate failed-check list disagrees with checks")
        if self.passed != (not bool(failed)):
            raise ValueError("gate pass state disagrees with checks")
        expected = {
            GateStage.SELECTION: (
                GateOutcome.ADVANCE_TO_FINAL
                if self.passed
                else GateOutcome.REJECT
            ),
            GateStage.FINAL_PROMOTION: (
                GateOutcome.PROMOTE
                if self.passed
                else GateOutcome.REJECT
            ),
            GateStage.POST_RELEASE: (
                GateOutcome.KEEP
                if self.passed
                else GateOutcome.ROLLBACK
            ),
        }[self.stage]
        if self.outcome != expected:
            raise ValueError("gate outcome disagrees with stage/pass state")
        if (
            self.stage == GateStage.POST_RELEASE
            and self.rollback_target_version_id is None
        ):
            raise ValueError("post-release gate requires a rollback target")
        return self


class GateApplicationRecord(ContractModel):
    application_id: str = Field(
        default_factory=lambda: new_id("gate_application")
    )
    gate_decision_id: str
    outcome: GateOutcome
    candidate_version_id: str
    version_transition_ids: tuple[str, ...]
    application_artifact_id: str
    applied_at: datetime

    @field_validator(
        "application_id",
        "gate_decision_id",
        "candidate_version_id",
        "version_transition_ids",
        "application_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _transition_requirement(self) -> "GateApplicationRecord":
        mutating = self.outcome in {
            GateOutcome.PROMOTE,
            GateOutcome.REJECT,
            GateOutcome.ROLLBACK,
        }
        if mutating != bool(self.version_transition_ids):
            raise ValueError(
                "mutating gate outcomes require version transitions only"
            )
        return self
