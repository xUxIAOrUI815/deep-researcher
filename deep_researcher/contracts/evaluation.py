from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier, validate_semantic_version
from .budgets import BudgetUsage
from .versioning import ComponentVersionSet


class DatasetSplit(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    SELECTION = "selection"
    TEST = "test"
    HIDDEN_TEST = "hidden-test"


class DatasetPurpose(str, Enum):
    TRAINING = "training"
    DEVELOPMENT = "development"
    CANDIDATE_SELECTION = "candidate_selection"
    FINAL_EVALUATION = "final_evaluation"
    RELEASE_GATE = "release_gate"
    AUDIT = "audit"


_DATASET_ACCESS: dict[DatasetPurpose, frozenset[DatasetSplit]] = {
    DatasetPurpose.TRAINING: frozenset({DatasetSplit.TRAIN}),
    DatasetPurpose.DEVELOPMENT: frozenset({DatasetSplit.TRAIN, DatasetSplit.DEV}),
    DatasetPurpose.CANDIDATE_SELECTION: frozenset({DatasetSplit.DEV, DatasetSplit.SELECTION}),
    DatasetPurpose.FINAL_EVALUATION: frozenset({DatasetSplit.TEST}),
    DatasetPurpose.RELEASE_GATE: frozenset({DatasetSplit.TEST, DatasetSplit.HIDDEN_TEST}),
    DatasetPurpose.AUDIT: frozenset(DatasetSplit),
}


def assert_dataset_access(split: DatasetSplit, purpose: DatasetPurpose) -> None:
    if split not in _DATASET_ACCESS[purpose]:
        raise ValueError(f"{purpose.value} cannot access the {split.value} split")


class DatasetDefinition(ContractModel):
    dataset_id: str = Field(default_factory=lambda: new_id("dataset"))
    name: str = Field(min_length=1, max_length=300)
    version: str
    split: DatasetSplit
    description: str = Field(min_length=1, max_length=4000)
    sample_schema: str = Field(min_length=1, max_length=255)
    manifest_artifact_id: str
    sample_count: int = Field(ge=0)
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_dataset_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("dataset_id", "manifest_artifact_id", "parent_dataset_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)


class DatasetSample(ContractModel):
    sample_id: str = Field(default_factory=lambda: new_id("sample"))
    dataset_id: str
    split: DatasetSplit
    input_artifact_id: str
    expected_artifact_id: str | None = None
    source_run_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sample_id", "dataset_id", "input_artifact_id", "expected_artifact_id", "source_run_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class DatasetAccessRequest(ContractModel):
    request_id: str = Field(default_factory=lambda: new_id("dataset_access"))
    actor_id: str
    dataset_id: str
    split: DatasetSplit
    purpose: DatasetPurpose
    run_id: str | None = None
    requested_at: datetime = Field(default_factory=utc_now)

    @field_validator("request_id", "actor_id", "dataset_id", "run_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _permitted(self) -> "DatasetAccessRequest":
        assert_dataset_access(self.split, self.purpose)
        return self


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"


class EvaluationMetric(ContractModel):
    name: str = Field(min_length=1, max_length=200)
    value: float
    direction: MetricDirection
    threshold: float | None = None
    passed: bool | None = None
    evaluator: str = Field(min_length=1, max_length=200)
    details_artifact_id: str | None = None

    @field_validator("details_artifact_id")
    @classmethod
    def _artifact_id(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _threshold_result(self) -> "EvaluationMetric":
        if self.threshold is None and self.passed is not None:
            raise ValueError("metric passed state requires a threshold")
        if self.threshold is not None:
            expected = self.value >= self.threshold if self.direction == MetricDirection.HIGHER_IS_BETTER else self.value <= self.threshold
            if self.passed is not None and self.passed != expected:
                raise ValueError("metric passed state disagrees with threshold")
        return self


class EvaluationResult(ContractModel):
    evaluation_id: str = Field(default_factory=lambda: new_id("evaluation"))
    run_id: str
    evaluator_id: str
    subject_version_id: str
    dataset_id: str
    dataset_split: DatasetSplit
    purpose: DatasetPurpose
    sample_count: int = Field(gt=0)
    metrics: tuple[EvaluationMetric, ...]
    aggregate_score: float
    passed: bool
    failed_gate_names: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    component_versions: ComponentVersionSet
    result_artifact_id: str | None = None
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evaluation_id", "run_id", "evaluator_id", "subject_version_id", "dataset_id", "result_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _consistent(self) -> "EvaluationResult":
        assert_dataset_access(self.dataset_split, self.purpose)
        if not self.metrics:
            raise ValueError("evaluation requires at least one metric")
        if self.passed and self.failed_gate_names:
            raise ValueError("a passed evaluation cannot contain failed gates")
        if not self.passed and not self.failed_gate_names:
            raise ValueError("a failed evaluation must identify at least one failed gate")
        if self.completed_at < self.started_at:
            raise ValueError("evaluation completion cannot precede start")
        return self


class FrozenReplay(ContractModel):
    replay_id: str = Field(default_factory=lambda: new_id("replay"))
    name: str = Field(min_length=1, max_length=300)
    version: str
    input_artifact_id: str
    expected_event_artifact_id: str
    expected_output_artifact_id: str
    fixture_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    deterministic_seed: int = Field(ge=0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("replay_id", "input_artifact_id", "expected_event_artifact_id", "expected_output_artifact_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)
