from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    BudgetUsage,
    ComponentVersionSet,
    ContractModel,
    DatasetAccessRequest,
    DatasetDefinition,
    DatasetPurpose,
    DatasetSample,
    DatasetSplit,
    EvaluationMetric,
    FrozenReplay,
    utc_now,
)
from deep_researcher.contracts._base import (
    new_id,
    validate_identifier,
    validate_semantic_version,
)


class EvaluationMode(str, Enum):
    FROZEN_REPLAY = "frozen_replay"
    LIVE_WEB = "live_web"


class SystemBaseline(str, Enum):
    LEGACY = "legacy"
    FIXED_WORKFLOW = "fixed_workflow"
    NEW_RUNTIME = "new_runtime"


class ExperimentRunStatus(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class DatasetBundle(ContractModel):
    bundle_id: str = Field(default_factory=lambda: new_id("dataset_bundle"))
    name: str = Field(min_length=1, max_length=300)
    version: str
    sample_schema: str = Field(min_length=1, max_length=255)
    split_dataset_ids: dict[DatasetSplit, str]
    manifest_artifact_id: str
    fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    parent_bundle_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "bundle_id",
        "manifest_artifact_id",
        "parent_bundle_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)

    @model_validator(mode="after")
    def _all_splits(self) -> "DatasetBundle":
        if set(self.split_dataset_ids) != set(DatasetSplit):
            raise ValueError(
                "a dataset bundle must define train, dev, selection, test, "
                "and hidden-test splits"
            )
        values = tuple(self.split_dataset_ids.values())
        for value in values:
            validate_identifier(value)
        if len(values) != len(set(values)):
            raise ValueError("dataset bundle split IDs must be unique")
        return self


class DatasetSampleSpec(ContractModel):
    input_artifact_id: str
    expected_artifact_id: str | None = None
    source_run_id: str | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "input_artifact_id",
        "expected_artifact_id",
        "source_run_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class DatasetAccessRecord(ContractModel):
    access_record_id: str = Field(
        default_factory=lambda: new_id("dataset_access_record")
    )
    request: DatasetAccessRequest
    bundle_id: str
    dataset_definition: DatasetDefinition
    sample_ids: tuple[str, ...]
    audit_artifact_id: str
    granted_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "access_record_id",
        "bundle_id",
        "audit_artifact_id",
        "sample_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("dataset access sample IDs must be unique")
            return value
        return validate_identifier(value)

    @model_validator(mode="after")
    def _consistent(self) -> "DatasetAccessRecord":
        definition = self.dataset_definition
        if (
            self.request.dataset_id != definition.dataset_id
            or self.request.split != definition.split
        ):
            raise ValueError(
                "dataset access request and granted definition differ"
            )
        return self


class SourceObservation(ContractModel):
    source_id: str
    canonical_url: str
    source_type: str
    source_level: str
    authority_score: float = Field(ge=0.0, le=1.0)
    publisher: str | None = None
    domain: str
    published_at: datetime | None = None
    fetched_at: datetime | None = None
    content_hash: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )

    @field_validator("source_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)


class CitationObservation(ContractModel):
    citation_id: str
    claim_id: str
    evidence_id: str
    source_id: str
    marker: str | None = Field(default=None, pattern=r"^\[[1-9][0-9]*\]$")
    canonical_url: str
    quote: str
    passage_text: str
    locator: str
    verified: bool
    used_in_report: bool

    @field_validator(
        "citation_id",
        "claim_id",
        "evidence_id",
        "source_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class SectionObservation(ContractModel):
    section_id: str
    required_claim_ids: tuple[str, ...]
    supported_claim_ids: tuple[str, ...]
    unsupported_claim_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]

    @field_validator(
        "section_id",
        "required_claim_ids",
        "supported_claim_ids",
        "unsupported_claim_ids",
        "citation_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)


class ToolCallObservation(ContractModel):
    call_id: str
    tool_name: str
    operation: str
    request_key: str
    is_search: bool = False
    valid: bool = True
    succeeded: bool = True
    evidence_count: int = Field(default=0, ge=0)
    recovered_after_retry: bool = False

    @field_validator("call_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)


class ExecutionCounters(ContractModel):
    task_count: int = Field(default=0, ge=0)
    failed_task_count: int = Field(default=0, ge=0)
    retried_operation_count: int = Field(default=0, ge=0)
    recovered_operation_count: int = Field(default=0, ge=0)
    idempotent_operation_count: int = Field(default=0, ge=0)
    idempotency_violation_count: int = Field(default=0, ge=0)
    protocol_operation_count: int = Field(default=0, ge=0)
    invalid_protocol_operation_count: int = Field(default=0, ge=0)
    convergence_turns: int = Field(default=0, ge=0)
    budget_violation_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def _bounded_counts(self) -> "ExecutionCounters":
        pairs = (
            (
                self.failed_task_count,
                self.task_count,
                "failed task count",
            ),
            (
                self.recovered_operation_count,
                self.retried_operation_count,
                "recovered operation count",
            ),
            (
                self.idempotency_violation_count,
                self.idempotent_operation_count,
                "idempotency violation count",
            ),
            (
                self.invalid_protocol_operation_count,
                self.protocol_operation_count,
                "invalid protocol operation count",
            ),
        )
        for numerator, denominator, label in pairs:
            if numerator > denominator:
                raise ValueError(f"{label} exceeds its total")
        return self


class EvaluationSnapshot(ContractModel):
    snapshot_id: str = Field(
        default_factory=lambda: new_id("evaluation_snapshot")
    )
    run_id: str
    report_id: str | None = None
    report_revision_id: str | None = None
    report_markdown: str = ""
    sources: tuple[SourceObservation, ...] = ()
    citations: tuple[CitationObservation, ...] = ()
    sections: tuple[SectionObservation, ...] = ()
    tool_calls: tuple[ToolCallObservation, ...] = ()
    counters: ExecutionCounters = Field(default_factory=ExecutionCounters)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    latency_ms: float = Field(default=0.0, ge=0.0)
    schema_errors: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    source_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "snapshot_id",
        "run_id",
        "report_id",
        "report_revision_id",
        "output_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _unique_observations(self) -> "EvaluationSnapshot":
        groups = (
            [item.source_id for item in self.sources],
            [item.citation_id for item in self.citations],
            [item.section_id for item in self.sections],
            [item.call_id for item in self.tool_calls],
        )
        if any(len(group) != len(set(group)) for group in groups):
            raise ValueError("evaluation observations require unique IDs")
        return self


class DeterministicEvaluationReport(ContractModel):
    report_id: str = Field(
        default_factory=lambda: new_id("deterministic_evaluation")
    )
    snapshot_id: str
    evaluator_version: str
    metrics: tuple[EvaluationMetric, ...]
    aggregate_score: float = Field(ge=0.0, le=1.0)
    details_artifact_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("report_id", "snapshot_id", "details_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("evaluator_version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)

    @model_validator(mode="after")
    def _unique_metrics(self) -> "DeterministicEvaluationReport":
        names = [item.name for item in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("deterministic metric names must be unique")
        return self


class ReplayExecutionResult(ContractModel):
    execution_id: str = Field(
        default_factory=lambda: new_id("replay_execution")
    )
    run_id: str
    output_artifact_id: str
    event_artifact_id: str
    snapshot: EvaluationSnapshot
    network_calls: int = Field(default=0, ge=0)
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "execution_id",
        "run_id",
        "output_artifact_id",
        "event_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class FrozenReplayEvaluation(ContractModel):
    frozen_result_id: str = Field(
        default_factory=lambda: new_id("frozen_result")
    )
    replay: FrozenReplay
    subject_version_id: str
    repeat_count: int = Field(ge=2, le=100)
    executions: tuple[ReplayExecutionResult, ...]
    evaluations: tuple[DeterministicEvaluationReport, ...]
    expected_output_match: bool
    expected_event_match: bool
    deterministic: bool
    network_free: bool
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "frozen_result_id",
        "subject_version_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _repeat_lengths(self) -> "FrozenReplayEvaluation":
        if (
            len(self.executions) != self.repeat_count
            or len(self.evaluations) != self.repeat_count
        ):
            raise ValueError("Frozen Replay repeat results are incomplete")
        return self


class LiveWebExecution(ContractModel):
    execution_id: str = Field(default_factory=lambda: new_id("live_execution"))
    repetition: int = Field(ge=0)
    seed: int = Field(ge=0)
    snapshot: EvaluationSnapshot
    evaluation: DeterministicEvaluationReport
    started_at: datetime
    completed_at: datetime

    @field_validator("execution_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _ordered(self) -> "LiveWebExecution":
        if self.completed_at < self.started_at:
            raise ValueError("live execution completion precedes start")
        return self


class LiveWebEvaluation(ContractModel):
    live_result_id: str = Field(default_factory=lambda: new_id("live_result"))
    subject_version_id: str
    bundle_id: str
    dataset_id: str
    dataset_access_record_id: str
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    sample_id: str
    repeat_count: int = Field(ge=2, le=100)
    executions: tuple[LiveWebExecution, ...]
    metric_means: dict[str, float]
    metric_variances: dict[str, float]
    changed_source_ids: tuple[str, ...]
    source_change_rate: float = Field(ge=0.0, le=1.0)
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "live_result_id",
        "subject_version_id",
        "bundle_id",
        "dataset_id",
        "dataset_access_record_id",
        "sample_id",
        "result_artifact_id",
        "changed_source_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _repeat_length(self) -> "LiveWebEvaluation":
        if len(self.executions) != self.repeat_count:
            raise ValueError("Live Web repeat results are incomplete")
        if set(self.metric_means) != set(self.metric_variances):
            raise ValueError("Live Web mean/variance metric sets differ")
        if not self.metric_means:
            raise ValueError("Live Web evaluation requires complete metrics")
        return self


class EnvironmentDescriptor(ContractModel):
    environment_id: str = Field(
        default_factory=lambda: new_id("environment")
    )
    platform: str
    python_version: str
    dependency_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    configuration_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    network_mode: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("environment_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)


class ExperimentDefinition(ContractModel):
    experiment_id: str = Field(
        default_factory=lambda: new_id("experiment")
    )
    name: str = Field(min_length=1, max_length=300)
    baseline: SystemBaseline
    subject_version_id: str
    mode: EvaluationMode
    bundle_id: str
    dataset_id: str
    dataset_split: DatasetSplit
    dataset_purpose: DatasetPurpose
    dataset_access_record_id: str
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    component_versions: ComponentVersionSet
    environment: EnvironmentDescriptor
    input_artifact_ids: tuple[str, ...]
    configuration_artifact_ids: tuple[str, ...] = ()
    deterministic_seed: int = Field(ge=0)
    repeat_count: int = Field(ge=2, le=100)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "experiment_id",
        "subject_version_id",
        "bundle_id",
        "dataset_id",
        "dataset_access_record_id",
        "input_artifact_ids",
        "configuration_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _mode_repeats(self) -> "ExperimentDefinition":
        DatasetAccessRequest(
            actor_id=self.subject_version_id,
            dataset_id=self.dataset_id,
            split=self.dataset_split,
            purpose=self.dataset_purpose,
        )
        if not self.input_artifact_ids:
            raise ValueError(
                "experiment definitions require input provenance artifacts"
            )
        return self


class ExperimentRun(ContractModel):
    experiment_run_id: str = Field(
        default_factory=lambda: new_id("experiment_run")
    )
    experiment_id: str
    baseline: SystemBaseline
    subject_version_id: str
    dataset_id: str
    dataset_split: DatasetSplit
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    component_versions: ComponentVersionSet
    environment_id: str
    status: ExperimentRunStatus
    metrics: tuple[EvaluationMetric, ...]
    evaluation_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    failure_summary: str | None = None
    started_at: datetime
    completed_at: datetime

    @field_validator(
        "experiment_run_id",
        "experiment_id",
        "subject_version_id",
        "dataset_id",
        "environment_id",
        "evaluation_artifact_ids",
        "output_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _consistent(self) -> "ExperimentRun":
        if self.completed_at < self.started_at:
            raise ValueError("experiment completion precedes start")
        failed = self.status == ExperimentRunStatus.FAILED
        if failed != bool(self.failure_summary):
            raise ValueError(
                "failed experiment runs require a failure summary only"
            )
        if not failed and not self.metrics:
            raise ValueError("successful experiment runs require metrics")
        if not failed and (
            not self.evaluation_artifact_ids
            or not self.output_artifact_ids
        ):
            raise ValueError(
                "successful experiment runs require evaluation and output "
                "artifact provenance"
            )
        names = [item.name for item in self.metrics]
        if len(names) != len(set(names)):
            raise ValueError("experiment metric names must be unique")
        return self


class MetricComparison(ContractModel):
    metric_name: str
    legacy_value: float
    fixed_workflow_value: float
    new_runtime_value: float
    new_vs_legacy_delta: float
    new_vs_fixed_delta: float
    direction: str


class ExperimentComparison(ContractModel):
    comparison_id: str = Field(
        default_factory=lambda: new_id("experiment_comparison")
    )
    experiment_run_ids: tuple[str, str, str]
    dataset_id: str
    dataset_split: DatasetSplit
    dataset_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    metrics: tuple[MetricComparison, ...]
    unavailable_metrics: dict[SystemBaseline, tuple[str, ...]]
    result_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator(
        "comparison_id",
        "experiment_run_ids",
        "dataset_id",
        "result_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, str, str]):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(set(value)) != 3:
                raise ValueError("comparison run IDs must be unique")
            return value
        return validate_identifier(value)


class RegisteredDataset(ContractModel):
    bundle: DatasetBundle
    definitions: tuple[DatasetDefinition, ...]
    samples: tuple[DatasetSample, ...]

    @model_validator(mode="after")
    def _consistent(self) -> "RegisteredDataset":
        definitions = {item.dataset_id: item for item in self.definitions}
        if len(definitions) != len(self.definitions):
            raise ValueError("registered dataset definitions must be unique")
        if set(definitions) != set(self.bundle.split_dataset_ids.values()):
            raise ValueError("registered definitions do not match bundle")
        sample_ids = [item.sample_id for item in self.samples]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("registered dataset sample IDs must be unique")
        for sample in self.samples:
            definition = definitions.get(sample.dataset_id)
            if definition is None or definition.split != sample.split:
                raise ValueError(
                    "registered sample does not match its split definition"
                )
        for split, dataset_id in self.bundle.split_dataset_ids.items():
            definition = definitions[dataset_id]
            if (
                definition.split != split
                or definition.manifest_artifact_id
                != self.bundle.manifest_artifact_id
            ):
                raise ValueError(
                    "registered definition disagrees with its bundle"
                )
            actual_count = sum(
                item.dataset_id == dataset_id for item in self.samples
            )
            if actual_count != definition.sample_count:
                raise ValueError(
                    "registered definition sample count is inconsistent"
                )
        return self
