from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import (
    ComponentKind,
    ContractModel,
    DatasetSplit,
    EvaluationMetric,
    VersionRef,
    utc_now,
)
from deep_researcher.contracts._base import new_id, validate_identifier


class OptimizationTarget(str, Enum):
    PLANNING = "planning"
    QUERY_GENERATION = "query_generation"
    SOURCE_SELECTION = "source_selection"
    EXTRACTION_CITATION = "extraction_citation"
    REPORT_WRITING = "report_writing"
    TOOL_ROUTING = "tool_routing"
    STOP_POLICY = "stop_policy"
    GRADER_RUBRIC = "grader_rubric"


_TARGET_COMPONENT_KINDS = {
    OptimizationTarget.PLANNING: frozenset(
        {ComponentKind.SKILL, ComponentKind.PROMPT}
    ),
    OptimizationTarget.QUERY_GENERATION: frozenset(
        {ComponentKind.SKILL, ComponentKind.PROMPT}
    ),
    OptimizationTarget.SOURCE_SELECTION: frozenset(
        {
            ComponentKind.SKILL,
            ComponentKind.PROMPT,
            ComponentKind.TOOL_POLICY,
        }
    ),
    OptimizationTarget.EXTRACTION_CITATION: frozenset(
        {ComponentKind.SKILL, ComponentKind.PROMPT}
    ),
    OptimizationTarget.REPORT_WRITING: frozenset(
        {ComponentKind.SKILL, ComponentKind.PROMPT}
    ),
    OptimizationTarget.TOOL_ROUTING: frozenset(
        {ComponentKind.TOOL_POLICY}
    ),
    OptimizationTarget.STOP_POLICY: frozenset(
        {ComponentKind.STOP_POLICY}
    ),
    OptimizationTarget.GRADER_RUBRIC: frozenset({ComponentKind.RUBRIC}),
}


def target_accepts_component(
    target: OptimizationTarget,
    component_kind: ComponentKind,
) -> bool:
    return component_kind in _TARGET_COMPONENT_KINDS[target]


class PoolSourceKind(str, Enum):
    SCORED_SUCCESS_TRACE = "scored_success_trace"
    SCORED_FAILURE_TRACE = "scored_failure_trace"
    BADCASE = "badcase"
    EVALUATION = "evaluation"


class PoolEntryStatus(str, Enum):
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    REJECTED = "rejected"


class MemoryLayer(str, Enum):
    RUNTIME = "runtime_memory"
    CROSS_TASK_EXPERIENCE = "cross_task_experience"
    FORMAL_SKILL_REGISTRY = "formal_skill_registry"


class CandidatePoolEntry(ContractModel):
    pool_entry_id: str = Field(
        default_factory=lambda: new_id("evolution_pool")
    )
    source_kind: PoolSourceKind
    source_artifact_id: str
    entry_artifact_id: str
    source_run_id: str | None = None
    source_span_id: str | None = None
    evaluation_artifact_ids: tuple[str, ...] = ()
    score_metrics: tuple[EvaluationMetric, ...] = ()
    observed_dataset_splits: tuple[DatasetSplit, ...] = ()
    production_trace: bool = False
    submitted_by: str
    submitted_at: datetime = Field(default_factory=utc_now)
    invokes_optimizer: bool = False
    publishes_versions: bool = False

    @field_validator(
        "pool_entry_id",
        "source_artifact_id",
        "entry_artifact_id",
        "source_run_id",
        "source_span_id",
        "submitted_by",
        "evaluation_artifact_ids",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            for item in value:
                validate_identifier(item)
            if len(value) != len(set(value)):
                raise ValueError("pool evaluation artifacts must be unique")
            return value
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _source_requirements(self) -> "CandidatePoolEntry":
        metric_names = [item.name for item in self.score_metrics]
        if len(metric_names) != len(set(metric_names)):
            raise ValueError("pool score metrics must be unique")
        trace_kinds = {
            PoolSourceKind.SCORED_SUCCESS_TRACE,
            PoolSourceKind.SCORED_FAILURE_TRACE,
        }
        if self.source_kind in trace_kinds:
            if self.source_run_id is None:
                raise ValueError("scored trace requires a source run")
            if not self.evaluation_artifact_ids or not self.score_metrics:
                raise ValueError(
                    "scored trace requires evaluation artifacts and metrics"
                )
        elif self.production_trace:
            raise ValueError(
                "only scored traces may be marked as production traces"
            )
        if self.source_kind == PoolSourceKind.EVALUATION:
            if self.evaluation_artifact_ids:
                raise ValueError(
                    "evaluation pool entry uses source_artifact_id directly"
                )
        if self.invokes_optimizer or self.publishes_versions:
            raise ValueError(
                "candidate-pool submission cannot optimize or publish"
            )
        return self

    @field_validator("observed_dataset_splits")
    @classmethod
    def _unique_splits(
        cls,
        value: tuple[DatasetSplit, ...],
    ) -> tuple[DatasetSplit, ...]:
        return tuple(dict.fromkeys(value))


class CandidatePoolReview(ContractModel):
    review_id: str = Field(
        default_factory=lambda: new_id("evolution_pool_review")
    )
    pool_entry_id: str
    review_artifact_id: str
    approved: bool
    assigned_split: DatasetSplit | None = None
    reviewer_id: str
    review_note: str = Field(min_length=1, max_length=8000)
    reviewed_at: datetime = Field(default_factory=utc_now)
    triggers_generation: bool = False

    @field_validator(
        "review_id",
        "pool_entry_id",
        "review_artifact_id",
        "reviewer_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _review_boundary(self) -> "CandidatePoolReview":
        if self.approved:
            if self.assigned_split not in {
                DatasetSplit.TRAIN,
                DatasetSplit.DEV,
            }:
                raise ValueError(
                    "approved pool sources may enter train or dev only"
                )
        elif self.assigned_split is not None:
            raise ValueError("rejected pool source cannot receive a split")
        if self.triggers_generation:
            raise ValueError("pool review cannot trigger candidate generation")
        return self


class ReviewedPoolEntry(ContractModel):
    entry: CandidatePoolEntry
    review: CandidatePoolReview | None = None
    status: PoolEntryStatus

    @model_validator(mode="after")
    def _projection(self) -> "ReviewedPoolEntry":
        expected = (
            PoolEntryStatus.PENDING_REVIEW
            if self.review is None
            else (
                PoolEntryStatus.APPROVED
                if self.review.approved
                else PoolEntryStatus.REJECTED
            )
        )
        if self.status != expected:
            raise ValueError("pool status disagrees with its review")
        if (
            self.review is not None
            and self.review.pool_entry_id != self.entry.pool_entry_id
        ):
            raise ValueError("pool review belongs to another entry")
        if self.status == PoolEntryStatus.APPROVED and any(
            split
            not in {
                DatasetSplit.TRAIN,
                DatasetSplit.DEV,
            }
            for split in self.entry.observed_dataset_splits
        ):
            raise ValueError(
                "selection/test/hidden-test sources cannot enter offline "
                "candidate generation"
            )
        if (
            self.status == PoolEntryStatus.APPROVED
            and self.entry.observed_dataset_splits
            and self.review is not None
            and (
                len(self.entry.observed_dataset_splits) != 1
                or self.review.assigned_split
                != self.entry.observed_dataset_splits[0]
            )
        ):
            raise ValueError(
                "candidate-pool review cannot relabel an observed dataset "
                "split"
            )
        return self


class PatchOperationKind(str, Enum):
    ADD = "add"
    DELETE = "delete"
    REPLACE = "replace"


class AddPlacement(str, Enum):
    START = "start"
    END = "end"
    BEFORE = "before"
    AFTER = "after"


class ExperienceEditSuggestion(ContractModel):
    operation: PatchOperationKind
    match_lines: tuple[str, ...] = ()
    new_lines: tuple[str, ...] = ()
    add_placement: AddPlacement | None = None

    @model_validator(mode="after")
    def _shape(self) -> "ExperienceEditSuggestion":
        if self.operation == PatchOperationKind.ADD:
            if not self.new_lines:
                raise ValueError("add suggestion requires new lines")
            if self.add_placement is None:
                raise ValueError("add suggestion requires placement")
            if (
                self.add_placement
                in {AddPlacement.BEFORE, AddPlacement.AFTER}
                and not self.match_lines
            ):
                raise ValueError(
                    "relative add suggestion requires anchor lines"
                )
            if (
                self.add_placement
                in {AddPlacement.START, AddPlacement.END}
                and self.match_lines
            ):
                raise ValueError(
                    "absolute add suggestion cannot carry anchor lines"
                )
        elif self.operation == PatchOperationKind.DELETE:
            if not self.match_lines or self.new_lines:
                raise ValueError(
                    "delete suggestion requires only matched lines"
                )
            if self.add_placement is not None:
                raise ValueError("delete suggestion has no add placement")
        else:
            if not self.match_lines or not self.new_lines:
                raise ValueError(
                    "replace suggestion requires old and new lines"
                )
            if self.add_placement is not None:
                raise ValueError("replace suggestion has no add placement")
        return self


class CrossTaskExperience(ContractModel):
    experience_id: str = Field(
        default_factory=lambda: new_id("evolution_experience")
    )
    target: OptimizationTarget
    summary: str = Field(min_length=1, max_length=4000)
    recommendation: str = Field(min_length=1, max_length=8000)
    suggestion: ExperienceEditSuggestion
    confidence: float = Field(gt=0.0, le=1.0)
    impact: float = Field(gt=0.0, le=1.0)
    pool_entry_ids: tuple[str, ...]
    source_artifact_ids: tuple[str, ...]
    created_by: str
    artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)
    memory_layer: MemoryLayer = MemoryLayer.CROSS_TASK_EXPERIENCE
    directly_deployable: bool = False

    @field_validator(
        "experience_id",
        "pool_entry_ids",
        "source_artifact_ids",
        "created_by",
        "artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            if not value:
                raise ValueError("experience provenance cannot be empty")
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _memory_boundary(self) -> "CrossTaskExperience":
        if self.memory_layer != MemoryLayer.CROSS_TASK_EXPERIENCE:
            raise ValueError("experience must stay in cross-task memory")
        if self.directly_deployable:
            raise ValueError("cross-task experience is not deployable")
        return self


class TextEditBudget(ContractModel):
    max_rounds: int = Field(ge=1, le=100)
    max_operations_per_round: int = Field(ge=1, le=100)
    max_added_lines_per_round: int = Field(ge=0, le=10000)
    max_deleted_lines_per_round: int = Field(ge=0, le=10000)
    max_changed_characters_per_round: int = Field(ge=1, le=1_000_000)
    max_edit_fraction_per_round: float = Field(gt=0.0, le=1.0)


class EvolutionCampaignRequest(ContractModel):
    campaign_id: str = Field(
        default_factory=lambda: new_id("evolution_campaign")
    )
    campaign_artifact_id: str
    target: OptimizationTarget
    component_kind: ComponentKind
    component_name: str = Field(min_length=1, max_length=200)
    base_version_id: str
    pool_entry_ids: tuple[str, ...]
    experience_ids: tuple[str, ...]
    edit_budget: TextEditBudget
    offline_environment_id: str
    created_by: str
    created_at: datetime = Field(default_factory=utc_now)
    allow_online_inference: bool = False
    allow_runtime_memory: bool = False
    allow_automatic_publish: bool = False

    @field_validator(
        "campaign_id",
        "campaign_artifact_id",
        "base_version_id",
        "pool_entry_ids",
        "experience_ids",
        "offline_environment_id",
        "created_by",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            if not value:
                raise ValueError("campaign inputs cannot be empty")
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _campaign_boundary(self) -> "EvolutionCampaignRequest":
        if not target_accepts_component(self.target, self.component_kind):
            raise ValueError(
                f"{self.target.value} cannot optimize "
                f"{self.component_kind.value}"
            )
        if (
            self.allow_online_inference
            or self.allow_runtime_memory
            or self.allow_automatic_publish
        ):
            raise ValueError(
                "offline evolution cannot use runtime memory, online "
                "inference, or automatic publication"
            )
        return self


class EvolutionInputSnapshot(ContractModel):
    snapshot_id: str = Field(
        default_factory=lambda: new_id("evolution_input")
    )
    campaign_id: str
    target: OptimizationTarget
    base_version: VersionRef
    base_content_artifact_id: str
    base_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    active_version_id_at_seal: str
    pool_entries: tuple[ReviewedPoolEntry, ...]
    experience_ids: tuple[str, ...]
    success_trace_artifact_ids: tuple[str, ...]
    failure_trace_artifact_ids: tuple[str, ...]
    badcase_artifact_ids: tuple[str, ...]
    evaluation_artifact_ids: tuple[str, ...]
    rejected_patch_fingerprints: tuple[str, ...] = ()
    snapshot_artifact_id: str
    sealed_at: datetime = Field(default_factory=utc_now)
    runtime_memory_used: bool = False
    online_inference_count: int = Field(default=0, ge=0)

    @field_validator(
        "snapshot_id",
        "campaign_id",
        "base_content_artifact_id",
        "active_version_id_at_seal",
        "experience_ids",
        "success_trace_artifact_ids",
        "failure_trace_artifact_ids",
        "badcase_artifact_ids",
        "evaluation_artifact_ids",
        "snapshot_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            if not value:
                raise ValueError(
                    "sealed evolution input category cannot be empty"
                )
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @field_validator("rejected_patch_fingerprints")
    @classmethod
    def _fingerprints(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if len(item) != 64 or any(
                char not in "0123456789abcdef" for char in item
            ):
                raise ValueError("rejected patch fingerprint must be SHA-256")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _sealed_boundary(self) -> "EvolutionInputSnapshot":
        if self.base_version.version_id != self.active_version_id_at_seal:
            raise ValueError("evolution must start from the active version")
        if self.base_version.artifact_id != self.base_content_artifact_id:
            raise ValueError("base version content artifact differs")
        if (
            self.base_version.content_hash is not None
            and self.base_version.content_hash != self.base_content_hash
        ):
            raise ValueError("base version content hash differs")
        if any(
            item.status != PoolEntryStatus.APPROVED
            or item.review is None
            or item.review.assigned_split
            not in {DatasetSplit.TRAIN, DatasetSplit.DEV}
            for item in self.pool_entries
        ):
            raise ValueError(
                "sealed evolution inputs require reviewed train/dev entries"
            )
        if self.runtime_memory_used or self.online_inference_count:
            raise ValueError(
                "offline input snapshot cannot use runtime memory or online "
                "inference"
            )
        return self


class TextPatchOperation(ContractModel):
    operation: PatchOperationKind
    start_line: int = Field(ge=0)
    end_line: int = Field(ge=0)
    old_lines: tuple[str, ...] = ()
    new_lines: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _operation_shape(self) -> "TextPatchOperation":
        if self.end_line < self.start_line:
            raise ValueError("patch line range is reversed")
        if self.operation == PatchOperationKind.ADD:
            if (
                self.start_line != self.end_line
                or self.old_lines
                or not self.new_lines
            ):
                raise ValueError("add patch requires an empty line range")
        elif self.operation == PatchOperationKind.DELETE:
            if (
                self.end_line <= self.start_line
                or not self.old_lines
                or self.new_lines
            ):
                raise ValueError("delete patch requires only old lines")
        elif (
            self.end_line <= self.start_line
            or not self.old_lines
            or not self.new_lines
        ):
            raise ValueError("replace patch requires old and new lines")
        if len(self.old_lines) != self.end_line - self.start_line:
            raise ValueError("patch old lines do not match its line range")
        return self


class PatchEditMetrics(ContractModel):
    operation_count: int = Field(ge=1)
    added_lines: int = Field(ge=0)
    deleted_lines: int = Field(ge=0)
    changed_characters: int = Field(ge=1)
    edit_fraction: float = Field(gt=0.0)


class OfflineGeneratorResult(ContractModel):
    operations: tuple[TextPatchOperation, ...]
    rationale: tuple[str, ...]
    consulted_experience_ids: tuple[str, ...]
    skipped_rejected_operation_fingerprints: tuple[str, ...] = ()
    online_inference_count: int = Field(default=0, ge=0)
    network_accessed: bool = False

    @field_validator("consulted_experience_ids")
    @classmethod
    def _experience_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("generator must consult cross-task experience")
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))

    @field_validator("skipped_rejected_operation_fingerprints")
    @classmethod
    def _operation_fingerprints(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        for item in value:
            if len(item) != 64:
                raise ValueError("operation fingerprint must be SHA-256")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _offline_only(self) -> "OfflineGeneratorResult":
        if not self.operations:
            raise ValueError("generator produced no structured edits")
        if not self.rationale:
            raise ValueError("generator must explain its structured edits")
        if self.online_inference_count or self.network_accessed:
            raise ValueError(
                "offline evolution generator accessed online inference"
            )
        return self


class EvolutionPatch(ContractModel):
    patch_id: str = Field(default_factory=lambda: new_id("evolution_patch"))
    campaign_id: str
    round_no: int = Field(ge=1)
    target: OptimizationTarget
    base_version_id: str
    base_artifact_id: str
    base_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    operations: tuple[TextPatchOperation, ...]
    rationale: tuple[str, ...]
    experience_ids: tuple[str, ...]
    rejected_patch_fingerprints_consulted: tuple[str, ...] = ()
    rejected_operation_fingerprints_skipped: tuple[str, ...] = ()
    operation_fingerprints: tuple[str, ...]
    edit_metrics: PatchEditMetrics
    patch_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)
    operation_vocabulary: tuple[PatchOperationKind, ...] = (
        PatchOperationKind.ADD,
        PatchOperationKind.DELETE,
        PatchOperationKind.REPLACE,
    )

    @field_validator(
        "patch_id",
        "campaign_id",
        "base_version_id",
        "base_artifact_id",
        "experience_ids",
        "artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...]):
        if isinstance(value, tuple):
            if not value:
                raise ValueError("patch requires experience provenance")
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value)

    @model_validator(mode="after")
    def _patch_shape(self) -> "EvolutionPatch":
        if not self.operations:
            raise ValueError("evolution patch cannot be empty")
        if len(self.operations) != self.edit_metrics.operation_count:
            raise ValueError("patch operation count disagrees with metrics")
        if len(self.operations) != len(self.operation_fingerprints):
            raise ValueError(
                "each patch operation needs an audit fingerprint"
            )
        if set(self.operation_vocabulary) != set(PatchOperationKind):
            raise ValueError("patch vocabulary must be add/delete/replace")
        return self


class CandidateStatus(str, Enum):
    GENERATED = "generated"
    SELECTION_PASSED = "selection_passed"
    SELECTION_REJECTED = "selection_rejected"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    FINAL_REJECTED = "final_rejected"
    PROMOTED = "promoted"
    KEPT = "kept"
    ROLLED_BACK = "rolled_back"


class EvolutionCandidate(ContractModel):
    candidate_id: str = Field(
        default_factory=lambda: new_id("evolution_candidate")
    )
    campaign_id: str
    round_no: int = Field(ge=1)
    patch_id: str
    patch_artifact_id: str
    base_version_id: str
    version_ref: VersionRef
    content_artifact_id: str
    candidate_artifact_id: str
    created_at: datetime = Field(default_factory=utc_now)
    online_inference_count: int = Field(default=0, ge=0)
    automatically_published: bool = False

    @field_validator(
        "candidate_id",
        "campaign_id",
        "patch_id",
        "patch_artifact_id",
        "base_version_id",
        "content_artifact_id",
        "candidate_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _candidate_boundary(self) -> "EvolutionCandidate":
        if self.version_ref.artifact_id != self.content_artifact_id:
            raise ValueError("candidate version content artifact differs")
        if self.online_inference_count or self.automatically_published:
            raise ValueError(
                "offline candidate cannot add online inference or publish"
            )
        return self


class RejectedEditMemory(ContractModel):
    rejection_id: str = Field(
        default_factory=lambda: new_id("rejected_edit")
    )
    campaign_id: str
    candidate_id: str
    target: OptimizationTarget
    base_version_id: str
    patch_id: str
    patch_fingerprint: str = Field(pattern=r"^[a-f0-9]{64}$")
    operation_fingerprints: tuple[str, ...]
    failed_check_names: tuple[str, ...]
    reasons: tuple[str, ...]
    gate_decision_id: str | None = None
    gate_decision_artifact_id: str | None = None
    source_experience_ids: tuple[str, ...]
    artifact_id: str
    rejected_at: datetime = Field(default_factory=utc_now)
    memory_layer: MemoryLayer = MemoryLayer.CROSS_TASK_EXPERIENCE
    directly_deployable: bool = False

    @field_validator(
        "rejection_id",
        "campaign_id",
        "candidate_id",
        "base_version_id",
        "patch_id",
        "gate_decision_id",
        "gate_decision_artifact_id",
        "source_experience_ids",
        "artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | tuple[str, ...] | None):
        if isinstance(value, tuple):
            if not value:
                raise ValueError("rejection requires experience provenance")
            for item in value:
                validate_identifier(item)
            return tuple(dict.fromkeys(value))
        return validate_identifier(value) if value is not None else None

    @field_validator("operation_fingerprints")
    @classmethod
    def _operation_hashes(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("rejection requires operation fingerprints")
        if any(len(item) != 64 for item in value):
            raise ValueError("operation fingerprints must be SHA-256")
        return tuple(dict.fromkeys(value))

    @model_validator(mode="after")
    def _rejected_memory(self) -> "RejectedEditMemory":
        if not self.failed_check_names or not self.reasons:
            raise ValueError("rejection requires checks and reasons")
        if self.memory_layer != MemoryLayer.CROSS_TASK_EXPERIENCE:
            raise ValueError("rejection history is cross-task experience")
        if self.directly_deployable:
            raise ValueError("rejected edit memory is not deployable")
        return self


class HumanGateOutcome(str, Enum):
    APPROVED = "approved"
    REJECTED = "rejected"


class EvolutionHumanDecision(ContractModel):
    human_decision_id: str = Field(
        default_factory=lambda: new_id("evolution_human")
    )
    campaign_id: str
    candidate_id: str
    selection_gate_decision_id: str
    selection_gate_artifact_id: str
    outcome: HumanGateOutcome
    reviewer_id: str
    note: str = Field(min_length=1, max_length=8000)
    decision_artifact_id: str
    decided_at: datetime = Field(default_factory=utc_now)
    triggers_final_evaluation: bool = False
    publishes_version: bool = False

    @field_validator(
        "human_decision_id",
        "campaign_id",
        "candidate_id",
        "selection_gate_decision_id",
        "selection_gate_artifact_id",
        "reviewer_id",
        "decision_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _human_boundary(self) -> "EvolutionHumanDecision":
        if self.triggers_final_evaluation or self.publishes_version:
            raise ValueError(
                "human decision records approval only; final evaluation and "
                "publication remain explicit actions"
            )
        return self


class BestSkillSnapshot(ContractModel):
    best_skill_id: str = Field(
        default_factory=lambda: new_id("best_skill")
    )
    target: OptimizationTarget
    component_name: str
    skill_version_id: str
    skill_content_artifact_id: str
    previous_best_skill_id: str | None = None
    release_gate_decision_id: str
    release_gate_artifact_id: str
    human_decision_id: str
    artifact_id: str
    published_at: datetime = Field(default_factory=utc_now)
    memory_layer: MemoryLayer = MemoryLayer.FORMAL_SKILL_REGISTRY
    static_versioned: bool = True
    online_inference_count: int = Field(default=0, ge=0)

    @field_validator(
        "best_skill_id",
        "skill_version_id",
        "skill_content_artifact_id",
        "previous_best_skill_id",
        "release_gate_decision_id",
        "release_gate_artifact_id",
        "human_decision_id",
        "artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _formal_registry_boundary(self) -> "BestSkillSnapshot":
        if self.memory_layer != MemoryLayer.FORMAL_SKILL_REGISTRY:
            raise ValueError("best_skill must live in the formal registry")
        if not self.static_versioned or self.online_inference_count:
            raise ValueError(
                "best_skill is a static version with no online inference"
            )
        return self


class GenerationAttemptStatus(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    ABANDONED = "abandoned"


class GenerationAttempt(ContractModel):
    attempt_id: str = Field(
        default_factory=lambda: new_id("evolution_generation")
    )
    campaign_id: str
    attempt_no: int = Field(ge=1)
    round_no: int = Field(ge=1)
    worker_id: str
    status: GenerationAttemptStatus
    started_at: datetime
    completed_at: datetime | None = None
    candidate_id: str | None = None
    error: str | None = Field(default=None, max_length=8000)

    @field_validator(
        "attempt_id",
        "campaign_id",
        "worker_id",
        "candidate_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _attempt_state(self) -> "GenerationAttempt":
        if self.status == GenerationAttemptStatus.RUNNING:
            if (
                self.completed_at is not None
                or self.candidate_id is not None
                or self.error is not None
            ):
                raise ValueError("running generation attempt is not terminal")
        else:
            if self.completed_at is None:
                raise ValueError("terminal generation attempt needs completion")
            if self.status == GenerationAttemptStatus.SUCCEEDED:
                if self.candidate_id is None or self.error is not None:
                    raise ValueError(
                        "successful generation needs only a candidate"
                    )
            elif self.candidate_id is not None or self.error is None:
                raise ValueError(
                    "abandoned generation needs only a structured error"
                )
        return self


class EvolutionCampaignStatus(str, Enum):
    DRAFT = "draft"
    READY = "ready"
    GENERATING = "generating"
    CANDIDATE_READY = "candidate_ready"
    AWAITING_HUMAN = "awaiting_human"
    READY_FOR_FINAL = "ready_for_final"
    PROMOTED = "promoted"
    ROLLED_BACK = "rolled_back"
    EXHAUSTED = "exhausted"


class EvolutionJournalKind(str, Enum):
    CREATED = "created"
    INPUTS_SEALED = "inputs_sealed"
    GENERATION_STARTED = "generation_started"
    GENERATION_ABANDONED = "generation_abandoned"
    CANDIDATE_GENERATED = "candidate_generated"
    SELECTION_PASSED = "selection_passed"
    SELECTION_REJECTED = "selection_rejected"
    HUMAN_APPROVED = "human_approved"
    HUMAN_REJECTED = "human_rejected"
    FINAL_REJECTED = "final_rejected"
    PROMOTED = "promoted"
    POST_RELEASE_KEPT = "post_release_kept"
    ROLLED_BACK = "rolled_back"
    EXHAUSTED = "exhausted"


class EvolutionJournalEntry(ContractModel):
    journal_event_id: str = Field(
        default_factory=lambda: new_id("evolution_event")
    )
    campaign_id: str
    sequence: int = Field(ge=1)
    kind: EvolutionJournalKind
    payload: dict[str, Any]
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("journal_event_id", "campaign_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)


class EvolutionCandidateRecord(ContractModel):
    candidate: EvolutionCandidate
    patch: EvolutionPatch
    status: CandidateStatus
    selection_gate_decision_id: str | None = None
    selection_gate_artifact_id: str | None = None
    human_decision: EvolutionHumanDecision | None = None
    final_gate_decision_id: str | None = None
    final_gate_artifact_id: str | None = None
    post_release_gate_decision_id: str | None = None
    post_release_gate_artifact_id: str | None = None
    rejection: RejectedEditMemory | None = None
    best_skill: BestSkillSnapshot | None = None

    @field_validator(
        "selection_gate_decision_id",
        "selection_gate_artifact_id",
        "final_gate_decision_id",
        "final_gate_artifact_id",
        "post_release_gate_decision_id",
        "post_release_gate_artifact_id",
    )
    @classmethod
    def _optional_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _candidate_projection(self) -> "EvolutionCandidateRecord":
        if self.candidate.patch_id != self.patch.patch_id:
            raise ValueError("candidate and patch differ")
        if self.status != CandidateStatus.GENERATED and (
            self.selection_gate_decision_id is None
            or self.selection_gate_artifact_id is None
        ):
            raise ValueError(
                "evaluated candidate needs a selection-gate decision"
            )
        if self.status in {
            CandidateStatus.HUMAN_APPROVED,
            CandidateStatus.HUMAN_REJECTED,
            CandidateStatus.FINAL_REJECTED,
            CandidateStatus.PROMOTED,
            CandidateStatus.KEPT,
            CandidateStatus.ROLLED_BACK,
        } and self.human_decision is None:
            raise ValueError("candidate status requires human decision")
        if self.status in {
            CandidateStatus.FINAL_REJECTED,
            CandidateStatus.PROMOTED,
            CandidateStatus.KEPT,
            CandidateStatus.ROLLED_BACK,
        } and (
            self.final_gate_decision_id is None
            or self.final_gate_artifact_id is None
        ):
            raise ValueError("final candidate status needs final gate")
        return self


class EvolutionCampaignRecord(ContractModel):
    request: EvolutionCampaignRequest
    status: EvolutionCampaignStatus
    revision: int = Field(ge=1)
    recovery_count: int = Field(ge=0)
    input_snapshot: EvolutionInputSnapshot | None = None
    attempts: tuple[GenerationAttempt, ...] = ()
    candidates: tuple[EvolutionCandidateRecord, ...] = ()

    @model_validator(mode="after")
    def _record_identity(self) -> "EvolutionCampaignRecord":
        campaign_id = self.request.campaign_id
        if self.input_snapshot is not None and (
            self.input_snapshot.campaign_id != campaign_id
        ):
            raise ValueError("input snapshot belongs to another campaign")
        if any(item.campaign_id != campaign_id for item in self.attempts):
            raise ValueError("generation attempt belongs to another campaign")
        if any(
            item.candidate.campaign_id != campaign_id
            for item in self.candidates
        ):
            raise ValueError("candidate belongs to another campaign")
        if len(self.candidates) > self.request.edit_budget.max_rounds:
            raise ValueError("campaign exceeded its round budget")
        return self


class EvolutionCampaignPage(ContractModel):
    items: tuple[EvolutionCampaignRecord, ...]
    next_cursor: str | None = None


class CandidatePoolPage(ContractModel):
    items: tuple[ReviewedPoolEntry, ...]
    next_cursor: str | None = None
