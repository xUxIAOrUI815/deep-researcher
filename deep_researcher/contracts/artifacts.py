from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier


class ArtifactKind(str, Enum):
    SEARCH_RESPONSE = "search_response"
    SOURCE_SNAPSHOT = "source_snapshot"
    CLEANED_CONTENT = "cleaned_content"
    PASSAGE = "passage"
    TOOL_RESULT = "tool_result"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    EVIDENCE_PACK = "evidence_pack"
    REPORT = "report"
    PROMPT = "prompt"
    SKILL = "skill"
    POLICY = "policy"
    RUBRIC = "rubric"
    DATASET_SAMPLE = "dataset_sample"
    DATASET_MANIFEST = "dataset_manifest"
    EVALUATION_RESULT = "evaluation_result"
    FROZEN_REPLAY_RESULT = "frozen_replay_result"
    LIVE_WEB_RESULT = "live_web_result"
    EXPERIMENT_RESULT = "experiment_result"
    SEMANTIC_EVALUATION_RESULT = "semantic_evaluation_result"
    JUDGE_EVALUATION = "judge_evaluation"
    JUDGE_CALIBRATION = "judge_calibration"
    RELEASE_GATE_DECISION = "release_gate_decision"
    RELEASE_GATE_APPLICATION = "release_gate_application"
    VERSION_MANIFEST = "version_manifest"
    VERSION_TRANSITION = "version_transition"
    VERIFICATION_RESULT = "verification_result"
    REPAIR_FEEDBACK = "repair_feedback"
    SUPERVISOR_PLAN = "supervisor_plan"
    WORKER_DELEGATION = "worker_delegation"
    WORKER_RESULT = "worker_result"
    RESEARCH_MERGE = "research_merge"
    CONVERGENCE_DECISION = "convergence_decision"
    CITATION_MAP = "citation_map"
    REPORT_REVIEW = "report_review"
    REPORT_LOOP_DECISION = "report_loop_decision"
    TRACE_EXPORT = "trace_export"
    STATE_PATCH = "state_patch"
    REPLAY_CAPSULE = "replay_capsule"
    STUDIO_REPLAY_RESULT = "studio_replay_result"
    STUDIO_COMPARISON = "studio_comparison"
    BADCASE = "badcase"
    EVOLUTION_POOL_ENTRY = "evolution_pool_entry"
    EVOLUTION_POOL_REVIEW = "evolution_pool_review"
    EVOLUTION_EXPERIENCE = "evolution_experience"
    EVOLUTION_CAMPAIGN = "evolution_campaign"
    EVOLUTION_INPUT = "evolution_input"
    EVOLUTION_PATCH = "evolution_patch"
    EVOLUTION_CANDIDATE = "evolution_candidate"
    EVOLUTION_HUMAN_DECISION = "evolution_human_decision"
    REJECTED_EDIT_MEMORY = "rejected_edit_memory"
    BEST_SKILL = "best_skill"
    OTHER = "other"


class ArtifactStatus(str, Enum):
    AVAILABLE = "available"
    QUARANTINED = "quarantined"
    EXPIRED = "expired"
    CORRUPT = "corrupt"
    DELETED = "deleted"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class ArtifactEnvelope(ContractModel):
    artifact_id: str = Field(default_factory=lambda: new_id("artifact"))
    kind: ArtifactKind
    content_uri: str = Field(min_length=1, max_length=2048)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    hash_algorithm: str = Field(default="sha256", pattern=r"^sha256$")
    byte_length: int = Field(ge=0)
    media_type: str = Field(min_length=1, max_length=255)
    content_schema: str | None = Field(default=None, max_length=255)
    producer_id: str
    run_id: str
    task_id: str | None = None
    source_artifact_ids: tuple[str, ...] = ()
    status: ArtifactStatus = ArtifactStatus.AVAILABLE
    sensitivity: Sensitivity = Sensitivity.INTERNAL
    created_at: datetime = Field(default_factory=utc_now)
    expires_at: datetime | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "artifact_id",
        "producer_id",
        "run_id",
        "task_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("source_artifact_ids")
    @classmethod
    def _source_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("source artifacts must be unique")
        return value

    @field_validator("created_at", "expires_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("artifact timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _expiry_after_creation(self) -> "ArtifactEnvelope":
        if self.expires_at is not None and self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if self.artifact_id in self.source_artifact_ids:
            raise ValueError("artifact cannot cite itself as a source")
        return self


class ArtifactLinkRelation(str, Enum):
    DERIVED_FROM = "derived_from"
    SUPERSEDES = "supersedes"
    REPRESENTS = "represents"
    EVALUATES = "evaluates"
    REPLAYS = "replays"


class ArtifactLink(ContractModel):
    source_artifact_id: str
    target_artifact_id: str
    relation: ArtifactLinkRelation
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_artifact_id", "target_artifact_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _not_self(self) -> "ArtifactLink":
        if self.source_artifact_id == self.target_artifact_id:
            raise ValueError("artifact link cannot be self-referential")
        return self
