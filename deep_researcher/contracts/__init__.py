"""Framework-independent, versioned contracts for the Background001 architecture."""

from ._base import CONTRACT_SCHEMA_VERSION, ContractModel, ExtensibleContract, VersionedEntity, new_id, utc_now
from .agents import AgentRole, AgentSpec, MiddlewareSpec, MiddlewareStage, ToolGrant
from .artifacts import ArtifactEnvelope, ArtifactKind, ArtifactLink, ArtifactLinkRelation, ArtifactStatus, Sensitivity
from .budgets import Budget, BudgetDimension, BudgetUsage, StopDecision, StopReason, combine_usage
from .commands import AgentDecisionSummary, Command, CommandKind, CommandStatus, Observation, ObservationStatus
from .errors import ErrorCategory, ErrorRecord
from .evaluation import (
    DatasetAccessRequest,
    DatasetDefinition,
    DatasetPurpose,
    DatasetSample,
    DatasetSplit,
    EvaluationMetric,
    EvaluationResult,
    FrozenReplay,
    MetricDirection,
    assert_dataset_access,
)
from .events import EventLevel, EventType, RunEvent, RunStatus, SpanKind
from .evidence import (
    AtomicFact,
    Citation,
    CitationStatus,
    Claim,
    ClaimStatus,
    Conflict,
    ConflictResolutionKind,
    ConflictSeverity,
    ConflictStatus,
    EntityProvenance,
    Evidence,
    EvidenceQuote,
    EvidenceRelation,
    EvidenceStatus,
    FactStatus,
    Passage,
    PassageStatus,
    Report,
    ReportStatus,
    Section,
    SectionCoverageStatus,
    SectionStatus,
    SnapshotStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    SourceType,
)
from .schema_evolution import SchemaMigrationRegistry, SerializedContract, canonical_contract_json, contract_fingerprint
from .tasks import TaskEnvelope, TaskKind, TaskResult, TaskResultStatus, TaskStatus
from .verification import RepairAction, RepairRequest, VerificationCategory, VerificationIssue, VerificationResult, VerificationSeverity
from .versioning import ComponentKind, ComponentVersionSet, VersionRef

__all__ = [name for name in globals() if not name.startswith("_")]
