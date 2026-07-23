from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, TypeVar

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier


class EntityProvenance(ContractModel):
    producer_id: str
    run_id: str
    task_id: str | None = None
    causation_event_id: str | None = None
    source_artifact_ids: tuple[str, ...] = ()

    @field_validator("producer_id", "run_id", "task_id", "causation_event_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("source_artifact_ids")
    @classmethod
    def _artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("source artifact references must be unique")
        return value


class SourceType(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    DATASET = "dataset"
    OFFICIAL_DOCUMENTATION = "official_documentation"
    ACADEMIC = "academic"
    NEWS = "news"
    COMMUNITY = "community"
    OTHER = "other"


class SourceLevel(str, Enum):
    PRIMARY = "primary"
    SECONDARY = "secondary"
    TERTIARY = "tertiary"
    UNKNOWN = "unknown"


class SourceStatus(str, Enum):
    DISCOVERED = "discovered"
    ACCESSIBLE = "accessible"
    UNREACHABLE = "unreachable"
    BLOCKED = "blocked"
    RETIRED = "retired"


class SnapshotStatus(str, Enum):
    CAPTURED = "captured"
    NORMALIZED = "normalized"
    FAILED = "failed"
    SUPERSEDED = "superseded"


class PassageStatus(str, Enum):
    CANDIDATE = "candidate"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class EvidenceRelation(str, Enum):
    SUPPORTS = "supports"
    REFUTES = "refutes"
    CONTEXTUALIZES = "contextualizes"


class EvidenceStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class FactStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    DISPUTED = "disputed"
    REJECTED = "rejected"


class ClaimStatus(str, Enum):
    DRAFT = "draft"
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONTRADICTED = "contradicted"
    CONFLICTED = "conflicted"
    UNSUPPORTED = "unsupported"
    STALE = "stale"
    CONTESTED = "contested"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class CitationStatus(str, Enum):
    PROPOSED = "proposed"
    VERIFIED = "verified"
    REJECTED = "rejected"


class ConflictStatus(str, Enum):
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    ACCEPTED_UNRESOLVED = "accepted_unresolved"


class ConflictSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ConflictResolutionKind(str, Enum):
    SOURCE_PRECEDENCE = "source_precedence"
    TEMPORAL = "temporal"
    SCOPE = "scope"
    DATA_CORRECTION = "data_correction"
    SYNTHESIZED = "synthesized"
    ACCEPTED_UNRESOLVED = "accepted_unresolved"


class SectionCoverageStatus(str, Enum):
    NOT_ASSESSED = "not_assessed"
    INSUFFICIENT = "insufficient"
    PARTIAL = "partial"
    COMPLETE = "complete"
    BLOCKED = "blocked"


class SectionStatus(str, Enum):
    PLANNED = "planned"
    DRAFTING = "drafting"
    NEEDS_REPAIR = "needs_repair"
    VERIFIED = "verified"
    APPROVED = "approved"


class ReportStatus(str, Enum):
    DRAFT = "draft"
    VERIFYING = "verifying"
    REVISION_REQUIRED = "revision_required"
    APPROVED = "approved"
    PUBLISHED = "published"
    FAILED = "failed"


_T = TypeVar("_T", bound=ContractModel)


def _transition(
    entity: _T,
    target: Enum,
    allowed: dict[Enum, frozenset[Enum]],
    **updates: Any,
) -> _T:
    current = getattr(entity, "status")
    if target not in allowed[current]:
        raise ValueError(
            f"invalid {type(entity).__name__} transition: {current.value} -> {target.value}"
        )
    values = entity.model_dump(mode="python")
    values.update(status=target, updated_at=utc_now(), **updates)
    return type(entity).model_validate(values)


class Source(ContractModel):
    source_id: str = Field(default_factory=lambda: new_id("source"))
    canonical_url: str = Field(pattern=r"^https?://[^\s]+$", max_length=2048)
    source_type: SourceType
    source_level: SourceLevel = SourceLevel.UNKNOWN
    status: SourceStatus = SourceStatus.DISCOVERED
    title: str | None = Field(default=None, max_length=1000)
    publisher: str | None = Field(default=None, max_length=500)
    authority_score: float = Field(default=0.0, ge=0.0, le=1.0)
    published_at: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("published_at", "discovered_at", "updated_at")
    @classmethod
    def _aware_times(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("source timestamps must be timezone-aware")
        return value

    def transition(self, target: SourceStatus) -> "Source":
        allowed = {
            SourceStatus.DISCOVERED: frozenset(
                {
                    SourceStatus.ACCESSIBLE,
                    SourceStatus.UNREACHABLE,
                    SourceStatus.BLOCKED,
                    SourceStatus.RETIRED,
                }
            ),
            SourceStatus.ACCESSIBLE: frozenset(
                {SourceStatus.UNREACHABLE, SourceStatus.BLOCKED, SourceStatus.RETIRED}
            ),
            SourceStatus.UNREACHABLE: frozenset(
                {SourceStatus.ACCESSIBLE, SourceStatus.BLOCKED, SourceStatus.RETIRED}
            ),
            SourceStatus.BLOCKED: frozenset(
                {SourceStatus.ACCESSIBLE, SourceStatus.RETIRED}
            ),
            SourceStatus.RETIRED: frozenset(),
        }
        return _transition(self, target, allowed)


class SourceSnapshot(ContractModel):
    snapshot_id: str = Field(default_factory=lambda: new_id("snapshot"))
    source_id: str
    artifact_id: str | None = None
    status: SnapshotStatus = SnapshotStatus.CAPTURED
    source_level: SourceLevel = SourceLevel.UNKNOWN
    source_version: int = Field(ge=1)
    content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    final_url: str = Field(pattern=r"^https?://[^\s]+$", max_length=2048)
    media_type: str | None = Field(default=None, max_length=255)
    http_status: int | None = Field(default=None, ge=100, le=599)
    fetched_at: datetime = Field(default_factory=utc_now)
    capture_method: str = Field(default="unknown", min_length=1, max_length=120)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("snapshot_id", "source_id", "artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _captured_content(self) -> "SourceSnapshot":
        if self.status in {
            SnapshotStatus.CAPTURED,
            SnapshotStatus.NORMALIZED,
            SnapshotStatus.SUPERSEDED,
        }:
            if self.artifact_id is None or self.content_hash is None:
                raise ValueError(
                    "captured snapshots require artifact_id and content_hash"
                )
        return self

    @field_validator("fetched_at", "updated_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("snapshot timestamps must be timezone-aware")
        return value

    def transition(self, target: SnapshotStatus) -> "SourceSnapshot":
        allowed = {
            SnapshotStatus.CAPTURED: frozenset(
                {
                    SnapshotStatus.NORMALIZED,
                    SnapshotStatus.FAILED,
                    SnapshotStatus.SUPERSEDED,
                }
            ),
            SnapshotStatus.NORMALIZED: frozenset({SnapshotStatus.SUPERSEDED}),
            SnapshotStatus.FAILED: frozenset({SnapshotStatus.CAPTURED}),
            SnapshotStatus.SUPERSEDED: frozenset(),
        }
        return _transition(self, target, allowed)


class Passage(ContractModel):
    passage_id: str = Field(default_factory=lambda: new_id("passage"))
    snapshot_id: str
    text_artifact_id: str
    ordinal: int = Field(ge=0)
    locator: str = Field(min_length=1, max_length=1000)
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    extraction_method: str = Field(default="unknown", min_length=1, max_length=120)
    extracted_at: datetime = Field(default_factory=utc_now)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    language: str | None = Field(default=None, min_length=2, max_length=35)
    status: PassageStatus = PassageStatus.CANDIDATE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("passage_id", "snapshot_id", "text_artifact_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _offsets(self) -> "Passage":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("passage offsets must be provided together")
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end <= self.char_start
        ):
            raise ValueError("char_end must be greater than char_start")
        return self

    @field_validator("extracted_at", "created_at", "updated_at")
    @classmethod
    def _aware_times(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("passage timestamps must be timezone-aware")
        return value

    def transition(self, target: PassageStatus) -> "Passage":
        allowed = {
            PassageStatus.CANDIDATE: frozenset(
                {PassageStatus.ACCEPTED, PassageStatus.REJECTED}
            ),
            PassageStatus.ACCEPTED: frozenset({PassageStatus.REJECTED}),
            PassageStatus.REJECTED: frozenset(),
        }
        return _transition(self, target, allowed)


class EvidenceQuote(ContractModel):
    passage_id: str
    quote: str = Field(min_length=1, max_length=3000)
    char_start: int = Field(ge=0)
    char_end: int = Field(gt=0)
    passage_content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    extraction_method: str = Field(min_length=1, max_length=120)
    extracted_at: datetime = Field(default_factory=utc_now)

    @field_validator("passage_id")
    @classmethod
    def _passage_id(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _valid_span(self) -> "EvidenceQuote":
        if self.char_end <= self.char_start:
            raise ValueError("evidence quote end must be greater than start")
        if self.char_end - self.char_start != len(self.quote):
            raise ValueError("evidence quote offsets must span the quote exactly")
        return self

    @field_validator("extracted_at")
    @classmethod
    def _aware_extracted_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("quote extraction timestamp must be timezone-aware")
        return value


class Evidence(ContractModel):
    evidence_id: str = Field(default_factory=lambda: new_id("evidence"))
    passage_ids: tuple[str, ...]
    relation: EvidenceRelation
    status: EvidenceStatus = EvidenceStatus.PROPOSED
    summary: str = Field(min_length=1, max_length=4000)
    confidence: float = Field(ge=0.0, le=1.0)
    relevance: float = Field(ge=0.0, le=1.0)
    source_quality: float = Field(ge=0.0, le=1.0)
    quotes: tuple[EvidenceQuote, ...] = ()
    verification_id: str | None = None
    verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("evidence_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("passage_ids")
    @classmethod
    def _passages(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("evidence requires at least one passage")
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("evidence passage references must be unique")
        return value

    @field_validator("verification_id")
    @classmethod
    def _verification_id(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _verified_details(self) -> "Evidence":
        quote_passages = {item.passage_id for item in self.quotes}
        if not quote_passages.issubset(set(self.passage_ids)):
            raise ValueError("evidence quotes must refer to evidence passages")
        verified = self.status == EvidenceStatus.VERIFIED
        if verified != (
            self.verification_id is not None and self.verified_at is not None
        ):
            raise ValueError(
                "verified evidence requires verification identity and timestamp"
            )
        return self

    def transition(
        self,
        target: EvidenceStatus,
        *,
        verification_id: str | None = None,
        verified_at: datetime | None = None,
    ) -> "Evidence":
        allowed = {
            EvidenceStatus.PROPOSED: frozenset(
                {
                    EvidenceStatus.VERIFIED,
                    EvidenceStatus.REJECTED,
                    EvidenceStatus.SUPERSEDED,
                }
            ),
            EvidenceStatus.VERIFIED: frozenset(
                {EvidenceStatus.REJECTED, EvidenceStatus.SUPERSEDED}
            ),
            EvidenceStatus.REJECTED: frozenset(),
            EvidenceStatus.SUPERSEDED: frozenset(),
        }
        updates: dict[str, Any] = {}
        if target == EvidenceStatus.VERIFIED:
            updates = {
                "verification_id": verification_id,
                "verified_at": verified_at or utc_now(),
            }
        else:
            updates = {
                "verification_id": None,
                "verified_at": None,
            }
        return _transition(self, target, allowed, **updates)


class AtomicFact(ContractModel):
    fact_id: str = Field(default_factory=lambda: new_id("fact"))
    statement: str = Field(min_length=1, max_length=4000)
    evidence_ids: tuple[str, ...]
    status: FactStatus = FactStatus.PROPOSED
    confidence: float = Field(ge=0.0, le=1.0)
    qualifiers: dict[str, str] = Field(default_factory=dict)
    verification_id: str | None = None
    verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator("fact_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("an atomic fact requires evidence")
        for item in value:
            validate_identifier(item)
        return value

    @field_validator("verification_id")
    @classmethod
    def _verification_id(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _verified_details(self) -> "AtomicFact":
        verified = self.status == FactStatus.VERIFIED
        if verified != (
            self.verification_id is not None and self.verified_at is not None
        ):
            raise ValueError(
                "verified facts require verification identity and timestamp"
            )
        return self

    def transition(
        self,
        target: FactStatus,
        *,
        verification_id: str | None = None,
        verified_at: datetime | None = None,
    ) -> "AtomicFact":
        allowed = {
            FactStatus.PROPOSED: frozenset(
                {FactStatus.VERIFIED, FactStatus.DISPUTED, FactStatus.REJECTED}
            ),
            FactStatus.VERIFIED: frozenset({FactStatus.DISPUTED, FactStatus.REJECTED}),
            FactStatus.DISPUTED: frozenset({FactStatus.VERIFIED, FactStatus.REJECTED}),
            FactStatus.REJECTED: frozenset(),
        }
        updates: dict[str, Any] = {}
        if target == FactStatus.VERIFIED:
            updates = {
                "verification_id": verification_id,
                "verified_at": verified_at or utc_now(),
            }
        else:
            updates = {
                "verification_id": None,
                "verified_at": None,
            }
        return _transition(self, target, allowed, **updates)


class Claim(ContractModel):
    claim_id: str = Field(default_factory=lambda: new_id("claim"))
    statement: str = Field(min_length=1, max_length=8000)
    fact_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    status: ClaimStatus = ClaimStatus.DRAFT
    confidence: float = Field(ge=0.0, le=1.0)
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    high_impact: bool = False
    support_score: float = Field(default=0.0, ge=0.0, le=1.0)
    verification_id: str | None = None
    verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator("claim_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("fact_ids", "evidence_ids")
    @classmethod
    def _refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("claim references must be unique")
        return value

    @field_validator("verification_id")
    @classmethod
    def _verification_id(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _support(self) -> "Claim":
        evaluated = {
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.CONFLICTED,
            ClaimStatus.UNSUPPORTED,
            ClaimStatus.STALE,
        }
        if self.status in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
        } and not (self.fact_ids or self.evidence_ids):
            raise ValueError("supported claims require fact or evidence references")
        if (self.status in evaluated) != (
            self.verification_id is not None and self.verified_at is not None
        ):
            raise ValueError(
                "evaluated claims require verification identity and timestamp"
            )
        return self

    def transition(
        self,
        target: ClaimStatus,
        *,
        verification_id: str | None = None,
        verified_at: datetime | None = None,
        support_score: float | None = None,
    ) -> "Claim":
        evaluated = frozenset(
            {
                ClaimStatus.SUPPORTED,
                ClaimStatus.PARTIALLY_SUPPORTED,
                ClaimStatus.CONTRADICTED,
                ClaimStatus.CONFLICTED,
                ClaimStatus.UNSUPPORTED,
                ClaimStatus.STALE,
                ClaimStatus.CONTESTED,
                ClaimStatus.REJECTED,
                ClaimStatus.SUPERSEDED,
            }
        )
        allowed = {
            ClaimStatus.DRAFT: evaluated,
            ClaimStatus.SUPPORTED: evaluated - {ClaimStatus.SUPPORTED},
            ClaimStatus.PARTIALLY_SUPPORTED: evaluated
            - {ClaimStatus.PARTIALLY_SUPPORTED},
            ClaimStatus.CONTRADICTED: evaluated - {ClaimStatus.CONTRADICTED},
            ClaimStatus.CONFLICTED: evaluated - {ClaimStatus.CONFLICTED},
            ClaimStatus.UNSUPPORTED: evaluated - {ClaimStatus.UNSUPPORTED},
            ClaimStatus.STALE: evaluated - {ClaimStatus.STALE},
            ClaimStatus.CONTESTED: evaluated - {ClaimStatus.CONTESTED},
            ClaimStatus.REJECTED: frozenset(),
            ClaimStatus.SUPERSEDED: frozenset(),
        }
        updates: dict[str, Any] = {}
        if target in {
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.CONFLICTED,
            ClaimStatus.UNSUPPORTED,
            ClaimStatus.STALE,
        }:
            updates = {
                "verification_id": verification_id,
                "verified_at": verified_at or utc_now(),
            }
            if support_score is not None:
                updates["support_score"] = support_score
        else:
            updates = {
                "verification_id": None,
                "verified_at": None,
                "support_score": 0.0,
            }
        return _transition(self, target, allowed, **updates)


class Citation(ContractModel):
    citation_id: str = Field(default_factory=lambda: new_id("citation"))
    claim_id: str
    evidence_id: str
    passage_id: str
    snapshot_id: str
    source_id: str
    locator: str = Field(min_length=1, max_length=1000)
    quote: str = Field(min_length=1, max_length=3000)
    quote_start: int | None = Field(default=None, ge=0)
    quote_end: int | None = Field(default=None, gt=0)
    extraction_method: str = Field(default="unknown", min_length=1, max_length=120)
    status: CitationStatus = CitationStatus.PROPOSED
    verification_id: str | None = None
    verified_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator(
        "citation_id",
        "claim_id",
        "evidence_id",
        "passage_id",
        "snapshot_id",
        "source_id",
        "verification_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _verified_details(self) -> "Citation":
        if (self.quote_start is None) != (self.quote_end is None):
            raise ValueError("citation quote offsets must be supplied together")
        if self.quote_start is not None and self.quote_end is not None:
            if self.quote_end - self.quote_start != len(self.quote):
                raise ValueError("citation offsets must span the quote exactly")
        verified = self.status == CitationStatus.VERIFIED
        if verified != (
            self.verification_id is not None
            and self.verified_at is not None
            and self.quote_start is not None
        ):
            raise ValueError(
                "verified citations require verification identity, timestamp, and quote offsets"
            )
        return self

    def transition(
        self,
        target: CitationStatus,
        *,
        verification_id: str | None = None,
        verified_at: datetime | None = None,
        quote_start: int | None = None,
        quote_end: int | None = None,
    ) -> "Citation":
        allowed = {
            CitationStatus.PROPOSED: frozenset(
                {CitationStatus.VERIFIED, CitationStatus.REJECTED}
            ),
            CitationStatus.VERIFIED: frozenset({CitationStatus.REJECTED}),
            CitationStatus.REJECTED: frozenset(),
        }
        updates: dict[str, Any] = {}
        if target == CitationStatus.VERIFIED:
            updates = {
                "verification_id": verification_id,
                "verified_at": verified_at or utc_now(),
                "quote_start": quote_start,
                "quote_end": quote_end,
            }
        else:
            updates = {
                "verification_id": None,
                "verified_at": None,
            }
        return _transition(self, target, allowed, **updates)


class Conflict(ContractModel):
    conflict_id: str = Field(default_factory=lambda: new_id("conflict"))
    claim_ids: tuple[str, ...]
    fact_ids: tuple[str, ...] = ()
    summary: str = Field(min_length=1, max_length=5000)
    status: ConflictStatus = ConflictStatus.OPEN
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    high_impact: bool = False
    resolution: str | None = Field(default=None, max_length=5000)
    resolution_kind: ConflictResolutionKind | None = None
    resolution_evidence_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator("conflict_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("claim_ids", "fact_ids", "resolution_evidence_ids")
    @classmethod
    def _refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("conflict references must be unique")
        return value

    @model_validator(mode="after")
    def _valid_conflict(self) -> "Conflict":
        if len(self.claim_ids) < 2:
            raise ValueError("a conflict requires at least two claims")
        if self.status == ConflictStatus.RESOLVED and (
            not self.resolution
            or not self.resolution_evidence_ids
            or self.resolution_kind is None
        ):
            raise ValueError(
                "resolved conflicts require a resolution kind and evidence"
            )
        if self.status != ConflictStatus.RESOLVED and self.resolution_kind not in {
            None,
            ConflictResolutionKind.ACCEPTED_UNRESOLVED,
        }:
            raise ValueError(
                "unresolved conflicts cannot use a definitive resolution kind"
            )
        return self

    def transition(
        self,
        target: ConflictStatus,
        *,
        resolution: str | None = None,
        resolution_kind: ConflictResolutionKind | None = None,
        resolution_evidence_ids: tuple[str, ...] = (),
    ) -> "Conflict":
        allowed = {
            ConflictStatus.OPEN: frozenset(
                {
                    ConflictStatus.INVESTIGATING,
                    ConflictStatus.RESOLVED,
                    ConflictStatus.ACCEPTED_UNRESOLVED,
                }
            ),
            ConflictStatus.INVESTIGATING: frozenset(
                {ConflictStatus.RESOLVED, ConflictStatus.ACCEPTED_UNRESOLVED}
            ),
            ConflictStatus.RESOLVED: frozenset(),
            ConflictStatus.ACCEPTED_UNRESOLVED: frozenset(
                {ConflictStatus.INVESTIGATING, ConflictStatus.RESOLVED}
            ),
        }
        updates: dict[str, Any] = {}
        if (
            resolution is not None
            or resolution_kind is not None
            or resolution_evidence_ids
        ):
            updates.update(
                resolution=resolution,
                resolution_kind=resolution_kind,
                resolution_evidence_ids=resolution_evidence_ids,
            )
        return _transition(self, target, allowed, **updates)


class Section(ContractModel):
    section_id: str = Field(default_factory=lambda: new_id("section"))
    report_id: str
    parent_section_id: str | None = None
    title: str = Field(min_length=1, max_length=500)
    goal: str = Field(min_length=1, max_length=4000)
    order: int = Field(ge=0)
    claim_ids: tuple[str, ...] = ()
    required_claim_ids: tuple[str, ...] = ()
    citation_ids: tuple[str, ...] = ()
    content_artifact_id: str | None = None
    status: SectionStatus = SectionStatus.PLANNED
    coverage_score: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    coverage_status: SectionCoverageStatus = SectionCoverageStatus.NOT_ASSESSED
    unsupported_claim_ids: tuple[str, ...] = ()
    conflicted_claim_ids: tuple[str, ...] = ()
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator(
        "section_id", "report_id", "parent_section_id", "content_artifact_id"
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator(
        "claim_ids",
        "required_claim_ids",
        "citation_ids",
        "unsupported_claim_ids",
        "conflicted_claim_ids",
    )
    @classmethod
    def _refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        if len(value) != len(set(value)):
            raise ValueError("section references must be unique")
        return value

    @model_validator(mode="after")
    def _valid_section(self) -> "Section":
        if self.section_id == self.parent_section_id:
            raise ValueError("section cannot parent itself")
        claim_ids = set(self.claim_ids)
        if not set(self.required_claim_ids).issubset(claim_ids):
            raise ValueError("required section claims must be included in claim_ids")
        if not set(self.unsupported_claim_ids).issubset(claim_ids):
            raise ValueError("unsupported section claims must be included in claim_ids")
        if not set(self.conflicted_claim_ids).issubset(claim_ids):
            raise ValueError("conflicted section claims must be included in claim_ids")
        if (
            self.status in {SectionStatus.VERIFIED, SectionStatus.APPROVED}
            and self.content_artifact_id is None
        ):
            raise ValueError("verified sections require a content artifact")
        return self

    def transition(self, target: SectionStatus) -> "Section":
        allowed = {
            SectionStatus.PLANNED: frozenset({SectionStatus.DRAFTING}),
            SectionStatus.DRAFTING: frozenset(
                {SectionStatus.NEEDS_REPAIR, SectionStatus.VERIFIED}
            ),
            SectionStatus.NEEDS_REPAIR: frozenset(
                {SectionStatus.DRAFTING, SectionStatus.VERIFIED}
            ),
            SectionStatus.VERIFIED: frozenset(
                {SectionStatus.NEEDS_REPAIR, SectionStatus.APPROVED}
            ),
            SectionStatus.APPROVED: frozenset({SectionStatus.NEEDS_REPAIR}),
        }
        return _transition(self, target, allowed)


class Report(ContractModel):
    report_id: str = Field(default_factory=lambda: new_id("report"))
    thread_id: str
    run_id: str
    title: str = Field(min_length=1, max_length=1000)
    research_question: str = Field(min_length=1, max_length=8000)
    section_ids: tuple[str, ...]
    status: ReportStatus = ReportStatus.DRAFT
    content_artifact_id: str | None = None
    version: int = Field(default=1, ge=1)
    quality_scores: dict[str, float] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    provenance: EntityProvenance

    @field_validator("report_id", "thread_id", "run_id", "content_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("section_ids")
    @classmethod
    def _sections(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("a report requires at least one section")
        for item in value:
            validate_identifier(item)
        if len(set(value)) != len(value):
            raise ValueError("report section references must be unique")
        return value

    @field_validator("quality_scores")
    @classmethod
    def _scores(cls, value: dict[str, float]) -> dict[str, float]:
        if any(score < 0.0 or score > 1.0 for score in value.values()):
            raise ValueError("quality scores must be between zero and one")
        return value

    @model_validator(mode="after")
    def _published_content(self) -> "Report":
        if (
            self.status in {ReportStatus.APPROVED, ReportStatus.PUBLISHED}
            and self.content_artifact_id is None
        ):
            raise ValueError("approved reports require a content artifact")
        return self

    def transition(self, target: ReportStatus) -> "Report":
        allowed = {
            ReportStatus.DRAFT: frozenset(
                {ReportStatus.VERIFYING, ReportStatus.FAILED}
            ),
            ReportStatus.VERIFYING: frozenset(
                {
                    ReportStatus.REVISION_REQUIRED,
                    ReportStatus.APPROVED,
                    ReportStatus.FAILED,
                }
            ),
            ReportStatus.REVISION_REQUIRED: frozenset(
                {ReportStatus.DRAFT, ReportStatus.VERIFYING, ReportStatus.FAILED}
            ),
            ReportStatus.APPROVED: frozenset(
                {ReportStatus.PUBLISHED, ReportStatus.REVISION_REQUIRED}
            ),
            ReportStatus.PUBLISHED: frozenset(),
            ReportStatus.FAILED: frozenset({ReportStatus.DRAFT}),
        }
        return _transition(self, target, allowed)
