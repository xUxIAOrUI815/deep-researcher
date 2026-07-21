from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol, TypeAlias

from deep_researcher.contracts import (
    AtomicFact,
    Citation,
    Claim,
    Conflict,
    Evidence,
    Passage,
    Report,
    Section,
    Source,
    SourceSnapshot,
)


KnowledgeEntity: TypeAlias = Source | SourceSnapshot | Passage | Evidence | AtomicFact | Claim | Citation | Conflict | Section | Report


class KnowledgeStoreError(RuntimeError):
    """Base class for durable knowledge graph failures."""


class KnowledgeConflict(KnowledgeStoreError):
    """An entity identity, natural key, revision, or run boundary conflicts."""


class KnowledgeNotFound(KnowledgeStoreError):
    """An entity or required relationship target is absent."""


class KnowledgeCorruption(KnowledgeStoreError):
    """Knowledge payloads or relationship indexes failed integrity checks."""


class KnowledgeRelation(str, Enum):
    SNAPSHOT_SOURCE = "snapshot_source"
    PASSAGE_SNAPSHOT = "passage_snapshot"
    EVIDENCE_PASSAGE = "evidence_passage"
    FACT_EVIDENCE = "fact_evidence"
    CLAIM_FACT = "claim_fact"
    CLAIM_EVIDENCE = "claim_evidence"
    CITATION_CLAIM = "citation_claim"
    CITATION_EVIDENCE = "citation_evidence"
    CITATION_PASSAGE = "citation_passage"
    CITATION_SNAPSHOT = "citation_snapshot"
    CITATION_SOURCE = "citation_source"
    CONFLICT_CLAIM = "conflict_claim"
    CONFLICT_FACT = "conflict_fact"
    CONFLICT_RESOLUTION_EVIDENCE = "conflict_resolution_evidence"
    SECTION_REPORT = "section_report"
    SECTION_PARENT = "section_parent"
    SECTION_CLAIM = "section_claim"
    SECTION_CITATION = "section_citation"
    REPORT_SECTION = "report_section"


@dataclass(frozen=True)
class SavedRevision:
    entity: KnowledgeEntity
    revision: int
    inserted: bool


@dataclass(frozen=True)
class KnowledgeQuery:
    run_id: str
    entity_types: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    after_created_at: datetime | None = None
    after_entity_id: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if (self.after_created_at is None) != (self.after_entity_id is None):
            raise ValueError("knowledge cursor fields must be supplied together")
        if self.after_created_at is not None and (
            self.after_created_at.tzinfo is None or self.after_created_at.utcoffset() is None
        ):
            raise ValueError("knowledge cursor timestamp must be timezone-aware")


@dataclass(frozen=True)
class KnowledgePage:
    items: tuple[SavedRevision, ...]
    next_cursor: tuple[datetime, str] | None


class KnowledgeStorage(Protocol):
    def save_batch(self, entities: tuple[KnowledgeEntity, ...]) -> tuple[SavedRevision, ...]:
        ...

    def get_latest(self, entity_id: str) -> SavedRevision | None:
        ...

    def get_history(self, entity_id: str) -> tuple[SavedRevision, ...]:
        ...

    def list_latest(self, query: KnowledgeQuery) -> KnowledgePage:
        ...

    def related(
        self,
        entity_id: str,
        relation: KnowledgeRelation,
        *,
        incoming: bool = False,
    ) -> tuple[SavedRevision, ...]:
        ...

    def find_by_natural_key(
        self,
        *,
        entity_type: str,
        run_id: str,
        scope_id: str,
        key_value: str,
    ) -> SavedRevision | None:
        ...

    def integrity_check(self) -> None:
        ...

    def close(self) -> None:
        ...
