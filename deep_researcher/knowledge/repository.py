from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast

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

from .normalization import canonicalize_url
from .storage import (
    KnowledgeEntity,
    KnowledgeNotFound,
    KnowledgePage,
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeStorage,
    SavedRevision,
)


EntityT = TypeVar("EntityT", bound=KnowledgeEntity)


class EntityRepository(Generic[EntityT]):
    def __init__(self, storage: KnowledgeStorage, model: type[EntityT]) -> None:
        self.storage = storage
        self.model = model

    def save(self, entity: EntityT) -> SavedRevision:
        if not isinstance(entity, self.model):
            raise TypeError(f"expected {self.model.__name__}")
        return self.storage.save_batch((entity,))[0]

    def get(self, entity_id: str) -> EntityT | None:
        saved = self.storage.get_latest(entity_id)
        if saved is None:
            return None
        if not isinstance(saved.entity, self.model):
            raise TypeError(f"{entity_id} is {type(saved.entity).__name__}, not {self.model.__name__}")
        return cast(EntityT, saved.entity)

    def require(self, entity_id: str) -> EntityT:
        entity = self.get(entity_id)
        if entity is None:
            raise KnowledgeNotFound(entity_id)
        return entity

    def history(self, entity_id: str) -> tuple[EntityT, ...]:
        history = self.storage.get_history(entity_id)
        if any(not isinstance(item.entity, self.model) for item in history):
            raise TypeError(f"history contains an unexpected entity type: {entity_id}")
        return tuple(cast(EntityT, item.entity) for item in history)

    def list(self, run_id: str, *, statuses: tuple[str, ...] = (), limit: int | None = None) -> tuple[EntityT, ...]:
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        items: list[EntityT] = []
        cursor = None
        while limit is None or len(items) < limit:
            page_size = min(1000, (limit - len(items)) if limit is not None else 1000)
            page = self.storage.list_latest(
                KnowledgeQuery(
                    run_id=run_id,
                    entity_types=(self.model.__name__,),
                    statuses=statuses,
                    after_created_at=cursor[0] if cursor else None,
                    after_entity_id=cursor[1] if cursor else None,
                    limit=page_size,
                )
            )
            items.extend(cast(EntityT, item.entity) for item in page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return tuple(items)


class KnowledgeRepository:
    def __init__(self, storage: KnowledgeStorage) -> None:
        self.storage = storage
        self.sources = EntityRepository(storage, Source)
        self.snapshots = EntityRepository(storage, SourceSnapshot)
        self.passages = EntityRepository(storage, Passage)
        self.evidence = EntityRepository(storage, Evidence)
        self.facts = EntityRepository(storage, AtomicFact)
        self.claims = EntityRepository(storage, Claim)
        self.citations = EntityRepository(storage, Citation)
        self.conflicts = EntityRepository(storage, Conflict)
        self.sections = EntityRepository(storage, Section)
        self.reports = EntityRepository(storage, Report)

    def save_graph(self, *entities: KnowledgeEntity) -> tuple[SavedRevision, ...]:
        return self.storage.save_batch(tuple(entities))

    def get(self, entity_id: str) -> KnowledgeEntity | None:
        saved = self.storage.get_latest(entity_id)
        return saved.entity if saved is not None else None

    def find_source_by_url(self, run_id: str, url: str) -> Source | None:
        saved = self.storage.find_by_natural_key(
            entity_type="Source",
            run_id=run_id,
            scope_id="canonical_url",
            key_value=canonicalize_url(url),
        )
        return cast(Source, saved.entity) if saved is not None else None

    def source_snapshots(self, source_id: str) -> tuple[SourceSnapshot, ...]:
        return tuple(
            cast(SourceSnapshot, item.entity)
            for item in self.storage.related(
                source_id,
                KnowledgeRelation.SNAPSHOT_SOURCE,
                incoming=True,
            )
        )

    def related(
        self,
        entity_id: str,
        relation: KnowledgeRelation,
        *,
        incoming: bool = False,
    ) -> tuple[KnowledgeEntity, ...]:
        return tuple(
            item.entity
            for item in self.storage.related(entity_id, relation, incoming=incoming)
        )
