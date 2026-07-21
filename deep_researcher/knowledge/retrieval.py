from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol

from .normalization import normalize_text
from .repository import KnowledgeRepository
from .sqlite_storage import entity_id
from .storage import KnowledgeEntity, KnowledgeQuery


class VectorRetrievalAdapter(Protocol):
    """Optional vector index; storage and correctness never depend on it."""

    def score(self, query: str, entities: tuple[KnowledgeEntity, ...]) -> dict[str, float]:
        ...


@dataclass(frozen=True)
class RetrievalHit:
    entity: KnowledgeEntity
    score: float
    lexical_score: float
    vector_score: float | None = None


class KnowledgeRetrievalService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        *,
        vector_adapter: VectorRetrievalAdapter | None = None,
        vector_weight: float = 0.35,
    ) -> None:
        if not 0.0 <= vector_weight <= 1.0:
            raise ValueError("vector_weight must be between zero and one")
        self.repository = repository
        self.vector_adapter = vector_adapter
        self.vector_weight = vector_weight

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(value).casefold()))

    @staticmethod
    def _search_text(entity: KnowledgeEntity) -> str:
        values = []
        for name in ("statement", "summary", "title", "research_question", "goal", "canonical_url"):
            value = getattr(entity, name, None)
            if value:
                values.append(str(value))
        metadata = getattr(entity, "metadata", None)
        if metadata:
            values.extend(str(value) for value in metadata.values() if isinstance(value, (str, int, float)))
        return " ".join(values)

    def search(
        self,
        *,
        run_id: str,
        query: str,
        entity_types: tuple[str, ...] = (),
        statuses: tuple[str, ...] = (),
        limit: int = 20,
    ) -> tuple[RetrievalHit, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        entities_list: list[KnowledgeEntity] = []
        cursor = None
        while True:
            page = self.repository.storage.list_latest(
                KnowledgeQuery(
                    run_id=run_id, entity_types=entity_types, statuses=statuses,
                    after_created_at=cursor[0] if cursor else None,
                    after_entity_id=cursor[1] if cursor else None, limit=1000,
                )
            )
            entities_list.extend(item.entity for item in page.items)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        entities = tuple(entities_list)
        query_tokens = self._tokens(query)
        vector_scores = self.vector_adapter.score(query, entities) if self.vector_adapter else {}
        hits: list[RetrievalHit] = []
        for entity in entities:
            tokens = self._tokens(self._search_text(entity))
            lexical = len(query_tokens & tokens) / len(query_tokens | tokens) if query_tokens | tokens else 0.0
            vector = vector_scores.get(entity_id(entity)) if self.vector_adapter else None
            if vector is not None and not 0.0 <= vector <= 1.0:
                raise ValueError("vector adapter scores must be between zero and one")
            score = lexical if vector is None else ((1.0 - self.vector_weight) * lexical + self.vector_weight * vector)
            if score > 0.0 or not query_tokens:
                hits.append(RetrievalHit(entity=entity, score=score, lexical_score=lexical, vector_score=vector))
        hits.sort(key=lambda hit: (-hit.score, entity_id(hit.entity)))
        return tuple(hits[:limit])
