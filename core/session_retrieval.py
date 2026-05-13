from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Dict, Iterable, List, Optional

from schemas.retrieval import SessionRetrievalQuery, SessionRetrievalResult


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", _normalize_text(value).lower()).strip()


def _tokenize(value: Any) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", _normalize_text(value).lower())


def _dedupe_rows(rows: Iterable[Dict[str, Any]], *keys: str) -> list[Dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[Dict[str, Any]] = []
    for row in rows:
        parts = [str(row.get(key, "")).strip().lower() for key in keys]
        dedup_key = "|".join(part for part in parts if part)
        if not dedup_key:
            dedup_key = str(id(row))
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        deduped.append(row)
    return deduped


def _embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class SessionRetrievalService:
    """Structured retrieval over persisted session knowledge."""

    def __init__(self, knowledge_manager: Any):
        self.knowledge_manager = knowledge_manager

    def retrieve(
        self,
        *,
        research_id: str,
        session_id: str,
        query: Optional[SessionRetrievalQuery] = None,
    ) -> SessionRetrievalResult:
        retrieval_query = query or SessionRetrievalQuery()
        snapshot = self.knowledge_manager.get_session_snapshot(research_id, session_id)

        section_packs = list(snapshot.get("section_evidence_packs", []))
        claims = list(snapshot.get("claims", []))
        facts = list(snapshot.get("facts", []))
        evidence = list(snapshot.get("evidence", []))
        conflicts = list(snapshot.get("conflicts", []))
        gaps = list(snapshot.get("open_gaps", []))
        source_registry = dict(snapshot.get("source_registry", {}) or {})
        sources = list(snapshot.get("sources", []))
        if not sources and source_registry:
            sources = [dict(row) for row in source_registry.values()]
        if not sources:
            sources = [
                {"source_id": str(source_id)}
                for source_id in (snapshot.get("knowledge_refs", {}) or {}).get("source_ids", [])
            ]

        related_pack_ids: set[str] = set()
        related_claim_ids: set[str] = set()
        related_fact_ids: set[str] = set()
        related_evidence_ids: set[str] = set()
        related_conflict_ids: set[str] = set()
        related_source_ids: set[str] = set()

        if retrieval_query.section_id:
            section_packs = [
                row for row in section_packs
                if str(row.get("section_id", "")).strip() == retrieval_query.section_id
            ]
            related_pack_ids.update(str(row.get("pack_id") or row.get("id") or "") for row in section_packs)
            for pack in section_packs:
                related_claim_ids.update(str(value) for value in pack.get("claim_ids", []) or [])
                related_fact_ids.update(str(value) for value in pack.get("fact_ids", []) or [])
                related_evidence_ids.update(str(value) for value in pack.get("evidence_ids", []) or [])
                related_conflict_ids.update(str(value) for value in pack.get("conflict_ids", []) or [])

        if retrieval_query.claim_id:
            related_claim_ids.add(retrieval_query.claim_id)
            for claim in claims:
                if str(claim.get("id", "")).strip() == retrieval_query.claim_id:
                    related_fact_ids.update(str(value) for value in claim.get("fact_ids", []) or [])
                    related_evidence_ids.update(str(value) for value in claim.get("evidence_ids", []) or [])
            for pack in snapshot.get("section_evidence_packs", []):
                claim_ids = {str(value) for value in pack.get("claim_ids", []) or []}
                if retrieval_query.claim_id in claim_ids:
                    related_pack_ids.add(str(pack.get("pack_id") or pack.get("id") or ""))

        if retrieval_query.source_id:
            related_source_ids.add(retrieval_query.source_id)

        if retrieval_query.source_url:
            for source in sources:
                if str(source.get("url", "")).strip() == retrieval_query.source_url:
                    related_source_ids.add(str(source.get("source_id", "")))

        filtered_claims = [
            row for row in claims
            if self._matches_row(
                row,
                retrieval_query,
                allowed_ids=related_claim_ids,
                id_key="id",
                section_key="section_id",
                task_key="task_id",
            )
        ]
        filtered_facts = [
            row for row in facts
            if self._matches_row(
                row,
                retrieval_query,
                allowed_ids=related_fact_ids,
                id_key="id",
                section_key="section_id",
                task_key="task_id",
            )
        ]
        filtered_evidence = [
            row for row in evidence
            if self._matches_evidence(row, retrieval_query, related_evidence_ids, related_source_ids)
        ]
        filtered_conflicts = [
            row for row in conflicts
            if self._matches_row(
                row,
                retrieval_query,
                allowed_ids=related_conflict_ids,
                id_key="id",
                section_key="section_id",
                task_key="task_id",
            )
        ]
        filtered_gaps = [
            row for row in gaps
            if self._matches_gap(row, retrieval_query)
        ]
        filtered_sources = [
            row for row in sources
            if self._matches_source(row, retrieval_query, related_source_ids)
        ]

        for row in filtered_evidence:
            source_id = str(row.get("source_id", "")).strip()
            if source_id:
                related_source_ids.add(source_id)
        if related_source_ids:
            filtered_sources = [
                row for row in sources
                if str(row.get("source_id", "")).strip() in related_source_ids or self._matches_source(row, retrieval_query, related_source_ids)
            ]

        if retrieval_query.include_section_packs:
            filtered_packs = [
                row for row in snapshot.get("section_evidence_packs", [])
                if self._matches_pack(row, retrieval_query, related_pack_ids, related_claim_ids, related_fact_ids, related_evidence_ids)
            ]
        else:
            filtered_packs = []

        if retrieval_query.deduplicate:
            filtered_packs = _dedupe_rows(filtered_packs, "pack_id", "section_id")
            filtered_claims = _dedupe_rows(filtered_claims, "id")
            filtered_facts = _dedupe_rows(filtered_facts, "id")
            filtered_evidence = _dedupe_rows(filtered_evidence, "id")
            filtered_conflicts = _dedupe_rows(filtered_conflicts, "id")
            filtered_gaps = _dedupe_rows(filtered_gaps, "gap_id", "gap_text")
            filtered_sources = _dedupe_rows(filtered_sources, "source_id", "url")

        filtered_packs = self._sort_rows(filtered_packs, retrieval_query, kind="pack")
        filtered_claims = self._sort_rows(filtered_claims, retrieval_query, kind="claim")
        filtered_facts = self._sort_rows(filtered_facts, retrieval_query, kind="fact")
        filtered_evidence = self._sort_rows(filtered_evidence, retrieval_query, kind="evidence")
        filtered_conflicts = self._sort_rows(filtered_conflicts, retrieval_query, kind="conflict")
        filtered_gaps = self._sort_rows(filtered_gaps, retrieval_query, kind="gap")
        filtered_sources = self._sort_rows(filtered_sources, retrieval_query, kind="source")

        limit = max(1, int(retrieval_query.limit_per_type))
        result = SessionRetrievalResult(
            research_id=research_id,
            session_id=session_id,
            query=retrieval_query,
            section_packs=filtered_packs[:limit] if retrieval_query.include_section_packs else [],
            claims=filtered_claims[:limit] if retrieval_query.include_claims else [],
            facts=filtered_facts[:limit] if retrieval_query.include_facts else [],
            evidence=filtered_evidence[:limit] if retrieval_query.include_evidence else [],
            conflicts=filtered_conflicts[:limit] if retrieval_query.include_conflicts else [],
            unresolved_gaps=filtered_gaps[:limit] if retrieval_query.include_gaps else [],
            sources=filtered_sources[:limit] if retrieval_query.include_sources else [],
            latest_coverage_snapshot=snapshot.get("latest_coverage_snapshot"),
            latest_novelty_snapshot=snapshot.get("latest_novelty_snapshot"),
            source_registry=source_registry,
            retrieval_meta={
                "scope": "session",
                "query": retrieval_query.model_dump(),
                "counts": {
                    "section_packs": len(filtered_packs),
                    "claims": len(filtered_claims),
                    "facts": len(filtered_facts),
                    "evidence": len(filtered_evidence),
                    "conflicts": len(filtered_conflicts),
                    "unresolved_gaps": len(filtered_gaps),
                    "sources": len(filtered_sources),
                },
                "semantic_enabled": bool(retrieval_query.semantic_query and retrieval_query.semantic_weight > 0.0),
            },
        )
        return result

    def _matches_row(
        self,
        row: Dict[str, Any],
        query: SessionRetrievalQuery,
        allowed_ids: set[str],
        *,
        id_key: str,
        section_key: str,
        task_key: str,
    ) -> bool:
        row_id = str(row.get(id_key, "")).strip()
        section_matches = bool(query.section_id and str(row.get(section_key, "")).strip() == query.section_id)
        if allowed_ids and row_id not in allowed_ids and not section_matches:
            return False
        if query.section_id and str(row.get(section_key, "")).strip() not in {"", query.section_id} and row_id not in allowed_ids:
            return False
        if query.task_id and str(row.get(task_key, "")).strip() not in {"", query.task_id}:
            return False
        return True

    def _matches_evidence(
        self,
        row: Dict[str, Any],
        query: SessionRetrievalQuery,
        allowed_ids: set[str],
        related_source_ids: set[str],
    ) -> bool:
        if not self._matches_row(row, query, allowed_ids, id_key="id", section_key="section_id", task_key="task_id"):
            return False
        source_id = str(row.get("source_id", "")).strip()
        if related_source_ids and source_id and source_id not in related_source_ids:
            return False
        if query.source_id and source_id != query.source_id:
            return False
        return True

    def _matches_gap(self, row: Dict[str, Any], query: SessionRetrievalQuery) -> bool:
        if query.section_id and str(row.get("section_id", "")).strip() not in {"", query.section_id}:
            return False
        if query.task_id and str(row.get("task_id", "")).strip() not in {"", query.task_id}:
            return False
        if query.gap_text:
            gap_key = _normalize_key(query.gap_text)
            if gap_key not in _normalize_key(row.get("gap_text", "")):
                return False
        return True

    def _matches_source(
        self,
        row: Dict[str, Any],
        query: SessionRetrievalQuery,
        related_source_ids: set[str],
    ) -> bool:
        source_id = str(row.get("source_id", "")).strip()
        url = str(row.get("url", "")).strip()
        if related_source_ids and source_id and source_id not in related_source_ids:
            return False
        if query.source_id and source_id != query.source_id:
            return False
        if query.source_url and url != query.source_url:
            return False
        if query.task_id and str(row.get("task_id", "")).strip() not in {"", query.task_id}:
            return False
        return True

    def _matches_pack(
        self,
        row: Dict[str, Any],
        query: SessionRetrievalQuery,
        related_pack_ids: set[str],
        related_claim_ids: set[str],
        related_fact_ids: set[str],
        related_evidence_ids: set[str],
    ) -> bool:
        pack_id = str(row.get("pack_id") or row.get("id") or "").strip()
        if related_pack_ids and pack_id not in related_pack_ids:
            return False
        if query.section_id and str(row.get("section_id", "")).strip() != query.section_id:
            if not ({str(value) for value in row.get("claim_ids", []) or []} & related_claim_ids):
                return False
        if query.task_id:
            task_id = str(row.get("task_id", "")).strip()
            meta_task_id = str((row.get("metadata_json", {}) or {}).get("task_id", "")).strip()
            if task_id not in {"", query.task_id} and meta_task_id not in {"", query.task_id}:
                return False
        if query.claim_id and query.claim_id not in {str(value) for value in row.get("claim_ids", []) or []}:
            return False
        if related_fact_ids and not ({str(value) for value in row.get("fact_ids", []) or []} & related_fact_ids):
            return False
        if related_evidence_ids and not ({str(value) for value in row.get("evidence_ids", []) or []} & related_evidence_ids):
            return False
        return True

    def _sort_rows(
        self,
        rows: List[Dict[str, Any]],
        query: SessionRetrievalQuery,
        *,
        kind: str,
    ) -> List[Dict[str, Any]]:
        semantic_vector = _embedding(query.semantic_query or query.topic or "") if query.semantic_query or query.topic else []

        def score(row: Dict[str, Any]) -> float:
            base = self._base_score(row, kind=kind, sort_by=query.sort_by)
            if query.section_id and str(row.get("section_id", "")).strip() == query.section_id:
                base += 1.5
            if query.task_id and str(row.get("task_id", "")).strip() == query.task_id:
                base += 1.0
            if query.claim_id and str(row.get("id", "")).strip() == query.claim_id:
                base += 2.0
            if semantic_vector and query.semantic_weight > 0.0:
                row_vector = _embedding(self._semantic_text(row, kind=kind))
                base += _cosine_similarity(semantic_vector, row_vector) * float(query.semantic_weight)
            return base

        return sorted(rows, key=score, reverse=query.sort_desc)

    def _base_score(self, row: Dict[str, Any], *, kind: str, sort_by: str) -> float:
        if kind == "pack":
            return (
                float(row.get("coverage_score", 0.0) or 0.0) * 3.0
                + len(row.get("evidence_ids", []) or []) * 0.2
                + len(row.get("claim_ids", []) or []) * 0.15
            )
        if kind == "source":
            return (
                float(row.get("authority_score", 0.0) or 0.0) * (2.0 if sort_by in {"authority", "relevance"} else 1.0)
                + float(row.get("freshness_score", 0.0) or 0.0)
            )
        if kind in {"claim", "fact", "evidence"}:
            confidence = float(row.get("confidence", 0.0) or 0.0)
            source_count = float(row.get("source_count", 0.0) or 0.0)
            evidence_count = float(row.get("evidence_count", 0.0) or 0.0)
            quality = float(row.get("quality_score", 0.0) or 0.0)
            verified_count = float(row.get("verified_count", 0.0) or 0.0)
            return confidence + (source_count * 0.15) + (evidence_count * 0.12) + (quality * 0.2) + (verified_count * 0.18)
        if kind == "conflict":
            severity_map = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}
            return severity_map.get(str(row.get("severity", "medium")).lower(), 1.0) + float(row.get("evidence_count", 0.0) or 0.0) * 0.1
        if kind == "gap":
            severity_map = {"critical": 3.0, "high": 2.0, "medium": 1.0, "low": 0.5}
            return severity_map.get(str(row.get("severity", "medium")).lower(), 1.0)
        return 0.0

    def _semantic_text(self, row: Dict[str, Any], *, kind: str) -> str:
        if kind == "pack":
            return " ".join(
                [
                    str(row.get("section_title", "")),
                    str(row.get("goal", "")),
                    str(row.get("notes", "")),
                    str(row.get("section_id", "")),
                ]
            )
        if kind == "source":
            return " ".join([str(row.get("title", "")), str(row.get("url", "")), str(row.get("domain", ""))])
        if kind == "gap":
            return str(row.get("gap_text", ""))
        return " ".join(
            [
                str(row.get("text", "")),
                str(row.get("summary", "")),
                str(row.get("snippet", "")),
                str(row.get("description", "")),
                str(row.get("source_url", "")),
            ]
        )
