import asyncio
import json
import os
import re
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx

from schemas.state import (
    AtomicFact,
    DistillerOutputs,
    KnowledgeRefs,
    SourceLevel,
)

SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY", "")
SILICON_FLOW_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"


class FactStatus(Enum):
    ACTIVE = "active"
    VERIFIED = "verified"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"


class EmbeddingModel:
    """Compatibility wrapper kept for older callers."""

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.api_key = SILICON_FLOW_API_KEY
        self._embedding_lock = asyncio.Lock()
        self._last_request_time = 0.0
        self._min_request_interval = 0.1

    async def get_embedding(self, text: str) -> List[float]:
        async with self._embedding_lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

        if not self.api_key:
            raise ValueError("SILICON_FLOW_API_KEY not set in environment")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": text}

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SILICON_FLOW_EMBEDDING_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return data.get("data", [{}])[0].get("embedding", [])

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("SILICON_FLOW_API_KEY not set in environment")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": texts}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                SILICON_FLOW_EMBEDDING_URL,
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return [item.get("embedding", []) for item in data.get("data", [])]


@dataclass
class FactConflict:
    fact_id_1: str
    fact_id_2: str
    conflict_description: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeStats:
    total_facts: int = 0
    verified_facts: int = 0
    conflicting_facts: int = 0
    conflicts_detected: int = 0
    duplicates_merged: int = 0


@dataclass
class UpsertFactResult:
    action: str
    fact_id: str
    confidence_change: float = 0.0
    conflict_with_id: Optional[str] = None
    conflict_description: Optional[str] = None


class SessionKnowledgeStore:
    """Session-scoped object store for distilled knowledge."""

    def __init__(self, db_path: str = "./session_knowledge.sqlite3"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS session_objects (
                object_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                object_type TEXT NOT NULL,
                task_id TEXT,
                section_id TEXT,
                source_id TEXT,
                status TEXT,
                dedupe_key TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_objects_scope ON session_objects(research_id, session_id, object_type)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_session_objects_dedupe ON session_objects(research_id, session_id, object_type, dedupe_key)"
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _serialize(self, obj: Any) -> str:
        return json.dumps(obj, ensure_ascii=False, default=self._to_json_serializable)

    def _to_json_serializable(self, value: Any) -> str:
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def _deserialize(self, payload: Optional[str]) -> Any:
        if not payload:
            return {}
        return self._restore_temporal_values(json.loads(payload))

    def _restore_temporal_values(self, value: Any) -> Any:
        if isinstance(value, dict):
            restored: Dict[str, Any] = {}
            for key, item in value.items():
                restored[key] = self._restore_temporal_values(item)
                if key in {"timestamp", "created_at", "updated_at"} and isinstance(restored[key], str):
                    try:
                        restored[key] = datetime.fromisoformat(restored[key])
                    except ValueError:
                        pass
            return restored
        if isinstance(value, list):
            return [self._restore_temporal_values(item) for item in value]
        return value

    def upsert_object(
        self,
        *,
        object_id: str,
        research_id: str,
        session_id: str,
        object_type: str,
        data: Dict[str, Any],
        task_id: Optional[str] = None,
        section_id: Optional[str] = None,
        source_id: Optional[str] = None,
        status: Optional[str] = None,
        dedupe_key: Optional[str] = None,
    ) -> str:
        now = self._now()
        existing = self.conn.execute(
            "SELECT created_at FROM session_objects WHERE object_id = ? AND research_id = ? AND session_id = ?",
            (object_id, research_id, session_id),
        ).fetchone()
        serialized_data = self._serialize(data)
        if existing:
            self.conn.execute(
                """
                UPDATE session_objects
                SET task_id = ?, section_id = ?, source_id = ?, status = ?, dedupe_key = ?, updated_at = ?, data = ?
                WHERE object_id = ? AND research_id = ? AND session_id = ?
                """,
                (
                    task_id,
                    section_id,
                    source_id,
                    status,
                    dedupe_key,
                    now,
                    serialized_data,
                    object_id,
                    research_id,
                    session_id,
                ),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO session_objects (
                    object_id, research_id, session_id, object_type, task_id, section_id,
                    source_id, status, dedupe_key, created_at, updated_at, data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    object_id,
                    research_id,
                    session_id,
                    object_type,
                    task_id,
                    section_id,
                    source_id,
                    status,
                    dedupe_key,
                    now,
                    now,
                    serialized_data,
                ),
            )
        self.conn.commit()
        return object_id

    def get_object(self, object_id: str, research_id: str, session_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(
            "SELECT * FROM session_objects WHERE object_id = ? AND research_id = ? AND session_id = ?",
            (object_id, research_id, session_id),
        ).fetchone()
        if not row:
            return None
        return self._row_to_dict(row)

    def list_objects(
        self,
        research_id: str,
        session_id: str,
        object_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM session_objects WHERE research_id = ? AND session_id = ?"
        params: Tuple[Any, ...] = (research_id, session_id)
        if object_type:
            query += " AND object_type = ?"
            params += (object_type,)
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def list_ids(
        self,
        research_id: str,
        session_id: str,
        object_type: Optional[str] = None,
    ) -> List[str]:
        query = "SELECT object_id FROM session_objects WHERE research_id = ? AND session_id = ?"
        params: Tuple[Any, ...] = (research_id, session_id)
        if object_type:
            query += " AND object_type = ?"
            params += (object_type,)
        rows = self.conn.execute(query, params).fetchall()
        return [row["object_id"] for row in rows]

    def get_distinct_source_ids(self, research_id: str, session_id: str) -> List[str]:
        rows = self.conn.execute(
            """
            SELECT DISTINCT source_id
            FROM session_objects
            WHERE research_id = ? AND session_id = ? AND source_id IS NOT NULL AND source_id != ''
            """,
            (research_id, session_id),
        ).fetchall()
        return [row["source_id"] for row in rows]

    def delete_session(self, research_id: str, session_id: str) -> None:
        self.conn.execute(
            "DELETE FROM session_objects WHERE research_id = ? AND session_id = ?",
            (research_id, session_id),
        )
        self.conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        data = self._deserialize(row["data"])
        data["id"] = row["object_id"]
        data["object_type"] = row["object_type"]
        data["research_id"] = row["research_id"]
        data["session_id"] = row["session_id"]
        data["task_id"] = row["task_id"]
        data["section_id"] = row["section_id"]
        data["source_id"] = row["source_id"]
        data["status"] = row["status"]
        data["dedupe_key"] = row["dedupe_key"]
        for field_name in ("created_at", "updated_at"):
            raw_value = row[field_name]
            try:
                data[field_name] = datetime.fromisoformat(raw_value) if raw_value else raw_value
            except ValueError:
                data[field_name] = raw_value
        return data


class KnowledgeManager:
    def __init__(
        self,
        base_storage_path: str = "./knowledge_data",
        sqlite_filename: str = "session_knowledge.sqlite3",
        storage_path: Optional[str] = None,
    ):
        self.base_storage_path = storage_path or base_storage_path
        os.makedirs(self.base_storage_path, exist_ok=True)
        self.db_path = ":memory:" if sqlite_filename == ":memory:" else os.path.join(self.base_storage_path, sqlite_filename)
        self.store = SessionKnowledgeStore(self.db_path)
        self._semaphore = asyncio.Semaphore(3)

    @classmethod
    def get_collection_name(cls, research_id: str) -> str:
        return f"research_{research_id}"

    def close(self) -> None:
        self.store.close()

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _normalize_text(self, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()

    def _normalize_key(self, value: Any) -> str:
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", self._normalize_text(value).lower()).strip()

    def _unique_ids(self, values: List[Any]) -> List[str]:
        seen: set[str] = set()
        output: List[str] = []
        for value in values:
            normalized = str(value).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    def _normalize_id(self, item: Dict[str, Any], object_type: str) -> str:
        if object_type == "section_evidence_pack":
            return str(item.get("pack_id") or item.get("id") or str(uuid.uuid4()))
        return str(item.get("id") or str(uuid.uuid4()))

    def _normalize_item(
        self,
        item: Any,
        *,
        research_id: str,
        session_id: str,
        object_type: str,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if hasattr(item, "model_dump"):
            data = item.model_dump()
        elif isinstance(item, dict):
            data = dict(item)
        else:
            raise TypeError(f"Unsupported knowledge item type: {type(item)}")

        object_id = self._normalize_id(data, object_type)
        now = self._now()
        data["id"] = object_id
        if object_type == "section_evidence_pack":
            data["pack_id"] = object_id

        data["research_id"] = str(data.get("research_id") or research_id)
        data["session_id"] = str(data.get("session_id") or session_id)
        data["task_id"] = data.get("task_id") or task_id
        data["section_id"] = str(data.get("section_id") or "")
        if object_type == "fact":
            data["source_id"] = str(data.get("source_id") or data.get("source_url") or "")
            data["status"] = str(
                data.get("status")
                or (FactStatus.CONFLICTING.value if data.get("is_conflict") else FactStatus.ACTIVE.value)
            )
        elif object_type == "evidence":
            data["source_id"] = str(data.get("source_id") or data.get("source_url") or "")
            data["status"] = str(data.get("status") or "active")
        elif object_type == "conflict":
            data["status"] = str(data.get("status") or "active")
            data["source_id"] = str(data.get("source_id") or "")
        else:
            data["source_id"] = str(data.get("source_id") or "")
            data["status"] = str(data.get("status") or "active")
        data.setdefault("created_at", now)
        data["updated_at"] = now
        return data

    def _dedupe_key(self, item: Dict[str, Any], object_type: str) -> str:
        if object_type == "fact":
            return f"fact:{self._normalize_key(item.get('text'))}:{self._normalize_key(item.get('source_url'))}"
        if object_type == "claim":
            return f"claim:{self._normalize_key(item.get('text'))}"
        if object_type == "evidence":
            quote = item.get("quote") or item.get("summary") or ""
            return f"evidence:{self._normalize_key(item.get('source_url') or item.get('source_id'))}:{self._normalize_key(quote)}"
        if object_type == "conflict":
            fact_ids = ",".join(sorted(str(v) for v in item.get("fact_ids", []) if v))
            return f"conflict:{fact_ids}:{self._normalize_key(item.get('description'))}"
        if object_type == "section_evidence_pack":
            return f"section_pack:{self._normalize_key(item.get('section_id'))}:{self._normalize_key(item.get('goal'))}"
        return f"{object_type}:{self._normalize_key(item.get('id'))}"

    def _merge_lists(self, left: List[Any], right: List[Any]) -> List[str]:
        return self._unique_ids(list(left or []) + list(right or []))

    def _merge_item(self, existing: Dict[str, Any], incoming: Dict[str, Any], object_type: str) -> Dict[str, Any]:
        merged = dict(existing)
        for key, value in incoming.items():
            if value is None:
                continue
            if isinstance(value, list):
                merged[key] = self._merge_lists(merged.get(key, []), value)
            elif value != "":
                merged[key] = value

        if object_type == "fact":
            merged["confidence"] = max(float(existing.get("confidence", 0.0)), float(incoming.get("confidence", 0.0)))
            merged["verified_count"] = max(
                int(existing.get("verified_count", 0)),
                int(incoming.get("verified_count", 0)),
            )
            merged["is_conflict"] = bool(existing.get("is_conflict")) or bool(incoming.get("is_conflict"))
            merged["conflict_with"] = self._merge_lists(existing.get("conflict_with", []), incoming.get("conflict_with", []))
            if merged["is_conflict"]:
                merged["status"] = FactStatus.CONFLICTING.value
        merged["updated_at"] = self._now()
        return merged

    def _build_existing_indexes(
        self,
        research_id: str,
        session_id: str,
        object_type: str,
    ) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        rows = self.store.list_objects(research_id, session_id, object_type=object_type)
        by_id = {str(row["id"]): row for row in rows}
        by_dedupe: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            key = str(row.get("dedupe_key") or self._dedupe_key(row, object_type))
            if key:
                by_dedupe[key] = row
        return by_id, by_dedupe

    def _persist_objects(
        self,
        items: List[Any],
        *,
        research_id: str,
        session_id: str,
        object_type: str,
        task_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        persisted: List[Dict[str, Any]] = []
        id_map: Dict[str, str] = {}
        existing_by_id, existing_by_dedupe = self._build_existing_indexes(research_id, session_id, object_type)

        for item in items:
            normalized = self._normalize_item(
                item,
                research_id=research_id,
                session_id=session_id,
                object_type=object_type,
                task_id=task_id,
            )
            incoming_id = normalized["id"]
            dedupe_key = self._dedupe_key(normalized, object_type)
            existing = existing_by_id.get(incoming_id) or existing_by_dedupe.get(dedupe_key)

            if existing:
                canonical_id = str(existing["id"])
                normalized["id"] = canonical_id
                if object_type == "section_evidence_pack":
                    normalized["pack_id"] = canonical_id
                merged = self._merge_item(existing, normalized, object_type)
            else:
                canonical_id = incoming_id
                merged = normalized

            self.store.upsert_object(
                object_id=canonical_id,
                research_id=research_id,
                session_id=session_id,
                object_type=object_type,
                data=merged,
                task_id=merged.get("task_id"),
                section_id=merged.get("section_id"),
                source_id=merged.get("source_id"),
                status=merged.get("status"),
                dedupe_key=dedupe_key,
            )

            saved = self.store.get_object(canonical_id, research_id, session_id) or merged
            existing_by_id[canonical_id] = saved
            existing_by_dedupe[dedupe_key] = saved
            id_map[str(incoming_id)] = canonical_id
            persisted.append(saved)

        return persisted, id_map

    def _remap_ids(self, values: List[Any], id_map: Dict[str, str]) -> List[str]:
        return self._unique_ids([id_map.get(str(value), str(value)) for value in values or []])

    def _remap_fact_links(self, payload: Dict[str, Any], fact_id_map: Dict[str, str]) -> None:
        for claim in payload.get("claims", []):
            claim["fact_ids"] = self._remap_ids(claim.get("fact_ids", []), fact_id_map)
        for evidence in payload.get("evidence", []):
            evidence["fact_ids"] = self._remap_ids(evidence.get("fact_ids", []), fact_id_map)
        for conflict in payload.get("conflicts", []):
            conflict["fact_ids"] = self._remap_ids(conflict.get("fact_ids", []), fact_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["fact_ids"] = self._remap_ids(pack.get("fact_ids", []), fact_id_map)

    def _remap_claim_links(self, payload: Dict[str, Any], claim_id_map: Dict[str, str]) -> None:
        for evidence in payload.get("evidence", []):
            evidence["claim_ids"] = self._remap_ids(evidence.get("claim_ids", []), claim_id_map)
        for conflict in payload.get("conflicts", []):
            conflict["claim_ids"] = self._remap_ids(conflict.get("claim_ids", []), claim_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["claim_ids"] = self._remap_ids(pack.get("claim_ids", []), claim_id_map)

    def _remap_evidence_links(self, payload: Dict[str, Any], evidence_id_map: Dict[str, str]) -> None:
        for conflict in payload.get("conflicts", []):
            conflict["evidence_ids"] = self._remap_ids(conflict.get("evidence_ids", []), evidence_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["evidence_ids"] = self._remap_ids(pack.get("evidence_ids", []), evidence_id_map)

    def _remap_conflict_links(self, payload: Dict[str, Any], conflict_id_map: Dict[str, str]) -> None:
        for pack in payload.get("section_evidence_packs", []):
            pack["conflict_ids"] = self._remap_ids(pack.get("conflict_ids", []), conflict_id_map)

    def _generate_knowledge_refs(
        self,
        original_refs: Dict[str, Any],
        research_id: str,
        session_id: str,
    ) -> Dict[str, Any]:
        refs = dict(original_refs or {})
        refs["collection_name"] = refs.get("collection_name") or self.get_collection_name(research_id)
        refs["fact_ids"] = self._unique_ids(self.store.list_ids(research_id, session_id, "fact"))
        refs["claim_ids"] = self._unique_ids(self.store.list_ids(research_id, session_id, "claim"))
        refs["evidence_ids"] = self._unique_ids(self.store.list_ids(research_id, session_id, "evidence"))
        refs["conflict_ids"] = self._unique_ids(self.store.list_ids(research_id, session_id, "conflict"))
        refs["section_pack_ids"] = self._unique_ids(self.store.list_ids(research_id, session_id, "section_evidence_pack"))
        refs["source_ids"] = self._unique_ids(
            list(refs.get("source_ids", [])) + self.store.get_distinct_source_ids(research_id, session_id)
        )
        return KnowledgeRefs(**refs).model_dump()

    def process_distiller_output(
        self,
        distiller_output: Any,
        research_id: str,
        session_id: str,
        task_id: Optional[str] = None,
    ) -> DistillerOutputs:
        payload = distiller_output.model_dump() if hasattr(distiller_output, "model_dump") else dict(distiller_output)
        payload["atomic_facts"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("atomic_facts", [])]
        payload["claims"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("claims", [])]
        payload["evidence"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("evidence", [])]
        payload["conflicts"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("conflicts", [])]
        payload["section_evidence_packs"] = [
            dict(item) if isinstance(item, dict) else item.model_dump()
            for item in payload.get("section_evidence_packs", [])
        ]

        original_refs = payload.get("knowledge_refs", {}) or {}

        persisted_facts, fact_id_map = self._persist_objects(
            payload["atomic_facts"],
            research_id=research_id,
            session_id=session_id,
            object_type="fact",
            task_id=task_id,
        )
        payload["atomic_facts"] = persisted_facts
        self._remap_fact_links(payload, fact_id_map)

        persisted_claims, claim_id_map = self._persist_objects(
            payload["claims"],
            research_id=research_id,
            session_id=session_id,
            object_type="claim",
            task_id=task_id,
        )
        payload["claims"] = persisted_claims
        self._remap_claim_links(payload, claim_id_map)

        persisted_evidence, evidence_id_map = self._persist_objects(
            payload["evidence"],
            research_id=research_id,
            session_id=session_id,
            object_type="evidence",
            task_id=task_id,
        )
        payload["evidence"] = persisted_evidence
        self._remap_evidence_links(payload, evidence_id_map)

        persisted_conflicts, conflict_id_map = self._persist_objects(
            payload["conflicts"],
            research_id=research_id,
            session_id=session_id,
            object_type="conflict",
            task_id=task_id,
        )
        payload["conflicts"] = persisted_conflicts
        self._remap_conflict_links(payload, conflict_id_map)

        persisted_packs, _ = self._persist_objects(
            payload["section_evidence_packs"],
            research_id=research_id,
            session_id=session_id,
            object_type="section_evidence_pack",
            task_id=task_id,
        )
        payload["section_evidence_packs"] = persisted_packs

        payload["knowledge_refs"] = self._generate_knowledge_refs(original_refs, research_id, session_id)
        payload["fact_ids"] = payload["knowledge_refs"]["fact_ids"]
        payload["claim_ids"] = payload["knowledge_refs"]["claim_ids"]
        payload["evidence_ids"] = payload["knowledge_refs"]["evidence_ids"]
        payload["conflict_ids"] = payload["knowledge_refs"]["conflict_ids"]

        return DistillerOutputs(**payload)

    def reload_session(self, research_id: str, session_id: str) -> List[Dict[str, Any]]:
        return self.store.list_objects(research_id, session_id)

    def get_session_snapshot(self, research_id: str, session_id: str) -> Dict[str, Any]:
        facts = self.store.list_objects(research_id, session_id, object_type="fact")
        claims = self.store.list_objects(research_id, session_id, object_type="claim")
        evidence = self.store.list_objects(research_id, session_id, object_type="evidence")
        conflicts = self.store.list_objects(research_id, session_id, object_type="conflict")
        section_packs = self.store.list_objects(research_id, session_id, object_type="section_evidence_pack")
        refs = self._generate_knowledge_refs({}, research_id, session_id)
        return {
            "facts": facts,
            "claims": claims,
            "evidence": evidence,
            "conflicts": conflicts,
            "section_evidence_packs": section_packs,
            "knowledge_refs": refs,
        }

    def clear_session(self, research_id: str, session_id: str) -> None:
        self.store.delete_session(research_id, session_id)

    # Compatibility helpers below are retained for older tests and experiments.
    async def _find_similar(
        self,
        collection_name: str,
        text: str,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        rows = self.store.list_objects("legacy", collection_name, object_type="fact")
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        query_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", self._normalize_text(text).lower()))
        for row in rows:
            row_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", self._normalize_text(row.get("text", "")).lower()))
            if not query_tokens or not row_tokens:
                continue
            score = len(query_tokens & row_tokens) / len(query_tokens | row_tokens)
            if score >= threshold:
                results.append((str(row["id"]), float(score), row))
        results.sort(key=lambda item: item[1], reverse=True)
        return results

    async def add_facts(
        self,
        facts: List[AtomicFact],
        collection_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        session_id = collection_name or "legacy_default"
        added_ids: List[str] = []
        for fact in facts:
            persisted, _ = self._persist_objects(
                [fact],
                research_id="legacy",
                session_id=session_id,
                object_type="fact",
                task_id=task_id,
            )
            if persisted:
                added_ids.append(str(persisted[0]["id"]))
        return self._unique_ids(added_ids), []

    async def upsert_fact_with_verification(
        self,
        fact: AtomicFact,
        collection_name: str,
    ) -> UpsertFactResult:
        persisted, id_map = self._persist_objects(
            [fact],
            research_id="legacy",
            session_id=collection_name or "legacy_default",
            object_type="fact",
            task_id=fact.task_id,
        )
        fact_id = id_map.get(fact.id, fact.id)
        action = "NEW"
        if persisted and fact_id != fact.id:
            action = "UPDATED"
        return UpsertFactResult(action=action, fact_id=fact_id)

    async def search_facts(
        self,
        query: str,
        collection_name: Optional[str] = None,
        task_id_filter: Optional[str] = None,
        status_filter: Optional[FactStatus] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        session_id = collection_name or "legacy_default"
        rows = self.store.list_objects("legacy", session_id, object_type="fact")
        query_key = self._normalize_key(query)
        results: List[Dict[str, Any]] = []
        for row in rows:
            if task_id_filter and row.get("task_id") != task_id_filter:
                continue
            if status_filter and row.get("status") != status_filter.value:
                continue
            text_key = self._normalize_key(row.get("text", ""))
            score = 1.0 if not query_key else float(query_key in text_key or text_key in query_key)
            if query_key and score <= 0.0:
                continue
            results.append(
                {
                    "id": row.get("id"),
                    "text": row.get("text", ""),
                    "source_url": row.get("source_url", ""),
                    "confidence": row.get("confidence", 0.0),
                    "status": row.get("status", FactStatus.ACTIVE.value),
                    "score": score,
                }
            )
        return results[:limit]

    async def get_fact_by_id(self, fact_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.store.get_object(fact_id, "legacy", collection_name or "legacy_default")

    async def get_conflicts(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        return self.store.list_objects("legacy", collection_name or "legacy_default", object_type="conflict")

    async def clear_collection(self, collection_name: Optional[str] = None) -> None:
        self.clear_session("legacy", collection_name or "legacy_default")

    def get_stats(self, collection_name: Optional[str] = None) -> KnowledgeStats:
        facts = self.store.list_objects("legacy", collection_name or "legacy_default", object_type="fact")
        conflicts = self.store.list_objects("legacy", collection_name or "legacy_default", object_type="conflict")
        return KnowledgeStats(
            total_facts=len(facts),
            verified_facts=sum(1 for fact in facts if fact.get("status") == FactStatus.VERIFIED.value),
            conflicting_facts=sum(1 for fact in facts if fact.get("status") == FactStatus.CONFLICTING.value),
            conflicts_detected=len(conflicts),
        )

    def get_source_level(self, url: str) -> SourceLevel:
        normalized = (url or "").lower()
        if any(domain in normalized for domain in ("nature.com", "sec.gov", "patents.google.com")):
            return SourceLevel.S
        if any(domain in normalized for domain in ("reuters.com", "bloomberg.com", "apple.com")):
            return SourceLevel.A
        if any(domain in normalized for domain in ("github.com", "reddit.com")):
            return SourceLevel.B
        return SourceLevel.C
