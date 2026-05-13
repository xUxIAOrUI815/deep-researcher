from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx

from schemas.session import (
    ResearchSessionRecord,
    SessionClaimRecord,
    SessionConflictRecord,
    SessionCoverageSnapshotRecord,
    SessionEvidenceRecord,
    SessionFactRecord,
    SessionNoveltySnapshotRecord,
    SessionSectionPackRecord,
    SessionSnapshot,
    SessionSourceRecord,
    SessionUnresolvedGapRecord,
)
from schemas.state import DistillerOutputs, KnowledgeRefs, SourceLevel

logger = logging.getLogger(__name__)

SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY", "")
SILICON_FLOW_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"


# TODO: 这里的状态设计没有生效
class FactStatus(Enum):
    ACTIVE = "active"
    VERIFIED = "verified"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"


class EmbeddingModel:
    """Compatibility wrapper retained for older tests and callers."""

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
    def __init__(self, db_path: str = "./session_knowledge.sqlite3"):
        self.db_path = db_path
        if db_path != ":memory:":
            os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._transaction_depth = 0
        self._init_db()

    def close(self) -> None:
        try:
            self.conn.commit()
            self.conn.close()
        except Exception:
            pass

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_sessions (
                research_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                root_query TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_active_at TEXT NOT NULL,
                current_round INTEGER NOT NULL DEFAULT 0,
                current_active_task_id TEXT,
                metadata_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_sources (
                research_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                source_type TEXT NOT NULL,
                authority_score REAL NOT NULL DEFAULT 0.0,
                freshness_score REAL NOT NULL DEFAULT 0.0,
                task_id TEXT,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (research_id, source_id),
                UNIQUE(research_id, url)
            );

            CREATE TABLE IF NOT EXISTS session_claims (
                claim_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                task_id TEXT,
                section_id TEXT,
                canonical_text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                source_count INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                fact_count INTEGER NOT NULL DEFAULT 0,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dedup_key TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, dedup_key)
            );

            CREATE TABLE IF NOT EXISTS session_facts (
                fact_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                task_id TEXT,
                section_id TEXT,
                canonical_text TEXT NOT NULL,
                raw_text TEXT NOT NULL,
                snippet TEXT NOT NULL,
                confidence REAL NOT NULL DEFAULT 0.0,
                verified_count INTEGER NOT NULL DEFAULT 0,
                source_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                dedup_key TEXT NOT NULL,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, dedup_key)
            );

            CREATE TABLE IF NOT EXISTS session_evidence (
                evidence_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                task_id TEXT,
                section_id TEXT,
                source_id TEXT NOT NULL,
                quote_text TEXT NOT NULL,
                summary_text TEXT NOT NULL,
                quality_score REAL NOT NULL DEFAULT 0.0,
                confidence REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                dedup_key TEXT NOT NULL,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, source_id, dedup_key)
            );

            CREATE TABLE IF NOT EXISTS session_conflicts (
                conflict_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                task_id TEXT,
                section_id TEXT,
                conflict_type TEXT NOT NULL,
                description TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                claim_count INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dedup_key TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, dedup_key)
            );

            CREATE TABLE IF NOT EXISTS session_section_packs (
                pack_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                section_id TEXT NOT NULL,
                section_title TEXT NOT NULL,
                goal TEXT NOT NULL,
                coverage_score REAL NOT NULL DEFAULT 0.0,
                status TEXT NOT NULL,
                claim_count INTEGER NOT NULL DEFAULT 0,
                fact_count INTEGER NOT NULL DEFAULT 0,
                evidence_count INTEGER NOT NULL DEFAULT 0,
                conflict_count INTEGER NOT NULL DEFAULT 0,
                notes TEXT NOT NULL,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_updated_round INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, section_id)
            );
            """
        )
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS session_claim_fact_links (
                research_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, claim_id, fact_id)
            );

            CREATE TABLE IF NOT EXISTS session_claim_evidence_links (
                research_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, claim_id, evidence_id)
            );

            CREATE TABLE IF NOT EXISTS session_pack_fact_links (
                research_id TEXT NOT NULL,
                pack_id TEXT NOT NULL,
                fact_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, pack_id, fact_id)
            );

            CREATE TABLE IF NOT EXISTS session_pack_claim_links (
                research_id TEXT NOT NULL,
                pack_id TEXT NOT NULL,
                claim_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, pack_id, claim_id)
            );

            CREATE TABLE IF NOT EXISTS session_pack_evidence_links (
                research_id TEXT NOT NULL,
                pack_id TEXT NOT NULL,
                evidence_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, pack_id, evidence_id)
            );

            CREATE TABLE IF NOT EXISTS session_pack_conflict_links (
                research_id TEXT NOT NULL,
                pack_id TEXT NOT NULL,
                conflict_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(research_id, pack_id, conflict_id)
            );

            CREATE TABLE IF NOT EXISTS session_coverage_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                avg_section_coverage REAL NOT NULL DEFAULT 0.0,
                evidence_density REAL NOT NULL DEFAULT 0.0,
                conflict_pressure REAL NOT NULL DEFAULT 0.0,
                sufficiency_level TEXT NOT NULL,
                completed_section_count INTEGER NOT NULL DEFAULT 0,
                partial_section_count INTEGER NOT NULL DEFAULT 0,
                uncovered_section_count INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                raw_summary_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS session_unresolved_gaps (
                gap_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                task_id TEXT,
                section_id TEXT,
                gap_text TEXT NOT NULL,
                gap_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                UNIQUE(research_id, gap_text)
            );

            CREATE TABLE IF NOT EXISTS session_novelty_snapshots (
                snapshot_id TEXT PRIMARY KEY,
                research_id TEXT NOT NULL,
                round_no INTEGER NOT NULL,
                new_fact_count INTEGER NOT NULL DEFAULT 0,
                merged_fact_count INTEGER NOT NULL DEFAULT 0,
                new_source_count INTEGER NOT NULL DEFAULT 0,
                new_claim_count INTEGER NOT NULL DEFAULT 0,
                new_evidence_count INTEGER NOT NULL DEFAULT 0,
                novelty_ratio REAL NOT NULL DEFAULT 0.0,
                novelty_level TEXT NOT NULL,
                created_at TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            """
        )
        self._migrate_session_sources_to_research_scoped_primary_key()
        self._commit()

    def _commit(self) -> None:
        if self._transaction_depth == 0:
            self.conn.commit()

    def _migrate_session_sources_to_research_scoped_primary_key(self) -> None:
        columns = self.conn.execute("PRAGMA table_info(session_sources)").fetchall()
        primary_key_columns = [row["name"] for row in sorted(columns, key=lambda item: item["pk"]) if row["pk"]]
        if primary_key_columns == ["research_id", "source_id"]:
            return

        self.conn.executescript(
            """
            DROP TABLE IF EXISTS session_sources_research_scoped;

            CREATE TABLE session_sources_research_scoped (
                research_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                url TEXT NOT NULL,
                title TEXT NOT NULL,
                domain TEXT NOT NULL,
                source_type TEXT NOT NULL,
                authority_score REAL NOT NULL DEFAULT 0.0,
                freshness_score REAL NOT NULL DEFAULT 0.0,
                task_id TEXT,
                first_seen_round INTEGER NOT NULL DEFAULT 0,
                last_seen_round INTEGER NOT NULL DEFAULT 0,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                is_active INTEGER NOT NULL DEFAULT 1,
                metadata_json TEXT NOT NULL,
                PRIMARY KEY (research_id, source_id),
                UNIQUE(research_id, url)
            );

            INSERT OR IGNORE INTO session_sources_research_scoped (
                research_id,
                source_id,
                url,
                title,
                domain,
                source_type,
                authority_score,
                freshness_score,
                task_id,
                first_seen_round,
                last_seen_round,
                first_seen_at,
                last_seen_at,
                is_active,
                metadata_json
            )
            SELECT
                research_id,
                source_id,
                url,
                title,
                domain,
                source_type,
                authority_score,
                freshness_score,
                task_id,
                first_seen_round,
                last_seen_round,
                first_seen_at,
                last_seen_at,
                is_active,
                metadata_json
            FROM session_sources;

            DROP TABLE session_sources;
            ALTER TABLE session_sources_research_scoped RENAME TO session_sources;
            """
        )

    @contextmanager
    def transaction(self):
        outermost = self._transaction_depth == 0
        if outermost:
            self.conn.execute("BEGIN")
        self._transaction_depth += 1
        try:
            yield
        except Exception:
            self._transaction_depth -= 1
            if outermost:
                self.conn.rollback()
            raise
        else:
            self._transaction_depth -= 1
            if outermost:
                self.conn.commit()

    def _now(self) -> str:
        return datetime.now().isoformat()

    def _dump(self, payload: Dict[str, Any]) -> str:
        return json.dumps(payload, ensure_ascii=False, default=str)

    def _load(self, payload: str) -> Dict[str, Any]:
        return json.loads(payload or "{}")

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return {key: row[key] for key in row.keys()}

    def _fetchone_dict(self, query: str, params: Tuple[Any, ...]) -> Optional[Dict[str, Any]]:
        row = self.conn.execute(query, params).fetchone()
        return self._row_to_dict(row) if row else None

    def _fetchall_dicts(self, query: str, params: Tuple[Any, ...]) -> List[Dict[str, Any]]:
        rows = self.conn.execute(query, params).fetchall()
        return [self._row_to_dict(row) for row in rows]

    def create_or_get_session(
        self,
        research_id: str,
        root_query: str,
        session_id: Optional[str] = None,
        status: str = "active",
        current_active_task_id: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> ResearchSessionRecord:
        existing = self._fetchone_dict("SELECT * FROM research_sessions WHERE research_id = ?", (research_id,))
        now = self._now()
        resolved_session_id = session_id or (existing or {}).get("session_id") or f"session_{research_id}"
        if existing:
            self.conn.execute(
                """
                UPDATE research_sessions
                SET session_id = ?, root_query = ?, status = ?, updated_at = ?, last_active_at = ?,
                    current_active_task_id = ?, metadata_json = ?
                WHERE research_id = ?
                """,
                (
                    resolved_session_id,
                    existing["root_query"],
                    status,
                    now,
                    now,
                    current_active_task_id,
                    self._dump(metadata_json or self._load(existing["metadata_json"])),
                    research_id,
                ),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO research_sessions (
                    research_id, session_id, root_query, status, created_at, updated_at,
                    last_active_at, current_round, current_active_task_id, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    research_id,
                    resolved_session_id,
                    root_query or research_id,
                    status,
                    now,
                    now,
                    now,
                    0,
                    current_active_task_id,
                    self._dump(metadata_json or {}),
                ),
            )
        self._commit()
        session = self._fetchone_dict("SELECT * FROM research_sessions WHERE research_id = ?", (research_id,))
        return ResearchSessionRecord(
            research_id=session["research_id"],
            session_id=session["session_id"],
            root_query=session["root_query"],
            status=session["status"],
            created_at=datetime.fromisoformat(session["created_at"]),
            updated_at=datetime.fromisoformat(session["updated_at"]),
            last_active_at=datetime.fromisoformat(session["last_active_at"]),
            current_round=int(session["current_round"]),
            current_active_task_id=session["current_active_task_id"],
            metadata_json=self._load(session["metadata_json"]),
        )

    def get_session(self, research_id: str) -> Optional[ResearchSessionRecord]:
        row = self._fetchone_dict("SELECT * FROM research_sessions WHERE research_id = ?", (research_id,))
        if not row:
            return None
        return ResearchSessionRecord(
            research_id=row["research_id"],
            session_id=row["session_id"],
            root_query=row["root_query"],
            status=row["status"],
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_active_at=datetime.fromisoformat(row["last_active_at"]),
            current_round=int(row["current_round"]),
            current_active_task_id=row["current_active_task_id"],
            metadata_json=self._load(row["metadata_json"]),
        )

    def update_session_round(
        self,
        research_id: str,
        round_no: int,
        current_active_task_id: Optional[str] = None,
        status: str = "active",
    ) -> None:
        now = self._now()
        self.conn.execute(
            """
            UPDATE research_sessions
            SET current_round = ?, current_active_task_id = ?, status = ?, updated_at = ?, last_active_at = ?
            WHERE research_id = ?
            """,
            (round_no, current_active_task_id, status, now, now, research_id),
        )
        self._commit()

    def update_session_status(
        self,
        research_id: str,
        *,
        status: str,
        current_active_task_id: Optional[str] = None,
    ) -> None:
        now = self._now()
        self.conn.execute(
            """
            UPDATE research_sessions
            SET status = ?, current_active_task_id = ?, updated_at = ?, last_active_at = ?
            WHERE research_id = ?
            """,
            (status, current_active_task_id, now, now, research_id),
        )
        self._commit()

    def _upsert_rows(
        self,
        *,
        table: str,
        id_field: str,
        unique_lookup_field: Optional[str],
        rows: List[Dict[str, Any]],
        merge_fn: Any,
    ) -> Dict[str, Any]:
        inserted = 0
        updated = 0
        id_map: Dict[str, str] = {}
        persisted: List[Dict[str, Any]] = []
        for row in rows:
            lookup_value = row.get(unique_lookup_field) if unique_lookup_field else None
            existing: Optional[Dict[str, Any]] = None
            if unique_lookup_field:
                existing = self._fetchone_dict(
                    f"SELECT * FROM {table} WHERE research_id = ? AND {unique_lookup_field} = ?",
                    (row["research_id"], lookup_value),
                )
            if not existing:
                existing = self._fetchone_dict(
                    f"SELECT * FROM {table} WHERE {id_field} = ? AND research_id = ?",
                    (row[id_field], row["research_id"]),
                )
            if existing:
                merged = merge_fn(existing, row)
                columns = [key for key in merged.keys() if key != id_field]
                assignments = ", ".join(f"{column} = ?" for column in columns)
                self.conn.execute(
                    f"UPDATE {table} SET {assignments} WHERE {id_field} = ? AND research_id = ?",
                    tuple(merged[column] for column in columns) + (existing[id_field], row["research_id"]),
                )
                canonical_id = existing[id_field]
                updated += 1
            else:
                columns = list(row.keys())
                placeholders = ", ".join("?" for _ in columns)
                self.conn.execute(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                    tuple(row[column] for column in columns),
                )
                canonical_id = row[id_field]
                inserted += 1
            id_map[str(row[id_field])] = str(canonical_id)
            persisted_row = self._fetchone_dict(
                f"SELECT * FROM {table} WHERE {id_field} = ? AND research_id = ?",
                (canonical_id, row["research_id"]),
            )
            if persisted_row:
                persisted.append(persisted_row)
        self._commit()
        return {"inserted": inserted, "updated": updated, "id_map": id_map, "rows": persisted}

    def _upsert_link_rows(self, table: str, columns: Tuple[str, ...], rows: List[Tuple[Any, ...]]) -> None:
        if not rows:
            return
        now = self._now()
        placeholders = ", ".join("?" for _ in (*columns, "created_at"))
        insert_columns = ", ".join((*columns, "created_at"))
        for row in rows:
            self.conn.execute(
                f"INSERT OR IGNORE INTO {table} ({insert_columns}) VALUES ({placeholders})",
                (*row, now),
            )
        self._commit()

    def upsert_sources(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(
                {
                    "title": incoming["title"] or existing["title"],
                    "domain": incoming["domain"] or existing["domain"],
                    "source_type": incoming["source_type"] or existing["source_type"],
                    "authority_score": max(float(existing["authority_score"]), float(incoming["authority_score"])),
                    "freshness_score": max(float(existing["freshness_score"]), float(incoming["freshness_score"])),
                    "task_id": incoming["task_id"] or existing["task_id"],
                    "last_seen_round": max(int(existing["last_seen_round"]), int(incoming["last_seen_round"])),
                    "last_seen_at": incoming["last_seen_at"],
                    "is_active": incoming["is_active"],
                    "metadata_json": incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_sources",
            id_field="source_id",
            unique_lookup_field="url",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_claims(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(
                {
                    "task_id": incoming["task_id"] or existing["task_id"],
                    "section_id": incoming["section_id"] or existing["section_id"],
                    "raw_text": incoming["raw_text"] or existing["raw_text"],
                    "confidence": max(float(existing["confidence"]), float(incoming["confidence"])),
                    "status": incoming["status"] or existing["status"],
                    "source_count": max(int(existing["source_count"]), int(incoming["source_count"])),
                    "evidence_count": max(int(existing["evidence_count"]), int(incoming["evidence_count"])),
                    "fact_count": max(int(existing["fact_count"]), int(incoming["fact_count"])),
                    "last_seen_round": max(int(existing["last_seen_round"]), int(incoming["last_seen_round"])),
                    "updated_at": incoming["updated_at"],
                    "metadata_json": incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_claims",
            id_field="claim_id",
            unique_lookup_field="dedup_key",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_facts(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(
                {
                    "task_id": incoming["task_id"] or existing["task_id"],
                    "section_id": incoming["section_id"] or existing["section_id"],
                    "raw_text": incoming["raw_text"] or existing["raw_text"],
                    "snippet": incoming["snippet"] or existing["snippet"],
                    "confidence": max(float(existing["confidence"]), float(incoming["confidence"])),
                    "verified_count": max(int(existing["verified_count"]), int(incoming["verified_count"])),
                    "source_count": max(int(existing["source_count"]), int(incoming["source_count"])),
                    "status": incoming["status"] or existing["status"],
                    "last_seen_round": max(int(existing["last_seen_round"]), int(incoming["last_seen_round"])),
                    "updated_at": incoming["updated_at"],
                    "metadata_json": incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_facts",
            id_field="fact_id",
            unique_lookup_field="dedup_key",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_evidence(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(
                {
                    "task_id": incoming["task_id"] or existing["task_id"],
                    "section_id": incoming["section_id"] or existing["section_id"],
                    "quality_score": max(float(existing["quality_score"]), float(incoming["quality_score"])),
                    "confidence": max(float(existing["confidence"]), float(incoming["confidence"])),
                    "status": incoming["status"] or existing["status"],
                    "last_seen_round": max(int(existing["last_seen_round"]), int(incoming["last_seen_round"])),
                    "updated_at": incoming["updated_at"],
                    "metadata_json": incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_evidence",
            id_field="evidence_id",
            unique_lookup_field="dedup_key",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_conflicts(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            merged.update(
                {
                    "task_id": incoming["task_id"] or existing["task_id"],
                    "section_id": incoming["section_id"] or existing["section_id"],
                    "severity": incoming["severity"] or existing["severity"],
                    "status": incoming["status"] or existing["status"],
                    "claim_count": max(int(existing["claim_count"]), int(incoming["claim_count"])),
                    "evidence_count": max(int(existing["evidence_count"]), int(incoming["evidence_count"])),
                    "last_seen_round": max(int(existing["last_seen_round"]), int(incoming["last_seen_round"])),
                    "updated_at": incoming["updated_at"],
                    "metadata_json": incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_conflicts",
            id_field="conflict_id",
            unique_lookup_field="dedup_key",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_section_packs(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        def metadata_ref_count(payload: str) -> int:
            try:
                meta = self._load(payload)
            except Exception:
                meta = {}
            return sum(len(meta.get(key, []) or []) for key in ("claim_ids", "fact_ids", "evidence_ids", "conflict_ids"))

        def linked_ref_count(research_id: str, pack_id: str) -> int:
            total = 0
            for table in (
                "session_pack_claim_links",
                "session_pack_fact_links",
                "session_pack_evidence_links",
                "session_pack_conflict_links",
            ):
                total += int(
                    self.conn.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE research_id = ? AND pack_id = ?",
                        (research_id, pack_id),
                    ).fetchone()[0]
                )
            return total

        def merge(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
            merged = dict(existing)
            incoming_ref_count = metadata_ref_count(incoming["metadata_json"]) + sum(
                int(incoming.get(key, 0) or 0)
                for key in ("claim_count", "fact_count", "evidence_count", "conflict_count")
            )
            existing_ref_count = metadata_ref_count(existing["metadata_json"]) + sum(
                int(existing.get(key, 0) or 0)
                for key in ("claim_count", "fact_count", "evidence_count", "conflict_count")
            ) + linked_ref_count(
                existing["research_id"],
                existing["pack_id"],
            )
            preserve_existing_refs = incoming_ref_count == 0 and existing_ref_count > 0
            merged.update(
                {
                    "section_title": incoming["section_title"] or existing["section_title"],
                    "goal": incoming["goal"] or existing["goal"],
                    "coverage_score": float(existing["coverage_score"] if preserve_existing_refs else incoming["coverage_score"]),
                    "status": incoming["status"] or existing["status"],
                    "claim_count": int(existing["claim_count"] if preserve_existing_refs else incoming["claim_count"]),
                    "fact_count": int(existing["fact_count"] if preserve_existing_refs else incoming["fact_count"]),
                    "evidence_count": int(existing["evidence_count"] if preserve_existing_refs else incoming["evidence_count"]),
                    "conflict_count": int(existing["conflict_count"] if preserve_existing_refs else incoming["conflict_count"]),
                    "notes": existing["notes"] if preserve_existing_refs else incoming["notes"] or existing["notes"],
                    "last_updated_round": int(existing["last_updated_round"] if preserve_existing_refs else incoming["last_updated_round"]),
                    "updated_at": incoming["updated_at"],
                    "metadata_json": existing["metadata_json"] if preserve_existing_refs else incoming["metadata_json"] or existing["metadata_json"],
                }
            )
            return merged

        return self._upsert_rows(
            table="session_section_packs",
            id_field="pack_id",
            unique_lookup_field="section_id",
            rows=rows,
            merge_fn=merge,
        )

    def upsert_claim_fact_links(self, rows: List[Tuple[str, str, str]]) -> None:
        self._upsert_link_rows("session_claim_fact_links", ("research_id", "claim_id", "fact_id"), rows)

    def upsert_claim_evidence_links(self, rows: List[Tuple[str, str, str]]) -> None:
        self._upsert_link_rows("session_claim_evidence_links", ("research_id", "claim_id", "evidence_id"), rows)

    def upsert_pack_links(
        self,
        *,
        fact_rows: List[Tuple[str, str, str]],
        claim_rows: List[Tuple[str, str, str]],
        evidence_rows: List[Tuple[str, str, str]],
        conflict_rows: List[Tuple[str, str, str]],
        replace_pack_ids: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        if replace_pack_ids:
            for table in (
                "session_pack_fact_links",
                "session_pack_claim_links",
                "session_pack_evidence_links",
                "session_pack_conflict_links",
            ):
                for research_id, pack_id in replace_pack_ids:
                    self.conn.execute(
                        f"DELETE FROM {table} WHERE research_id = ? AND pack_id = ?",
                        (research_id, pack_id),
                    )
        self._upsert_link_rows("session_pack_fact_links", ("research_id", "pack_id", "fact_id"), fact_rows)
        self._upsert_link_rows("session_pack_claim_links", ("research_id", "pack_id", "claim_id"), claim_rows)
        self._upsert_link_rows("session_pack_evidence_links", ("research_id", "pack_id", "evidence_id"), evidence_rows)
        self._upsert_link_rows("session_pack_conflict_links", ("research_id", "pack_id", "conflict_id"), conflict_rows)
        self._commit()

    def append_coverage_snapshot(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO session_coverage_snapshots (
                snapshot_id, research_id, round_no, avg_section_coverage, evidence_density,
                conflict_pressure, sufficiency_level, completed_section_count,
                partial_section_count, uncovered_section_count, created_at, raw_summary_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["snapshot_id"],
                row["research_id"],
                row["round_no"],
                row["avg_section_coverage"],
                row["evidence_density"],
                row["conflict_pressure"],
                row["sufficiency_level"],
                row["completed_section_count"],
                row["partial_section_count"],
                row["uncovered_section_count"],
                row["created_at"],
                row["raw_summary_json"],
            ),
        )
        self._commit()
        logger.info(
            "coverage snapshot appended research_id=%s round=%s sufficiency=%s",
            row["research_id"],
            row["round_no"],
            row["sufficiency_level"],
        )
        return self._fetchone_dict(
            "SELECT * FROM session_coverage_snapshots WHERE snapshot_id = ?",
            (row["snapshot_id"],),
        ) or row

    def upsert_unresolved_gaps(self, rows: List[Dict[str, Any]], close_missing: bool = True) -> Dict[str, Any]:
        inserted = 0
        updated = 0
        open_keys = {str(row["gap_text"]).strip().lower() for row in rows}
        for row in rows:
            existing = self._fetchone_dict(
                "SELECT * FROM session_unresolved_gaps WHERE research_id = ? AND gap_text = ?",
                (row["research_id"], row["gap_text"]),
            )
            if existing:
                self.conn.execute(
                    """
                    UPDATE session_unresolved_gaps
                    SET round_no = ?, task_id = ?, section_id = ?, gap_type = ?, severity = ?, status = ?,
                        updated_at = ?, metadata_json = ?
                    WHERE gap_id = ? AND research_id = ?
                    """,
                    (
                        row["round_no"],
                        row["task_id"],
                        row["section_id"],
                        row["gap_type"],
                        row["severity"],
                        row["status"],
                        row["updated_at"],
                        row["metadata_json"],
                        existing["gap_id"],
                        row["research_id"],
                    ),
                )
                updated += 1
            else:
                self.conn.execute(
                    """
                    INSERT INTO session_unresolved_gaps (
                        gap_id, research_id, round_no, task_id, section_id, gap_text, gap_type,
                        severity, status, created_at, updated_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["gap_id"],
                        row["research_id"],
                        row["round_no"],
                        row["task_id"],
                        row["section_id"],
                        row["gap_text"],
                        row["gap_type"],
                        row["severity"],
                        row["status"],
                        row["created_at"],
                        row["updated_at"],
                        row["metadata_json"],
                    ),
                )
                inserted += 1
        if rows and close_missing:
            research_id = rows[0]["research_id"]
            existing_rows = self._fetchall_dicts(
                "SELECT gap_id, gap_text FROM session_unresolved_gaps WHERE research_id = ? AND status = 'open'",
                (research_id,),
            )
            for existing in existing_rows:
                if str(existing["gap_text"]).strip().lower() not in open_keys:
                    self.conn.execute(
                        "UPDATE session_unresolved_gaps SET status = 'resolved', updated_at = ? WHERE gap_id = ?",
                        (self._now(), existing["gap_id"]),
                    )
        self._commit()
        if rows:
            logger.info(
                "unresolved gaps upserted research_id=%s inserted=%s updated=%s",
                rows[0]["research_id"],
                inserted,
                updated,
            )
        return {"inserted": inserted, "updated": updated}

    def append_novelty_snapshot(self, row: Dict[str, Any]) -> Dict[str, Any]:
        self.conn.execute(
            """
            INSERT INTO session_novelty_snapshots (
                snapshot_id, research_id, round_no, new_fact_count, merged_fact_count,
                new_source_count, new_claim_count, new_evidence_count, novelty_ratio,
                novelty_level, created_at, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["snapshot_id"],
                row["research_id"],
                row["round_no"],
                row["new_fact_count"],
                row["merged_fact_count"],
                row["new_source_count"],
                row["new_claim_count"],
                row["new_evidence_count"],
                row["novelty_ratio"],
                row["novelty_level"],
                row["created_at"],
                row["metadata_json"],
            ),
        )
        self._commit()
        return self._fetchone_dict(
            "SELECT * FROM session_novelty_snapshots WHERE snapshot_id = ?",
            (row["snapshot_id"],),
        ) or row

    def _inflate_json_fields(self, rows: List[Dict[str, Any]], *fields: str) -> List[Dict[str, Any]]:
        inflated: List[Dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            for field_name in fields:
                if field_name in item:
                    item[field_name] = self._load(item.get(field_name))
            inflated.append(item)
        return inflated

    def build_session_snapshot(self, research_id: str) -> Dict[str, Any]:
        session = self.get_session(research_id)
        if session is None:
            return SessionSnapshot().model_dump()

        sources = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_sources WHERE research_id = ? ORDER BY first_seen_round, source_id",
                (research_id,),
            ),
            "metadata_json",
        )
        claims = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_claims WHERE research_id = ? ORDER BY first_seen_round, claim_id",
                (research_id,),
            ),
            "metadata_json",
        )
        facts = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_facts WHERE research_id = ? ORDER BY first_seen_round, fact_id",
                (research_id,),
            ),
            "metadata_json",
        )
        evidence = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_evidence WHERE research_id = ? ORDER BY first_seen_round, evidence_id",
                (research_id,),
            ),
            "metadata_json",
        )
        conflicts = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_conflicts WHERE research_id = ? ORDER BY first_seen_round, conflict_id",
                (research_id,),
            ),
            "metadata_json",
        )
        packs = self._inflate_json_fields(
            self._fetchall_dicts(
                "SELECT * FROM session_section_packs WHERE research_id = ? ORDER BY section_id",
                (research_id,),
            ),
            "metadata_json",
        )
        latest_coverage = self._fetchone_dict(
            "SELECT * FROM session_coverage_snapshots WHERE research_id = ? ORDER BY round_no DESC, created_at DESC LIMIT 1",
            (research_id,),
        )
        if latest_coverage:
            latest_coverage["raw_summary_json"] = self._load(latest_coverage["raw_summary_json"])
        latest_novelty = self._fetchone_dict(
            "SELECT * FROM session_novelty_snapshots WHERE research_id = ? ORDER BY round_no DESC, created_at DESC LIMIT 1",
            (research_id,),
        )
        if latest_novelty:
            latest_novelty["metadata_json"] = self._load(latest_novelty["metadata_json"])
        open_gaps = self._inflate_json_fields(
            self._fetchall_dicts(
                """
                SELECT * FROM session_unresolved_gaps
                WHERE research_id = ? AND status = 'open'
                ORDER BY round_no DESC, updated_at DESC
                """,
                (research_id,),
            ),
            "metadata_json",
        )
        refs = KnowledgeRefs(
            collection_name=f"research_{research_id}",
            fact_ids=[row["fact_id"] for row in facts],
            claim_ids=[row["claim_id"] for row in claims],
            evidence_ids=[row["evidence_id"] for row in evidence],
            conflict_ids=[row["conflict_id"] for row in conflicts],
            section_pack_ids=[row["pack_id"] for row in packs],
            source_ids=[row["source_id"] for row in sources],
        ).model_dump()
        snapshot = SessionSnapshot(
            session=session,
            knowledge_refs=refs,
            sources=sources,
            claims=claims,
            facts=facts,
            evidence=evidence,
            conflicts=conflicts,
            section_evidence_packs=packs,
            latest_coverage_snapshot=latest_coverage,
            open_gaps=open_gaps,
            latest_novelty_snapshot=latest_novelty,
            stats={
                "source_count": len(sources),
                "fact_count": len(facts),
                "claim_count": len(claims),
                "evidence_count": len(evidence),
                "conflict_count": len(conflicts),
                "section_pack_count": len(packs),
            },
        ).model_dump()
        snapshot["source_registry"] = {row["source_id"]: row for row in sources}
        logger.info(
            "session snapshot built research_id=%s round=%s facts=%s evidence=%s packs=%s gaps=%s",
            research_id,
            session.current_round,
            len(facts),
            len(evidence),
            len(packs),
            len(open_gaps),
        )
        return snapshot

    def get_session_snapshot(self, research_id: str) -> Dict[str, Any]:
        return self.build_session_snapshot(research_id)


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

    def _apply_pack_section_ids(self, payload: Dict[str, Any]) -> None:
        indexes = {
            "claim_ids": {
                str(item.get("id", "")).strip(): item
                for item in payload.get("claims", []) or []
                if str(item.get("id", "")).strip()
            },
            "fact_ids": {
                str(item.get("id", "")).strip(): item
                for item in payload.get("atomic_facts", []) or []
                if str(item.get("id", "")).strip()
            },
            "evidence_ids": {
                str(item.get("id", "")).strip(): item
                for item in payload.get("evidence", []) or []
                if str(item.get("id", "")).strip()
            },
            "conflict_ids": {
                str(item.get("id", "")).strip(): item
                for item in payload.get("conflicts", []) or []
                if str(item.get("id", "")).strip()
            },
        }
        for pack in payload.get("section_evidence_packs", []) or []:
            section_id = str(pack.get("section_id", "")).strip()
            if not section_id:
                continue
            for id_field, item_index in indexes.items():
                for item_id in pack.get(id_field, []) or []:
                    item = item_index.get(str(item_id).strip())
                    if item is not None and not str(item.get("section_id", "")).strip():
                        item["section_id"] = section_id

    def _has_current_round_support(self, payload: Dict[str, Any]) -> bool:
        if any(payload.get(field) for field in ("sources", "atomic_facts", "claims", "evidence", "conflicts")):
            return True
        for pack in payload.get("section_evidence_packs", []) or []:
            if any(pack.get(field) for field in ("claim_ids", "fact_ids", "evidence_ids", "conflict_ids")):
                return True
        return False

    def create_or_get_session(
        self,
        research_id: str,
        root_query: str,
        session_id: Optional[str] = None,
        current_active_task_id: Optional[str] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> ResearchSessionRecord:
        return self.store.create_or_get_session(
            research_id=research_id,
            root_query=root_query,
            session_id=session_id,
            current_active_task_id=current_active_task_id,
            metadata_json=metadata_json,
        )

    def get_session_snapshot(self, research_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self._build_legacy_snapshot(self.store.get_session_snapshot(research_id))

    def reload_session(self, research_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        return self.get_session_snapshot(research_id, session_id)

    def clear_session(self, research_id: str, session_id: Optional[str] = None) -> None:
        for table in (
            "session_claim_fact_links",
            "session_claim_evidence_links",
            "session_pack_fact_links",
            "session_pack_claim_links",
            "session_pack_evidence_links",
            "session_pack_conflict_links",
            "session_sources",
            "session_claims",
            "session_facts",
            "session_evidence",
            "session_conflicts",
            "session_section_packs",
            "session_coverage_snapshots",
            "session_unresolved_gaps",
            "session_novelty_snapshots",
            "research_sessions",
        ):
            self.store.conn.execute(f"DELETE FROM {table} WHERE research_id = ?", (research_id,))
        self.store._commit()

    def _round_no(self, research_id: str) -> int:
        session = self.store.get_session(research_id)
        return (session.current_round + 1) if session else 1

    def _section_title_lookup(self, payload: Dict[str, Any]) -> Dict[str, str]:
        titles: Dict[str, str] = {}
        for section in (payload.get("report_outline", {}) or {}).get("sections", []) or []:
            section_id = str(section.get("section_id", "")).strip()
            if section_id:
                titles[section_id] = str(section.get("title", "")).strip()
        return titles

    def _section_goal_lookup(self, payload: Dict[str, Any]) -> Dict[str, str]:
        goals: Dict[str, str] = {}
        for goal in payload.get("section_goals", []) or []:
            section_id = str(goal.get("section_id", "")).strip()
            if section_id:
                goals[section_id] = str(goal.get("goal", "")).strip()
        return goals

    def _normalize_sources(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        now = self._now()
        for item in items:
            url = str(item.get("url", "")).strip()
            if not url:
                continue
            source_id = str(item.get("source_id") or f"src_{uuid.uuid5(uuid.NAMESPACE_URL, url).hex[:16]}")
            domain = urlparse(url).netloc.lower()
            rows.append(
                {
                    "source_id": source_id,
                    "research_id": research_id,
                    "url": url,
                    "title": str(item.get("title", "") or url),
                    "domain": domain,
                    "source_type": str(item.get("source_type") or "web"),
                    "authority_score": float(item.get("authority_score") or item.get("score") or 0.0),
                    "freshness_score": float(item.get("freshness_score") or 0.0),
                    "task_id": item.get("task_id") or task_id,
                    "first_seen_round": round_no,
                    "last_seen_round": round_no,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "is_active": 1,
                    "metadata_json": self.store._dump(
                        {
                            "query": item.get("query"),
                            "snippet": item.get("snippet"),
                            "status": item.get("status"),
                            "text_length": item.get("text_length"),
                            "extraction_method": item.get("extraction_method"),
                        }
                    ),
                }
            )
        return rows

    def _normalize_claims(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        now = self._now()
        rows: List[Dict[str, Any]] = []
        for item in items:
            text = self._normalize_text(item.get("text"))
            if not text:
                continue
            rows.append(
                {
                    "claim_id": str(item.get("id") or uuid.uuid4()),
                    "research_id": research_id,
                    "task_id": item.get("task_id") or task_id,
                    "section_id": str(item.get("section_id", "")),
                    "canonical_text": text,
                    "raw_text": str(item.get("text", "")),
                    "confidence": float(item.get("confidence") or 0.0),
                    "status": str(item.get("status") or "active"),
                    "source_count": len(self._unique_ids(item.get("source_ids", []))),
                    "evidence_count": len(self._unique_ids(item.get("evidence_ids", []))),
                    "fact_count": len(self._unique_ids(item.get("fact_ids", []))),
                    "first_seen_round": round_no,
                    "last_seen_round": round_no,
                    "created_at": now,
                    "updated_at": now,
                    "dedup_key": self._normalize_key(text),
                    "metadata_json": self.store._dump(
                        {
                            "fact_ids": item.get("fact_ids", []),
                            "evidence_ids": item.get("evidence_ids", []),
                            "created_at": item.get("created_at"),
                        }
                    ),
                }
            )
        return rows

    def _normalize_facts(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        now = self._now()
        rows: List[Dict[str, Any]] = []
        for item in items:
            text = self._normalize_text(item.get("text"))
            if not text:
                continue
            source_id = str(item.get("source_id") or item.get("source_url") or "")
            rows.append(
                {
                    "fact_id": str(item.get("id") or uuid.uuid4()),
                    "research_id": research_id,
                    "task_id": item.get("task_id") or task_id,
                    "section_id": str(item.get("section_id", "")),
                    "canonical_text": text,
                    "raw_text": str(item.get("text", "")),
                    "snippet": str(item.get("snippet", "")),
                    "confidence": float(item.get("confidence") or 0.0),
                    "verified_count": int(item.get("verified_count") or 0),
                    "source_count": len(self._unique_ids([source_id] + list(item.get("source_ids", [])))),
                    "status": str(
                        item.get("status")
                        or (FactStatus.CONFLICTING.value if item.get("is_conflict") else FactStatus.ACTIVE.value)
                    ),
                    "dedup_key": self._normalize_key(text),
                    "first_seen_round": round_no,
                    "last_seen_round": round_no,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": self.store._dump(
                        {
                            "source_url": item.get("source_url"),
                            "source_level": item.get("source_level"),
                            "conflict_with": item.get("conflict_with", []),
                            "confidence_reason": item.get("confidence_reason"),
                        }
                    ),
                }
            )
        return rows

    def _normalize_evidence(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        now = self._now()
        rows: List[Dict[str, Any]] = []
        for item in items:
            quote = str(item.get("quote") or item.get("summary") or "").strip()
            source_id = str(item.get("source_id") or item.get("source_url") or "")
            dedupe_key = self._normalize_key(f"{source_id} {quote}")
            rows.append(
                {
                    "evidence_id": str(item.get("id") or uuid.uuid4()),
                    "research_id": research_id,
                    "task_id": item.get("task_id") or task_id,
                    "section_id": str(item.get("section_id", "")),
                    "source_id": source_id,
                    "quote_text": str(item.get("quote", "")),
                    "summary_text": str(item.get("summary", "")),
                    "quality_score": float(item.get("quality_score") or 0.0),
                    "confidence": float(item.get("confidence") or item.get("quality_score") or 0.0),
                    "status": str(item.get("status") or "active"),
                    "dedup_key": dedupe_key,
                    "first_seen_round": round_no,
                    "last_seen_round": round_no,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": self.store._dump(
                        {
                            "fact_ids": item.get("fact_ids", []),
                            "claim_ids": item.get("claim_ids", []),
                            "source_url": item.get("source_url"),
                            "created_at": item.get("created_at"),
                        }
                    ),
                }
            )
        return rows

    def _normalize_conflicts(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        now = self._now()
        rows: List[Dict[str, Any]] = []
        for item in items:
            fact_ids = self._unique_ids(item.get("fact_ids", []))
            description = str(item.get("description", "")).strip()
            dedupe_key = self._normalize_key(f"{','.join(sorted(fact_ids))} {description}")
            rows.append(
                {
                    "conflict_id": str(item.get("id") or uuid.uuid4()),
                    "research_id": research_id,
                    "task_id": item.get("task_id") or task_id,
                    "section_id": str(item.get("section_id", "")),
                    "conflict_type": str(item.get("conflict_type") or "semantic_conflict"),
                    "description": description,
                    "severity": str(item.get("severity") or "medium"),
                    "status": str(item.get("status") or "active"),
                    "claim_count": len(self._unique_ids(item.get("claim_ids", []))),
                    "evidence_count": len(self._unique_ids(item.get("evidence_ids", []))),
                    "first_seen_round": round_no,
                    "last_seen_round": round_no,
                    "created_at": now,
                    "updated_at": now,
                    "dedup_key": dedupe_key,
                    "metadata_json": self.store._dump(
                        {
                            "fact_ids": fact_ids,
                            "claim_ids": item.get("claim_ids", []),
                            "evidence_ids": item.get("evidence_ids", []),
                            "created_at": item.get("created_at"),
                        }
                    ),
                }
            )
        return rows

    def _normalize_section_packs(
        self,
        items: List[Dict[str, Any]],
        *,
        research_id: str,
        round_no: int,
        section_titles: Dict[str, str],
        section_goals: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        now = self._now()
        rows: List[Dict[str, Any]] = []
        for item in items:
            section_id = str(item.get("section_id", "")).strip()
            rows.append(
                {
                    "pack_id": str(item.get("pack_id") or item.get("id") or uuid.uuid4()),
                    "research_id": research_id,
                    "section_id": section_id,
                    "section_title": str(item.get("section_title") or section_titles.get(section_id, "")),
                    "goal": str(item.get("goal") or section_goals.get(section_id, "")),
                    "coverage_score": float(item.get("coverage_score") or 0.0),
                    "status": str(item.get("status") or "active"),
                    "claim_count": len(self._unique_ids(item.get("claim_ids", []))),
                    "fact_count": len(self._unique_ids(item.get("fact_ids", []))),
                    "evidence_count": len(self._unique_ids(item.get("evidence_ids", []))),
                    "conflict_count": len(self._unique_ids(item.get("conflict_ids", []))),
                    "notes": str(item.get("notes") or ""),
                    "first_seen_round": round_no,
                    "last_updated_round": round_no,
                    "created_at": now,
                    "updated_at": now,
                    "metadata_json": self.store._dump(
                        {
                            "claim_ids": item.get("claim_ids", []),
                            "fact_ids": item.get("fact_ids", []),
                            "evidence_ids": item.get("evidence_ids", []),
                            "conflict_ids": item.get("conflict_ids", []),
                        }
                    ),
                }
            )
        return rows

    def _remap_ids(self, values: List[Any], id_map: Dict[str, str]) -> List[str]:
        return self._unique_ids([id_map.get(str(value), str(value)) for value in values or []])

    def _build_source_rows_from_payload(
        self,
        payload: Dict[str, Any],
        research_id: str,
        task_id: Optional[str],
        round_no: int,
    ) -> List[Dict[str, Any]]:
        sources = list(payload.get("sources", []) or [])
        if sources:
            return self._normalize_sources(sources, research_id=research_id, task_id=task_id, round_no=round_no)

        derived_sources: List[Dict[str, Any]] = []
        seen_urls: set[str] = set()
        for evidence in payload.get("evidence", []):
            url = str(evidence.get("source_url") or evidence.get("source_id") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            derived_sources.append({"url": url, "title": url, "source_id": evidence.get("source_id") or url})
        for fact in payload.get("atomic_facts", []):
            url = str(fact.get("source_url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            derived_sources.append({"url": url, "title": url, "source_id": fact.get("source_id") or url})
        return self._normalize_sources(derived_sources, research_id=research_id, task_id=task_id, round_no=round_no)

    def _link_map(self, table: str, id_column: str, research_id: str) -> Dict[str, List[str]]:
        rows = self.store.conn.execute(
            f"SELECT pack_id, {id_column} FROM {table} WHERE research_id = ? ORDER BY pack_id, {id_column}",
            (research_id,),
        ).fetchall()
        linked: Dict[str, List[str]] = {}
        for row in rows:
            linked.setdefault(str(row["pack_id"]), []).append(str(row[id_column]))
        return linked

    def _coerce_datetime(self, value: Any) -> Any:
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return value
        return value

    def _legacy_fact(self, row: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(row.get("metadata_json", {}) or {})
        return {
            "id": row["fact_id"],
            "fact_id": row["fact_id"],
            "text": row.get("canonical_text", ""),
            "source_url": meta.get("source_url", ""),
            "confidence": row.get("confidence", 0.0),
            "task_id": row.get("task_id"),
            "section_id": row.get("section_id", ""),
            "snippet": row.get("snippet", ""),
            "source_level": meta.get("source_level", SourceLevel.C.value),
            "verified_count": row.get("verified_count", 0),
            "is_conflict": row.get("status") == FactStatus.CONFLICTING.value,
            "conflict_with": meta.get("conflict_with", []),
            "confidence_reason": meta.get("confidence_reason"),
            "status": row.get("status", FactStatus.ACTIVE.value),
            "research_id": row.get("research_id"),
            "created_at": self._coerce_datetime(row.get("created_at")),
            "updated_at": self._coerce_datetime(row.get("updated_at")),
        }

    def _legacy_claim(self, row: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(row.get("metadata_json", {}) or {})
        return {
            "id": row["claim_id"],
            "claim_id": row["claim_id"],
            "text": row.get("canonical_text", ""),
            "fact_ids": self._unique_ids(meta.get("fact_ids", [])),
            "evidence_ids": self._unique_ids(meta.get("evidence_ids", [])),
            "confidence": row.get("confidence", 0.0),
            "task_id": row.get("task_id"),
            "section_id": row.get("section_id", ""),
            "created_at": self._coerce_datetime(meta.get("created_at") or row.get("created_at")),
            "status": row.get("status", "active"),
            "research_id": row.get("research_id"),
        }

    def _legacy_evidence(self, row: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(row.get("metadata_json", {}) or {})
        return {
            "id": row["evidence_id"],
            "evidence_id": row["evidence_id"],
            "source_id": row.get("source_id", ""),
            "source_url": meta.get("source_url", row.get("source_id", "")),
            "quote": row.get("quote_text", ""),
            "summary": row.get("summary_text", ""),
            "fact_ids": self._unique_ids(meta.get("fact_ids", [])),
            "claim_ids": self._unique_ids(meta.get("claim_ids", [])),
            "quality_score": row.get("quality_score", 0.0),
            "confidence": row.get("confidence", 0.0),
            "task_id": row.get("task_id"),
            "section_id": row.get("section_id", ""),
            "created_at": self._coerce_datetime(meta.get("created_at") or row.get("created_at")),
            "status": row.get("status", "active"),
            "research_id": row.get("research_id"),
        }

    def _legacy_conflict(self, row: Dict[str, Any]) -> Dict[str, Any]:
        meta = dict(row.get("metadata_json", {}) or {})
        return {
            "id": row["conflict_id"],
            "conflict_id": row["conflict_id"],
            "fact_ids": self._unique_ids(meta.get("fact_ids", [])),
            "claim_ids": self._unique_ids(meta.get("claim_ids", [])),
            "evidence_ids": self._unique_ids(meta.get("evidence_ids", [])),
            "description": row.get("description", ""),
            "severity": row.get("severity", "medium"),
            "task_id": row.get("task_id"),
            "section_id": row.get("section_id", ""),
            "created_at": self._coerce_datetime(meta.get("created_at") or row.get("created_at")),
            "status": row.get("status", "active"),
            "research_id": row.get("research_id"),
        }

    def _build_legacy_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        if not snapshot or not snapshot.get("session"):
            return {
                "session": None,
                "knowledge_refs": KnowledgeRefs().model_dump(),
                "sources": [],
                "claims": [],
                "facts": [],
                "evidence": [],
                "conflicts": [],
                "section_evidence_packs": [],
                "latest_coverage_snapshot": None,
                "open_gaps": [],
                "latest_novelty_snapshot": None,
                "stats": {},
                "source_registry": {},
            }

        research_id = snapshot["session"]["research_id"]
        pack_fact_links = self._link_map("session_pack_fact_links", "fact_id", research_id)
        pack_claim_links = self._link_map("session_pack_claim_links", "claim_id", research_id)
        pack_evidence_links = self._link_map("session_pack_evidence_links", "evidence_id", research_id)
        pack_conflict_links = self._link_map("session_pack_conflict_links", "conflict_id", research_id)

        claims = [self._legacy_claim(row) for row in snapshot.get("claims", [])]
        facts = [self._legacy_fact(row) for row in snapshot.get("facts", [])]
        evidence = [self._legacy_evidence(row) for row in snapshot.get("evidence", [])]
        conflicts = [self._legacy_conflict(row) for row in snapshot.get("conflicts", [])]

        packs: List[Dict[str, Any]] = []
        for row in snapshot.get("section_evidence_packs", []):
            meta = dict(row.get("metadata_json", {}) or {})
            pack_id = str(row["pack_id"])
            packs.append(
                {
                    "id": pack_id,
                    "pack_id": pack_id,
                    "section_id": row.get("section_id", ""),
                    "section_title": row.get("section_title", ""),
                    "goal": row.get("goal", ""),
                    "claim_ids": self._unique_ids(pack_claim_links.get(pack_id, meta.get("claim_ids", []))),
                    "fact_ids": self._unique_ids(pack_fact_links.get(pack_id, meta.get("fact_ids", []))),
                    "evidence_ids": self._unique_ids(pack_evidence_links.get(pack_id, meta.get("evidence_ids", []))),
                    "conflict_ids": self._unique_ids(pack_conflict_links.get(pack_id, meta.get("conflict_ids", []))),
                    "coverage_score": row.get("coverage_score", 0.0),
                    "notes": row.get("notes", ""),
                    "status": row.get("status", "active"),
                    "research_id": row.get("research_id"),
                    "created_at": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                }
            )

        knowledge_refs = dict(snapshot.get("knowledge_refs", {}) or {})
        knowledge_refs["fact_ids"] = [item["id"] for item in facts]
        knowledge_refs["claim_ids"] = [item["id"] for item in claims]
        knowledge_refs["evidence_ids"] = [item["id"] for item in evidence]
        knowledge_refs["conflict_ids"] = [item["id"] for item in conflicts]
        knowledge_refs["section_pack_ids"] = [item["id"] for item in packs]
        knowledge_refs["source_ids"] = self._unique_ids(
            list(knowledge_refs.get("source_ids", []))
            + [item.get("source_url", "") for item in facts if item.get("source_url")]
            + [item.get("source_url", "") for item in evidence if item.get("source_url")]
        )

        return {
            "session": snapshot.get("session"),
            "knowledge_refs": KnowledgeRefs(**knowledge_refs).model_dump(),
            "sources": snapshot.get("sources", []),
            "claims": claims,
            "facts": facts,
            "evidence": evidence,
            "conflicts": conflicts,
            "section_evidence_packs": packs,
            "latest_coverage_snapshot": snapshot.get("latest_coverage_snapshot"),
            "open_gaps": snapshot.get("open_gaps", []),
            "latest_novelty_snapshot": snapshot.get("latest_novelty_snapshot"),
            "stats": snapshot.get("stats", {}),
            "source_registry": snapshot.get("source_registry", {}),
        }

    def process_distiller_output(
        self,
        distiller_output: Any,
        research_id: str,
        session_id: str,
        task_id: Optional[str] = None,
    ) -> DistillerOutputs:
        with self.store.transaction():
            return self._process_distiller_output_in_transaction(
                distiller_output,
                research_id=research_id,
                session_id=session_id,
                task_id=task_id,
            )

    def _process_distiller_output_in_transaction(
        self,
        distiller_output: Any,
        research_id: str,
        session_id: str,
        task_id: Optional[str] = None,
    ) -> DistillerOutputs:
        payload = distiller_output.model_dump() if hasattr(distiller_output, "model_dump") else dict(distiller_output)
        payload["sources"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("sources", [])]
        payload["atomic_facts"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("atomic_facts", [])]
        payload["claims"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("claims", [])]
        payload["evidence"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("evidence", [])]
        payload["conflicts"] = [dict(item) if isinstance(item, dict) else item.model_dump() for item in payload.get("conflicts", [])]
        payload["section_evidence_packs"] = [
            dict(item) if isinstance(item, dict) else item.model_dump()
            for item in payload.get("section_evidence_packs", [])
        ]
        self._apply_pack_section_ids(payload)
        has_current_round_support = self._has_current_round_support(payload)

        session = self.create_or_get_session(
            research_id=research_id,
            root_query=str(payload.get("root_query") or research_id),
            session_id=session_id,
            current_active_task_id=task_id,
        )
        round_no = session.current_round + 1
        self.store.update_session_round(research_id, round_no=round_no, current_active_task_id=task_id, status="active")

        sources = self._build_source_rows_from_payload(payload, research_id, task_id, round_no)
        if sources:
            self.store.upsert_sources(sources)

        facts_result = self.store.upsert_facts(
            self._normalize_facts(payload.get("atomic_facts", []), research_id=research_id, task_id=task_id, round_no=round_no)
        )
        fact_id_map = facts_result["id_map"]
        for claim in payload.get("claims", []):
            claim["fact_ids"] = self._remap_ids(claim.get("fact_ids", []), fact_id_map)
        for evidence_row in payload.get("evidence", []):
            evidence_row["fact_ids"] = self._remap_ids(evidence_row.get("fact_ids", []), fact_id_map)
        for conflict in payload.get("conflicts", []):
            conflict["fact_ids"] = self._remap_ids(conflict.get("fact_ids", []), fact_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["fact_ids"] = self._remap_ids(pack.get("fact_ids", []), fact_id_map)

        claims_result = self.store.upsert_claims(
            self._normalize_claims(payload.get("claims", []), research_id=research_id, task_id=task_id, round_no=round_no)
        )
        claim_id_map = claims_result["id_map"]
        for evidence_row in payload.get("evidence", []):
            evidence_row["claim_ids"] = self._remap_ids(evidence_row.get("claim_ids", []), claim_id_map)
        for conflict in payload.get("conflicts", []):
            conflict["claim_ids"] = self._remap_ids(conflict.get("claim_ids", []), claim_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["claim_ids"] = self._remap_ids(pack.get("claim_ids", []), claim_id_map)

        evidence_result = self.store.upsert_evidence(
            self._normalize_evidence(payload.get("evidence", []), research_id=research_id, task_id=task_id, round_no=round_no)
        )
        evidence_id_map = evidence_result["id_map"]
        for conflict in payload.get("conflicts", []):
            conflict["evidence_ids"] = self._remap_ids(conflict.get("evidence_ids", []), evidence_id_map)
        for pack in payload.get("section_evidence_packs", []):
            pack["evidence_ids"] = self._remap_ids(pack.get("evidence_ids", []), evidence_id_map)

        conflicts_result = self.store.upsert_conflicts(
            self._normalize_conflicts(payload.get("conflicts", []), research_id=research_id, task_id=task_id, round_no=round_no)
        )
        conflict_id_map = conflicts_result["id_map"]
        for pack in payload.get("section_evidence_packs", []):
            pack["conflict_ids"] = self._remap_ids(pack.get("conflict_ids", []), conflict_id_map)

        section_titles = self._section_title_lookup(payload)
        section_goals = self._section_goal_lookup(payload)
        packs_result = self.store.upsert_section_packs(
            self._normalize_section_packs(
                payload.get("section_evidence_packs", []),
                research_id=research_id,
                round_no=round_no,
                section_titles=section_titles,
                section_goals=section_goals,
            )
        )
        pack_id_map = packs_result["id_map"]
        replace_pack_ids = [
            (
                research_id,
                pack_id_map.get(str(pack.get("pack_id") or pack.get("id")), str(pack.get("pack_id") or pack.get("id"))),
            )
            for pack in payload.get("section_evidence_packs", [])
            if any(pack.get(field) for field in ("fact_ids", "claim_ids", "evidence_ids", "conflict_ids"))
        ]

        claim_fact_rows = []
        for claim in payload.get("claims", []):
            claim_id = claim_id_map.get(str(claim.get("id")), str(claim.get("id")))
            for fact_id in claim.get("fact_ids", []):
                claim_fact_rows.append((research_id, claim_id, fact_id))

        claim_evidence_rows = []
        for evidence_item in payload.get("evidence", []):
            evidence_id = evidence_id_map.get(str(evidence_item.get("id")), str(evidence_item.get("id")))
            for claim_id in evidence_item.get("claim_ids", []):
                claim_evidence_rows.append((research_id, claim_id, evidence_id))

        self.store.upsert_claim_fact_links(claim_fact_rows)
        self.store.upsert_claim_evidence_links(claim_evidence_rows)
        self.store.upsert_pack_links(
            fact_rows=[
                (research_id, pack_id_map.get(str(pack.get("pack_id") or pack.get("id")), str(pack.get("pack_id") or pack.get("id"))), fact_id)
                for pack in payload.get("section_evidence_packs", [])
                for fact_id in pack.get("fact_ids", [])
            ],
            claim_rows=[
                (research_id, pack_id_map.get(str(pack.get("pack_id") or pack.get("id")), str(pack.get("pack_id") or pack.get("id"))), claim_id)
                for pack in payload.get("section_evidence_packs", [])
                for claim_id in pack.get("claim_ids", [])
            ],
            evidence_rows=[
                (research_id, pack_id_map.get(str(pack.get("pack_id") or pack.get("id")), str(pack.get("pack_id") or pack.get("id"))), evidence_id)
                for pack in payload.get("section_evidence_packs", [])
                for evidence_id in pack.get("evidence_ids", [])
            ],
            conflict_rows=[
                (research_id, pack_id_map.get(str(pack.get("pack_id") or pack.get("id")), str(pack.get("pack_id") or pack.get("id"))), conflict_id)
                for pack in payload.get("section_evidence_packs", [])
                for conflict_id in pack.get("conflict_ids", [])
            ],
            replace_pack_ids=replace_pack_ids,
        )

        coverage_summary = dict(payload.get("coverage_summary", {}) or {})
        if coverage_summary and has_current_round_support:
            covered = coverage_summary.get("covered_sections", []) or []
            uncovered = coverage_summary.get("uncovered_sections", []) or []
            all_sections = covered + uncovered
            self.store.append_coverage_snapshot(
                {
                    "snapshot_id": f"coverage-{research_id}-{round_no}",
                    "research_id": research_id,
                    "round_no": round_no,
                    "avg_section_coverage": float(coverage_summary.get("avg_section_coverage") or 0.0),
                    "evidence_density": float(coverage_summary.get("evidence_density") or 0.0),
                    "conflict_pressure": float(coverage_summary.get("conflict_pressure") or 0.0),
                    "sufficiency_level": str(coverage_summary.get("sufficiency_level") or "insufficient"),
                    "completed_section_count": len(covered),
                    "partial_section_count": max(0, len(all_sections) - len(covered) - len(uncovered)),
                    "uncovered_section_count": len(uncovered),
                    "created_at": self._now(),
                    "raw_summary_json": self.store._dump(coverage_summary),
                }
            )

        self.store.append_novelty_snapshot(
            {
                "snapshot_id": f"novelty-{research_id}-{round_no}",
                "research_id": research_id,
                "round_no": round_no,
                "new_fact_count": int(facts_result["inserted"]),
                "merged_fact_count": int(facts_result["updated"]),
                "new_source_count": int(len(sources)),
                "new_claim_count": int(claims_result["inserted"]),
                "new_evidence_count": int(evidence_result["inserted"]),
                "novelty_ratio": float(facts_result["inserted"]) / max(1, len(payload.get("atomic_facts", []))),
                "novelty_level": "medium" if facts_result["inserted"] else "low",
                "created_at": self._now(),
                "metadata_json": self.store._dump({"task_id": task_id}),
            }
        )

        gaps = payload.get("unresolved_gaps", []) or []
        gap_rows = [
            {
                "gap_id": f"gap-{uuid.uuid5(uuid.NAMESPACE_DNS, f'{research_id}:{gap}').hex[:16]}",
                "research_id": research_id,
                "round_no": round_no,
                "task_id": task_id,
                "section_id": "",
                "gap_text": str(gap),
                "gap_type": "coverage_gap",
                "severity": "medium",
                "status": "open",
                "created_at": self._now(),
                "updated_at": self._now(),
                "metadata_json": self.store._dump({"task_id": task_id}),
            }
            for gap in gaps
            if str(gap).strip()
        ]
        if gap_rows and has_current_round_support:
            self.store.upsert_unresolved_gaps(gap_rows, close_missing=True)

        snapshot = self.get_session_snapshot(research_id, session_id)
        payload["atomic_facts"] = snapshot["facts"]
        payload["claims"] = snapshot["claims"]
        payload["evidence"] = snapshot["evidence"]
        payload["conflicts"] = snapshot["conflicts"]
        payload["section_evidence_packs"] = snapshot["section_evidence_packs"]
        payload["knowledge_refs"] = snapshot["knowledge_refs"]
        payload["fact_ids"] = snapshot["knowledge_refs"]["fact_ids"]
        payload["claim_ids"] = snapshot["knowledge_refs"]["claim_ids"]
        payload["evidence_ids"] = snapshot["knowledge_refs"]["evidence_ids"]
        payload["conflict_ids"] = snapshot["knowledge_refs"]["conflict_ids"]
        return DistillerOutputs(**payload)

    async def _find_similar(
        self,
        collection_name: str,
        text: str,
        threshold: float = 0.7,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        research_id = collection_name.replace("research_", "", 1)
        snapshot = self.get_session_snapshot(research_id, collection_name)
        query_tokens = set(re.findall(r"[\w\u4e00-\u9fff]+", self._normalize_text(text).lower()))
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        for row in snapshot.get("facts", []):
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
        facts: List[Any],
        collection_name: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Tuple[List[str], List[str]]:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        result = self.process_distiller_output(
            DistillerOutputs(atomic_facts=facts),
            research_id=research_id,
            session_id=collection_name or "legacy_default",
            task_id=task_id,
        )
        return self._unique_ids(result.fact_ids), []

    async def upsert_fact_with_verification(
        self,
        fact: Any,
        collection_name: str,
    ) -> UpsertFactResult:
        similar = await self._find_similar(collection_name, getattr(fact, "text", ""), threshold=0.6)
        result = self.process_distiller_output(
            DistillerOutputs(atomic_facts=[fact]),
            research_id=collection_name.replace("research_", "", 1),
            session_id=collection_name,
            task_id=getattr(fact, "task_id", None),
        )
        fact_id = result.atomic_facts[0].id if result.atomic_facts else getattr(fact, "id", "")
        if similar:
            return UpsertFactResult(action="MUTUAL_VERIFICATION", fact_id=fact_id, confidence_change=0.0)
        return UpsertFactResult(action="NEW", fact_id=fact_id)

    async def search_facts(
        self,
        query: str,
        collection_name: Optional[str] = None,
        task_id_filter: Optional[str] = None,
        status_filter: Optional[FactStatus] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        snapshot = self.get_session_snapshot(research_id, collection_name)
        query_key = self._normalize_key(query)
        results: List[Dict[str, Any]] = []
        for row in snapshot.get("facts", []):
            if task_id_filter and row.get("task_id") != task_id_filter:
                continue
            if status_filter and row.get("status") != status_filter.value:
                continue
            text_key = self._normalize_key(row.get("text", ""))
            score = 1.0 if not query_key else float(query_key in text_key or text_key in query_key)
            if query_key and score <= 0.0:
                continue
            results.append({**row, "score": score})
        return results[:limit]

    async def get_fact_by_id(self, fact_id: str, collection_name: Optional[str] = None) -> Optional[Dict[str, Any]]:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        snapshot = self.get_session_snapshot(research_id, collection_name)
        return next((item for item in snapshot.get("facts", []) if item["id"] == fact_id), None)

    async def get_conflicts(self, collection_name: Optional[str] = None) -> List[Dict[str, Any]]:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        return self.get_session_snapshot(research_id, collection_name).get("conflicts", [])

    async def clear_collection(self, collection_name: Optional[str] = None) -> None:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        self.clear_session(research_id, collection_name)

    def get_stats(self, collection_name: Optional[str] = None) -> KnowledgeStats:
        research_id = (collection_name or "legacy_default").replace("research_", "", 1)
        snapshot = self.get_session_snapshot(research_id, collection_name)
        facts = snapshot.get("facts", [])
        conflicts = snapshot.get("conflicts", [])
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
