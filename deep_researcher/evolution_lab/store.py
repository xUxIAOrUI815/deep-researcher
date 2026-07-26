from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.contracts import canonical_contract_json, utc_now

from .models import (
    BestSkillSnapshot,
    CandidatePoolEntry,
    CandidatePoolPage,
    CandidatePoolReview,
    CandidateStatus,
    CrossTaskExperience,
    EvolutionCampaignPage,
    EvolutionCampaignRecord,
    EvolutionCampaignRequest,
    EvolutionCampaignStatus,
    EvolutionCandidate,
    EvolutionCandidateRecord,
    EvolutionHumanDecision,
    EvolutionInputSnapshot,
    EvolutionJournalEntry,
    EvolutionJournalKind,
    EvolutionPatch,
    GenerationAttempt,
    GenerationAttemptStatus,
    HumanGateOutcome,
    OptimizationTarget,
    PoolEntryStatus,
    RejectedEditMemory,
    ReviewedPoolEntry,
)


class EvolutionStoreError(RuntimeError):
    pass


class EvolutionConflict(EvolutionStoreError):
    pass


class EvolutionCorruption(EvolutionStoreError):
    pass


class SQLiteEvolutionStore:
    """Append-only offline-evolution journal with rebuildable projections."""

    CURRENT_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path),
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        if str(path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _json(value: Any) -> str:
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json")
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def _projection_checksum(
        cls,
        campaign_id: str,
        status: EvolutionCampaignStatus,
        revision: int,
        recovery_count: int,
    ) -> str:
        return cls._checksum(
            cls._json(
                {
                    "campaign_id": campaign_id,
                    "status": status.value,
                    "revision": revision,
                    "recovery_count": recovery_count,
                }
            )
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()
        return f"{prefix}_{digest[:32]}"

    def _migrate(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS evolution_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO evolution_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS evolution_pool_entries(
                    pool_entry_id TEXT PRIMARY KEY,
                    submitted_at TEXT NOT NULL,
                    source_kind TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evolution_pool_catalog
                ON evolution_pool_entries(
                    submitted_at, pool_entry_id, source_kind
                );

                CREATE TABLE IF NOT EXISTS evolution_pool_reviews(
                    review_id TEXT PRIMARY KEY,
                    pool_entry_id TEXT NOT NULL UNIQUE,
                    reviewed_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(pool_entry_id)
                      REFERENCES evolution_pool_entries(pool_entry_id)
                      ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS evolution_experiences(
                    experience_id TEXT PRIMARY KEY,
                    target TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_evolution_experience_target
                ON evolution_experiences(target, created_at, experience_id);

                CREATE TABLE IF NOT EXISTS evolution_campaigns(
                    campaign_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    target TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evolution_journal(
                    journal_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    journal_event_id TEXT NOT NULL UNIQUE,
                    campaign_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(campaign_id, sequence),
                    FOREIGN KEY(campaign_id)
                      REFERENCES evolution_campaigns(campaign_id)
                      ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_evolution_journal_campaign
                ON evolution_journal(campaign_id, sequence);

                CREATE TABLE IF NOT EXISTS evolution_projection(
                    campaign_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    recovery_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(campaign_id)
                      REFERENCES evolution_campaigns(campaign_id)
                      ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS evolution_rejections(
                    rejection_id TEXT PRIMARY KEY,
                    campaign_id TEXT NOT NULL,
                    target TEXT NOT NULL,
                    base_version_id TEXT NOT NULL,
                    rejected_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(campaign_id)
                      REFERENCES evolution_campaigns(campaign_id)
                      ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_evolution_rejection_memory
                ON evolution_rejections(
                    target, base_version_id, rejected_at, rejection_id
                );

                CREATE TABLE IF NOT EXISTS evolution_best_skills(
                    best_skill_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    best_skill_id TEXT NOT NULL UNIQUE,
                    target TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evolution_best_skill_projection(
                    target TEXT NOT NULL,
                    component_name TEXT NOT NULL,
                    best_skill_id TEXT NOT NULL UNIQUE,
                    checksum TEXT NOT NULL,
                    PRIMARY KEY(target, component_name),
                    FOREIGN KEY(best_skill_id)
                      REFERENCES evolution_best_skills(best_skill_id)
                      ON DELETE RESTRICT
                );

                CREATE TRIGGER IF NOT EXISTS evolution_pool_no_update
                BEFORE UPDATE ON evolution_pool_entries
                BEGIN SELECT RAISE(ABORT, 'pool entries are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_pool_no_delete
                BEFORE DELETE ON evolution_pool_entries
                BEGIN SELECT RAISE(ABORT, 'pool entries are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_pool_review_no_update
                BEFORE UPDATE ON evolution_pool_reviews
                BEGIN SELECT RAISE(ABORT, 'pool reviews are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_pool_review_no_delete
                BEFORE DELETE ON evolution_pool_reviews
                BEGIN SELECT RAISE(ABORT, 'pool reviews are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_experience_no_update
                BEFORE UPDATE ON evolution_experiences
                BEGIN SELECT RAISE(ABORT, 'experiences are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_experience_no_delete
                BEFORE DELETE ON evolution_experiences
                BEGIN SELECT RAISE(ABORT, 'experiences are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_campaign_no_update
                BEFORE UPDATE ON evolution_campaigns
                BEGIN SELECT RAISE(ABORT, 'campaigns are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_campaign_no_delete
                BEFORE DELETE ON evolution_campaigns
                BEGIN SELECT RAISE(ABORT, 'campaigns are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_journal_no_update
                BEFORE UPDATE ON evolution_journal
                BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_journal_no_delete
                BEFORE DELETE ON evolution_journal
                BEGIN SELECT RAISE(ABORT, 'journal is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_rejection_no_update
                BEFORE UPDATE ON evolution_rejections
                BEGIN SELECT RAISE(ABORT, 'rejections are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_rejection_no_delete
                BEFORE DELETE ON evolution_rejections
                BEGIN SELECT RAISE(ABORT, 'rejections are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_best_skill_no_update
                BEFORE UPDATE ON evolution_best_skills
                BEGIN SELECT RAISE(ABORT, 'best skills are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS evolution_best_skill_no_delete
                BEFORE DELETE ON evolution_best_skills
                BEGIN SELECT RAISE(ABORT, 'best skills are immutable'); END;
                """
            )
            row = self._connection.execute(
                "SELECT version FROM evolution_schema WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != self.CURRENT_VERSION:
                raise EvolutionCorruption(
                    "unsupported offline-evolution store schema"
                )

    def submit_pool_entry(
        self,
        entry: CandidatePoolEntry,
    ) -> ReviewedPoolEntry:
        payload = canonical_contract_json(entry)
        checksum = self._checksum(payload)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT payload_json, checksum FROM evolution_pool_entries "
                "WHERE pool_entry_id=?",
                (entry.pool_entry_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checksum"]) != checksum
                    or str(existing["payload_json"]) != payload
                ):
                    raise EvolutionConflict(
                        "pool entry ID was reused with different content"
                    )
                return self._pool_entry_locked(
                    connection,
                    entry.pool_entry_id,
                )
            connection.execute(
                """
                INSERT INTO evolution_pool_entries(
                    pool_entry_id, submitted_at, source_kind,
                    payload_json, checksum
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    entry.pool_entry_id,
                    entry.submitted_at.isoformat(),
                    entry.source_kind.value,
                    payload,
                    checksum,
                ),
            )
            return self._pool_entry_locked(
                connection,
                entry.pool_entry_id,
            )

    def review_pool_entry(
        self,
        review: CandidatePoolReview,
    ) -> ReviewedPoolEntry:
        payload = canonical_contract_json(review)
        checksum = self._checksum(payload)
        with self.transaction() as connection:
            self._pool_entry_locked(
                connection,
                review.pool_entry_id,
            )
            existing = connection.execute(
                "SELECT payload_json, checksum FROM evolution_pool_reviews "
                "WHERE pool_entry_id=?",
                (review.pool_entry_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checksum"]) != checksum
                    or str(existing["payload_json"]) != payload
                ):
                    raise EvolutionConflict(
                        "pool source already has another immutable review"
                    )
                return self._pool_entry_locked(
                    connection,
                    review.pool_entry_id,
                )
            connection.execute(
                """
                INSERT INTO evolution_pool_reviews(
                    review_id, pool_entry_id, reviewed_at,
                    payload_json, checksum
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    review.review_id,
                    review.pool_entry_id,
                    review.reviewed_at.isoformat(),
                    payload,
                    checksum,
                ),
            )
            return self._pool_entry_locked(
                connection,
                review.pool_entry_id,
            )

    def pool_entry(
        self,
        pool_entry_id: str,
    ) -> ReviewedPoolEntry | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM evolution_pool_entries "
                "WHERE pool_entry_id=?",
                (pool_entry_id,),
            ).fetchone()
            if row is None:
                return None
            return self._pool_entry_locked(
                self._connection,
                pool_entry_id,
            )

    def list_pool(
        self,
        *,
        statuses: tuple[PoolEntryStatus, ...] = (),
        cursor: str | None = None,
        limit: int = 100,
    ) -> CandidatePoolPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        after_time = None
        after_id = None
        if cursor is not None:
            try:
                decoded = json.loads(
                    base64.urlsafe_b64decode(
                        cursor + "=" * (-len(cursor) % 4)
                    ).decode()
                )
                after_time = str(decoded["submitted_at"])
                after_id = str(decoded["pool_entry_id"])
            except Exception as exc:
                raise ValueError("invalid candidate-pool cursor") from exc
        clauses = ["1=1"]
        params: list[Any] = []
        if after_time is not None and after_id is not None:
            clauses.append(
                "(submitted_at>? OR "
                "(submitted_at=? AND pool_entry_id>?))"
            )
            params.extend((after_time, after_time, after_id))
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT pool_entry_id, submitted_at
                FROM evolution_pool_entries
                WHERE {' AND '.join(clauses)}
                ORDER BY submitted_at, pool_entry_id
                """,
                params,
            ).fetchall()
            records = [
                self._pool_entry_locked(
                    self._connection,
                    str(row["pool_entry_id"]),
                )
                for row in rows
            ]
        if statuses:
            allowed = set(statuses)
            records = [item for item in records if item.status in allowed]
        selected = records[:limit]
        next_cursor = None
        if len(records) > limit and selected:
            final = selected[-1].entry
            payload = self._json(
                {
                    "submitted_at": final.submitted_at.isoformat(),
                    "pool_entry_id": final.pool_entry_id,
                }
            ).encode()
            next_cursor = base64.urlsafe_b64encode(payload).decode().rstrip(
                "="
            )
        return CandidatePoolPage(
            items=tuple(selected),
            next_cursor=next_cursor,
        )

    def save_experience(
        self,
        experience: CrossTaskExperience,
    ) -> CrossTaskExperience:
        self._save_immutable(
            table="evolution_experiences",
            id_column="experience_id",
            identifier=experience.experience_id,
            columns={
                "target": experience.target.value,
                "created_at": experience.created_at.isoformat(),
            },
            value=experience,
        )
        return experience

    def experience(
        self,
        experience_id: str,
    ) -> CrossTaskExperience | None:
        value = self._load_immutable(
            "evolution_experiences",
            "experience_id",
            experience_id,
        )
        return (
            CrossTaskExperience.model_validate(value, strict=False)
            if value is not None
            else None
        )

    def experiences(
        self,
        *,
        target: OptimizationTarget | None = None,
    ) -> tuple[CrossTaskExperience, ...]:
        query = "SELECT experience_id FROM evolution_experiences"
        params: tuple[Any, ...] = ()
        if target is not None:
            query += " WHERE target=?"
            params = (target.value,)
        query += " ORDER BY created_at, experience_id"
        with self._lock:
            rows = self._connection.execute(query, params).fetchall()
        return tuple(
            item
            for row in rows
            if (
                item := self.experience(str(row["experience_id"]))
            )
            is not None
        )

    def create_campaign(
        self,
        request: EvolutionCampaignRequest,
    ) -> EvolutionCampaignRecord:
        payload = canonical_contract_json(request)
        checksum = self._checksum(payload)
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT request_json, checksum FROM evolution_campaigns "
                "WHERE campaign_id=?",
                (request.campaign_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checksum"]) != checksum
                    or str(existing["request_json"]) != payload
                ):
                    raise EvolutionConflict(
                        "campaign ID was reused with different content"
                    )
                return self._record_locked(
                    connection,
                    request.campaign_id,
                )
            connection.execute(
                """
                INSERT INTO evolution_campaigns(
                    campaign_id, created_at, target,
                    request_json, checksum
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    request.campaign_id,
                    request.created_at.isoformat(),
                    request.target.value,
                    payload,
                    checksum,
                ),
            )
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=request.campaign_id,
                    sequence=1,
                    kind=EvolutionJournalKind.CREATED,
                    payload={
                        "initial_status": (
                            EvolutionCampaignStatus.DRAFT.value
                        )
                    },
                    occurred_at=request.created_at,
                ),
                EvolutionCampaignStatus.DRAFT,
                recovery_count=0,
            )
            return self._record_locked(
                connection,
                request.campaign_id,
            )

    def seal_inputs(
        self,
        snapshot: EvolutionInputSnapshot,
    ) -> EvolutionCampaignRecord:
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                snapshot.campaign_id,
            )
            if record.status == EvolutionCampaignStatus.READY:
                if record.input_snapshot != snapshot:
                    raise EvolutionConflict(
                        "campaign already has another sealed input snapshot"
                    )
                return record
            if record.status != EvolutionCampaignStatus.DRAFT:
                raise EvolutionConflict(
                    "only a draft campaign can seal inputs"
                )
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=snapshot.campaign_id,
                    sequence=record.revision + 1,
                    kind=EvolutionJournalKind.INPUTS_SEALED,
                    payload={
                        "snapshot": snapshot.model_dump(mode="json")
                    },
                    occurred_at=snapshot.sealed_at,
                ),
                EvolutionCampaignStatus.READY,
                recovery_count=record.recovery_count,
            )
            return self._record_locked(
                connection,
                snapshot.campaign_id,
            )

    def campaign(
        self,
        campaign_id: str,
    ) -> EvolutionCampaignRecord | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM evolution_campaigns WHERE campaign_id=?",
                (campaign_id,),
            ).fetchone()
            if row is None:
                return None
            return self._record_locked(
                self._connection,
                campaign_id,
            )

    def list_campaigns(
        self,
        *,
        statuses: tuple[EvolutionCampaignStatus, ...] = (),
        cursor: str | None = None,
        limit: int = 100,
    ) -> EvolutionCampaignPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses = ["1=1"]
        params: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"p.status IN ({placeholders})")
            params.extend(item.value for item in statuses)
        if cursor is not None:
            try:
                decoded = json.loads(
                    base64.urlsafe_b64decode(
                        cursor + "=" * (-len(cursor) % 4)
                    ).decode()
                )
                created_at = str(decoded["created_at"])
                campaign_id = str(decoded["campaign_id"])
            except Exception as exc:
                raise ValueError("invalid evolution campaign cursor") from exc
            clauses.append(
                "(c.created_at>? OR "
                "(c.created_at=? AND c.campaign_id>?))"
            )
            params.extend((created_at, created_at, campaign_id))
        params.append(limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT c.campaign_id, c.created_at
                FROM evolution_campaigns c
                JOIN evolution_projection p
                  ON p.campaign_id=c.campaign_id
                WHERE {' AND '.join(clauses)}
                ORDER BY c.created_at, c.campaign_id
                LIMIT ?
                """,
                params,
            ).fetchall()
            selected = rows[:limit]
            records = tuple(
                self._record_locked(
                    self._connection,
                    str(row["campaign_id"]),
                )
                for row in selected
            )
        next_cursor = None
        if len(rows) > limit and selected:
            value = self._json(
                {
                    "created_at": str(selected[-1]["created_at"]),
                    "campaign_id": str(selected[-1]["campaign_id"]),
                }
            ).encode()
            next_cursor = base64.urlsafe_b64encode(value).decode().rstrip(
                "="
            )
        return EvolutionCampaignPage(
            items=records,
            next_cursor=next_cursor,
        )

    def claim_generation(
        self,
        campaign_id: str,
        *,
        worker_id: str,
        at: datetime | None = None,
    ) -> GenerationAttempt:
        started_at = at or utc_now()
        with self.transaction() as connection:
            record = self._record_locked(connection, campaign_id)
            if record.status != EvolutionCampaignStatus.READY:
                raise EvolutionConflict(
                    "only a ready campaign can generate a candidate"
                )
            round_no = len(record.candidates) + 1
            if round_no > record.request.edit_budget.max_rounds:
                raise EvolutionConflict(
                    "campaign exhausted its candidate-round budget"
                )
            attempt_no = len(record.attempts) + 1
            attempt = GenerationAttempt(
                attempt_id=self._stable_id(
                    "evolution_generation",
                    campaign_id,
                    str(attempt_no),
                ),
                campaign_id=campaign_id,
                attempt_no=attempt_no,
                round_no=round_no,
                worker_id=worker_id,
                status=GenerationAttemptStatus.RUNNING,
                started_at=started_at,
            )
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=campaign_id,
                    sequence=record.revision + 1,
                    kind=EvolutionJournalKind.GENERATION_STARTED,
                    payload={
                        "attempt": attempt.model_dump(mode="json")
                    },
                    occurred_at=started_at,
                ),
                EvolutionCampaignStatus.GENERATING,
                recovery_count=record.recovery_count,
            )
            return attempt

    def finish_generation(
        self,
        *,
        attempt_id: str,
        patch: EvolutionPatch,
        candidate: EvolutionCandidate,
        at: datetime | None = None,
    ) -> EvolutionCampaignRecord:
        completed_at = at or utc_now()
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                candidate.campaign_id,
            )
            active = self._active_attempt(record, attempt_id)
            if (
                patch.campaign_id != record.request.campaign_id
                or candidate.campaign_id != record.request.campaign_id
                or patch.round_no != active.round_no
                or candidate.round_no != active.round_no
                or candidate.patch_id != patch.patch_id
            ):
                raise EvolutionConflict(
                    "generation result does not match its active attempt"
                )
            terminal = active.model_copy(
                update={
                    "status": GenerationAttemptStatus.SUCCEEDED,
                    "completed_at": completed_at,
                    "candidate_id": candidate.candidate_id,
                }
            )
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=candidate.campaign_id,
                    sequence=record.revision + 1,
                    kind=EvolutionJournalKind.CANDIDATE_GENERATED,
                    payload={
                        "attempt": terminal.model_dump(mode="json"),
                        "patch": patch.model_dump(mode="json"),
                        "candidate": candidate.model_dump(mode="json"),
                    },
                    occurred_at=completed_at,
                ),
                EvolutionCampaignStatus.CANDIDATE_READY,
                recovery_count=record.recovery_count,
            )
            return self._record_locked(
                connection,
                candidate.campaign_id,
            )

    def abandon_generation(
        self,
        campaign_id: str,
        *,
        attempt_id: str,
        error: str,
        at: datetime | None = None,
    ) -> EvolutionCampaignRecord:
        completed_at = at or utc_now()
        with self.transaction() as connection:
            record = self._record_locked(connection, campaign_id)
            active = self._active_attempt(record, attempt_id)
            terminal = active.model_copy(
                update={
                    "status": GenerationAttemptStatus.ABANDONED,
                    "completed_at": completed_at,
                    "error": error,
                }
            )
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=campaign_id,
                    sequence=record.revision + 1,
                    kind=EvolutionJournalKind.GENERATION_ABANDONED,
                    payload={
                        "attempt": terminal.model_dump(mode="json")
                    },
                    occurred_at=completed_at,
                ),
                EvolutionCampaignStatus.READY,
                recovery_count=record.recovery_count + 1,
            )
            return self._record_locked(connection, campaign_id)

    def recover_interrupted(
        self,
        *,
        at: datetime | None = None,
    ) -> tuple[EvolutionCampaignRecord, ...]:
        recovered_at = at or utc_now()
        with self._lock:
            rows = self._connection.execute(
                "SELECT campaign_id FROM evolution_projection "
                "WHERE status=? ORDER BY campaign_id",
                (EvolutionCampaignStatus.GENERATING.value,),
            ).fetchall()
        recovered = []
        for row in rows:
            campaign_id = str(row["campaign_id"])
            with self.transaction() as connection:
                record = self._record_locked(connection, campaign_id)
                if record.status != EvolutionCampaignStatus.GENERATING:
                    continue
                active = record.attempts[-1]
                terminal = active.model_copy(
                    update={
                        "status": GenerationAttemptStatus.ABANDONED,
                        "completed_at": recovered_at,
                        "error": (
                            "offline evolution worker restarted before "
                            "candidate commit; any unregistered output is "
                            "discarded and the campaign is ready for a new "
                            "immutable attempt"
                        ),
                    }
                )
                self._append_locked(
                    connection,
                    EvolutionJournalEntry(
                        campaign_id=campaign_id,
                        sequence=record.revision + 1,
                        kind=EvolutionJournalKind.GENERATION_ABANDONED,
                        payload={
                            "attempt": terminal.model_dump(mode="json")
                        },
                        occurred_at=recovered_at,
                    ),
                    EvolutionCampaignStatus.READY,
                    recovery_count=record.recovery_count + 1,
                )
                recovered.append(
                    self._record_locked(connection, campaign_id)
                )
        return tuple(recovered)

    def record_selection(
        self,
        campaign_id: str,
        *,
        candidate_id: str,
        passed: bool,
        gate_decision_id: str,
        gate_artifact_id: str,
        failed_check_names: tuple[str, ...],
        rejection: RejectedEditMemory | None,
        at: datetime,
    ) -> EvolutionCampaignRecord:
        with self.transaction() as connection:
            record = self._record_locked(connection, campaign_id)
            if record.status != EvolutionCampaignStatus.CANDIDATE_READY:
                raise EvolutionConflict(
                    "selection gate requires a generated candidate"
                )
            candidate = self._latest_candidate(record, candidate_id)
            if passed:
                if failed_check_names or rejection is not None:
                    raise ValueError(
                        "passed selection cannot carry rejection data"
                    )
                kind = EvolutionJournalKind.SELECTION_PASSED
                status = EvolutionCampaignStatus.AWAITING_HUMAN
            else:
                if not failed_check_names or rejection is None:
                    raise ValueError(
                        "failed selection requires rejected-edit memory"
                    )
                self._save_rejection_locked(connection, rejection)
                kind = EvolutionJournalKind.SELECTION_REJECTED
                status = EvolutionCampaignStatus.READY
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=campaign_id,
                    sequence=record.revision + 1,
                    kind=kind,
                    payload={
                        "candidate_id": candidate.candidate.candidate_id,
                        "gate_decision_id": gate_decision_id,
                        "gate_artifact_id": gate_artifact_id,
                        "failed_check_names": failed_check_names,
                        "rejection": (
                            rejection.model_dump(mode="json")
                            if rejection is not None
                            else None
                        ),
                    },
                    occurred_at=at,
                ),
                status,
                recovery_count=record.recovery_count,
            )
            updated = self._record_locked(connection, campaign_id)
            return self._exhaust_locked(connection, updated, at)

    def record_human_decision(
        self,
        decision: EvolutionHumanDecision,
        *,
        rejection: RejectedEditMemory | None = None,
    ) -> EvolutionCampaignRecord:
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                decision.campaign_id,
            )
            if record.status != EvolutionCampaignStatus.AWAITING_HUMAN:
                raise EvolutionConflict(
                    "human gate requires a selection-passed candidate"
                )
            self._latest_candidate(record, decision.candidate_id)
            if decision.outcome == HumanGateOutcome.APPROVED:
                if rejection is not None:
                    raise ValueError(
                        "approved human decision cannot carry rejection"
                    )
                kind = EvolutionJournalKind.HUMAN_APPROVED
                status = EvolutionCampaignStatus.READY_FOR_FINAL
            else:
                if rejection is None:
                    raise ValueError(
                        "human rejection requires rejected-edit memory"
                    )
                self._save_rejection_locked(connection, rejection)
                kind = EvolutionJournalKind.HUMAN_REJECTED
                status = EvolutionCampaignStatus.READY
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=decision.campaign_id,
                    sequence=record.revision + 1,
                    kind=kind,
                    payload={
                        "decision": decision.model_dump(mode="json"),
                        "rejection": (
                            rejection.model_dump(mode="json")
                            if rejection is not None
                            else None
                        ),
                    },
                    occurred_at=decision.decided_at,
                ),
                status,
                recovery_count=record.recovery_count,
            )
            updated = self._record_locked(
                connection,
                decision.campaign_id,
            )
            return self._exhaust_locked(
                connection,
                updated,
                decision.decided_at,
            )

    def record_final_gate(
        self,
        campaign_id: str,
        *,
        candidate_id: str,
        passed: bool,
        gate_decision_id: str,
        gate_artifact_id: str,
        failed_check_names: tuple[str, ...],
        rejection: RejectedEditMemory | None,
        best_skill: BestSkillSnapshot | None,
        at: datetime,
    ) -> EvolutionCampaignRecord:
        with self.transaction() as connection:
            record = self._record_locked(connection, campaign_id)
            if record.status != EvolutionCampaignStatus.READY_FOR_FINAL:
                raise EvolutionConflict(
                    "final gate requires human-approved candidate"
                )
            self._latest_candidate(record, candidate_id)
            if passed:
                if failed_check_names or rejection is not None:
                    raise ValueError(
                        "passed final gate cannot carry rejection"
                    )
                if best_skill is not None:
                    self._save_best_skill_locked(connection, best_skill)
                kind = EvolutionJournalKind.PROMOTED
                status = EvolutionCampaignStatus.PROMOTED
            else:
                if not failed_check_names or rejection is None:
                    raise ValueError(
                        "failed final gate requires rejected-edit memory"
                    )
                if best_skill is not None:
                    raise ValueError(
                        "failed final gate cannot publish best_skill"
                    )
                self._save_rejection_locked(connection, rejection)
                kind = EvolutionJournalKind.FINAL_REJECTED
                status = EvolutionCampaignStatus.READY
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=campaign_id,
                    sequence=record.revision + 1,
                    kind=kind,
                    payload={
                        "candidate_id": candidate_id,
                        "gate_decision_id": gate_decision_id,
                        "gate_artifact_id": gate_artifact_id,
                        "failed_check_names": failed_check_names,
                        "rejection": (
                            rejection.model_dump(mode="json")
                            if rejection is not None
                            else None
                        ),
                        "best_skill": (
                            best_skill.model_dump(mode="json")
                            if best_skill is not None
                            else None
                        ),
                    },
                    occurred_at=at,
                ),
                status,
                recovery_count=record.recovery_count,
            )
            updated = self._record_locked(connection, campaign_id)
            return self._exhaust_locked(connection, updated, at)

    def record_post_release(
        self,
        campaign_id: str,
        *,
        candidate_id: str,
        kept: bool,
        gate_decision_id: str,
        gate_artifact_id: str,
        best_skill: BestSkillSnapshot | None,
        at: datetime,
    ) -> EvolutionCampaignRecord:
        with self.transaction() as connection:
            record = self._record_locked(connection, campaign_id)
            if record.status != EvolutionCampaignStatus.PROMOTED:
                raise EvolutionConflict(
                    "post-release gate requires a promoted campaign"
                )
            self._latest_candidate(record, candidate_id)
            if kept:
                if best_skill is not None:
                    raise ValueError(
                        "keep decision does not create another best_skill"
                    )
                kind = EvolutionJournalKind.POST_RELEASE_KEPT
                status = EvolutionCampaignStatus.PROMOTED
            else:
                if best_skill is not None:
                    self._save_best_skill_locked(connection, best_skill)
                kind = EvolutionJournalKind.ROLLED_BACK
                status = EvolutionCampaignStatus.ROLLED_BACK
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=campaign_id,
                    sequence=record.revision + 1,
                    kind=kind,
                    payload={
                        "candidate_id": candidate_id,
                        "gate_decision_id": gate_decision_id,
                        "gate_artifact_id": gate_artifact_id,
                        "best_skill": (
                            best_skill.model_dump(mode="json")
                            if best_skill is not None
                            else None
                        ),
                    },
                    occurred_at=at,
                ),
                status,
                recovery_count=record.recovery_count,
            )
            return self._record_locked(connection, campaign_id)

    def rejection_memory(
        self,
        *,
        target: OptimizationTarget,
        base_version_id: str,
    ) -> tuple[RejectedEditMemory, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json, checksum
                FROM evolution_rejections
                WHERE target=? AND base_version_id=?
                ORDER BY rejected_at, rejection_id
                """,
                (target.value, base_version_id),
            ).fetchall()
        values = []
        for row in rows:
            payload = self._verified_payload(
                row,
                "evolution rejection",
            )
            values.append(
                RejectedEditMemory.model_validate(payload, strict=False)
            )
        return tuple(values)

    def best_skill(
        self,
        *,
        target: OptimizationTarget,
        component_name: str,
    ) -> BestSkillSnapshot | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT s.payload_json, s.checksum
                FROM evolution_best_skill_projection p
                JOIN evolution_best_skills s
                  ON s.best_skill_id=p.best_skill_id
                WHERE p.target=? AND p.component_name=?
                """,
                (target.value, component_name),
            ).fetchone()
        if row is None:
            return None
        return BestSkillSnapshot.model_validate(
            self._verified_payload(row, "best_skill"),
            strict=False,
        )

    def best_skill_history(
        self,
        *,
        target: OptimizationTarget,
        component_name: str,
    ) -> tuple[BestSkillSnapshot, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload_json, checksum
                FROM evolution_best_skills
                WHERE target=? AND component_name=?
                ORDER BY best_skill_order
                """,
                (target.value, component_name),
            ).fetchall()
        return tuple(
            BestSkillSnapshot.model_validate(
                self._verified_payload(row, "best_skill"),
                strict=False,
            )
            for row in rows
        )

    def rebuild_projections(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM evolution_projection")
            rows = connection.execute(
                "SELECT campaign_id FROM evolution_campaigns "
                "ORDER BY created_at, campaign_id"
            ).fetchall()
            for row in rows:
                campaign_id = str(row["campaign_id"])
                status, revision, recovery_count = (
                    self._replay_projection_locked(
                        connection,
                        campaign_id,
                    )
                )
                connection.execute(
                    """
                    INSERT INTO evolution_projection(
                        campaign_id, status, revision,
                        recovery_count, checksum
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        campaign_id,
                        status.value,
                        revision,
                        recovery_count,
                        self._projection_checksum(
                            campaign_id,
                            status,
                            revision,
                            recovery_count,
                        ),
                    ),
                )
            connection.execute(
                "DELETE FROM evolution_best_skill_projection"
            )
            best_rows = connection.execute(
                """
                SELECT payload_json, checksum
                FROM evolution_best_skills
                ORDER BY best_skill_order
                """
            ).fetchall()
            for row in best_rows:
                best = BestSkillSnapshot.model_validate(
                    self._verified_payload(row, "best_skill"),
                    strict=False,
                )
                self._project_best_skill_locked(connection, best)

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
            if result is None or str(result[0]).lower() != "ok":
                raise EvolutionCorruption("SQLite integrity check failed")
            for table, column in (
                ("evolution_pool_entries", "payload_json"),
                ("evolution_pool_reviews", "payload_json"),
                ("evolution_experiences", "payload_json"),
                ("evolution_campaigns", "request_json"),
                ("evolution_journal", "entry_json"),
                ("evolution_rejections", "payload_json"),
                ("evolution_best_skills", "payload_json"),
            ):
                rows = self._connection.execute(
                    f"SELECT {column}, checksum FROM {table}"
                ).fetchall()
                for row in rows:
                    if self._checksum(str(row[column])) != str(
                        row["checksum"]
                    ):
                        raise EvolutionCorruption(
                            f"{table} checksum mismatch"
                        )
            campaigns = self._connection.execute(
                "SELECT campaign_id FROM evolution_campaigns"
            ).fetchall()
            for row in campaigns:
                campaign_id = str(row["campaign_id"])
                expected = self._replay_projection_locked(
                    self._connection,
                    campaign_id,
                )
                actual = self._projection_locked(
                    self._connection,
                    campaign_id,
                )
                if expected != actual[:3]:
                    raise EvolutionCorruption(
                        "evolution projection disagrees with journal"
                    )
                self._record_locked(self._connection, campaign_id)
            active_rows = self._connection.execute(
                "SELECT * FROM evolution_best_skill_projection"
            ).fetchall()
            for row in active_rows:
                expected = self._checksum(
                    self._json(
                        {
                            "target": str(row["target"]),
                            "component_name": str(row["component_name"]),
                            "best_skill_id": str(row["best_skill_id"]),
                        }
                    )
                )
                if str(row["checksum"]) != expected:
                    raise EvolutionCorruption(
                        "best_skill projection checksum mismatch"
                    )

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            connection = sqlite3.connect(str(target))
            try:
                self._connection.backup(connection)
            finally:
                connection.close()
        return target

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _pool_entry_locked(
        self,
        connection: sqlite3.Connection,
        pool_entry_id: str,
    ) -> ReviewedPoolEntry:
        row = connection.execute(
            "SELECT payload_json, checksum FROM evolution_pool_entries "
            "WHERE pool_entry_id=?",
            (pool_entry_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown candidate-pool entry: {pool_entry_id}")
        entry = CandidatePoolEntry.model_validate(
            self._verified_payload(row, "candidate-pool entry"),
            strict=False,
        )
        review_row = connection.execute(
            "SELECT payload_json, checksum FROM evolution_pool_reviews "
            "WHERE pool_entry_id=?",
            (pool_entry_id,),
        ).fetchone()
        review = (
            CandidatePoolReview.model_validate(
                self._verified_payload(
                    review_row,
                    "candidate-pool review",
                ),
                strict=False,
            )
            if review_row is not None
            else None
        )
        status = (
            PoolEntryStatus.PENDING_REVIEW
            if review is None
            else (
                PoolEntryStatus.APPROVED
                if review.approved
                else PoolEntryStatus.REJECTED
            )
        )
        return ReviewedPoolEntry(
            entry=entry,
            review=review,
            status=status,
        )

    def _request_locked(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
    ) -> EvolutionCampaignRequest:
        row = connection.execute(
            "SELECT request_json, checksum FROM evolution_campaigns "
            "WHERE campaign_id=?",
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown evolution campaign: {campaign_id}")
        payload = str(row["request_json"])
        if self._checksum(payload) != str(row["checksum"]):
            raise EvolutionCorruption("campaign request checksum mismatch")
        return EvolutionCampaignRequest.model_validate_json(payload)

    def _entries_locked(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
    ) -> tuple[EvolutionJournalEntry, ...]:
        rows = connection.execute(
            """
            SELECT entry_json, checksum FROM evolution_journal
            WHERE campaign_id=? ORDER BY sequence
            """,
            (campaign_id,),
        ).fetchall()
        entries = []
        for row in rows:
            payload = str(row["entry_json"])
            if self._checksum(payload) != str(row["checksum"]):
                raise EvolutionCorruption(
                    "evolution journal checksum mismatch"
                )
            entries.append(
                EvolutionJournalEntry.model_validate_json(payload)
            )
        if any(
            item.sequence != index
            for index, item in enumerate(entries, 1)
        ):
            raise EvolutionCorruption(
                "evolution journal sequence is not contiguous"
            )
        return tuple(entries)

    def _projection_locked(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
    ) -> tuple[EvolutionCampaignStatus, int, int, str]:
        row = connection.execute(
            "SELECT * FROM evolution_projection WHERE campaign_id=?",
            (campaign_id,),
        ).fetchone()
        if row is None:
            raise EvolutionCorruption("evolution projection is missing")
        status = EvolutionCampaignStatus(str(row["status"]))
        revision = int(row["revision"])
        recovery_count = int(row["recovery_count"])
        checksum = str(row["checksum"])
        if checksum != self._projection_checksum(
            campaign_id,
            status,
            revision,
            recovery_count,
        ):
            raise EvolutionCorruption(
                "evolution projection checksum mismatch"
            )
        return status, revision, recovery_count, checksum

    def _append_locked(
        self,
        connection: sqlite3.Connection,
        entry: EvolutionJournalEntry,
        status: EvolutionCampaignStatus,
        *,
        recovery_count: int,
    ) -> None:
        row = connection.execute(
            "SELECT revision FROM evolution_projection "
            "WHERE campaign_id=?",
            (entry.campaign_id,),
        ).fetchone()
        expected = 1 if row is None else int(row["revision"]) + 1
        if entry.sequence != expected:
            raise EvolutionConflict(
                f"expected evolution journal sequence {expected}, "
                f"got {entry.sequence}"
            )
        payload = canonical_contract_json(entry)
        connection.execute(
            """
            INSERT INTO evolution_journal(
                journal_event_id, campaign_id, sequence, kind,
                occurred_at, entry_json, checksum
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.journal_event_id,
                entry.campaign_id,
                entry.sequence,
                entry.kind.value,
                entry.occurred_at.isoformat(),
                payload,
                self._checksum(payload),
            ),
        )
        checksum = self._projection_checksum(
            entry.campaign_id,
            status,
            entry.sequence,
            recovery_count,
        )
        connection.execute(
            """
            INSERT INTO evolution_projection(
                campaign_id, status, revision,
                recovery_count, checksum
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(campaign_id) DO UPDATE SET
                status=excluded.status,
                revision=excluded.revision,
                recovery_count=excluded.recovery_count,
                checksum=excluded.checksum
            """,
            (
                entry.campaign_id,
                status.value,
                entry.sequence,
                recovery_count,
                checksum,
            ),
        )

    def _replay_projection_locked(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
    ) -> tuple[EvolutionCampaignStatus, int, int]:
        entries = self._entries_locked(connection, campaign_id)
        if not entries or entries[0].kind != EvolutionJournalKind.CREATED:
            raise EvolutionCorruption(
                "evolution journal must begin with created"
            )
        status = EvolutionCampaignStatus(
            entries[0].payload["initial_status"]
        )
        recovery_count = 0
        for entry in entries[1:]:
            kind = entry.kind
            if kind == EvolutionJournalKind.INPUTS_SEALED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.DRAFT,
                    kind,
                )
                status = EvolutionCampaignStatus.READY
            elif kind == EvolutionJournalKind.GENERATION_STARTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.READY,
                    kind,
                )
                status = EvolutionCampaignStatus.GENERATING
            elif kind == EvolutionJournalKind.GENERATION_ABANDONED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.GENERATING,
                    kind,
                )
                status = EvolutionCampaignStatus.READY
                recovery_count += 1
            elif kind == EvolutionJournalKind.CANDIDATE_GENERATED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.GENERATING,
                    kind,
                )
                status = EvolutionCampaignStatus.CANDIDATE_READY
            elif kind == EvolutionJournalKind.SELECTION_PASSED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.CANDIDATE_READY,
                    kind,
                )
                status = EvolutionCampaignStatus.AWAITING_HUMAN
            elif kind == EvolutionJournalKind.SELECTION_REJECTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.CANDIDATE_READY,
                    kind,
                )
                status = EvolutionCampaignStatus.READY
            elif kind == EvolutionJournalKind.HUMAN_APPROVED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.AWAITING_HUMAN,
                    kind,
                )
                status = EvolutionCampaignStatus.READY_FOR_FINAL
            elif kind == EvolutionJournalKind.HUMAN_REJECTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.AWAITING_HUMAN,
                    kind,
                )
                status = EvolutionCampaignStatus.READY
            elif kind == EvolutionJournalKind.FINAL_REJECTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.READY_FOR_FINAL,
                    kind,
                )
                status = EvolutionCampaignStatus.READY
            elif kind == EvolutionJournalKind.PROMOTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.READY_FOR_FINAL,
                    kind,
                )
                status = EvolutionCampaignStatus.PROMOTED
            elif kind == EvolutionJournalKind.POST_RELEASE_KEPT:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.PROMOTED,
                    kind,
                )
            elif kind == EvolutionJournalKind.ROLLED_BACK:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.PROMOTED,
                    kind,
                )
                status = EvolutionCampaignStatus.ROLLED_BACK
            elif kind == EvolutionJournalKind.EXHAUSTED:
                self._require_status(
                    status,
                    EvolutionCampaignStatus.READY,
                    kind,
                )
                status = EvolutionCampaignStatus.EXHAUSTED
            else:
                raise EvolutionCorruption(
                    f"unsupported evolution journal kind: {kind.value}"
                )
        return status, len(entries), recovery_count

    @staticmethod
    def _require_status(
        actual: EvolutionCampaignStatus,
        expected: EvolutionCampaignStatus,
        kind: EvolutionJournalKind,
    ) -> None:
        if actual != expected:
            raise EvolutionCorruption(
                f"{kind.value} occurred in {actual.value}, "
                f"expected {expected.value}"
            )

    def _record_locked(
        self,
        connection: sqlite3.Connection,
        campaign_id: str,
    ) -> EvolutionCampaignRecord:
        request = self._request_locked(connection, campaign_id)
        entries = self._entries_locked(connection, campaign_id)
        status, revision, recovery_count, _ = self._projection_locked(
            connection,
            campaign_id,
        )
        expected = self._replay_projection_locked(
            connection,
            campaign_id,
        )
        if expected != (status, revision, recovery_count):
            raise EvolutionCorruption(
                "evolution projection disagrees with journal"
            )
        snapshot = None
        attempts: list[GenerationAttempt] = []
        candidates: list[EvolutionCandidateRecord] = []
        candidate_indexes: dict[str, int] = {}
        for entry in entries:
            payload = entry.payload
            if entry.kind == EvolutionJournalKind.INPUTS_SEALED:
                snapshot = EvolutionInputSnapshot.model_validate(
                    payload["snapshot"],
                    strict=False,
                )
            elif entry.kind == EvolutionJournalKind.GENERATION_STARTED:
                attempts.append(
                    GenerationAttempt.model_validate(
                        payload["attempt"],
                        strict=False,
                    )
                )
            elif entry.kind == EvolutionJournalKind.GENERATION_ABANDONED:
                if not attempts:
                    raise EvolutionCorruption(
                        "abandoned generation has no attempt"
                    )
                attempts[-1] = GenerationAttempt.model_validate(
                    payload["attempt"],
                    strict=False,
                )
            elif entry.kind == EvolutionJournalKind.CANDIDATE_GENERATED:
                if not attempts:
                    raise EvolutionCorruption(
                        "generated candidate has no attempt"
                    )
                attempts[-1] = GenerationAttempt.model_validate(
                    payload["attempt"],
                    strict=False,
                )
                patch = EvolutionPatch.model_validate(
                    payload["patch"],
                    strict=False,
                )
                candidate = EvolutionCandidate.model_validate(
                    payload["candidate"],
                    strict=False,
                )
                candidate_indexes[candidate.candidate_id] = len(candidates)
                candidates.append(
                    EvolutionCandidateRecord(
                        candidate=candidate,
                        patch=patch,
                        status=CandidateStatus.GENERATED,
                    )
                )
            elif entry.kind in {
                EvolutionJournalKind.SELECTION_PASSED,
                EvolutionJournalKind.SELECTION_REJECTED,
            }:
                current = self._candidate_from_payload(
                    candidates,
                    candidate_indexes,
                    payload,
                )
                rejection = (
                    RejectedEditMemory.model_validate(
                        payload["rejection"],
                        strict=False,
                    )
                    if payload.get("rejection") is not None
                    else None
                )
                candidates[candidate_indexes[payload["candidate_id"]]] = (
                    current.model_copy(
                        update={
                            "status": (
                                CandidateStatus.SELECTION_PASSED
                                if entry.kind
                                == EvolutionJournalKind.SELECTION_PASSED
                                else CandidateStatus.SELECTION_REJECTED
                            ),
                            "selection_gate_decision_id": payload[
                                "gate_decision_id"
                            ],
                            "selection_gate_artifact_id": payload[
                                "gate_artifact_id"
                            ],
                            "rejection": rejection,
                        }
                    )
                )
            elif entry.kind in {
                EvolutionJournalKind.HUMAN_APPROVED,
                EvolutionJournalKind.HUMAN_REJECTED,
            }:
                decision = EvolutionHumanDecision.model_validate(
                    payload["decision"],
                    strict=False,
                )
                current = self._candidate_by_id(
                    candidates,
                    candidate_indexes,
                    decision.candidate_id,
                )
                rejection = (
                    RejectedEditMemory.model_validate(
                        payload["rejection"],
                        strict=False,
                    )
                    if payload.get("rejection") is not None
                    else None
                )
                candidates[candidate_indexes[decision.candidate_id]] = (
                    current.model_copy(
                        update={
                            "status": (
                                CandidateStatus.HUMAN_APPROVED
                                if entry.kind
                                == EvolutionJournalKind.HUMAN_APPROVED
                                else CandidateStatus.HUMAN_REJECTED
                            ),
                            "human_decision": decision,
                            "rejection": rejection,
                        }
                    )
                )
            elif entry.kind in {
                EvolutionJournalKind.FINAL_REJECTED,
                EvolutionJournalKind.PROMOTED,
            }:
                current = self._candidate_from_payload(
                    candidates,
                    candidate_indexes,
                    payload,
                )
                rejection = (
                    RejectedEditMemory.model_validate(
                        payload["rejection"],
                        strict=False,
                    )
                    if payload.get("rejection") is not None
                    else None
                )
                best = (
                    BestSkillSnapshot.model_validate(
                        payload["best_skill"],
                        strict=False,
                    )
                    if payload.get("best_skill") is not None
                    else None
                )
                candidates[candidate_indexes[payload["candidate_id"]]] = (
                    current.model_copy(
                        update={
                            "status": (
                                CandidateStatus.PROMOTED
                                if entry.kind
                                == EvolutionJournalKind.PROMOTED
                                else CandidateStatus.FINAL_REJECTED
                            ),
                            "final_gate_decision_id": payload[
                                "gate_decision_id"
                            ],
                            "final_gate_artifact_id": payload[
                                "gate_artifact_id"
                            ],
                            "rejection": rejection,
                            "best_skill": best,
                        }
                    )
                )
            elif entry.kind in {
                EvolutionJournalKind.POST_RELEASE_KEPT,
                EvolutionJournalKind.ROLLED_BACK,
            }:
                current = self._candidate_from_payload(
                    candidates,
                    candidate_indexes,
                    payload,
                )
                best = (
                    BestSkillSnapshot.model_validate(
                        payload["best_skill"],
                        strict=False,
                    )
                    if payload.get("best_skill") is not None
                    else None
                )
                candidates[candidate_indexes[payload["candidate_id"]]] = (
                    current.model_copy(
                        update={
                            "status": (
                                CandidateStatus.KEPT
                                if entry.kind
                                == EvolutionJournalKind.POST_RELEASE_KEPT
                                else CandidateStatus.ROLLED_BACK
                            ),
                            "post_release_gate_decision_id": payload[
                                "gate_decision_id"
                            ],
                            "post_release_gate_artifact_id": payload[
                                "gate_artifact_id"
                            ],
                            "best_skill": best or current.best_skill,
                        }
                    )
                )
        return EvolutionCampaignRecord(
            request=request,
            status=status,
            revision=revision,
            recovery_count=recovery_count,
            input_snapshot=snapshot,
            attempts=tuple(attempts),
            candidates=tuple(candidates),
        )

    @staticmethod
    def _candidate_from_payload(
        candidates: list[EvolutionCandidateRecord],
        indexes: dict[str, int],
        payload: dict[str, Any],
    ) -> EvolutionCandidateRecord:
        return SQLiteEvolutionStore._candidate_by_id(
            candidates,
            indexes,
            str(payload["candidate_id"]),
        )

    @staticmethod
    def _candidate_by_id(
        candidates: list[EvolutionCandidateRecord],
        indexes: dict[str, int],
        candidate_id: str,
    ) -> EvolutionCandidateRecord:
        if candidate_id not in indexes:
            raise EvolutionCorruption(
                "journal references unknown evolution candidate"
            )
        return candidates[indexes[candidate_id]]

    @staticmethod
    def _active_attempt(
        record: EvolutionCampaignRecord,
        attempt_id: str,
    ) -> GenerationAttempt:
        if (
            record.status != EvolutionCampaignStatus.GENERATING
            or not record.attempts
        ):
            raise EvolutionConflict(
                "campaign has no active generation attempt"
            )
        active = record.attempts[-1]
        if (
            active.attempt_id != attempt_id
            or active.status != GenerationAttemptStatus.RUNNING
        ):
            raise EvolutionConflict(
                "generation result does not own the active attempt"
            )
        return active

    @staticmethod
    def _latest_candidate(
        record: EvolutionCampaignRecord,
        candidate_id: str,
    ) -> EvolutionCandidateRecord:
        if (
            not record.candidates
            or record.candidates[-1].candidate.candidate_id
            != candidate_id
        ):
            raise EvolutionConflict(
                "gate decision does not target the latest candidate"
            )
        return record.candidates[-1]

    def _exhaust_locked(
        self,
        connection: sqlite3.Connection,
        record: EvolutionCampaignRecord,
        at: datetime,
    ) -> EvolutionCampaignRecord:
        if (
            record.status == EvolutionCampaignStatus.READY
            and len(record.candidates)
            >= record.request.edit_budget.max_rounds
        ):
            self._append_locked(
                connection,
                EvolutionJournalEntry(
                    campaign_id=record.request.campaign_id,
                    sequence=record.revision + 1,
                    kind=EvolutionJournalKind.EXHAUSTED,
                    payload={
                        "rounds": len(record.candidates),
                        "reason": (
                            "candidate round budget exhausted without "
                            "promotion"
                        ),
                    },
                    occurred_at=at,
                ),
                EvolutionCampaignStatus.EXHAUSTED,
                recovery_count=record.recovery_count,
            )
            return self._record_locked(
                connection,
                record.request.campaign_id,
            )
        return record

    def _save_rejection_locked(
        self,
        connection: sqlite3.Connection,
        rejection: RejectedEditMemory,
    ) -> None:
        payload = canonical_contract_json(rejection)
        checksum = self._checksum(payload)
        row = connection.execute(
            "SELECT payload_json, checksum FROM evolution_rejections "
            "WHERE rejection_id=?",
            (rejection.rejection_id,),
        ).fetchone()
        if row is not None:
            if (
                str(row["checksum"]) != checksum
                or str(row["payload_json"]) != payload
            ):
                raise EvolutionConflict(
                    "rejection ID was reused with different content"
                )
            return
        connection.execute(
            """
            INSERT INTO evolution_rejections(
                rejection_id, campaign_id, target, base_version_id,
                rejected_at, payload_json, checksum
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rejection.rejection_id,
                rejection.campaign_id,
                rejection.target.value,
                rejection.base_version_id,
                rejection.rejected_at.isoformat(),
                payload,
                checksum,
            ),
        )

    def _save_best_skill_locked(
        self,
        connection: sqlite3.Connection,
        best: BestSkillSnapshot,
    ) -> None:
        payload = canonical_contract_json(best)
        checksum = self._checksum(payload)
        row = connection.execute(
            "SELECT payload_json, checksum FROM evolution_best_skills "
            "WHERE best_skill_id=?",
            (best.best_skill_id,),
        ).fetchone()
        if row is not None:
            if (
                str(row["checksum"]) != checksum
                or str(row["payload_json"]) != payload
            ):
                raise EvolutionConflict(
                    "best_skill ID was reused with different content"
                )
        else:
            connection.execute(
                """
                INSERT INTO evolution_best_skills(
                    best_skill_id, target, component_name, published_at,
                    payload_json, checksum
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    best.best_skill_id,
                    best.target.value,
                    best.component_name,
                    best.published_at.isoformat(),
                    payload,
                    checksum,
                ),
            )
        self._project_best_skill_locked(connection, best)

    def _project_best_skill_locked(
        self,
        connection: sqlite3.Connection,
        best: BestSkillSnapshot,
    ) -> None:
        checksum = self._checksum(
            self._json(
                {
                    "target": best.target.value,
                    "component_name": best.component_name,
                    "best_skill_id": best.best_skill_id,
                }
            )
        )
        connection.execute(
            """
            INSERT INTO evolution_best_skill_projection(
                target, component_name, best_skill_id, checksum
            ) VALUES(?, ?, ?, ?)
            ON CONFLICT(target, component_name) DO UPDATE SET
                best_skill_id=excluded.best_skill_id,
                checksum=excluded.checksum
            """,
            (
                best.target.value,
                best.component_name,
                best.best_skill_id,
                checksum,
            ),
        )

    def _save_immutable(
        self,
        *,
        table: str,
        id_column: str,
        identifier: str,
        columns: dict[str, str],
        value: Any,
    ) -> None:
        payload = canonical_contract_json(value)
        checksum = self._checksum(payload)
        with self.transaction() as connection:
            existing = connection.execute(
                f"SELECT payload_json, checksum FROM {table} "
                f"WHERE {id_column}=?",
                (identifier,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checksum"]) != checksum
                    or str(existing["payload_json"]) != payload
                ):
                    raise EvolutionConflict(
                        f"{id_column} was reused with different content"
                    )
                return
            names = [id_column, *columns, "payload_json", "checksum"]
            placeholders = ",".join("?" for _ in names)
            connection.execute(
                f"INSERT INTO {table}({','.join(names)}) "
                f"VALUES({placeholders})",
                (
                    identifier,
                    *columns.values(),
                    payload,
                    checksum,
                ),
            )

    def _load_immutable(
        self,
        table: str,
        id_column: str,
        identifier: str,
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                f"SELECT payload_json, checksum FROM {table} "
                f"WHERE {id_column}=?",
                (identifier,),
            ).fetchone()
        if row is None:
            return None
        return self._verified_payload(row, table)

    def _verified_payload(
        self,
        row: sqlite3.Row,
        label: str,
    ) -> dict[str, Any]:
        payload = str(row["payload_json"])
        if self._checksum(payload) != str(row["checksum"]):
            raise EvolutionCorruption(f"{label} checksum mismatch")
        return json.loads(payload)
