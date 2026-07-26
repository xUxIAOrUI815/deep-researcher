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

from deep_researcher.contracts import (
    ErrorCategory,
    ErrorRecord,
    canonical_contract_json,
    utc_now,
)

from .advanced_models import (
    ReplayApprovalGrant,
    ReplayAttempt,
    ReplayAttemptStatus,
    ReplayExecutionOutcome,
    ReplayJournalEntry,
    ReplayJournalKind,
    ReplayRecord,
    ReplayRecordPage,
    ReplayRequest,
    ReplayRequestStatus,
    StudioABComparison,
    StudioBadcase,
)


class StudioAdvancedStoreError(RuntimeError):
    pass


class StudioAdvancedConflict(StudioAdvancedStoreError):
    pass


class StudioAdvancedCorruption(StudioAdvancedStoreError):
    pass


class SQLiteStudioAdvancedStore:
    """Checksummed, append-only Studio replay/comparison/badcase store."""

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
        request_id: str,
        status: ReplayRequestStatus,
        revision: int,
        recovery_count: int,
    ) -> str:
        return cls._checksum(
            cls._json(
                {
                    "request_id": request_id,
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
                CREATE TABLE IF NOT EXISTS studio_advanced_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO studio_advanced_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS studio_replay_requests(
                    replay_request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS studio_replay_journal(
                    journal_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    journal_event_id TEXT NOT NULL UNIQUE,
                    replay_request_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    entry_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(replay_request_id, sequence),
                    FOREIGN KEY(replay_request_id)
                      REFERENCES studio_replay_requests(replay_request_id)
                      ON DELETE RESTRICT
                );
                CREATE INDEX IF NOT EXISTS idx_studio_replay_journal_request
                ON studio_replay_journal(replay_request_id, sequence);

                CREATE TABLE IF NOT EXISTS studio_replay_projection(
                    replay_request_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    recovery_count INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(replay_request_id)
                      REFERENCES studio_replay_requests(replay_request_id)
                      ON DELETE RESTRICT
                );

                CREATE TABLE IF NOT EXISTS studio_comparisons(
                    comparison_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS studio_badcases(
                    badcase_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    checksum TEXT NOT NULL
                );

                CREATE TRIGGER IF NOT EXISTS studio_replay_requests_no_update
                BEFORE UPDATE ON studio_replay_requests
                BEGIN SELECT RAISE(ABORT, 'replay requests are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studio_replay_requests_no_delete
                BEFORE DELETE ON studio_replay_requests
                BEGIN SELECT RAISE(ABORT, 'replay requests are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studio_replay_journal_no_update
                BEFORE UPDATE ON studio_replay_journal
                BEGIN SELECT RAISE(ABORT, 'replay journal is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS studio_replay_journal_no_delete
                BEFORE DELETE ON studio_replay_journal
                BEGIN SELECT RAISE(ABORT, 'replay journal is append-only'); END;
                CREATE TRIGGER IF NOT EXISTS studio_comparisons_no_update
                BEFORE UPDATE ON studio_comparisons
                BEGIN SELECT RAISE(ABORT, 'comparisons are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studio_comparisons_no_delete
                BEFORE DELETE ON studio_comparisons
                BEGIN SELECT RAISE(ABORT, 'comparisons are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studio_badcases_no_update
                BEFORE UPDATE ON studio_badcases
                BEGIN SELECT RAISE(ABORT, 'badcases are immutable'); END;
                CREATE TRIGGER IF NOT EXISTS studio_badcases_no_delete
                BEFORE DELETE ON studio_badcases
                BEGIN SELECT RAISE(ABORT, 'badcases are immutable'); END;
                """
            )
            row = self._connection.execute(
                "SELECT version FROM studio_advanced_schema "
                "WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != self.CURRENT_VERSION:
                raise StudioAdvancedCorruption(
                    "unsupported Studio advanced store schema"
                )

    def create_request(self, request: ReplayRequest) -> ReplayRecord:
        payload = canonical_contract_json(request)
        checksum = self._checksum(payload)
        initial = (
            ReplayRequestStatus.WAITING_APPROVAL
            if request.required_approval_fingerprints
            else ReplayRequestStatus.QUEUED
        )
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT request_json, checksum FROM studio_replay_requests "
                "WHERE replay_request_id=?",
                (request.replay_request_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["checksum"]) != checksum
                    or str(existing["request_json"]) != payload
                ):
                    raise StudioAdvancedConflict(
                        "replay request ID was reused with different content"
                    )
                return self._record_locked(
                    connection,
                    request.replay_request_id,
                )
            connection.execute(
                """
                INSERT INTO studio_replay_requests(
                    replay_request_id, created_at, request_json, checksum
                ) VALUES(?, ?, ?, ?)
                """,
                (
                    request.replay_request_id,
                    request.created_at.isoformat(),
                    payload,
                    checksum,
                ),
            )
            self._append_locked(
                connection,
                ReplayJournalEntry(
                    replay_request_id=request.replay_request_id,
                    sequence=1,
                    kind=ReplayJournalKind.CREATED,
                    payload={"initial_status": initial.value},
                    occurred_at=request.created_at,
                ),
                initial,
                recovery_count=0,
            )
            return self._record_locked(
                connection,
                request.replay_request_id,
            )

    def get(self, replay_request_id: str) -> ReplayRecord | None:
        with self._lock:
            exists = self._connection.execute(
                "SELECT 1 FROM studio_replay_requests "
                "WHERE replay_request_id=?",
                (replay_request_id,),
            ).fetchone()
            if exists is None:
                return None
            return self._record_locked(
                self._connection,
                replay_request_id,
            )

    def list(
        self,
        *,
        statuses: tuple[ReplayRequestStatus, ...] = (),
        cursor: str | None = None,
        limit: int = 100,
    ) -> ReplayRecordPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses = ["1=1"]
        params: list[Any] = []
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            clauses.append(f"p.status IN ({placeholders})")
            params.extend(item.value for item in statuses)
        if cursor:
            try:
                decoded = json.loads(
                    base64.urlsafe_b64decode(
                        cursor + "=" * (-len(cursor) % 4)
                    ).decode()
                )
                created_at = str(decoded["created_at"])
                request_id = str(decoded["replay_request_id"])
            except Exception as exc:
                raise ValueError("invalid replay cursor") from exc
            clauses.append(
                "(r.created_at>? OR "
                "(r.created_at=? AND r.replay_request_id>?))"
            )
            params.extend((created_at, created_at, request_id))
        params.append(limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"""
                SELECT r.replay_request_id, r.created_at
                FROM studio_replay_requests r
                JOIN studio_replay_projection p
                  ON p.replay_request_id=r.replay_request_id
                WHERE {' AND '.join(clauses)}
                ORDER BY r.created_at, r.replay_request_id
                LIMIT ?
                """,
                params,
            ).fetchall()
            selected = rows[:limit]
            records = tuple(
                self._record_locked(
                    self._connection,
                    str(row["replay_request_id"]),
                )
                for row in selected
            )
        next_cursor = None
        if len(rows) > limit and selected:
            value = self._json(
                {
                    "created_at": str(selected[-1]["created_at"]),
                    "replay_request_id": str(
                        selected[-1]["replay_request_id"]
                    ),
                }
            ).encode()
            next_cursor = base64.urlsafe_b64encode(value).decode().rstrip("=")
        return ReplayRecordPage(items=records, next_cursor=next_cursor)

    def grant_approval(
        self,
        grant: ReplayApprovalGrant,
    ) -> ReplayRecord:
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                grant.replay_request_id,
            )
            if record.status != ReplayRequestStatus.WAITING_APPROVAL:
                raise StudioAdvancedConflict(
                    "replay request is not waiting for approval"
                )
            required = set(
                record.request.required_approval_fingerprints
            )
            if record.attempts:
                required.update(
                    record.attempts[-1].pending_approval_fingerprints
                )
            if grant.command_fingerprint not in required:
                raise ValueError(
                    "approval does not match a required replay command"
                )
            by_fingerprint = {
                item.command_fingerprint: item
                for item in record.approvals
            }
            existing = by_fingerprint.get(grant.command_fingerprint)
            if existing is not None:
                if existing != grant:
                    raise StudioAdvancedConflict(
                        "replay command already has another approval grant"
                    )
                return record
            self._append_locked(
                connection,
                ReplayJournalEntry(
                    replay_request_id=grant.replay_request_id,
                    sequence=record.revision + 1,
                    kind=ReplayJournalKind.APPROVAL_GRANTED,
                    payload={"grant": grant.model_dump(mode="json")},
                    occurred_at=grant.granted_at,
                ),
                ReplayRequestStatus.WAITING_APPROVAL,
                recovery_count=record.recovery_count,
            )
            updated = self._record_locked(
                connection,
                grant.replay_request_id,
            )
            granted = {
                item.command_fingerprint for item in updated.approvals
            }
            if required.issubset(granted):
                self._append_locked(
                    connection,
                    ReplayJournalEntry(
                        replay_request_id=grant.replay_request_id,
                        sequence=updated.revision + 1,
                        kind=ReplayJournalKind.QUEUED,
                        payload={
                            "reason": "all fresh replay approvals granted"
                        },
                        occurred_at=grant.granted_at,
                    ),
                    ReplayRequestStatus.QUEUED,
                    recovery_count=updated.recovery_count,
                )
            return self._record_locked(
                connection,
                grant.replay_request_id,
            )

    def claim(
        self,
        replay_request_id: str,
        *,
        at: datetime | None = None,
    ) -> ReplayAttempt:
        started_at = at or utc_now()
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                replay_request_id,
            )
            if record.status != ReplayRequestStatus.QUEUED:
                raise StudioAdvancedConflict(
                    "only a queued replay request can be claimed"
                )
            attempt_no = len(record.attempts) + 1
            attempt_id = self._stable_id(
                "replay_attempt",
                replay_request_id,
                str(attempt_no),
            )
            attempt = ReplayAttempt(
                attempt_id=attempt_id,
                replay_request_id=replay_request_id,
                attempt_no=attempt_no,
                target_run_id=self._stable_id(
                    "run",
                    replay_request_id,
                    str(attempt_no),
                ),
                target_thread_id=self._stable_id(
                    "thread",
                    replay_request_id,
                ),
                target_trace_id=self._stable_id(
                    "trace",
                    replay_request_id,
                    str(attempt_no),
                ),
                status=ReplayAttemptStatus.RUNNING,
                started_at=started_at,
            )
            self._append_locked(
                connection,
                ReplayJournalEntry(
                    replay_request_id=replay_request_id,
                    sequence=record.revision + 1,
                    kind=ReplayJournalKind.ATTEMPT_STARTED,
                    payload={"attempt": attempt.model_dump(mode="json")},
                    occurred_at=started_at,
                ),
                ReplayRequestStatus.RUNNING,
                recovery_count=record.recovery_count,
            )
            return attempt

    def finish(
        self,
        outcome: ReplayExecutionOutcome,
    ) -> ReplayRecord:
        with self.transaction() as connection:
            record = self._record_locked(
                connection,
                outcome.replay_request_id,
            )
            if record.status != ReplayRequestStatus.RUNNING:
                raise StudioAdvancedConflict(
                    "only a running replay request can finish"
                )
            if not record.attempts:
                raise StudioAdvancedCorruption(
                    "running replay request has no attempt"
                )
            active = record.attempts[-1]
            if (
                active.attempt_id != outcome.attempt_id
                or active.target_run_id != outcome.target_run_id
                or active.status != ReplayAttemptStatus.RUNNING
            ):
                raise StudioAdvancedConflict(
                    "replay outcome does not match the active attempt"
                )
            terminal = active.model_copy(
                update={
                    "status": outcome.status,
                    "completed_at": outcome.completed_at,
                    "result_artifact_id": outcome.result_artifact_id,
                    "error": outcome.error,
                    "pending_approval_fingerprints": (
                        outcome.pending_approval_fingerprints
                    ),
                }
            )
            if outcome.status == ReplayAttemptStatus.SUCCEEDED:
                kind = ReplayJournalKind.ATTEMPT_SUCCEEDED
                request_status = ReplayRequestStatus.SUCCEEDED
            elif outcome.status == ReplayAttemptStatus.WAITING_APPROVAL:
                kind = ReplayJournalKind.ATTEMPT_WAITING_APPROVAL
                request_status = ReplayRequestStatus.WAITING_APPROVAL
            elif outcome.status == ReplayAttemptStatus.FAILED:
                kind = ReplayJournalKind.ATTEMPT_FAILED
                request_status = ReplayRequestStatus.FAILED
            else:
                raise ValueError(
                    "finish accepts succeeded, failed, or approval-waiting"
                )
            self._append_locked(
                connection,
                ReplayJournalEntry(
                    replay_request_id=outcome.replay_request_id,
                    sequence=record.revision + 1,
                    kind=kind,
                    payload={
                        "attempt": terminal.model_dump(mode="json"),
                        "outcome": outcome.model_dump(mode="json"),
                    },
                    occurred_at=outcome.completed_at,
                ),
                request_status,
                recovery_count=record.recovery_count,
            )
            return self._record_locked(
                connection,
                outcome.replay_request_id,
            )

    def recover_incomplete(
        self,
        *,
        at: datetime | None = None,
    ) -> tuple[ReplayRecord, ...]:
        recovered_at = at or utc_now()
        recovered: list[ReplayRecord] = []
        with self.transaction() as connection:
            rows = connection.execute(
                "SELECT replay_request_id FROM studio_replay_projection "
                "WHERE status=? ORDER BY replay_request_id",
                (ReplayRequestStatus.RUNNING.value,),
            ).fetchall()
            for row in rows:
                request_id = str(row["replay_request_id"])
                record = self._record_locked(connection, request_id)
                if not record.attempts:
                    raise StudioAdvancedCorruption(
                        "running replay request has no recoverable attempt"
                    )
                active = record.attempts[-1]
                error = ErrorRecord(
                    category=ErrorCategory.INTERNAL,
                    code="replay_worker_restart",
                    message=(
                        "Replay worker restarted before the attempt "
                        "committed a terminal result. The partial target run "
                        "is preserved and a new immutable attempt is queued."
                    ),
                    retryable=True,
                    fatal=False,
                )
                abandoned = active.model_copy(
                    update={
                        "status": ReplayAttemptStatus.ABANDONED,
                        "completed_at": recovered_at,
                        "error": error,
                    }
                )
                self._append_locked(
                    connection,
                    ReplayJournalEntry(
                        replay_request_id=request_id,
                        sequence=record.revision + 1,
                        kind=ReplayJournalKind.ATTEMPT_ABANDONED,
                        payload={
                            "attempt": abandoned.model_dump(mode="json"),
                            "preserved_target_run": active.target_run_id,
                        },
                        occurred_at=recovered_at,
                    ),
                    ReplayRequestStatus.QUEUED,
                    recovery_count=record.recovery_count + 1,
                )
                recovered.append(
                    self._record_locked(connection, request_id)
                )
        return tuple(recovered)

    def save_comparison(
        self,
        comparison: StudioABComparison,
    ) -> StudioABComparison:
        self._save_immutable(
            table="studio_comparisons",
            id_column="comparison_id",
            identifier=comparison.comparison_id,
            created_at=comparison.created_at,
            value=comparison,
        )
        return comparison

    def comparison(
        self,
        comparison_id: str,
    ) -> StudioABComparison | None:
        value = self._load_immutable(
            "studio_comparisons",
            "comparison_id",
            comparison_id,
        )
        return (
            StudioABComparison.model_validate(value, strict=False)
            if value is not None
            else None
        )

    def save_badcase(self, badcase: StudioBadcase) -> StudioBadcase:
        self._save_immutable(
            table="studio_badcases",
            id_column="badcase_id",
            identifier=badcase.badcase_id,
            created_at=badcase.created_at,
            value=badcase,
        )
        return badcase

    def badcase(self, badcase_id: str) -> StudioBadcase | None:
        value = self._load_immutable(
            "studio_badcases",
            "badcase_id",
            badcase_id,
        )
        return (
            StudioBadcase.model_validate(value, strict=False)
            if value is not None
            else None
        )

    def rebuild_projections(self) -> None:
        with self.transaction() as connection:
            connection.execute("DELETE FROM studio_replay_projection")
            rows = connection.execute(
                "SELECT replay_request_id FROM studio_replay_requests "
                "ORDER BY created_at, replay_request_id"
            ).fetchall()
            for row in rows:
                request_id = str(row["replay_request_id"])
                status, revision, recovery_count = (
                    self._replay_projection_locked(
                        connection,
                        request_id,
                    )
                )
                connection.execute(
                    """
                    INSERT INTO studio_replay_projection(
                        replay_request_id, status, revision,
                        recovery_count, checksum
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (
                        request_id,
                        status.value,
                        revision,
                        recovery_count,
                        self._projection_checksum(
                            request_id,
                            status,
                            revision,
                            recovery_count,
                        ),
                    ),
                )

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()
            if result is None or str(result[0]).lower() != "ok":
                raise StudioAdvancedCorruption(
                    "SQLite integrity check failed"
                )
            for table, json_column in (
                ("studio_replay_requests", "request_json"),
                ("studio_replay_journal", "entry_json"),
                ("studio_comparisons", "payload_json"),
                ("studio_badcases", "payload_json"),
            ):
                for row in self._connection.execute(
                    f"SELECT {json_column}, checksum FROM {table}"
                ).fetchall():
                    if self._checksum(str(row[json_column])) != str(
                        row["checksum"]
                    ):
                        raise StudioAdvancedCorruption(
                            f"{table} checksum mismatch"
                        )
            request_rows = self._connection.execute(
                "SELECT replay_request_id FROM studio_replay_requests"
            ).fetchall()
            for row in request_rows:
                request_id = str(row["replay_request_id"])
                expected = self._replay_projection_locked(
                    self._connection,
                    request_id,
                )
                actual = self._projection_locked(
                    self._connection,
                    request_id,
                )
                if expected != actual[:3]:
                    raise StudioAdvancedCorruption(
                        "replay projection disagrees with journal"
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

    def _save_immutable(
        self,
        *,
        table: str,
        id_column: str,
        identifier: str,
        created_at: datetime,
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
                    raise StudioAdvancedConflict(
                        f"{id_column} was reused with different content"
                    )
                return
            connection.execute(
                f"INSERT INTO {table}("
                f"{id_column}, created_at, payload_json, checksum"
                ") VALUES(?, ?, ?, ?)",
                (
                    identifier,
                    created_at.isoformat(),
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
        payload = str(row["payload_json"])
        if self._checksum(payload) != str(row["checksum"]):
            raise StudioAdvancedCorruption(
                f"{table} checksum mismatch"
            )
        return json.loads(payload)

    def _request_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> ReplayRequest:
        row = connection.execute(
            "SELECT request_json, checksum FROM studio_replay_requests "
            "WHERE replay_request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"unknown replay request: {request_id}")
        payload = str(row["request_json"])
        if self._checksum(payload) != str(row["checksum"]):
            raise StudioAdvancedCorruption(
                "replay request checksum mismatch"
            )
        return ReplayRequest.model_validate_json(payload)

    def _entries_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> tuple[ReplayJournalEntry, ...]:
        rows = connection.execute(
            """
            SELECT entry_json, checksum
            FROM studio_replay_journal
            WHERE replay_request_id=?
            ORDER BY sequence
            """,
            (request_id,),
        ).fetchall()
        entries: list[ReplayJournalEntry] = []
        for row in rows:
            payload = str(row["entry_json"])
            if self._checksum(payload) != str(row["checksum"]):
                raise StudioAdvancedCorruption(
                    "replay journal checksum mismatch"
                )
            entries.append(
                ReplayJournalEntry.model_validate_json(payload)
            )
        if any(
            item.sequence != index
            for index, item in enumerate(entries, 1)
        ):
            raise StudioAdvancedCorruption(
                "replay journal sequence is not contiguous"
            )
        return tuple(entries)

    def _projection_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> tuple[ReplayRequestStatus, int, int, str]:
        row = connection.execute(
            "SELECT * FROM studio_replay_projection "
            "WHERE replay_request_id=?",
            (request_id,),
        ).fetchone()
        if row is None:
            raise StudioAdvancedCorruption(
                "replay projection is missing"
            )
        status = ReplayRequestStatus(str(row["status"]))
        revision = int(row["revision"])
        recovery_count = int(row["recovery_count"])
        checksum = str(row["checksum"])
        if checksum != self._projection_checksum(
            request_id,
            status,
            revision,
            recovery_count,
        ):
            raise StudioAdvancedCorruption(
                "replay projection checksum mismatch"
            )
        return status, revision, recovery_count, checksum

    def _append_locked(
        self,
        connection: sqlite3.Connection,
        entry: ReplayJournalEntry,
        status: ReplayRequestStatus,
        *,
        recovery_count: int,
    ) -> None:
        existing = connection.execute(
            "SELECT revision FROM studio_replay_projection "
            "WHERE replay_request_id=?",
            (entry.replay_request_id,),
        ).fetchone()
        expected = 1 if existing is None else int(existing["revision"]) + 1
        if entry.sequence != expected:
            raise StudioAdvancedConflict(
                f"expected replay journal sequence {expected}, "
                f"got {entry.sequence}"
            )
        payload = canonical_contract_json(entry)
        connection.execute(
            """
            INSERT INTO studio_replay_journal(
                journal_event_id, replay_request_id, sequence, kind,
                occurred_at, entry_json, checksum
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.journal_event_id,
                entry.replay_request_id,
                entry.sequence,
                entry.kind.value,
                entry.occurred_at.isoformat(),
                payload,
                self._checksum(payload),
            ),
        )
        checksum = self._projection_checksum(
            entry.replay_request_id,
            status,
            entry.sequence,
            recovery_count,
        )
        connection.execute(
            """
            INSERT INTO studio_replay_projection(
                replay_request_id, status, revision,
                recovery_count, checksum
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(replay_request_id) DO UPDATE SET
                status=excluded.status,
                revision=excluded.revision,
                recovery_count=excluded.recovery_count,
                checksum=excluded.checksum
            """,
            (
                entry.replay_request_id,
                status.value,
                entry.sequence,
                recovery_count,
                checksum,
            ),
        )

    def _replay_projection_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> tuple[ReplayRequestStatus, int, int]:
        entries = self._entries_locked(connection, request_id)
        if not entries or entries[0].kind != ReplayJournalKind.CREATED:
            raise StudioAdvancedCorruption(
                "replay journal must begin with created"
            )
        status = ReplayRequestStatus(
            entries[0].payload["initial_status"]
        )
        recovery_count = 0
        for entry in entries[1:]:
            if entry.kind == ReplayJournalKind.APPROVAL_GRANTED:
                if status != ReplayRequestStatus.WAITING_APPROVAL:
                    raise StudioAdvancedCorruption(
                        "approval grant occurred outside approval wait"
                    )
            elif entry.kind == ReplayJournalKind.QUEUED:
                if status != ReplayRequestStatus.WAITING_APPROVAL:
                    raise StudioAdvancedCorruption(
                        "queued transition requires approval wait"
                    )
                status = ReplayRequestStatus.QUEUED
            elif entry.kind == ReplayJournalKind.ATTEMPT_STARTED:
                if status != ReplayRequestStatus.QUEUED:
                    raise StudioAdvancedCorruption(
                        "attempt started outside queued state"
                    )
                status = ReplayRequestStatus.RUNNING
            elif entry.kind == ReplayJournalKind.ATTEMPT_SUCCEEDED:
                if status != ReplayRequestStatus.RUNNING:
                    raise StudioAdvancedCorruption(
                        "attempt succeeded outside running state"
                    )
                status = ReplayRequestStatus.SUCCEEDED
            elif entry.kind == ReplayJournalKind.ATTEMPT_FAILED:
                if status != ReplayRequestStatus.RUNNING:
                    raise StudioAdvancedCorruption(
                        "attempt failed outside running state"
                    )
                status = ReplayRequestStatus.FAILED
            elif (
                entry.kind
                == ReplayJournalKind.ATTEMPT_WAITING_APPROVAL
            ):
                if status != ReplayRequestStatus.RUNNING:
                    raise StudioAdvancedCorruption(
                        "attempt approval wait occurred outside running"
                    )
                status = ReplayRequestStatus.WAITING_APPROVAL
            elif entry.kind == ReplayJournalKind.ATTEMPT_ABANDONED:
                if status != ReplayRequestStatus.RUNNING:
                    raise StudioAdvancedCorruption(
                        "attempt abandoned outside running state"
                    )
                status = ReplayRequestStatus.QUEUED
                recovery_count += 1
            else:
                raise StudioAdvancedCorruption(
                    f"unsupported replay journal kind: {entry.kind.value}"
                )
        return status, len(entries), recovery_count

    def _record_locked(
        self,
        connection: sqlite3.Connection,
        request_id: str,
    ) -> ReplayRecord:
        request = self._request_locked(connection, request_id)
        entries = self._entries_locked(connection, request_id)
        status, revision, recovery_count, _ = self._projection_locked(
            connection,
            request_id,
        )
        expected = self._replay_projection_locked(
            connection,
            request_id,
        )
        if expected != (status, revision, recovery_count):
            raise StudioAdvancedCorruption(
                "replay projection disagrees with journal"
            )
        approvals: list[ReplayApprovalGrant] = []
        attempts: list[ReplayAttempt] = []
        latest_outcome: ReplayExecutionOutcome | None = None
        for entry in entries:
            if entry.kind == ReplayJournalKind.APPROVAL_GRANTED:
                approvals.append(
                    ReplayApprovalGrant.model_validate(
                        entry.payload["grant"],
                        strict=False,
                    )
                )
            elif entry.kind == ReplayJournalKind.ATTEMPT_STARTED:
                attempts.append(
                    ReplayAttempt.model_validate(
                        entry.payload["attempt"],
                        strict=False,
                    )
                )
            elif entry.kind in {
                ReplayJournalKind.ATTEMPT_SUCCEEDED,
                ReplayJournalKind.ATTEMPT_FAILED,
                ReplayJournalKind.ATTEMPT_WAITING_APPROVAL,
            }:
                if not attempts:
                    raise StudioAdvancedCorruption(
                        "terminal replay attempt has no start"
                    )
                attempts[-1] = ReplayAttempt.model_validate(
                    entry.payload["attempt"],
                    strict=False,
                )
                latest_outcome = ReplayExecutionOutcome.model_validate(
                    entry.payload["outcome"],
                    strict=False,
                )
            elif entry.kind == ReplayJournalKind.ATTEMPT_ABANDONED:
                if not attempts:
                    raise StudioAdvancedCorruption(
                        "abandoned replay attempt has no start"
                    )
                attempts[-1] = ReplayAttempt.model_validate(
                    entry.payload["attempt"],
                    strict=False,
                )
                latest_outcome = None
        return ReplayRecord(
            request=request,
            status=status,
            revision=revision,
            recovery_count=recovery_count,
            approvals=tuple(approvals),
            attempts=tuple(attempts),
            latest_outcome=latest_outcome,
        )
