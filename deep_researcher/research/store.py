from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.contracts import utc_now

from .models import (
    ConvergenceDecision,
    DedupDecision,
    DedupDecisionKind,
    DedupKind,
    DedupStatus,
    MergedResearchResult,
    ResearchWorkerResult,
    TaskReservation,
)


class ResearchCoordinationError(RuntimeError):
    pass


class ResearchCoordinationConflict(ResearchCoordinationError):
    pass


class ResearchCoordinationCorruption(ResearchCoordinationError):
    pass


class SQLiteResearchCoordinationStore:
    """Durable cross-worker deduplication, result, and convergence journal."""

    def __init__(self, path: str | Path, *, clock=utc_now) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS research_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO research_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS research_dedup_claims (
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    PRIMARY KEY(run_id, kind, key_hash)
                );

                CREATE TABLE IF NOT EXISTS research_task_reservations (
                    run_id TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    PRIMARY KEY(run_id, fingerprint),
                    UNIQUE(run_id, task_id)
                );

                CREATE TABLE IF NOT EXISTS research_novelty (
                    run_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    key_hash TEXT NOT NULL,
                    normalized_key TEXT NOT NULL,
                    first_task_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    PRIMARY KEY(run_id, kind, key_hash)
                );

                CREATE TABLE IF NOT EXISTS research_worker_results (
                    worker_result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    task_attempt INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, task_id, task_attempt)
                );
                CREATE INDEX IF NOT EXISTS idx_research_worker_results_run
                ON research_worker_results(run_id, created_at, worker_result_id);

                CREATE TABLE IF NOT EXISTS research_merges (
                    merge_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_research_merges_run
                ON research_merges(run_id, created_at, merge_id);

                CREATE TABLE IF NOT EXISTS research_convergence (
                    decision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    cycle INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, cycle)
                );
                CREATE INDEX IF NOT EXISTS idx_research_convergence_run
                ON research_convergence(run_id, cycle);
                """
            )
            row = connection.execute(
                "SELECT version FROM research_schema WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != 1:
                raise ResearchCoordinationCorruption(
                    "unsupported research coordination schema"
                )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def claim_key(
        self,
        *,
        run_id: str,
        kind: DedupKind,
        normalized_key: str,
        task_id: str,
        lease_seconds: float = 60.0,
    ) -> DedupDecision:
        normalized_key = self._normalize_key(kind, normalized_key)
        if not normalized_key:
            raise ValueError("deduplication key cannot be empty")
        if lease_seconds <= 0:
            raise ValueError("deduplication lease must be positive")
        key_hash = self.key_hash(normalized_key)
        now = self.clock()
        expires = now + timedelta(seconds=lease_seconds)
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT payload, checksum
                FROM research_dedup_claims
                WHERE run_id=? AND kind=? AND key_hash=?
                """,
                (run_id, kind.value, key_hash),
            ).fetchone()
            if row is None:
                payload = self._claim_payload(
                    run_id=run_id,
                    kind=kind,
                    key_hash=key_hash,
                    normalized_key=normalized_key,
                    owner_task_id=task_id,
                    status=DedupStatus.CLAIMED,
                    artifact_ids=(),
                    lease_expires_at=expires,
                    updated_at=now,
                    error=None,
                )
                self._write_claim(connection, payload)
                return self._decision(
                    payload,
                    requesting_task_id=task_id,
                    decision=DedupDecisionKind.NEW,
                )
            payload = self._verified_payload(row, "dedup claim")
            status = DedupStatus(payload["status"])
            owner = str(payload["owner_task_id"])
            artifact_ids = tuple(payload.get("artifact_ids", ()))
            lease_expires_at = self._datetime(payload.get("lease_expires_at"))
            if status == DedupStatus.COMPLETED:
                return self._decision(
                    payload,
                    requesting_task_id=task_id,
                    decision=DedupDecisionKind.DUPLICATE_COMPLETED,
                )
            if (
                status == DedupStatus.CLAIMED
                and owner == task_id
                and lease_expires_at is not None
                and lease_expires_at > now
            ):
                return self._decision(
                    payload,
                    requesting_task_id=task_id,
                    decision=DedupDecisionKind.OWNED_REPLAY,
                )
            if (
                status == DedupStatus.CLAIMED
                and lease_expires_at is not None
                and lease_expires_at > now
            ):
                return self._decision(
                    payload,
                    requesting_task_id=task_id,
                    decision=DedupDecisionKind.DUPLICATE_IN_PROGRESS,
                )
            payload = self._claim_payload(
                run_id=run_id,
                kind=kind,
                key_hash=key_hash,
                normalized_key=normalized_key,
                owner_task_id=task_id,
                status=DedupStatus.CLAIMED,
                artifact_ids=artifact_ids,
                lease_expires_at=expires,
                updated_at=now,
                error=None,
            )
            self._write_claim(connection, payload)
            return self._decision(
                payload,
                requesting_task_id=task_id,
                decision=DedupDecisionKind.RETRY,
            )

    def complete_key(
        self,
        *,
        run_id: str,
        kind: DedupKind,
        normalized_key: str,
        task_id: str,
        artifact_ids: tuple[str, ...],
    ) -> DedupDecision:
        return self._resolve_key(
            run_id=run_id,
            kind=kind,
            normalized_key=normalized_key,
            task_id=task_id,
            status=DedupStatus.COMPLETED,
            artifact_ids=artifact_ids,
            error=None,
        )

    def fail_key(
        self,
        *,
        run_id: str,
        kind: DedupKind,
        normalized_key: str,
        task_id: str,
        error: str,
    ) -> DedupDecision:
        return self._resolve_key(
            run_id=run_id,
            kind=kind,
            normalized_key=normalized_key,
            task_id=task_id,
            status=DedupStatus.FAILED,
            artifact_ids=(),
            error=error[:2000],
        )

    def _resolve_key(
        self,
        *,
        run_id: str,
        kind: DedupKind,
        normalized_key: str,
        task_id: str,
        status: DedupStatus,
        artifact_ids: tuple[str, ...],
        error: str | None,
    ) -> DedupDecision:
        normalized_key = self._normalize_key(kind, normalized_key)
        key_hash = self.key_hash(normalized_key)
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT payload, checksum
                FROM research_dedup_claims
                WHERE run_id=? AND kind=? AND key_hash=?
                """,
                (run_id, kind.value, key_hash),
            ).fetchone()
            if row is None:
                raise ResearchCoordinationConflict(
                    "deduplication key was not claimed"
                )
            existing = self._verified_payload(row, "dedup claim")
            if str(existing["owner_task_id"]) != task_id:
                if (
                    DedupStatus(existing["status"]) == DedupStatus.COMPLETED
                    and status == DedupStatus.COMPLETED
                    and tuple(existing.get("artifact_ids", ()))
                    == tuple(dict.fromkeys(artifact_ids))
                ):
                    return self._decision(
                        existing,
                        requesting_task_id=task_id,
                        decision=DedupDecisionKind.DUPLICATE_COMPLETED,
                    )
                raise ResearchCoordinationConflict(
                    "only the deduplication owner may resolve a claim"
                )
            now = self.clock()
            payload = self._claim_payload(
                run_id=run_id,
                kind=kind,
                key_hash=key_hash,
                normalized_key=str(existing["normalized_key"]),
                owner_task_id=task_id,
                status=status,
                artifact_ids=tuple(dict.fromkeys(artifact_ids)),
                lease_expires_at=None,
                updated_at=now,
                error=error,
            )
            self._write_claim(connection, payload)
            return self._decision(
                payload,
                requesting_task_id=task_id,
                decision=(
                    DedupDecisionKind.DUPLICATE_COMPLETED
                    if status == DedupStatus.COMPLETED
                    else DedupDecisionKind.RETRY
                ),
            )

    def reserve_task(
        self,
        *,
        run_id: str,
        fingerprint: str,
        task_id: str,
    ) -> TaskReservation:
        if len(fingerprint) != 64:
            raise ValueError("task fingerprint must be a SHA-256 hex digest")
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT task_id, payload, checksum
                FROM research_task_reservations
                WHERE run_id=? AND fingerprint=?
                """,
                (run_id, fingerprint),
            ).fetchone()
            if row is not None:
                payload = self._verified_payload(row, "task reservation")
                canonical_task_id = str(payload["task_id"])
                return TaskReservation(
                    run_id=run_id,
                    task_id=task_id,
                    canonical_task_id=canonical_task_id,
                    fingerprint=fingerprint,
                    inserted=False,
                )
            payload = {
                "run_id": run_id,
                "fingerprint": fingerprint,
                "task_id": task_id,
                "created_at": self.clock().isoformat(),
            }
            encoded = self._json(payload)
            collision = connection.execute(
                """
                SELECT fingerprint, payload, checksum
                FROM research_task_reservations
                WHERE run_id=? AND task_id=?
                """,
                (run_id, task_id),
            ).fetchone()
            if collision is not None:
                self._verified_payload(collision, "task reservation")
                raise ResearchCoordinationConflict(
                    "task ID is already reserved for a different semantic "
                    "fingerprint"
                )
            connection.execute(
                """
                INSERT INTO research_task_reservations(
                    run_id, fingerprint, task_id, payload, checksum
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    fingerprint,
                    task_id,
                    encoded,
                    self._checksum(encoded),
                ),
            )
            return TaskReservation(
                run_id=run_id,
                task_id=task_id,
                canonical_task_id=task_id,
                fingerprint=fingerprint,
                inserted=True,
            )

    def register_novelty(
        self,
        *,
        run_id: str,
        kind: DedupKind,
        normalized_key: str,
        task_id: str,
    ) -> bool:
        normalized_key = self._normalize_key(kind, normalized_key)
        if not normalized_key:
            return False
        key_hash = self.key_hash(normalized_key)
        created_at = self.clock().isoformat()
        encoded = self._json(
            {
                "run_id": run_id,
                "kind": kind.value,
                "key_hash": key_hash,
                "normalized_key": normalized_key,
                "first_task_id": task_id,
                "created_at": created_at,
            }
        )
        with self.transaction() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO research_novelty(
                    run_id, kind, key_hash, normalized_key,
                    first_task_id, created_at, checksum
                ) VALUES(?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    kind.value,
                    key_hash,
                    normalized_key,
                    task_id,
                    created_at,
                    self._checksum(encoded),
                ),
            )
            return cursor.rowcount == 1

    def save_worker_result(self, result: ResearchWorkerResult) -> None:
        row = self._connection.execute(
            """
            SELECT worker_result_id, payload, checksum
            FROM research_worker_results
            WHERE run_id=? AND task_id=? AND task_attempt=?
            """,
            (result.run_id, result.task_id, result.task_attempt),
        ).fetchone()
        if row is not None and str(row["worker_result_id"]) != result.worker_result_id:
            self._verified_payload(row, "worker result")
            raise ResearchCoordinationConflict(
                "a task attempt already has a different Worker result"
            )
        try:
            self._save_contract(
                table="research_worker_results",
                id_column="worker_result_id",
                identity=result.worker_result_id,
                run_id=result.run_id,
                value=result,
            )
        except sqlite3.IntegrityError as exc:
            row = self._connection.execute(
                """
                SELECT worker_result_id, payload, checksum
                FROM research_worker_results
                WHERE run_id=? AND task_id=? AND task_attempt=?
                """,
                (result.run_id, result.task_id, result.task_attempt),
            ).fetchone()
            if row is None:
                raise
            payload = self._verified_payload(row, "worker result")
            if (
                str(row["worker_result_id"]) == result.worker_result_id
                and payload == result.model_dump(mode="json")
            ):
                return
            raise ResearchCoordinationConflict(
                "a task attempt already has a different Worker result"
            ) from exc

    def worker_results(self, run_id: str) -> tuple[ResearchWorkerResult, ...]:
        rows = self._connection.execute(
            """
            SELECT payload, checksum
            FROM research_worker_results
            WHERE run_id=?
            ORDER BY created_at, worker_result_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            ResearchWorkerResult.model_validate(
                self._verified_payload(row, "worker result"),
                strict=False,
            )
            for row in rows
        )

    def save_merge(self, result: MergedResearchResult) -> None:
        self._save_contract(
            table="research_merges",
            id_column="merge_id",
            identity=result.merge_id,
            run_id=result.run_id,
            value=result,
        )

    def merges(self, run_id: str) -> tuple[MergedResearchResult, ...]:
        rows = self._connection.execute(
            """
            SELECT payload, checksum
            FROM research_merges
            WHERE run_id=?
            ORDER BY created_at, merge_id
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            MergedResearchResult.model_validate(
                self._verified_payload(row, "research merge"),
                strict=False,
            )
            for row in rows
        )

    def save_convergence(self, decision: ConvergenceDecision) -> None:
        encoded = self._json(decision.model_dump(mode="json"))
        checksum = self._checksum(encoded)
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT decision_id, payload, checksum
                FROM research_convergence
                WHERE run_id=? AND cycle=?
                """,
                (decision.run_id, decision.cycle),
            ).fetchone()
            if row is not None:
                self._verify(
                    str(row["payload"]),
                    str(row["checksum"]),
                    "convergence decision",
                )
                if (
                    str(row["decision_id"]) != decision.decision_id
                    or str(row["payload"]) != encoded
                ):
                    raise ResearchCoordinationConflict(
                        "convergence cycle was reused for a different decision"
                    )
                return
            connection.execute(
                """
                INSERT INTO research_convergence(
                    decision_id, run_id, cycle, payload, checksum, created_at
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    decision.decision_id,
                    decision.run_id,
                    decision.cycle,
                    encoded,
                    checksum,
                    decision.created_at.isoformat(),
                ),
            )

    def convergence_decisions(
        self,
        run_id: str,
    ) -> tuple[ConvergenceDecision, ...]:
        rows = self._connection.execute(
            """
            SELECT payload, checksum
            FROM research_convergence
            WHERE run_id=?
            ORDER BY cycle
            """,
            (run_id,),
        ).fetchall()
        return tuple(
            ConvergenceDecision.model_validate(
                self._verified_payload(row, "convergence decision"),
                strict=False,
            )
            for row in rows
        )

    def integrity_check(self) -> None:
        check = self._connection.execute("PRAGMA quick_check").fetchone()
        if check is None or str(check[0]).casefold() != "ok":
            raise ResearchCoordinationCorruption(
                f"SQLite integrity check failed: {check[0] if check else 'missing'}"
            )
        tables = (
            ("research_dedup_claims", "dedup claim"),
            ("research_task_reservations", "task reservation"),
            ("research_worker_results", "worker result"),
            ("research_merges", "research merge"),
            ("research_convergence", "convergence decision"),
        )
        for table, label in tables:
            rows = self._connection.execute(
                f"SELECT payload, checksum FROM {table}"
            ).fetchall()
            for row in rows:
                self._verified_payload(row, label)
        rows = self._connection.execute(
            """
            SELECT run_id, kind, key_hash, normalized_key,
                   first_task_id, created_at, checksum
            FROM research_novelty
            """
        ).fetchall()
        for row in rows:
            encoded = self._json(
                {
                    "run_id": row["run_id"],
                    "kind": row["kind"],
                    "key_hash": row["key_hash"],
                    "normalized_key": row["normalized_key"],
                    "first_task_id": row["first_task_id"],
                    "created_at": row["created_at"],
                }
            )
            self._verify(encoded, str(row["checksum"]), "novelty entry")

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            output = sqlite3.connect(target)
            try:
                self._connection.backup(output)
            finally:
                output.close()
        return target

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteResearchCoordinationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _save_contract(
        self,
        *,
        table: str,
        id_column: str,
        identity: str,
        run_id: str,
        value: Any,
    ) -> None:
        encoded = self._json(value.model_dump(mode="json"))
        checksum = self._checksum(encoded)
        created_at = value.created_at.isoformat()
        with self.transaction() as connection:
            row = connection.execute(
                f"SELECT payload, checksum FROM {table} WHERE {id_column}=?",
                (identity,),
            ).fetchone()
            if row is not None:
                self._verify(str(row["payload"]), str(row["checksum"]), table)
                if str(row["payload"]) != encoded:
                    raise ResearchCoordinationConflict(
                        f"{id_column} was reused for different content"
                    )
                return
            if table == "research_worker_results":
                connection.execute(
                    f"""
                    INSERT INTO {table}(
                        {id_column}, run_id, task_id, task_attempt, payload,
                        checksum, created_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identity,
                        run_id,
                        value.task_id,
                        value.task_attempt,
                        encoded,
                        checksum,
                        created_at,
                    ),
                )
            else:
                connection.execute(
                    f"""
                    INSERT INTO {table}(
                        {id_column}, run_id, payload, checksum, created_at
                    ) VALUES(?, ?, ?, ?, ?)
                    """,
                    (identity, run_id, encoded, checksum, created_at),
                )

    def _write_claim(
        self,
        connection: sqlite3.Connection,
        payload: dict[str, Any],
    ) -> None:
        encoded = self._json(payload)
        connection.execute(
            """
            INSERT INTO research_dedup_claims(
                run_id, kind, key_hash, payload, checksum
            ) VALUES(?, ?, ?, ?, ?)
            ON CONFLICT(run_id, kind, key_hash)
            DO UPDATE SET payload=excluded.payload, checksum=excluded.checksum
            """,
            (
                payload["run_id"],
                payload["kind"],
                payload["key_hash"],
                encoded,
                self._checksum(encoded),
            ),
        )

    @staticmethod
    def _claim_payload(
        *,
        run_id: str,
        kind: DedupKind,
        key_hash: str,
        normalized_key: str,
        owner_task_id: str,
        status: DedupStatus,
        artifact_ids: tuple[str, ...],
        lease_expires_at: datetime | None,
        updated_at: datetime,
        error: str | None,
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "kind": kind.value,
            "key_hash": key_hash,
            "normalized_key": normalized_key,
            "owner_task_id": owner_task_id,
            "status": status.value,
            "artifact_ids": list(artifact_ids),
            "lease_expires_at": (
                lease_expires_at.isoformat()
                if lease_expires_at is not None
                else None
            ),
            "updated_at": updated_at.isoformat(),
            "error": error,
        }

    @staticmethod
    def _decision(
        payload: dict[str, Any],
        *,
        requesting_task_id: str,
        decision: DedupDecisionKind,
    ) -> DedupDecision:
        return DedupDecision(
            run_id=str(payload["run_id"]),
            kind=DedupKind(payload["kind"]),
            key_hash=str(payload["key_hash"]),
            normalized_key=str(payload["normalized_key"]),
            owner_task_id=str(payload["owner_task_id"]),
            requesting_task_id=requesting_task_id,
            decision=decision,
            artifact_ids=tuple(payload.get("artifact_ids", ())),
            lease_expires_at=SQLiteResearchCoordinationStore._datetime(
                payload.get("lease_expires_at")
            ),
        )

    @staticmethod
    def key_hash(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _normalize_key(kind: DedupKind, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if kind == DedupKind.QUERY:
            normalized = normalized.casefold()
        return normalized

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _verify(cls, payload: str, checksum: str, label: str) -> None:
        if cls._checksum(payload) != checksum:
            raise ResearchCoordinationCorruption(
                f"{label} checksum mismatch"
            )

    @classmethod
    def _verified_payload(
        cls,
        row: sqlite3.Row,
        label: str,
    ) -> dict[str, Any]:
        payload = str(row["payload"])
        cls._verify(payload, str(row["checksum"]), label)
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ResearchCoordinationCorruption(
                f"{label} contains invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ResearchCoordinationCorruption(
                f"{label} payload must be an object"
            )
        return value

    @staticmethod
    def _datetime(value: Any) -> datetime | None:
        if value is None:
            return None
        parsed = datetime.fromisoformat(str(value))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ResearchCoordinationCorruption(
                "stored timestamp is not timezone-aware"
            )
        return parsed
