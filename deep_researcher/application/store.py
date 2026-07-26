from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Iterator

from .models import ApplicationRunRecord, ApplicationRunStatus


class ApplicationStoreError(RuntimeError):
    pass


class ApplicationStoreConflict(ApplicationStoreError):
    pass


class ApplicationStoreCorruption(ApplicationStoreError):
    pass


class SQLiteApplicationStore:
    """Durable run catalog with append-only transition journal."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA busy_timeout=30000")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialize()

    def _initialize(self) -> None:
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS application_runs(
                    research_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS application_run_journal(
                    research_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    PRIMARY KEY(research_id, revision),
                    FOREIGN KEY(research_id)
                      REFERENCES application_runs(research_id)
                );
                CREATE INDEX IF NOT EXISTS idx_application_runs_updated
                  ON application_runs(updated_at DESC, research_id);
                CREATE TRIGGER IF NOT EXISTS application_journal_no_update
                  BEFORE UPDATE ON application_run_journal
                  BEGIN SELECT RAISE(ABORT, 'application journal is immutable'); END;
                CREATE TRIGGER IF NOT EXISTS application_journal_no_delete
                  BEFORE DELETE ON application_run_journal
                  BEGIN SELECT RAISE(ABORT, 'application journal is immutable'); END;
                """
            )

    @staticmethod
    def _serialize(record: ApplicationRunRecord) -> tuple[str, str]:
        payload = json.dumps(
            record.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return payload, hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Cursor]:
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def create(self, record: ApplicationRunRecord) -> ApplicationRunRecord:
        if record.revision != 0:
            raise ValueError("new application run must start at revision zero")
        payload, checksum = self._serialize(record)
        with self._transaction() as cursor:
            existing = cursor.execute(
                "SELECT record_json, checksum FROM application_runs "
                "WHERE research_id=? OR run_id=?",
                (record.research_id, record.run_id),
            ).fetchone()
            if existing is not None:
                if existing["checksum"] == checksum:
                    return ApplicationRunRecord.model_validate_json(
                        existing["record_json"],
                        strict=False,
                    )
                raise ApplicationStoreConflict(
                    "run identity is already bound to different content"
                )
            cursor.execute(
                "INSERT INTO application_runs("
                "research_id, run_id, status, revision, record_json, checksum, updated_at"
                ") VALUES(?,?,?,?,?,?,?)",
                (
                    record.research_id,
                    record.run_id,
                    record.status.value,
                    record.revision,
                    payload,
                    checksum,
                    record.updated_at.isoformat(),
                ),
            )
            cursor.execute(
                "INSERT INTO application_run_journal("
                "research_id, revision, status, record_json, checksum, occurred_at"
                ") VALUES(?,?,?,?,?,?)",
                (
                    record.research_id,
                    record.revision,
                    record.status.value,
                    payload,
                    checksum,
                    record.updated_at.isoformat(),
                ),
            )
        return record

    def transition(
        self,
        research_id: str,
        *,
        status: ApplicationRunStatus,
        current_stage: str,
        report_artifact_id: str | None = None,
        error_code: str | None = None,
        error_message: str | None = None,
        resumed: bool | None = None,
        metadata: dict[str, object] | None = None,
    ) -> ApplicationRunRecord:
        with self._transaction() as cursor:
            row = cursor.execute(
                "SELECT record_json, checksum FROM application_runs "
                "WHERE research_id=?",
                (research_id,),
            ).fetchone()
            if row is None:
                raise KeyError(research_id)
            self._validate_row(row)
            current = ApplicationRunRecord.model_validate_json(
                row["record_json"],
                strict=False,
            )
            terminal = {
                ApplicationRunStatus.COMPLETED,
                ApplicationRunStatus.FAILED,
                ApplicationRunStatus.CANCELLED,
            }
            allowed = {
                ApplicationRunStatus.QUEUED: {
                    ApplicationRunStatus.RUNNING,
                    ApplicationRunStatus.CANCELLED,
                },
                ApplicationRunStatus.RUNNING: {
                    ApplicationRunStatus.RUNNING,
                    ApplicationRunStatus.WAITING_APPROVAL,
                    ApplicationRunStatus.COMPLETED,
                    ApplicationRunStatus.FAILED,
                    ApplicationRunStatus.CANCELLED,
                },
                ApplicationRunStatus.WAITING_APPROVAL: {
                    ApplicationRunStatus.QUEUED,
                    ApplicationRunStatus.FAILED,
                    ApplicationRunStatus.CANCELLED,
                },
            }
            if current.status in terminal:
                if current.status == status:
                    return current
                raise ApplicationStoreConflict(
                    f"terminal run cannot transition from "
                    f"{current.status.value} to {status.value}"
                )
            if status not in allowed[current.status]:
                raise ApplicationStoreConflict(
                    f"invalid application run transition: "
                    f"{current.status.value} -> {status.value}"
                )
            updated = current.model_copy(
                update={
                    "status": status,
                    "current_stage": current_stage,
                    "report_artifact_id": (
                        report_artifact_id
                        if report_artifact_id is not None
                        else current.report_artifact_id
                    ),
                    "error_code": error_code,
                    "error_message": error_message,
                    "resumed": current.resumed if resumed is None else resumed,
                    "revision": current.revision + 1,
                    "updated_at": datetime.now(timezone.utc),
                    "metadata": {
                        **current.metadata,
                        **(metadata or {}),
                    },
                }
            )
            payload, checksum = self._serialize(updated)
            cursor.execute(
                "UPDATE application_runs SET status=?, revision=?, record_json=?, "
                "checksum=?, updated_at=? WHERE research_id=? AND revision=?",
                (
                    updated.status.value,
                    updated.revision,
                    payload,
                    checksum,
                    updated.updated_at.isoformat(),
                    research_id,
                    current.revision,
                ),
            )
            if cursor.rowcount != 1:
                raise ApplicationStoreConflict(
                    "run transition lost an optimistic concurrency race"
                )
            cursor.execute(
                "INSERT INTO application_run_journal("
                "research_id, revision, status, record_json, checksum, occurred_at"
                ") VALUES(?,?,?,?,?,?)",
                (
                    research_id,
                    updated.revision,
                    updated.status.value,
                    payload,
                    checksum,
                    updated.updated_at.isoformat(),
                ),
            )
        return updated

    def get(self, research_id: str) -> ApplicationRunRecord | None:
        row = self.connection.execute(
            "SELECT record_json, checksum FROM application_runs WHERE research_id=?",
            (research_id,),
        ).fetchone()
        if row is None:
            return None
        self._validate_row(row)
        return ApplicationRunRecord.model_validate_json(
            row["record_json"],
            strict=False,
        )

    def get_by_run_id(self, run_id: str) -> ApplicationRunRecord | None:
        row = self.connection.execute(
            "SELECT record_json, checksum FROM application_runs WHERE run_id=?",
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        self._validate_row(row)
        return ApplicationRunRecord.model_validate_json(
            row["record_json"],
            strict=False,
        )

    def list(self, *, limit: int = 100) -> tuple[ApplicationRunRecord, ...]:
        if limit < 1 or limit > 10_000:
            raise ValueError("limit must be between 1 and 10000")
        rows = self.connection.execute(
            "SELECT record_json, checksum FROM application_runs "
            "ORDER BY updated_at DESC, research_id LIMIT ?",
            (limit,),
        ).fetchall()
        result: list[ApplicationRunRecord] = []
        for row in rows:
            self._validate_row(row)
            result.append(
                ApplicationRunRecord.model_validate_json(
                    row["record_json"],
                    strict=False,
                )
            )
        return tuple(result)

    def history(self, research_id: str) -> tuple[ApplicationRunRecord, ...]:
        rows = self.connection.execute(
            "SELECT record_json, checksum FROM application_run_journal "
            "WHERE research_id=? ORDER BY revision",
            (research_id,),
        ).fetchall()
        result: list[ApplicationRunRecord] = []
        for row in rows:
            self._validate_row(row)
            result.append(
                ApplicationRunRecord.model_validate_json(
                    row["record_json"],
                    strict=False,
                )
            )
        return tuple(result)

    @staticmethod
    def _validate_row(row: sqlite3.Row) -> None:
        actual = hashlib.sha256(row["record_json"].encode("utf-8")).hexdigest()
        if actual != row["checksum"]:
            raise ApplicationStoreCorruption(
                "application run checksum mismatch"
            )

    def integrity_check(self) -> None:
        result = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise ApplicationStoreCorruption(str(result))
        for record in self.list(limit=10_000):
            history = self.history(record.research_id)
            if not history or history[-1] != record:
                raise ApplicationStoreCorruption(
                    f"run journal/projection mismatch: {record.research_id}"
                )
            revisions = [item.revision for item in history]
            if revisions != list(range(len(revisions))):
                raise ApplicationStoreCorruption(
                    f"non-contiguous run journal: {record.research_id}"
                )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SQLiteApplicationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
