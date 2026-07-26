from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from .models import (
    RunControl,
    SchedulerEvent,
    SchedulerEventPage,
    SchedulerEventQuery,
    SchedulerTaskPage,
    SchedulerTaskQuery,
    TaskRecord,
)


class SchedulerStoreError(RuntimeError):
    pass


class SchedulerMutationConflict(SchedulerStoreError):
    pass


class SchedulerCorruption(SchedulerStoreError):
    pass


class SQLiteSchedulerStore:
    """Append-only scheduler facts plus rebuildable task/DAG projections."""

    CURRENT_SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS scheduler_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO scheduler_schema(singleton, version) VALUES (1, 1);

                CREATE TABLE IF NOT EXISTS scheduler_events (
                    run_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    mutation_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    task_id TEXT,
                    event_type TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    occurred_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, sequence_no)
                );
                CREATE INDEX IF NOT EXISTS scheduler_events_run_task
                    ON scheduler_events(run_id, task_id, sequence_no);
                CREATE INDEX IF NOT EXISTS scheduler_events_type
                    ON scheduler_events(run_id, event_type, sequence_no);

                CREATE TABLE IF NOT EXISTS scheduler_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    max_concurrency INTEGER NOT NULL,
                    projection_revision INTEGER NOT NULL,
                    control_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS scheduler_tasks (
                    task_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES scheduler_runs(run_id) ON DELETE CASCADE,
                    status TEXT NOT NULL,
                    priority REAL NOT NULL,
                    deadline TEXT,
                    available_at TEXT NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at TEXT,
                    revision INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS scheduler_tasks_ready
                    ON scheduler_tasks(run_id, status, available_at, priority DESC, deadline);
                CREATE INDEX IF NOT EXISTS scheduler_tasks_lease
                    ON scheduler_tasks(run_id, status, lease_expires_at);

                CREATE TABLE IF NOT EXISTS scheduler_dependencies (
                    task_id TEXT NOT NULL REFERENCES scheduler_tasks(task_id) ON DELETE CASCADE,
                    dependency_task_id TEXT NOT NULL REFERENCES scheduler_tasks(task_id),
                    PRIMARY KEY(task_id, dependency_task_id),
                    CHECK(task_id <> dependency_task_id)
                );
                CREATE INDEX IF NOT EXISTS scheduler_dependencies_reverse
                    ON scheduler_dependencies(dependency_task_id, task_id);
                """
            )
            row = self._connection.execute("SELECT version FROM scheduler_schema WHERE singleton=1").fetchone()
            if row is None or int(row["version"]) != self.CURRENT_SCHEMA_VERSION:
                raise SchedulerStoreError(f"unsupported scheduler schema version: {row['version'] if row else 'missing'}")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def get_control(self, run_id: str, *, connection: sqlite3.Connection | None = None) -> RunControl | None:
        conn = connection or self._connection
        row = conn.execute("SELECT control_json, checksum FROM scheduler_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            return None
        self._verify(row["control_json"], row["checksum"], f"run {run_id}")
        return RunControl.model_validate_json(row["control_json"], strict=False)

    def get_task(self, task_id: str, *, connection: sqlite3.Connection | None = None) -> TaskRecord | None:
        conn = connection or self._connection
        row = conn.execute("SELECT record_json, checksum FROM scheduler_tasks WHERE task_id=?", (task_id,)).fetchone()
        if row is None:
            return None
        self._verify(row["record_json"], row["checksum"], f"task {task_id}")
        return TaskRecord.model_validate_json(row["record_json"], strict=False)

    def list_tasks(self, run_id: str, *, connection: sqlite3.Connection | None = None) -> tuple[TaskRecord, ...]:
        records: list[TaskRecord] = []
        cursor: str | None = None
        while True:
            page = self.list_task_page(
                SchedulerTaskQuery(
                    run_id=run_id,
                    after_task_id=cursor,
                    limit=1000,
                ),
                connection=connection,
            )
            records.extend(page.items)
            if page.next_after_task_id is None:
                break
            cursor = page.next_after_task_id
        return tuple(records)

    def list_task_page(
        self,
        query: SchedulerTaskQuery,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> SchedulerTaskPage:
        clauses = ["run_id=?"]
        parameters: list[Any] = [query.run_id]
        if query.after_task_id is not None:
            clauses.append("task_id>?")
            parameters.append(query.after_task_id)
        if query.statuses:
            placeholders = ",".join("?" for _ in query.statuses)
            clauses.append(f"status IN ({placeholders})")
            parameters.extend(status.value for status in query.statuses)
        parameters.append(query.limit + 1)
        conn = connection or self._connection
        with self._lock:
            rows = conn.execute(
                f"SELECT task_id, record_json, checksum FROM scheduler_tasks "
                f"WHERE {' AND '.join(clauses)} ORDER BY task_id LIMIT ?",
                parameters,
            ).fetchall()
        selected = rows[: query.limit]
        items: list[TaskRecord] = []
        for row in selected:
            self._verify(
                row["record_json"],
                row["checksum"],
                f"run {query.run_id} task projection",
            )
            items.append(
                TaskRecord.model_validate_json(
                    row["record_json"],
                    strict=False,
                )
            )
        cursor = (
            str(selected[-1]["task_id"])
            if len(rows) > query.limit and selected
            else None
        )
        return SchedulerTaskPage(
            items=tuple(items),
            next_after_task_id=cursor,
        )

    def mutation_event(
        self,
        mutation_id: str,
        fingerprint: str,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> SchedulerEvent | None:
        conn = connection or self._connection
        row = conn.execute(
            "SELECT event_json, checksum, fingerprint FROM scheduler_events WHERE mutation_id=?", (mutation_id,)
        ).fetchone()
        if row is None:
            return None
        if row["fingerprint"] != fingerprint:
            raise SchedulerMutationConflict(f"mutation ID was reused with different content: {mutation_id}")
        self._verify(row["event_json"], row["checksum"], f"mutation {mutation_id}")
        return SchedulerEvent.model_validate_json(row["event_json"], strict=False)

    def append_event(self, event: SchedulerEvent, *, connection: sqlite3.Connection) -> None:
        event = SchedulerEvent.model_validate(event.model_dump(mode="python"), strict=False)
        payload = self._json(event.model_dump(mode="json"))
        connection.execute(
            "INSERT INTO scheduler_events(run_id, sequence_no, event_id, mutation_id, fingerprint, task_id, "
            "event_type, actor_id, event_json, checksum, occurred_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.run_id,
                event.sequence_no,
                event.event_id,
                event.mutation_id,
                event.fingerprint,
                event.task_id,
                event.event_type.value,
                event.actor_id,
                payload,
                self._checksum(payload),
                event.occurred_at.isoformat(),
            ),
        )

    def write_control(self, control: RunControl, *, connection: sqlite3.Connection) -> None:
        control = RunControl.model_validate(control.model_dump(mode="python"), strict=False)
        payload = self._json(control.model_dump(mode="json"))
        connection.execute(
            "INSERT INTO scheduler_runs(run_id, status, max_concurrency, projection_revision, control_json, checksum, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET status=excluded.status, "
            "max_concurrency=excluded.max_concurrency, projection_revision=excluded.projection_revision, "
            "control_json=excluded.control_json, checksum=excluded.checksum, updated_at=excluded.updated_at",
            (
                control.run_id,
                control.status.value,
                control.max_concurrency,
                control.projection_revision,
                payload,
                self._checksum(payload),
                control.updated_at.isoformat(),
            ),
        )

    def write_task(self, record: TaskRecord, *, connection: sqlite3.Connection) -> None:
        record = TaskRecord.model_validate(record.model_dump(mode="python"), strict=False)
        payload = self._json(record.model_dump(mode="json"))
        envelope = record.envelope
        connection.execute(
            "INSERT INTO scheduler_tasks(task_id, run_id, status, priority, deadline, available_at, lease_owner, "
            "lease_expires_at, revision, record_json, checksum, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(task_id) DO UPDATE SET run_id=excluded.run_id, status=excluded.status, priority=excluded.priority, "
            "deadline=excluded.deadline, available_at=excluded.available_at, lease_owner=excluded.lease_owner, "
            "lease_expires_at=excluded.lease_expires_at, revision=excluded.revision, record_json=excluded.record_json, "
            "checksum=excluded.checksum, updated_at=excluded.updated_at",
            (
                envelope.task_id,
                envelope.run_id,
                envelope.status.value,
                envelope.priority,
                envelope.deadline.isoformat() if envelope.deadline else None,
                record.available_at.isoformat(),
                record.lease_owner,
                record.lease_expires_at.isoformat() if record.lease_expires_at else None,
                record.revision,
                payload,
                self._checksum(payload),
                record.updated_at.isoformat(),
            ),
        )
        connection.execute("DELETE FROM scheduler_dependencies WHERE task_id=?", (envelope.task_id,))
        connection.executemany(
            "INSERT INTO scheduler_dependencies(task_id, dependency_task_id) VALUES (?, ?)",
            ((envelope.task_id, dependency) for dependency in envelope.dependency_task_ids),
        )

    def list_events(self, run_id: str) -> tuple[SchedulerEvent, ...]:
        events: list[SchedulerEvent] = []
        cursor = 0
        while True:
            page = self.list_event_page(
                SchedulerEventQuery(
                    run_id=run_id,
                    after_sequence=cursor,
                    limit=1000,
                )
            )
            events.extend(page.items)
            if page.next_after_sequence is None:
                break
            cursor = page.next_after_sequence
        return tuple(events)

    def list_event_page(
        self,
        query: SchedulerEventQuery,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> SchedulerEventPage:
        clauses = ["run_id=?", "sequence_no>?"]
        parameters: list[Any] = [query.run_id, query.after_sequence]
        if query.event_types:
            placeholders = ",".join("?" for _ in query.event_types)
            clauses.append(f"event_type IN ({placeholders})")
            parameters.extend(event_type.value for event_type in query.event_types)
        if query.task_id is not None:
            clauses.append("task_id=?")
            parameters.append(query.task_id)
        parameters.append(query.limit + 1)
        conn = connection or self._connection
        with self._lock:
            rows = conn.execute(
                f"SELECT sequence_no, event_json, checksum FROM scheduler_events "
                f"WHERE {' AND '.join(clauses)} ORDER BY sequence_no LIMIT ?",
                parameters,
            ).fetchall()
        selected = rows[: query.limit]
        items: list[SchedulerEvent] = []
        for row in selected:
            self._verify(
                row["event_json"],
                row["checksum"],
                f"run {query.run_id} event",
            )
            items.append(
                SchedulerEvent.model_validate_json(
                    row["event_json"],
                    strict=False,
                )
            )
        cursor = (
            int(selected[-1]["sequence_no"])
            if len(rows) > query.limit and selected
            else None
        )
        return SchedulerEventPage(
            items=tuple(items),
            next_after_sequence=cursor,
        )

    def rebuild_projection(self, run_id: str) -> None:
        events = self.list_events(run_id)
        if not events:
            raise KeyError(f"unknown scheduler run: {run_id}")
        expected = list(range(1, len(events) + 1))
        if [event.sequence_no for event in events] != expected:
            raise SchedulerCorruption(f"scheduler event sequence gap for run {run_id}")
        with self.transaction() as connection:
            connection.execute("DELETE FROM scheduler_dependencies WHERE task_id IN (SELECT task_id FROM scheduler_tasks WHERE run_id=?)", (run_id,))
            connection.execute("DELETE FROM scheduler_tasks WHERE run_id=?", (run_id,))
            connection.execute("DELETE FROM scheduler_runs WHERE run_id=?", (run_id,))
            for event in events:
                payload = event.payload
                control_payload = payload.get("control")
                if control_payload is not None:
                    self.write_control(RunControl.model_validate(control_payload, strict=False), connection=connection)
                for item in payload.get("records", []):
                    self.write_task(TaskRecord.model_validate(item, strict=False), connection=connection)

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise SchedulerCorruption(f"scheduler SQLite integrity failure: {result[0] if result else 'missing'}")
            rows = self._connection.execute(
                "SELECT event_json AS value, checksum FROM scheduler_events "
                "UNION ALL SELECT control_json, checksum FROM scheduler_runs "
                "UNION ALL SELECT record_json, checksum FROM scheduler_tasks"
            ).fetchall()
            runs = self._connection.execute("SELECT run_id, projection_revision FROM scheduler_runs").fetchall()
        for row in rows:
            self._verify(row["value"], row["checksum"], "scheduler integrity")
        for run in runs:
            events = self.list_events(run["run_id"])
            if len(events) != int(run["projection_revision"]):
                raise SchedulerCorruption(f"projection revision mismatch for run {run['run_id']}")

    def backup_to(self, destination: str | Path) -> Path:
        if self.path == ":memory:":
            raise ValueError("in-memory scheduler store cannot be backed up by path")
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._connection.execute("PRAGMA wal_checkpoint(FULL)")
            backup = sqlite3.connect(target)
            try:
                self._connection.backup(backup)
            finally:
                backup.close()
        return target

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteSchedulerStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    @staticmethod
    def fingerprint(operation: str, payload: dict[str, Any]) -> str:
        return hashlib.sha256(SQLiteSchedulerStore._json([operation, payload]).encode("utf-8")).hexdigest()

    @staticmethod
    def event_id(mutation_id: str) -> str:
        return f"event_scheduler_{hashlib.sha256(mutation_id.encode('utf-8')).hexdigest()[:24]}"

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _verify(cls, value: str, checksum: str, label: str) -> None:
        if cls._checksum(value) != checksum:
            raise SchedulerCorruption(f"checksum mismatch: {label}")
