from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.contracts import EventType, RunEvent, RunStatus, canonical_contract_json

from .redaction import RedactionPolicy
from .store import (
    AppendResult,
    DuplicateEventConflict,
    EventPage,
    EventQuery,
    EventStoreCorruption,
    PendingExport,
    RunPage,
    RunQuery,
    RunRecord,
    RunTerminalError,
    SequenceConflict,
    SpanLifecycleError,
)


CURRENT_SCHEMA_VERSION = 2
_RUN_TERMINAL_TYPES = {EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}
_SPAN_START_TYPES = {EventType.SPAN_STARTED, EventType.MODEL_STARTED, EventType.TOOL_STARTED}
_SPAN_TERMINAL_TYPES = {
    EventType.SPAN_COMPLETED,
    EventType.SPAN_FAILED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_FAILED,
    EventType.TOOL_COMPLETED,
    EventType.TOOL_FAILED,
}


class SQLiteEventStore:
    """Transactional append-only event store with run and span invariants."""

    def __init__(self, path: str | Path, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self._lock = threading.RLock()
        self._connection = self._connect(str(path))
        self._migrate()

    @staticmethod
    def _connect(path: str) -> sqlite3.Connection:
        connection = sqlite3.connect(path, timeout=30.0, isolation_level=None, check_same_thread=False)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        if path != ":memory:":
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _migrate(self) -> None:
        with self._lock:
            version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
            if version > CURRENT_SCHEMA_VERSION:
                raise EventStoreCorruption(
                    f"event store schema {version} is newer than supported {CURRENT_SCHEMA_VERSION}"
                )
            if version < 1:
                self._migration_1()
                self._connection.execute("PRAGMA user_version=1")
                version = 1
            if version < 2:
                self._migration_2()
                self._connection.execute("PRAGMA user_version=2")

    def _migration_1(self) -> None:
        self._connection.executescript(
            """
            BEGIN IMMEDIATE;
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                thread_id TEXT NOT NULL,
                trace_id TEXT NOT NULL,
                root_span_id TEXT NOT NULL,
                status TEXT NOT NULL,
                next_sequence INTEGER NOT NULL,
                terminal_event_id TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY(terminal_event_id) REFERENCES events(event_id)
            );
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                sequence_no INTEGER NOT NULL,
                trace_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                event_type TEXT NOT NULL,
                status TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                task_id TEXT,
                occurred_at TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                event_json TEXT NOT NULL,
                UNIQUE(run_id, sequence_no)
            );
            CREATE TABLE IF NOT EXISTS spans (
                run_id TEXT NOT NULL,
                span_id TEXT NOT NULL,
                parent_span_id TEXT,
                span_kind TEXT NOT NULL,
                started_event_id TEXT NOT NULL,
                terminal_event_id TEXT,
                is_open INTEGER NOT NULL CHECK(is_open IN (0, 1)),
                PRIMARY KEY(run_id, span_id),
                FOREIGN KEY(started_event_id) REFERENCES events(event_id),
                FOREIGN KEY(terminal_event_id) REFERENCES events(event_id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_run_type_seq ON events(run_id, event_type, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_events_trace_span_seq ON events(trace_id, span_id, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_events_run_actor_seq ON events(run_id, actor_id, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_events_run_task_seq ON events(run_id, task_id, sequence_no);
            CREATE INDEX IF NOT EXISTS idx_events_occurred ON events(occurred_at);
            CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(run_id, parent_span_id);
            COMMIT;
            """
        )

    def _migration_2(self) -> None:
        columns = {row[1] for row in self._connection.execute("PRAGMA table_info(events)").fetchall()}
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            if "checksum" not in columns:
                self._connection.execute("ALTER TABLE events ADD COLUMN checksum TEXT NOT NULL DEFAULT ''")
            for row in self._connection.execute("SELECT event_id, event_json FROM events WHERE checksum = ''").fetchall():
                checksum = hashlib.sha256(row["event_json"].encode("utf-8")).hexdigest()
                self._connection.execute("UPDATE events SET checksum=? WHERE event_id=?", (checksum, row["event_id"]))
            self._connection.execute(
                """CREATE TABLE IF NOT EXISTS event_exports (
                    event_id TEXT NOT NULL,
                    exporter_name TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('pending', 'exported')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    last_error TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(event_id, exporter_name),
                    FOREIGN KEY(event_id) REFERENCES events(event_id) ON DELETE RESTRICT
                )"""
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_event_exports_pending ON event_exports(exporter_name, status, updated_at)"
            )
            self._connection.execute("COMMIT")
        except Exception:
            self._connection.execute("ROLLBACK")
            raise

    @staticmethod
    def _checksum(serialized: str) -> str:
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode(serialized: str) -> RunEvent:
        return RunEvent.model_validate_json(serialized)

    def append(self, event: RunEvent, *, export_targets: tuple[str, ...] = ()) -> AppendResult:
        event = self.redaction_policy.redact_event(event)
        serialized = canonical_contract_json(event)
        checksum = self._checksum(serialized)
        targets = tuple(dict.fromkeys(name.strip() for name in export_targets if name.strip()))
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT event_json, checksum FROM events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if existing is not None:
                if existing["checksum"] != checksum or existing["event_json"] != serialized:
                    raise DuplicateEventConflict(f"event ID reused with different content: {event.event_id}")
                self._enqueue_exports(connection, event.event_id, targets)
                return AppendResult(event=self._decode(existing["event_json"]), inserted=False)

            run = connection.execute("SELECT * FROM runs WHERE run_id=?", (event.run_id,)).fetchone()
            if run is None:
                if event.sequence_no != 1:
                    raise SequenceConflict("a new run must start at sequence 1")
                if event.event_type != EventType.RUN_STARTED:
                    raise RunTerminalError("the first event of a run must be run_started")
                connection.execute(
                    "INSERT INTO runs(run_id, thread_id, trace_id, root_span_id, status, next_sequence, created_at) VALUES (?, ?, ?, ?, ?, 1, ?)",
                    (event.run_id, event.thread_id, event.trace_id, event.span_id, RunStatus.RUNNING.value, event.occurred_at.isoformat()),
                )
                run = connection.execute("SELECT * FROM runs WHERE run_id=?", (event.run_id,)).fetchone()
            else:
                if run["terminal_event_id"] is not None:
                    raise RunTerminalError(f"run is already terminal: {event.run_id}")
                if event.thread_id != run["thread_id"] or event.trace_id != run["trace_id"]:
                    raise EventStoreCorruption("run thread/trace identity changed")
            if event.sequence_no != int(run["next_sequence"]):
                raise SequenceConflict(
                    f"expected sequence {run['next_sequence']} for {event.run_id}, got {event.sequence_no}"
                )

            self._validate_span_before_insert(connection, event, run)
            connection.execute(
                """
                INSERT INTO events(
                    event_id, run_id, sequence_no, trace_id, span_id,
                    parent_span_id, event_type, status, actor_id, task_id,
                    occurred_at, recorded_at, event_json, checksum
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    event.run_id,
                    event.sequence_no,
                    event.trace_id,
                    event.span_id,
                    event.parent_span_id,
                    event.event_type.value,
                    event.status.value,
                    event.actor_id,
                    event.task_id,
                    event.occurred_at.isoformat(),
                    event.recorded_at.isoformat(),
                    serialized,
                    checksum,
                ),
            )
            self._update_span_after_insert(connection, event, run)
            terminal_id = event.event_id if event.event_type in _RUN_TERMINAL_TYPES else None
            status = event.status.value if terminal_id else RunStatus.RUNNING.value
            connection.execute(
                "UPDATE runs SET next_sequence=?, status=?, terminal_event_id=COALESCE(?, terminal_event_id) WHERE run_id=?",
                (event.sequence_no + 1, status, terminal_id, event.run_id),
            )
            self._enqueue_exports(connection, event.event_id, targets)
        return AppendResult(event=event, inserted=True)

    def _validate_span_before_insert(self, connection: sqlite3.Connection, event: RunEvent, run: sqlite3.Row) -> None:
        span = connection.execute(
            "SELECT * FROM spans WHERE run_id=? AND span_id=?", (event.run_id, event.span_id)
        ).fetchone()
        if event.event_type == EventType.RUN_STARTED:
            if event.parent_span_id is not None or event.span_id != run["root_span_id"]:
                raise SpanLifecycleError("run root span must have no parent and match the run root")
            if span is not None:
                raise SpanLifecycleError("root span already exists")
            return
        if event.event_type in _SPAN_START_TYPES:
            if span is not None:
                raise SpanLifecycleError("span already started")
            if event.parent_span_id is None:
                raise SpanLifecycleError("non-root spans require a parent")
            parent = connection.execute(
                "SELECT * FROM spans WHERE run_id=? AND span_id=?", (event.run_id, event.parent_span_id)
            ).fetchone()
            if parent is None or not parent["is_open"]:
                raise SpanLifecycleError("parent span is missing or closed")
            return
        if span is None:
            raise SpanLifecycleError(f"event references an orphan span: {event.span_id}")
        if not span["is_open"]:
            raise SpanLifecycleError(f"event references a closed span: {event.span_id}")
        if event.parent_span_id != span["parent_span_id"]:
            raise SpanLifecycleError("event parent span does not match span lifecycle")
        if event.event_type in _SPAN_TERMINAL_TYPES or event.event_type in _RUN_TERMINAL_TYPES:
            open_child = connection.execute(
                "SELECT 1 FROM spans WHERE run_id=? AND parent_span_id=? AND is_open=1 LIMIT 1",
                (event.run_id, event.span_id),
            ).fetchone()
            if open_child is not None:
                raise SpanLifecycleError("a span cannot close while a child span is open")
        if event.event_type in _RUN_TERMINAL_TYPES and event.span_id != run["root_span_id"]:
            raise SpanLifecycleError("run terminal event must close the root span")

    def _update_span_after_insert(self, connection: sqlite3.Connection, event: RunEvent, run: sqlite3.Row) -> None:
        if event.event_type == EventType.RUN_STARTED or event.event_type in _SPAN_START_TYPES:
            connection.execute(
                "INSERT INTO spans(run_id, span_id, parent_span_id, span_kind, started_event_id, is_open) VALUES (?, ?, ?, ?, ?, 1)",
                (event.run_id, event.span_id, event.parent_span_id, event.span_kind.value, event.event_id),
            )
        if event.event_type in _SPAN_TERMINAL_TYPES or event.event_type in _RUN_TERMINAL_TYPES:
            connection.execute(
                "UPDATE spans SET is_open=0, terminal_event_id=? WHERE run_id=? AND span_id=?",
                (event.event_id, event.run_id, event.span_id),
            )

    @staticmethod
    def _enqueue_exports(connection: sqlite3.Connection, event_id: str, targets: tuple[str, ...]) -> None:
        connection.executemany(
            "INSERT OR IGNORE INTO event_exports(event_id, exporter_name, status) VALUES (?, ?, 'pending')",
            ((event_id, target) for target in targets),
        )

    def get(self, event_id: str) -> RunEvent | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT event_json, checksum FROM events WHERE event_id=?", (event_id,)
            ).fetchone()
        if row is None:
            return None
        if self._checksum(row["event_json"]) != row["checksum"]:
            raise EventStoreCorruption(f"checksum mismatch for event {event_id}")
        return self._decode(row["event_json"])

    def list(self, query: EventQuery) -> EventPage:
        clauses = ["run_id=?", "sequence_no>?"]
        params: list[Any] = [query.run_id, query.after_sequence]
        if query.event_types:
            placeholders = ",".join("?" for _ in query.event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_type.value for event_type in query.event_types)
        for column, value in (
            ("trace_id", query.trace_id),
            ("span_id", query.span_id),
            ("actor_id", query.actor_id),
            ("task_id", query.task_id),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if query.occurred_from is not None:
            clauses.append("occurred_at>=?")
            params.append(query.occurred_from.isoformat())
        if query.occurred_to is not None:
            clauses.append("occurred_at<=?")
            params.append(query.occurred_to.isoformat())
        params.append(query.limit + 1)
        sql = f"SELECT event_json, checksum FROM events WHERE {' AND '.join(clauses)} ORDER BY sequence_no LIMIT ?"
        with self._lock:
            rows = self._connection.execute(sql, params).fetchall()
        has_more = len(rows) > query.limit
        selected = rows[: query.limit]
        events: list[RunEvent] = []
        for row in selected:
            if self._checksum(row["event_json"]) != row["checksum"]:
                raise EventStoreCorruption("event checksum mismatch during pagination")
            events.append(self._decode(row["event_json"]))
        cursor = events[-1].sequence_no if has_more and events else None
        return EventPage(items=tuple(events), next_after_sequence=cursor)

    def next_sequence(self, run_id: str) -> int:
        with self._lock:
            row = self._connection.execute("SELECT next_sequence FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return int(row["next_sequence"]) if row is not None else 1

    @staticmethod
    def _run_record(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            thread_id=row["thread_id"],
            trace_id=row["trace_id"],
            root_span_id=row["root_span_id"],
            status=RunStatus(row["status"]),
            next_sequence=int(row["next_sequence"]),
            terminal_event_id=row["terminal_event_id"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return self._run_record(row) if row is not None else None

    def list_runs(self, query: RunQuery) -> RunPage:
        clauses = ["1=1"]
        params: list[Any] = []
        if query.thread_id is not None:
            clauses.append("thread_id=?")
            params.append(query.thread_id)
        if query.statuses:
            placeholders = ",".join("?" for _ in query.statuses)
            clauses.append(f"status IN ({placeholders})")
            params.extend(status.value for status in query.statuses)
        if query.after_created_at is not None and query.after_run_id is not None:
            clauses.append("(created_at>? OR (created_at=? AND run_id>?))")
            timestamp = query.after_created_at.isoformat()
            params.extend((timestamp, timestamp, query.after_run_id))
        params.append(query.limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM runs WHERE {' AND '.join(clauses)} ORDER BY created_at, run_id LIMIT ?",
                params,
            ).fetchall()
        has_more = len(rows) > query.limit
        records = tuple(self._run_record(row) for row in rows[: query.limit])
        cursor = (records[-1].created_at, records[-1].run_id) if has_more and records else None
        return RunPage(items=records, next_cursor=cursor)

    def mark_exported(self, event_id: str, exporter_name: str) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE event_exports SET status='exported', attempts=attempts+1, last_error=NULL, updated_at=CURRENT_TIMESTAMP WHERE event_id=? AND exporter_name=?",
                (event_id, exporter_name),
            ).rowcount
            if not changed:
                raise KeyError(f"unknown export target: {event_id}/{exporter_name}")

    def mark_export_failed(self, event_id: str, exporter_name: str, error: str) -> None:
        with self._transaction() as connection:
            changed = connection.execute(
                "UPDATE event_exports SET status='pending', attempts=attempts+1, last_error=?, updated_at=CURRENT_TIMESTAMP WHERE event_id=? AND exporter_name=?",
                (error[:2000], event_id, exporter_name),
            ).rowcount
            if not changed:
                raise KeyError(f"unknown export target: {event_id}/{exporter_name}")

    def pending_exports(self, *, exporter_name: str | None = None, limit: int = 100) -> tuple[PendingExport, ...]:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        clauses = ["x.status='pending'"]
        params: list[Any] = []
        if exporter_name is not None:
            clauses.append("x.exporter_name=?")
            params.append(exporter_name)
        params.append(limit)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT e.event_json, e.checksum, x.exporter_name, x.attempts, x.last_error FROM event_exports x JOIN events e ON e.event_id=x.event_id WHERE {' AND '.join(clauses)} ORDER BY e.run_id, e.sequence_no LIMIT ?",
                params,
            ).fetchall()
        return tuple(
            PendingExport(
                event=self._decode(row["event_json"]),
                exporter_name=row["exporter_name"],
                attempts=int(row["attempts"]),
                last_error=row["last_error"],
            )
            for row in rows
        )

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise EventStoreCorruption(f"SQLite integrity check failed: {result}")
            foreign_key_errors = self._connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise EventStoreCorruption(f"foreign key integrity failed: {foreign_key_errors[0]}")
            rows = self._connection.execute(
                "SELECT * FROM events ORDER BY run_id, sequence_no"
            ).fetchall()
            expected_by_run: dict[str, int] = {}
            for row in rows:
                if self._checksum(row["event_json"]) != row["checksum"]:
                    raise EventStoreCorruption(f"checksum mismatch for event {row['event_id']}")
                event = self._decode(row["event_json"])
                expected = expected_by_run.get(row["run_id"], 1)
                if row["sequence_no"] != expected or event.sequence_no != expected:
                    raise EventStoreCorruption(f"non-contiguous sequence in run {row['run_id']}")
                if (
                    event.event_id != row["event_id"]
                    or event.run_id != row["run_id"]
                    or event.trace_id != row["trace_id"]
                    or event.span_id != row["span_id"]
                    or event.event_type.value != row["event_type"]
                ):
                    raise EventStoreCorruption(f"indexed event columns disagree with payload: {row['event_id']}")
                expected_by_run[row["run_id"]] = expected + 1
            for run in self._connection.execute("SELECT * FROM runs").fetchall():
                if int(run["next_sequence"]) != expected_by_run.get(run["run_id"], 1):
                    raise EventStoreCorruption(f"run sequence cursor mismatch: {run['run_id']}")
                terminals = self._connection.execute(
                    "SELECT event_id FROM events WHERE run_id=? AND event_type IN (?, ?, ?)",
                    (
                        run["run_id"],
                        EventType.RUN_COMPLETED.value,
                        EventType.RUN_FAILED.value,
                        EventType.RUN_CANCELLED.value,
                    ),
                ).fetchall()
                if len(terminals) > 1:
                    raise EventStoreCorruption(f"multiple terminal events in run {run['run_id']}")
                terminal_id = terminals[0]["event_id"] if terminals else None
                if terminal_id != run["terminal_event_id"]:
                    raise EventStoreCorruption(f"run terminal pointer mismatch: {run['run_id']}")
                if terminal_id is not None:
                    open_span = self._connection.execute(
                        "SELECT 1 FROM spans WHERE run_id=? AND is_open=1 LIMIT 1", (run["run_id"],)
                    ).fetchone()
                    if open_span is not None:
                        raise EventStoreCorruption(f"terminal run has open spans: {run['run_id']}")
            for span in self._connection.execute("SELECT * FROM spans").fetchall():
                started = self._connection.execute(
                    "SELECT event_type FROM events WHERE event_id=?", (span["started_event_id"],)
                ).fetchone()
                if started is None or started["event_type"] not in {
                    EventType.RUN_STARTED.value,
                    *(event_type.value for event_type in _SPAN_START_TYPES),
                }:
                    raise EventStoreCorruption(f"span has invalid start event: {span['span_id']}")
                if bool(span["is_open"]) == (span["terminal_event_id"] is not None):
                    raise EventStoreCorruption(f"span terminal state mismatch: {span['span_id']}")

    def backup_to(self, destination: str | Path) -> None:
        destination_path = Path(destination).resolve()
        if str(self.path) != ":memory:" and destination_path == self.path.resolve():
            raise ValueError("backup destination must differ from the active database")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(str(destination_path))
            try:
                self._connection.backup(target)
            finally:
                target.close()
        verification = SQLiteEventStore(destination_path, redaction_policy=self.redaction_policy)
        try:
            verification.integrity_check()
        finally:
            verification.close()

    @classmethod
    def restore_backup(cls, backup: str | Path, destination: str | Path) -> "SQLiteEventStore":
        backup_path = Path(backup).resolve()
        destination_path = Path(destination).resolve()
        if backup_path == destination_path:
            raise ValueError("backup and restore destination must differ")
        source = sqlite3.connect(f"file:{backup_path.as_posix()}?mode=ro", uri=True)
        try:
            result = source.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise EventStoreCorruption(f"backup integrity check failed: {result}")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            target = sqlite3.connect(str(destination_path))
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        restored = cls(destination_path)
        restored.integrity_check()
        return restored

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteEventStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
