from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.contracts import EventType, RunEvent
from deep_researcher.events import EventQuery, EventStore, RunQuery
from deep_researcher.events.redaction import RedactionPolicy

from .models import (
    ProjectionConflict,
    ProjectionCorruption,
    ProjectionGap,
    ProjectionPage,
    TimelinePage,
    TimelineQuery,
)


CURRENT_STUDIO_SCHEMA_VERSION = 1
_MODEL_TERMINALS = {EventType.MODEL_COMPLETED, EventType.MODEL_FAILED}
_TOOL_TERMINALS = {EventType.TOOL_COMPLETED, EventType.TOOL_FAILED}
_SPAN_TERMINALS = {
    EventType.SPAN_COMPLETED,
    EventType.SPAN_FAILED,
    EventType.MODEL_COMPLETED,
    EventType.MODEL_FAILED,
    EventType.TOOL_COMPLETED,
    EventType.TOOL_FAILED,
    EventType.RUN_COMPLETED,
    EventType.RUN_FAILED,
    EventType.RUN_CANCELLED,
}
_SUCCESS_SPAN_TERMINALS = {
    EventType.SPAN_COMPLETED,
    EventType.MODEL_COMPLETED,
    EventType.TOOL_COMPLETED,
    EventType.RUN_COMPLETED,
}


class SQLiteStudioProjectionStore:
    """Persistent, rebuildable Studio read models derived exclusively from RunEvents."""

    def __init__(self, path: str | Path, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(str(path), timeout=30.0, isolation_level=None, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        if str(path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

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
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CURRENT_STUDIO_SCHEMA_VERSION:
            raise ProjectionCorruption("Studio projection schema is newer than this runtime")
        if version < 1:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE studio_threads (
                    thread_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    latest_run_id TEXT NOT NULL,
                    run_count INTEGER NOT NULL
                );
                CREATE TABLE studio_runs (
                    run_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    root_span_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    last_sequence INTEGER NOT NULL,
                    event_count INTEGER NOT NULL,
                    model_calls INTEGER NOT NULL,
                    tool_calls INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    retries INTEGER NOT NULL,
                    errors INTEGER NOT NULL,
                    terminal_event_id TEXT,
                    component_versions_json TEXT NOT NULL,
                    artifact_ids_json TEXT NOT NULL,
                    error_json TEXT,
                    FOREIGN KEY(thread_id) REFERENCES studio_threads(thread_id) ON DELETE RESTRICT
                );
                CREATE TABLE studio_spans (
                    run_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    span_kind TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    started_sequence INTEGER NOT NULL,
                    completed_sequence INTEGER,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    event_count INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    cost_usd REAL NOT NULL,
                    latency_ms REAL NOT NULL,
                    attempts INTEGER NOT NULL,
                    component_versions_json TEXT NOT NULL,
                    artifact_ids_json TEXT NOT NULL,
                    permission_json TEXT NOT NULL,
                    error_json TEXT,
                    PRIMARY KEY(run_id, span_id),
                    FOREIGN KEY(run_id) REFERENCES studio_runs(run_id) ON DELETE CASCADE
                );
                CREATE TABLE studio_timeline (
                    event_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    thread_id TEXT NOT NULL,
                    trace_id TEXT NOT NULL,
                    span_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    span_kind TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    level TEXT NOT NULL,
                    actor_id TEXT NOT NULL,
                    task_id TEXT,
                    occurred_at TEXT NOT NULL,
                    message TEXT NOT NULL,
                    search_text TEXT NOT NULL,
                    has_error INTEGER NOT NULL,
                    has_artifacts INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(run_id, sequence_no),
                    FOREIGN KEY(run_id) REFERENCES studio_runs(run_id) ON DELETE CASCADE
                );
                CREATE INDEX idx_studio_runs_thread_started ON studio_runs(thread_id, started_at, run_id);
                CREATE INDEX idx_studio_spans_run_kind ON studio_spans(run_id, span_kind, started_sequence);
                CREATE INDEX idx_studio_timeline_filters ON studio_timeline(run_id, event_type, span_kind, status, sequence_no);
                CREATE INDEX idx_studio_timeline_actor ON studio_timeline(run_id, actor_id, sequence_no);
                CREATE INDEX idx_studio_timeline_task ON studio_timeline(run_id, task_id, sequence_no);
                PRAGMA user_version=1;
                COMMIT;
                """
            )

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _masked(self, event: RunEvent) -> dict[str, Any]:
        payload = self.redaction_policy.redact(event.model_dump(mode="json"))
        if payload.get("error"):
            payload["error"]["message"] = self.redaction_policy.redact_text(str(payload["error"].get("message", "")))
            if payload["error"].get("details"):
                payload["error"]["details"] = self.redaction_policy.redact(payload["error"]["details"])
        return payload

    @staticmethod
    def _artifacts(payload: dict[str, Any]) -> list[str]:
        values = [*(payload.get("input_artifact_ids") or []), *(payload.get("output_artifact_ids") or [])]
        if payload.get("state_artifact_id"):
            values.append(payload["state_artifact_id"])
        return list(dict.fromkeys(str(value) for value in values))

    def apply(self, event: RunEvent) -> bool:
        masked = self._masked(event)
        serialized = json.dumps(masked, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checksum = self._checksum(serialized)
        message = str(masked.get("payload", {}).get("message") or "")
        artifacts = self._artifacts(masked)
        versions_json = json.dumps(masked.get("component_versions", {}), sort_keys=True, separators=(",", ":"))
        error_json = json.dumps(masked.get("error"), sort_keys=True, separators=(",", ":")) if masked.get("error") else None
        permission = {
            key: masked.get("payload", {}).get(key)
            for key in ("permission", "permissions", "approval", "risk_level", "policy_decision")
            if key in masked.get("payload", {})
        }
        with self._transaction() as connection:
            existing_event = connection.execute(
                "SELECT checksum FROM studio_timeline WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if existing_event is not None:
                if existing_event["checksum"] != checksum:
                    raise ProjectionConflict(f"event projection conflict: {event.event_id}")
                return False
            run = connection.execute("SELECT * FROM studio_runs WHERE run_id=?", (event.run_id,)).fetchone()
            if run is None:
                if event.sequence_no != 1 or event.event_type != EventType.RUN_STARTED:
                    raise ProjectionGap(f"projection requires run_started at sequence 1: {event.run_id}")
                thread = connection.execute("SELECT * FROM studio_threads WHERE thread_id=?", (event.thread_id,)).fetchone()
                if thread is None:
                    connection.execute(
                        "INSERT INTO studio_threads VALUES (?, ?, ?, ?, 1)",
                        (event.thread_id, event.occurred_at.isoformat(), event.occurred_at.isoformat(), event.run_id),
                    )
                else:
                    connection.execute(
                        "UPDATE studio_threads SET updated_at=?, latest_run_id=?, run_count=run_count+1 WHERE thread_id=?",
                        (event.occurred_at.isoformat(), event.run_id, event.thread_id),
                    )
                connection.execute(
                    """INSERT INTO studio_runs VALUES (?, ?, ?, ?, ?, ?, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, ?, '[]', NULL)""",
                    (event.run_id, event.thread_id, event.trace_id, event.span_id, event.status.value, event.occurred_at.isoformat(), versions_json),
                )
                run = connection.execute("SELECT * FROM studio_runs WHERE run_id=?", (event.run_id,)).fetchone()
            if event.thread_id != run["thread_id"] or event.trace_id != run["trace_id"]:
                raise ProjectionConflict("run projection identity changed")
            expected = int(run["last_sequence"]) + 1
            if event.sequence_no != expected:
                raise ProjectionGap(f"expected sequence {expected}, got {event.sequence_no}")

            connection.execute(
                """INSERT INTO studio_timeline VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id, event.run_id, event.sequence_no, event.thread_id, event.trace_id,
                    event.span_id, event.parent_span_id, event.span_kind.value, event.event_type.value,
                    event.status.value, event.level.value, event.actor_id, event.task_id,
                    event.occurred_at.isoformat(), message,
                    " ".join((event.event_type.value, event.actor_id, event.task_id or "", message, serialized)).casefold(),
                    int(event.error is not None), int(bool(artifacts)), serialized, checksum,
                ),
            )
            span = connection.execute(
                "SELECT * FROM studio_spans WHERE run_id=? AND span_id=?", (event.run_id, event.span_id)
            ).fetchone()
            if span is None:
                connection.execute(
                    """INSERT INTO studio_spans VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, NULL, 0, 0, 0, 0, 0, ?, ?, ?, ?, ?)""",
                    (
                        event.run_id, event.span_id, event.parent_span_id, event.span_kind.value,
                        event.actor_id, event.status.value, event.sequence_no, event.occurred_at.isoformat(),
                        event.attempt, versions_json, json.dumps(artifacts), json.dumps(permission), error_json,
                    ),
                )
            elif event.parent_span_id != span["parent_span_id"] or event.span_kind.value != span["span_kind"]:
                raise ProjectionConflict(f"span projection identity changed: {event.span_id}")

            is_model = event.event_type in _MODEL_TERMINALS
            is_tool = event.event_type in _TOOL_TERMINALS
            usage = event.usage
            terminal = event.event_type in _SPAN_TERMINALS
            span_status = (
                "succeeded" if event.event_type in _SUCCESS_SPAN_TERMINALS
                else "failed" if event.event_type in {EventType.SPAN_FAILED, EventType.MODEL_FAILED, EventType.TOOL_FAILED, EventType.RUN_FAILED}
                else "cancelled" if event.event_type == EventType.RUN_CANCELLED
                else event.status.value
            )
            current_span_artifacts = json.loads((span or {"artifact_ids_json": "[]"})["artifact_ids_json"])
            span_artifacts = list(dict.fromkeys([*current_span_artifacts, *artifacts]))
            span_permission = {
                **json.loads((span or {"permission_json": "{}"})["permission_json"]),
                **permission,
            }
            connection.execute(
                """UPDATE studio_spans SET status=?, completed_sequence=CASE WHEN ? THEN ? ELSE completed_sequence END,
                   completed_at=CASE WHEN ? THEN ? ELSE completed_at END, event_count=event_count+1,
                   input_tokens=input_tokens+?, output_tokens=output_tokens+?, cost_usd=cost_usd+?,
                   latency_ms=latency_ms+?, attempts=MAX(attempts, ?), component_versions_json=?,
                   artifact_ids_json=?, permission_json=CASE WHEN ?!='{}' THEN ? ELSE permission_json END,
                   error_json=COALESCE(?, error_json) WHERE run_id=? AND span_id=?""",
                (
                    span_status, int(terminal), event.sequence_no, int(terminal), event.occurred_at.isoformat(),
                    usage.input_tokens if is_model else 0, usage.output_tokens if is_model else 0,
                    usage.cost_usd if is_model else 0.0, event.latency_ms if (is_model or is_tool) else 0.0,
                    event.attempt, versions_json, json.dumps(span_artifacts), json.dumps(span_permission),
                    json.dumps(span_permission), error_json, event.run_id, event.span_id,
                ),
            )
            run_artifacts = list(dict.fromkeys([*json.loads(run["artifact_ids_json"]), *artifacts]))
            terminal_run = event.event_type in {EventType.RUN_COMPLETED, EventType.RUN_FAILED, EventType.RUN_CANCELLED}
            connection.execute(
                """UPDATE studio_runs SET status=CASE WHEN ? THEN ? ELSE status END, completed_at=CASE WHEN ? THEN ? ELSE completed_at END,
                   last_sequence=?, event_count=event_count+1, model_calls=model_calls+?, tool_calls=tool_calls+?,
                   input_tokens=input_tokens+?, output_tokens=output_tokens+?, cost_usd=cost_usd+?,
                   latency_ms=latency_ms+?, retries=retries+?, errors=errors+?,
                   terminal_event_id=CASE WHEN ? THEN ? ELSE terminal_event_id END,
                   component_versions_json=?, artifact_ids_json=?, error_json=COALESCE(?, error_json)
                   WHERE run_id=?""",
                (
                    int(terminal_run), event.status.value, int(terminal_run), event.occurred_at.isoformat(), event.sequence_no,
                    int(is_model), int(is_tool), usage.input_tokens if is_model else 0,
                    usage.output_tokens if is_model else 0, usage.cost_usd if is_model else 0.0,
                    event.latency_ms if (is_model or is_tool) else 0.0,
                    int(event.event_type == EventType.RETRY_SCHEDULED), int(event.error is not None),
                    int(terminal_run), event.event_id, versions_json, json.dumps(run_artifacts), error_json, event.run_id,
                ),
            )
            connection.execute(
                """UPDATE studio_threads SET updated_at=MAX(updated_at, ?),
                   latest_run_id=(SELECT run_id FROM studio_runs WHERE thread_id=? ORDER BY started_at DESC, run_id DESC LIMIT 1)
                   WHERE thread_id=?""",
                (event.occurred_at.isoformat(), event.thread_id, event.thread_id),
            )
        return True

    @staticmethod
    def _decode_row(row: sqlite3.Row, json_fields: tuple[str, ...]) -> dict[str, Any]:
        value = dict(row)
        for field in json_fields:
            raw = value.pop(field, None)
            value[field.removesuffix("_json")] = json.loads(raw) if raw is not None else None
        return value

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM studio_threads WHERE thread_id=?", (thread_id,)).fetchone()
        return dict(row) if row else None

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute("SELECT * FROM studio_runs WHERE run_id=?", (run_id,)).fetchone()
        return self._decode_row(row, ("component_versions_json", "artifact_ids_json", "error_json")) if row else None

    def list_threads(self, *, after_created_at: str | None = None, after_thread_id: str | None = None, limit: int = 100) -> ProjectionPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if (after_created_at is None) != (after_thread_id is None):
            raise ValueError("thread cursor fields must be supplied together")
        clauses = ["1=1"]
        params: list[Any] = []
        if after_created_at is not None:
            clauses.append("(created_at>? OR (created_at=? AND thread_id>?))")
            params.extend((after_created_at, after_created_at, after_thread_id))
        params.append(limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM studio_threads WHERE {' AND '.join(clauses)} ORDER BY created_at, thread_id LIMIT ?", params
            ).fetchall()
        selected = rows[:limit]
        items = tuple(dict(row) for row in selected)
        cursor = (selected[-1]["created_at"], selected[-1]["thread_id"]) if len(rows) > limit else None
        return ProjectionPage(items=items, next_cursor=cursor)

    def list_runs(self, *, thread_id: str | None = None, after_started_at: str | None = None, after_run_id: str | None = None, limit: int = 100) -> ProjectionPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if (after_started_at is None) != (after_run_id is None):
            raise ValueError("run cursor fields must be supplied together")
        clauses = ["1=1"]
        params: list[Any] = []
        if thread_id:
            clauses.append("thread_id=?")
            params.append(thread_id)
        if after_started_at is not None:
            clauses.append("(started_at>? OR (started_at=? AND run_id>?))")
            params.extend((after_started_at, after_started_at, after_run_id))
        params.append(limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT * FROM studio_runs WHERE {' AND '.join(clauses)} ORDER BY started_at, run_id LIMIT ?", params
            ).fetchall()
        selected = rows[:limit]
        items = tuple(self._decode_row(row, ("component_versions_json", "artifact_ids_json", "error_json")) for row in selected)
        cursor = (selected[-1]["started_at"], selected[-1]["run_id"]) if len(rows) > limit else None
        return ProjectionPage(items=items, next_cursor=cursor)

    def list_spans(self, run_id: str, *, after_started_sequence: int = 0, limit: int = 100) -> ProjectionPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM studio_spans WHERE run_id=? AND started_sequence>? ORDER BY started_sequence, span_id LIMIT ?",
                (run_id, after_started_sequence, limit + 1),
            ).fetchall()
        selected = rows[:limit]
        items = tuple(self._decode_row(row, ("component_versions_json", "artifact_ids_json", "permission_json", "error_json")) for row in selected)
        cursor = (str(selected[-1]["started_sequence"]), selected[-1]["span_id"]) if len(rows) > limit else None
        return ProjectionPage(items=items, next_cursor=cursor)

    def timeline(self, query: TimelineQuery) -> TimelinePage:
        clauses = ["run_id=?", "sequence_no>?"]
        params: list[Any] = [query.run_id, query.after_sequence]
        for column, values in (("event_type", query.event_types), ("span_kind", query.span_kinds), ("status", query.statuses)):
            if values:
                placeholders = ",".join("?" for _ in values)
                clauses.append(f"{column} IN ({placeholders})")
                params.extend(values)
        for column, value in (("actor_id", query.actor_id), ("task_id", query.task_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if query.text:
            escaped = query.text.casefold().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("search_text LIKE ? ESCAPE '\\'")
            params.append(f"%{escaped}%")
        if query.error_only:
            clauses.append("has_error=1")
        if query.has_artifacts is not None:
            clauses.append("has_artifacts=?")
            params.append(int(query.has_artifacts))
        if query.occurred_from:
            clauses.append("occurred_at>=?")
            params.append(query.occurred_from.isoformat())
        if query.occurred_to:
            clauses.append("occurred_at<=?")
            params.append(query.occurred_to.isoformat())
        params.append(query.limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT event_json, checksum FROM studio_timeline WHERE {' AND '.join(clauses)} ORDER BY sequence_no LIMIT ?", params
            ).fetchall()
        selected = rows[:query.limit]
        items = []
        for row in selected:
            if self._checksum(row["event_json"]) != row["checksum"]:
                raise ProjectionCorruption("timeline checksum mismatch")
            items.append(json.loads(row["event_json"]))
        cursor = int(items[-1]["sequence_no"]) if len(rows) > query.limit else None
        return TimelinePage(items=tuple(items), next_after_sequence=cursor)

    def delete_run(self, run_id: str) -> None:
        with self._transaction() as connection:
            row = connection.execute("SELECT thread_id FROM studio_runs WHERE run_id=?", (run_id,)).fetchone()
            if row is None:
                return
            thread_id = row["thread_id"]
            connection.execute("DELETE FROM studio_runs WHERE run_id=?", (run_id,))
            remaining = connection.execute(
                "SELECT run_id, started_at FROM studio_runs WHERE thread_id=? ORDER BY started_at DESC, run_id DESC", (thread_id,)
            ).fetchall()
            if not remaining:
                connection.execute("DELETE FROM studio_threads WHERE thread_id=?", (thread_id,))
            else:
                connection.execute(
                    "UPDATE studio_threads SET latest_run_id=?, run_count=? WHERE thread_id=?",
                    (remaining[0]["run_id"], len(remaining), thread_id),
                )

    def integrity_check(self) -> None:
        with self._lock:
            if self._connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ProjectionCorruption("Studio SQLite integrity check failed")
            if self._connection.execute("PRAGMA foreign_key_check").fetchall():
                raise ProjectionCorruption("Studio projection foreign key failure")
            for run in self._connection.execute("SELECT * FROM studio_runs").fetchall():
                rows = self._connection.execute(
                    "SELECT * FROM studio_timeline WHERE run_id=? ORDER BY sequence_no", (run["run_id"],)
                ).fetchall()
                if [row["sequence_no"] for row in rows] != list(range(1, int(run["last_sequence"]) + 1)):
                    raise ProjectionCorruption(f"non-contiguous projected timeline: {run['run_id']}")
                if int(run["event_count"]) != len(rows):
                    raise ProjectionCorruption(f"run event count mismatch: {run['run_id']}")
                for row in rows:
                    if self._checksum(row["event_json"]) != row["checksum"]:
                        raise ProjectionCorruption(f"timeline checksum mismatch: {row['event_id']}")
            for span in self._connection.execute("SELECT * FROM studio_spans").fetchall():
                count = self._connection.execute(
                    "SELECT COUNT(*) FROM studio_timeline WHERE run_id=? AND span_id=?",
                    (span["run_id"], span["span_id"]),
                ).fetchone()[0]
                if int(span["event_count"]) != int(count):
                    raise ProjectionCorruption(f"span event count mismatch: {span['run_id']}/{span['span_id']}")
            for thread in self._connection.execute("SELECT * FROM studio_threads").fetchall():
                runs = self._connection.execute(
                    "SELECT run_id FROM studio_runs WHERE thread_id=? ORDER BY started_at DESC, run_id DESC",
                    (thread["thread_id"],),
                ).fetchall()
                if len(runs) != int(thread["run_count"]):
                    raise ProjectionCorruption(f"thread run count mismatch: {thread['thread_id']}")
                if not runs or runs[0]["run_id"] != thread["latest_run_id"]:
                    raise ProjectionCorruption(f"thread latest run mismatch: {thread['thread_id']}")

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class StudioProjector:
    def __init__(self, event_store: EventStore, projection_store: SQLiteStudioProjectionStore) -> None:
        self.event_store = event_store
        self.projection_store = projection_store

    def sync_run(self, run_id: str) -> int:
        projected = self.projection_store.get_run(run_id)
        cursor = int(projected["last_sequence"]) if projected else 0
        applied = 0
        while True:
            page = self.event_store.list(EventQuery(run_id, after_sequence=cursor, limit=1000))
            for event in page.items:
                applied += int(self.projection_store.apply(event))
                cursor = event.sequence_no
            if page.next_after_sequence is None:
                break
        return applied

    def sync_all(self) -> int:
        cursor = None
        applied = 0
        while True:
            page = self.event_store.list_runs(
                RunQuery(
                    after_created_at=cursor[0] if cursor else None,
                    after_run_id=cursor[1] if cursor else None,
                    limit=1000,
                )
            )
            for run in page.items:
                applied += self.sync_run(run.run_id)
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return applied

    def rebuild_run(self, run_id: str) -> int:
        self.projection_store.delete_run(run_id)
        return self.sync_run(run_id)

    def export_trace(self, run_id: str) -> dict[str, Any]:
        self.sync_run(run_id)
        run = self.projection_store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        spans: list[dict[str, Any]] = []
        span_cursor = 0
        while True:
            page = self.projection_store.list_spans(run_id, after_started_sequence=span_cursor, limit=1000)
            spans.extend(page.items)
            if page.next_cursor is None:
                break
            span_cursor = int(page.next_cursor[0])
        events: list[dict[str, Any]] = []
        sequence = 0
        while True:
            page = self.projection_store.timeline(TimelineQuery(run_id, after_sequence=sequence, limit=1000))
            events.extend(page.items)
            if page.next_after_sequence is None:
                break
            sequence = page.next_after_sequence
        return {
            "schema": "StudioTraceExport@1",
            "thread": self.projection_store.get_thread(run["thread_id"]),
            "run": run,
            "spans": spans,
            "events": events,
        }


class StudioProjectionExporter:
    name = "studio_projection"

    def __init__(self, store: SQLiteStudioProjectionStore) -> None:
        self.store = store

    def export(self, event: RunEvent) -> None:
        self.store.apply(event)
