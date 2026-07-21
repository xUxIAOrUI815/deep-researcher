from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sqlite3
import threading

import httpx
import pytest

from deep_researcher.contracts import (
    ComponentKind,
    ComponentVersionSet,
    ErrorCategory,
    ErrorRecord,
    EventType,
    RunEvent,
    RunStatus,
    SpanKind,
    VersionRef,
    utc_now,
)
from deep_researcher.events import (
    DuplicateEventConflict,
    EventExportError,
    EventQuery,
    EventRecorder,
    EventStoreCorruption,
    OTLPHTTPConfig,
    OTLPHTTPEventExporter,
    RunQuery,
    RunTerminalError,
    SQLiteEventStore,
    SequenceConflict,
    SpanLifecycleError,
)


def _version(kind: ComponentKind, name: str) -> VersionRef:
    return VersionRef(kind=kind, name=name, version="1.0.0")


VERSIONS = ComponentVersionSet(
    runtime=_version(ComponentKind.RUNTIME, "test-runtime"),
    scheduler=_version(ComponentKind.SCHEDULER, "test-scheduler"),
)


def _event(
    sequence: int,
    event_type: EventType,
    *,
    run_id: str = "run_event_store",
    span_id: str = "span_root",
    parent_span_id: str | None = None,
    span_kind: SpanKind = SpanKind.RUN,
    status: RunStatus = RunStatus.RUNNING,
    event_id: str | None = None,
    payload: dict | None = None,
    error: ErrorRecord | None = None,
) -> RunEvent:
    values = {
        "sequence_no": sequence,
        "event_type": event_type,
        "status": status,
        "trace_id": f"trace_{run_id}",
        "span_id": span_id,
        "parent_span_id": parent_span_id,
        "span_kind": span_kind,
        "correlation_id": f"correlation_{run_id}",
        "run_id": run_id,
        "thread_id": "thread_event_store",
        "actor_id": "actor_test",
        "producer_id": "producer_test",
        "component_versions": VERSIONS,
        "payload": payload or {},
        "error": error,
    }
    if event_id is not None:
        values["event_id"] = event_id
    return RunEvent(**values)


def _start(store: SQLiteEventStore, run_id: str = "run_event_store") -> RunEvent:
    event = _event(1, EventType.RUN_STARTED, run_id=run_id)
    store.append(event)
    return event


def _complete(store: SQLiteEventStore, sequence: int, run_id: str = "run_event_store") -> RunEvent:
    event = _event(sequence, EventType.RUN_COMPLETED, run_id=run_id, status=RunStatus.SUCCEEDED)
    store.append(event)
    return event


def test_append_is_idempotent_but_rejects_event_id_content_reuse(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        started = _start(store)
        duplicate = store.append(started)
        assert duplicate.inserted is False
        changed = started.model_copy(update={"payload": {"different": True}})
        with pytest.raises(DuplicateEventConflict):
            store.append(changed)
        assert len(store.list(EventQuery(started.run_id)).items) == 1


def test_run_sequence_and_single_terminal_are_transactionally_enforced(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        _start(store)
        with pytest.raises(SequenceConflict):
            store.append(_event(3, EventType.CUSTOM))
        _complete(store, 2)
        with pytest.raises(RunTerminalError):
            store.append(_event(3, EventType.CUSTOM))
        store.integrity_check()


def test_span_lifecycle_rejects_orphans_and_open_children(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        _start(store)
        with pytest.raises(SpanLifecycleError, match="orphan"):
            store.append(
                _event(
                    2,
                    EventType.CUSTOM,
                    span_id="span_missing",
                    parent_span_id="span_root",
                    span_kind=SpanKind.TOOL,
                )
            )
        child_start = _event(
            2,
            EventType.TOOL_STARTED,
            span_id="span_tool",
            parent_span_id="span_root",
            span_kind=SpanKind.TOOL,
        )
        store.append(child_start)
        with pytest.raises(SpanLifecycleError, match="child span"):
            _complete(store, 3)
        store.append(
            _event(
                3,
                EventType.TOOL_COMPLETED,
                span_id="span_tool",
                parent_span_id="span_root",
                span_kind=SpanKind.TOOL,
            )
        )
        _complete(store, 4)
        store.integrity_check()


def test_pagination_and_filters_are_stable_for_long_runs(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        _start(store)
        for sequence in range(2, 27):
            store.append(_event(sequence, EventType.TASK_STATE_CHANGED))
        _complete(store, 27)
        first = store.list(EventQuery("run_event_store", limit=10, event_types=(EventType.TASK_STATE_CHANGED,)))
        second = store.list(
            EventQuery(
                "run_event_store",
                after_sequence=first.next_after_sequence or 0,
                limit=10,
                event_types=(EventType.TASK_STATE_CHANGED,),
            )
        )
        third = store.list(
            EventQuery(
                "run_event_store",
                after_sequence=second.next_after_sequence or 0,
                limit=10,
                event_types=(EventType.TASK_STATE_CHANGED,),
            )
        )
        sequences = [event.sequence_no for page in (first, second, third) for event in page.items]
        assert sequences == list(range(2, 27))
        assert third.next_after_sequence is None


def test_run_catalog_supports_status_filter_and_stable_cursor(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        _start(store, "run_catalog_a")
        _complete(store, 2, "run_catalog_a")
        _start(store, "run_catalog_b")
        first = store.list_runs(RunQuery(thread_id="thread_event_store", limit=1))
        assert len(first.items) == 1
        assert first.next_cursor is not None
        second = store.list_runs(
            RunQuery(
                thread_id="thread_event_store",
                after_created_at=first.next_cursor[0],
                after_run_id=first.next_cursor[1],
                limit=1,
            )
        )
        assert {first.items[0].run_id, second.items[0].run_id} == {"run_catalog_a", "run_catalog_b"}
        running = store.list_runs(RunQuery(statuses=(RunStatus.RUNNING,)))
        assert [run.run_id for run in running.items] == ["run_catalog_b"]
        assert store.get_run("run_catalog_a").terminal_event_id is not None


def test_concurrent_append_allocates_a_contiguous_run_sequence(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        _start(store)
        barrier = threading.Barrier(8)

        def append_many(worker: int) -> None:
            barrier.wait()
            for index in range(10):
                event = _event(
                    store.next_sequence("run_event_store"),
                    EventType.CUSTOM,
                    payload={"worker": worker, "index": index},
                )
                while True:
                    try:
                        store.append(event)
                        break
                    except SequenceConflict:
                        event = event.model_copy(
                            update={"sequence_no": store.next_sequence("run_event_store")}
                        )

        with ThreadPoolExecutor(max_workers=8) as executor:
            list(executor.map(append_many, range(8)))
        _complete(store, 82)
        events = store.list(EventQuery("run_event_store", limit=100)).items
        assert [event.sequence_no for event in events] == list(range(1, 83))
        store.integrity_check()


def test_concurrent_store_instances_serialize_cross_connection_writes(tmp_path):
    database = tmp_path / "events.sqlite3"
    primary = SQLiteEventStore(database)
    _start(primary)
    stores = [SQLiteEventStore(database) for _ in range(4)]
    barrier = threading.Barrier(4)

    def append_from_connection(worker: int) -> None:
        store = stores[worker]
        barrier.wait()
        for index in range(5):
            event = _event(
                store.next_sequence("run_event_store"),
                EventType.CUSTOM,
                payload={"connection": worker, "index": index},
            )
            while True:
                try:
                    store.append(event)
                    break
                except SequenceConflict:
                    event = event.model_copy(
                        update={"sequence_no": store.next_sequence("run_event_store")}
                    )

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append_from_connection, range(4)))
    for store in stores:
        store.close()
    _complete(primary, 22)
    assert [event.sequence_no for event in primary.list(EventQuery("run_event_store", limit=100)).items] == list(range(1, 23))
    primary.integrity_check()
    primary.close()


def test_restart_backup_and_restore_preserve_all_events(tmp_path):
    database = tmp_path / "events.sqlite3"
    backup = tmp_path / "events.backup.sqlite3"
    restored_path = tmp_path / "events.restored.sqlite3"
    store = SQLiteEventStore(database)
    root = _start(store)
    store.append(_event(2, EventType.TASK_CREATED))
    _complete(store, 3)
    store.backup_to(backup)
    store.close()

    restarted = SQLiteEventStore(database)
    assert restarted.get(root.event_id) == root
    restarted.integrity_check()
    restarted.close()

    restored = SQLiteEventStore.restore_backup(backup, restored_path)
    try:
        assert [event.sequence_no for event in restored.list(EventQuery(root.run_id)).items] == [1, 2, 3]
    finally:
        restored.close()


def test_checksum_corruption_is_detected_on_read_and_audit(tmp_path):
    database = tmp_path / "events.sqlite3"
    store = SQLiteEventStore(database)
    root = _start(store)
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("UPDATE events SET event_json='{}' WHERE event_id=?", (root.event_id,))
    connection.commit()
    connection.close()
    reopened = SQLiteEventStore(database)
    try:
        with pytest.raises(EventStoreCorruption, match="checksum"):
            reopened.get(root.event_id)
        with pytest.raises(EventStoreCorruption, match="checksum"):
            reopened.integrity_check()
    finally:
        reopened.close()


def test_v1_database_migrates_to_checksum_and_export_outbox(tmp_path):
    database = tmp_path / "events.sqlite3"
    store = SQLiteEventStore(database)
    root = _start(store)
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE event_exports")
    connection.execute("ALTER TABLE events DROP COLUMN checksum")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()
    migrated = SQLiteEventStore(database)
    try:
        columns = {row[1] for row in migrated._connection.execute("PRAGMA table_info(events)")}
        assert "checksum" in columns
        assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert migrated._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='event_exports'"
        ).fetchone()
        assert migrated.get(root.event_id) == root
        migrated.integrity_check()
    finally:
        migrated.close()


def test_store_redacts_secrets_and_decision_payloads_before_commit(tmp_path):
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        root = _event(
            1,
            EventType.RUN_STARTED,
            payload={
                "authorization": "Bearer top-secret-token",
                "nested": {"api_key": "sk-abcdefghijklmnop", "safe": "visible"},
            },
        )
        stored = store.append(root).event
        assert stored.payload["authorization"] == "[REDACTED]"
        assert stored.payload["nested"]["api_key"] == "[REDACTED]"
        decision = _event(
            2,
            EventType.DECISION_RECORDED,
            payload={
                "observation_summary": "Sources disagree.",
                "selected_command_ids": ["command_search"],
                "raw_model_response": "must not persist",
                "metadata": {"safe": True, "raw_model_response": "also forbidden"},
            },
        )
        stored_decision = store.append(decision).event
        assert "raw_model_response" not in stored_decision.payload
        assert "raw_model_response" not in stored_decision.payload["metadata"]
        assert stored_decision.payload["observation_summary"] == "Sources disagree."


class _FlakyExporter:
    name = "flaky"

    def __init__(self) -> None:
        self.calls = 0

    def export(self, event: RunEvent) -> None:
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("collector unavailable")


def test_export_failure_never_loses_local_event_and_is_restart_retryable(tmp_path):
    database = tmp_path / "events.sqlite3"
    exporter = _FlakyExporter()
    store = SQLiteEventStore(database)
    recorder = EventRecorder(store, (exporter,))
    root = _event(1, EventType.RUN_STARTED)
    outcome = recorder.record(root)
    assert outcome.append.inserted is True
    assert "flaky" in outcome.failed_exports
    assert store.get(root.event_id) is not None
    assert store.pending_exports()[0].attempts == 1
    store.close()

    restarted = SQLiteEventStore(database)
    retrying = EventRecorder(restarted, (exporter,))
    retried = retrying.retry_pending()
    assert retried[0].exported_to == ("flaky",)
    assert restarted.pending_exports() == ()
    restarted.close()


def test_otlp_http_exporter_emits_standard_log_envelope():
    captured: list[dict] = []

    def handle(request: httpx.Request) -> httpx.Response:
        captured.append(json.loads(request.content))
        assert request.url.path == "/v1/logs"
        return httpx.Response(200)

    client = httpx.Client(transport=httpx.MockTransport(handle))
    exporter = OTLPHTTPEventExporter(OTLPHTTPConfig(endpoint="https://collector.test"), client=client)
    exporter.export(_event(1, EventType.RUN_STARTED))
    record = captured[0]["resourceLogs"][0]["scopeLogs"][0]["logRecords"][0]
    assert len(record["traceId"]) == 32
    assert len(record["spanId"]) == 16
    assert any(item["key"] == "event.sequence" for item in record["attributes"])
    client.close()


def test_otlp_http_exporter_classifies_transport_failures():
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    client = httpx.Client(transport=httpx.MockTransport(handle))
    exporter = OTLPHTTPEventExporter(OTLPHTTPConfig(endpoint="https://collector.test/v1/logs"), client=client)
    with pytest.raises(EventExportError, match="OTLP export failed"):
        exporter.export(_event(1, EventType.RUN_STARTED))
    client.close()
