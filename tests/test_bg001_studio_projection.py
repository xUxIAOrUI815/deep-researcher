from __future__ import annotations

import json
import sqlite3

import pytest

from deep_researcher.contracts import EventType, RunStatus, SpanKind
from deep_researcher.events import EventRecorder, SQLiteEventStore
from deep_researcher.studio import (
    ProjectionCorruption,
    ProjectionGap,
    SQLiteStudioProjectionStore,
    StudioProjector,
    StudioProjectionExporter,
    TimelineQuery,
)
from tests.test_bg001_event_store import _event


def _build_trace(event_store: SQLiteEventStore, count: int = 25) -> str:
    run_id = "run_studio"
    event_store.append(_event(1, EventType.RUN_STARTED, run_id=run_id, payload={"authorization": "Bearer top-secret"}))
    for sequence in range(2, count + 2):
        event_store.append(_event(
            sequence, EventType.TASK_STATE_CHANGED, run_id=run_id,
            payload={"message": f"timeline needle {sequence}", "permission": "read"},
        ))
    event_store.append(_event(count + 2, EventType.RUN_COMPLETED, run_id=run_id, status=RunStatus.SUCCEEDED))
    return run_id


def test_projection_is_persistent_paginated_searchable_masked_and_rebuildable(tmp_path):
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    run_id = _build_trace(event_store, count=2050)
    projection = SQLiteStudioProjectionStore(tmp_path / "studio.sqlite3")
    projector = StudioProjector(event_store, projection)
    assert projector.sync_all() == 2052
    run = projection.get_run(run_id)
    assert run["event_count"] == 2052
    assert run["status"] == "succeeded"
    assert projection.get_thread(run["thread_id"])["run_count"] == 1

    sequences = []
    cursor = 0
    while True:
        page = projection.timeline(TimelineQuery(run_id, after_sequence=cursor, limit=137))
        sequences.extend(item["sequence_no"] for item in page.items)
        if page.next_after_sequence is None:
            break
        cursor = page.next_after_sequence
    assert sequences == list(range(1, 2053))
    searched = projection.timeline(TimelineQuery(run_id, text="needle 1777", limit=10))
    assert [item["sequence_no"] for item in searched.items] == [1777]
    assert not projection.timeline(TimelineQuery(run_id, text="top-secret", limit=10)).items
    assert "top-secret" not in json.dumps(projector.export_trace(run_id))
    projection.integrity_check()
    projection.close()

    restarted = SQLiteStudioProjectionStore(tmp_path / "studio.sqlite3")
    restarted_projector = StudioProjector(event_store, restarted)
    assert restarted_projector.sync_run(run_id) == 0
    before = restarted_projector.export_trace(run_id)
    assert restarted_projector.rebuild_run(run_id) == 2052
    assert restarted_projector.export_trace(run_id) == before
    restarted.integrity_check()
    restarted.close()
    event_store.close()


def test_span_details_filters_artifacts_usage_permissions_and_errors(tmp_path):
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteStudioProjectionStore(tmp_path / "studio.sqlite3")
    run_id = "run_details"
    root = _event(1, EventType.RUN_STARTED, run_id=run_id)
    event_store.append(root)
    started = _event(
        2, EventType.TOOL_STARTED, run_id=run_id, span_id="span_tool",
        parent_span_id="span_root", span_kind=SpanKind.TOOL,
        payload={"message": "tool request", "permission": "network", "risk_level": "low"},
    ).model_copy(update={"input_artifact_ids": ("artifact_input",)})
    event_store.append(started)
    completed = _event(
        3, EventType.TOOL_COMPLETED, run_id=run_id, span_id="span_tool",
        parent_span_id="span_root", span_kind=SpanKind.TOOL,
        payload={"message": "tool response", "approval": "approved"},
    ).model_copy(update={"output_artifact_ids": ("artifact_output",), "latency_ms": 42.5, "attempt": 2})
    event_store.append(completed)
    event_store.append(_event(4, EventType.RUN_COMPLETED, run_id=run_id, status=RunStatus.SUCCEEDED))
    projector = StudioProjector(event_store, projection)
    projector.sync_run(run_id)
    spans = projection.list_spans(run_id, limit=10).items
    tool = next(item for item in spans if item["span_kind"] == "tool")
    assert tool["latency_ms"] == 42.5
    assert tool["attempts"] == 2
    assert tool["status"] == "succeeded"
    assert tool["artifact_ids"] == ["artifact_input", "artifact_output"]
    assert tool["permission"] == {"permission": "network", "risk_level": "low", "approval": "approved"}
    filtered = projection.timeline(TimelineQuery(
        run_id, event_types=("tool_completed",), span_kinds=("tool",),
        actor_id="actor_test", has_artifacts=True,
    ))
    assert [item["event_id"] for item in filtered.items] == [completed.event_id]
    trace = projector.export_trace(run_id)
    assert trace["run"]["tool_calls"] == 1
    assert trace["events"][2]["attempt"] == 2
    projection.close()
    event_store.close()


def test_projection_rejects_gaps_and_detects_corruption(tmp_path):
    database = tmp_path / "studio.sqlite3"
    projection = SQLiteStudioProjectionStore(database)
    with pytest.raises(ProjectionGap):
        projection.apply(_event(2, EventType.TASK_CREATED, run_id="run_gap"))
    projection.apply(_event(1, EventType.RUN_STARTED, run_id="run_corrupt"))
    projection.close()
    connection = sqlite3.connect(database)
    connection.execute("UPDATE studio_timeline SET event_json='{}'")
    connection.commit()
    connection.close()
    reopened = SQLiteStudioProjectionStore(database)
    try:
        with pytest.raises(ProjectionCorruption, match="checksum"):
            reopened.timeline(TimelineQuery("run_corrupt"))
        with pytest.raises(ProjectionCorruption, match="checksum"):
            reopened.integrity_check()
    finally:
        reopened.close()


def test_projection_delivery_failure_is_outboxed_and_retryable(tmp_path, monkeypatch):
    event_store = SQLiteEventStore(tmp_path / "events.sqlite3")
    projection = SQLiteStudioProjectionStore(tmp_path / "studio.sqlite3")
    exporter = StudioProjectionExporter(projection)
    recorder = EventRecorder(event_store, (exporter,))
    original = projection.apply
    calls = 0

    def fail_once(event):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("projection temporarily unavailable")
        return original(event)

    monkeypatch.setattr(projection, "apply", fail_once)
    root = _event(1, EventType.RUN_STARTED, run_id="run_outbox")
    outcome = recorder.record(root)
    assert outcome.append.inserted
    assert "studio_projection" in outcome.failed_exports
    assert projection.get_run(root.run_id) is None
    assert event_store.pending_exports(exporter_name="studio_projection")
    retried = recorder.retry_pending(exporter_name="studio_projection")
    assert retried[0].exported_to == ("studio_projection",)
    assert projection.get_run(root.run_id)["event_count"] == 1
    assert not event_store.pending_exports(exporter_name="studio_projection")
    projection.close()
    event_store.close()
