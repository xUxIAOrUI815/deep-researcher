from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re

from deep_researcher.application import (
    ApplicationRunEventController,
    ApplicationRunRecord,
    ApplicationRunStatus,
    SQLiteApplicationStore,
    application_component_versions,
)
from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    BudgetUsage,
    EventType,
    RunEvent,
    RunStatus,
    SpanKind,
)
from deep_researcher.events import (
    EventQuery,
    EventRecorder,
    SQLiteEventStore,
)


def _record() -> ApplicationRunRecord:
    return ApplicationRunRecord(
        research_id="research_lock_stress",
        thread_id="thread_lock_stress",
        session_id="session_lock_stress",
        run_id="run_lock_stress",
        trace_id="trace_lock_stress",
        root_task_id="task_lock_stress",
        report_id="report_lock_stress",
        query="Exercise concurrent durable transitions.",
    )


def test_application_store_serializes_concurrent_writers_and_journals_all(
    tmp_path: Path,
):
    database = tmp_path / "application.sqlite3"
    initial = SQLiteApplicationStore(database)
    initial.create(_record())
    initial.close()

    def transition(index: int) -> int:
        store = SQLiteApplicationStore(database)
        try:
            return store.transition(
                "research_lock_stress",
                status=ApplicationRunStatus.RUNNING,
                current_stage=f"concurrent_writer_{index}",
            ).revision
        finally:
            store.close()

    with ThreadPoolExecutor(max_workers=12) as executor:
        revisions = list(executor.map(transition, range(24)))

    reopened = SQLiteApplicationStore(database)
    try:
        assert sorted(revisions) == list(range(1, 25))
        history = reopened.history("research_lock_stress")
        assert [item.revision for item in history] == list(range(25))
        assert history[-1].status == ApplicationRunStatus.RUNNING
        reopened.integrity_check()
    finally:
        reopened.close()


def test_large_report_and_structured_secret_artifact_round_trip(
    tmp_path: Path,
):
    body = (
        "# Verified long report\n\n"
        + ("Evidence-backed paragraph with citation [C1].\n" * 50_000)
    )
    assert len(body.encode("utf-8")) > 2_000_000
    with SQLiteArtifactStore(tmp_path / "artifacts.sqlite3") as store:
        report = store.put_text(
            body,
            kind=ArtifactKind.REPORT,
            producer_id="runtime_reporting",
            run_id="run_large_report",
            idempotency_key="large-report",
        )
        secret = store.put_json(
            {
                "authorization": "Bearer top-secret-token",
                "nested": {
                    "api_key": "sk-12345678901234567890",
                    "safe": "retained",
                },
            },
            kind=ArtifactKind.MODEL_INPUT,
            producer_id="runtime_security_test",
            run_id="run_large_report",
        )
        assert store.read_bytes(report.artifact_id).decode("utf-8") == body
        secret_payload = store.read_json(secret.artifact_id)
        assert secret_payload == {
            "authorization": "[REDACTED]",
            "nested": {
                "api_key": "[REDACTED]",
                "safe": "retained",
            },
        }
        store.integrity_check()


def test_long_trace_paginates_and_redacts_without_losing_terminal_event(
    tmp_path: Path,
):
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    recorder = EventRecorder(store)
    controller = ApplicationRunEventController(
        recorder,
        run_id="run_long_trace",
        thread_id="thread_long_trace",
        trace_id="trace_long_trace",
        correlation_id="correlation_long_trace",
        component_versions=application_component_versions(),
    )
    try:
        controller.start(
            query=(
                "Trace request with "
                "sk-12345678901234567890 that must be redacted."
            ),
            research_id="research_long_trace",
        )
        for index in range(1_505):
            controller.stage(
                "stress",
                payload={
                    "index": index,
                    "authorization": "Bearer top-secret-token",
                },
            )
        controller.complete(
            output_artifact_ids=("artifact_long_report",),
            usage=BudgetUsage(model_calls=25, tool_calls=40),
        )

        first = store.list(EventQuery("run_long_trace", limit=1000))
        assert first.next_after_sequence is not None
        second = store.list(
            EventQuery(
                "run_long_trace",
                after_sequence=first.next_after_sequence,
                limit=1000,
            )
        )
        events = (*first.items, *second.items)
        assert len(events) == 1_507
        assert events[0].event_type == EventType.RUN_STARTED
        assert events[-1].event_type == EventType.RUN_COMPLETED
        serialized = "\n".join(
            item.model_dump_json() for item in events
        )
        assert "sk-123" not in serialized
        assert "top-secret-token" not in serialized
        assert serialized.count("[REDACTED]") >= 1_506
        store.integrity_check()
    finally:
        store.close()


def test_resumed_controller_closes_detached_child_spans_before_cancel(
    tmp_path: Path,
):
    store = SQLiteEventStore(tmp_path / "detached-spans.sqlite3")
    recorder = EventRecorder(store)
    controller = ApplicationRunEventController(
        recorder,
        run_id="run_detached_spans",
        thread_id="thread_detached_spans",
        trace_id="trace_detached_spans",
        correlation_id="correlation_detached_spans",
        component_versions=application_component_versions(),
    )
    controller.start(
        query="Recover stale spans.",
        research_id="research_detached_spans",
    )
    root = store.get_run("run_detached_spans").root_span_id
    versions = application_component_versions()
    for sequence, event_type, span_id, parent, kind in (
        (2, EventType.SPAN_STARTED, "span_detached_agent", root, SpanKind.AGENT),
        (
            3,
            EventType.MODEL_STARTED,
            "span_detached_model",
            "span_detached_agent",
            SpanKind.MODEL,
        ),
    ):
        store.append(
            RunEvent(
                sequence_no=sequence,
                event_type=event_type,
                status=RunStatus.RUNNING,
                trace_id="trace_detached_spans",
                span_id=span_id,
                parent_span_id=parent,
                span_kind=kind,
                correlation_id="correlation_detached_spans",
                run_id="run_detached_spans",
                thread_id="thread_detached_spans",
                task_id="task_detached_spans",
                actor_id="agent_detached",
                producer_id="test_detached",
                component_versions=versions,
            )
        )

    resumed = ApplicationRunEventController(
        recorder,
        run_id="run_detached_spans",
        thread_id="thread_detached_spans",
        trace_id="trace_detached_spans",
        correlation_id="correlation_detached_spans",
        component_versions=versions,
        attempt=2,
    )
    resumed.start(
        query="Recover stale spans.",
        research_id="research_detached_spans",
    )
    resumed.cancel(reason="Superseded acceptance run.", usage=BudgetUsage())

    events = store.list(
        EventQuery("run_detached_spans", limit=100)
    ).items
    assert [item.event_type for item in events[-3:]] == [
        EventType.SPAN_FAILED,
        EventType.SPAN_FAILED,
        EventType.RUN_CANCELLED,
    ]
    assert [
        item.span_id for item in events[-3:-1]
    ] == ["span_detached_model", "span_detached_agent"]
    store.integrity_check()
    store.close()


def test_production_source_tree_has_no_superseded_runtime_imports():
    root = Path(__file__).resolve().parents[1]
    forbidden_import = re.compile(
        r"^\s*(?:from|import)\s+"
        r"(?:core|agents|providers|schemas|langgraph)(?:\b|\.)",
        re.MULTILINE,
    )
    violations: list[str] = []
    for path in sorted((root / "deep_researcher").rglob("*.py")):
        if forbidden_import.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(root)))
    for path in sorted((root / "console_app").rglob("*.py")):
        if forbidden_import.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(root)))
    for path in (root / "run_research.py", root / "run_console.py"):
        if forbidden_import.search(path.read_text(encoding="utf-8")):
            violations.append(str(path.relative_to(root)))
    assert violations == []

    requirements = (root / "requirements.txt").read_text(
        encoding="utf-8"
    ).casefold()
    for dependency in (
        "langgraph",
        "langchain",
        "qdrant",
        "sqlalchemy",
        "aiosqlite",
        "numpy",
    ):
        assert dependency not in requirements

    for obsolete in ("agents", "core", "providers", "schemas"):
        assert not list((root / obsolete).glob("*.py"))
