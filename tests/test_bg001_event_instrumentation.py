from __future__ import annotations

import json

import httpx
import pytest

import agents.researcher as researcher_module
import core.graph as graph_module
from core.observability import EventType as LegacyType
from core.observability import get_observer, set_observer
from core.run_context import RunContext
from deep_researcher.contracts import EventType, RunStatus
from deep_researcher.events import EventQuery, EventRecorder, PersistentEventObserver, SQLiteEventStore
from tests.fixtures.offline_research_inputs import build_initial_graph_state


@pytest.mark.asyncio
async def test_complete_offline_graph_persists_task_tool_evidence_report_budget_and_spans(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    observer = PersistentEventObserver(EventRecorder(store))
    previous = get_observer()
    set_observer(observer)
    try:
        graph = graph_module.create_research_graph(None)
        result = await graph.ainvoke(
            build_initial_graph_state(),
            {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
        )
    finally:
        set_observer(previous)

    run_id = f"run_{result['run_metadata']['run_id']}" if not result["run_metadata"]["run_id"].startswith("run_") else result["run_metadata"]["run_id"]
    events = store.list(EventQuery(run_id, limit=1000)).items
    types = {event.event_type for event in events}
    assert events[0].event_type == EventType.RUN_STARTED
    assert events[-1].event_type == EventType.RUN_COMPLETED
    assert events[-1].status == RunStatus.SUCCEEDED
    assert EventType.TASK_CREATED in types
    assert EventType.TOOL_STARTED in types
    assert EventType.TOOL_COMPLETED in types
    assert EventType.EVIDENCE_CHANGED in types
    assert EventType.REPORT_CHANGED in types
    assert EventType.BUDGET_CHANGED in types
    assert EventType.SPAN_STARTED in types
    assert EventType.SPAN_COMPLETED in types
    assert all(event.component_versions.runtime.name == "draft-runtime" for event in events)
    store.integrity_check()
    store.close()


@pytest.mark.asyncio
async def test_graph_failure_closes_node_and_run_with_structured_error(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    async def fail_researcher(state, context):
        raise RuntimeError("provider exploded with token=super-secret")

    monkeypatch.setattr(graph_module, "_call_researcher_agent", fail_researcher)
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    observer = PersistentEventObserver(EventRecorder(store))
    previous = get_observer()
    set_observer(observer)
    state = build_initial_graph_state()
    try:
        graph = graph_module.create_research_graph(None)
        with pytest.raises(RuntimeError, match="provider exploded"):
            await graph.ainvoke(
                state,
                {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
            )
    finally:
        set_observer(previous)
    run_id = state["run_metadata"]["run_id"]
    events = store.list(EventQuery(run_id, limit=1000)).items
    assert events[-1].event_type == EventType.RUN_FAILED
    assert events[-1].error is not None
    assert "super-secret" not in events[-1].error.message
    assert any(event.event_type == EventType.SPAN_FAILED for event in events)
    store.integrity_check()
    store.close()


@pytest.mark.asyncio
async def test_real_model_call_path_emits_model_span_and_usage(tmp_path, monkeypatch):
    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "choices": [{"message": {"content": "9"}}],
                "usage": {"prompt_tokens": 17, "completion_tokens": 1},
            }

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return None

        async def post(self, *args, **kwargs):
            return Response()

    monkeypatch.setenv("RESEARCHER_USE_LLM_SCORING", "1")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setattr(researcher_module.httpx, "AsyncClient", FakeAsyncClient)
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    observer = PersistentEventObserver(EventRecorder(store))
    previous = get_observer()
    set_observer(observer)
    context = RunContext.from_config(
        {"configurable": {"thread_id": "model-thread", "research_id": "model-research"}}
    )
    try:
        score, method = await researcher_module._score_query_relevance(
            "AI accelerator demand",
            "AI accelerator market demand",
            run_context=context,
            task_id="task-model",
        )
        observer.record_run_event(context, LegacyType.RUN_COMPLETED, message="Model instrumentation test completed.")
    finally:
        set_observer(previous)
    events = store.list(EventQuery(context.run_id, limit=100)).items
    model_events = [event for event in events if event.event_type in {EventType.MODEL_STARTED, EventType.MODEL_COMPLETED}]
    assert (score, method) == (9, "llm")
    assert [event.event_type for event in model_events] == [EventType.MODEL_STARTED, EventType.MODEL_COMPLETED]
    assert model_events[-1].usage.total_tokens == 18
    assert model_events[-1].usage.model_calls == 1
    store.integrity_check()
    store.close()


def test_legacy_payload_hidden_reasoning_is_removed_before_contract_creation(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    observer = PersistentEventObserver(EventRecorder(store))
    context = RunContext.from_config(
        {"configurable": {"thread_id": "redact-thread", "research_id": "redact-research"}}
    )
    observer.record_run_event(
        context,
        LegacyType.BUDGET_SNAPSHOT,
        message="safe summary",
        payload={"chain_of_thought": "private", "usage": {"tool_calls": 1}},
    )
    observer.record_run_event(context, LegacyType.RUN_COMPLETED, message="done")
    events = store.list(EventQuery(context.run_id, limit=100)).items
    assert all("chain_of_thought" not in json.dumps(event.payload) for event in events)
    store.close()


def test_observer_restart_attaches_to_existing_open_run(tmp_path):
    store = SQLiteEventStore(tmp_path / "events.sqlite3")
    context = RunContext.from_config(
        {"configurable": {"thread_id": "resume-thread", "research_id": "resume-research"}}
    )
    first_observer = PersistentEventObserver(EventRecorder(store))
    first_observer.record_run_event(
        context,
        LegacyType.BUDGET_SNAPSHOT,
        message="before observer restart",
        payload={"usage": {"tool_calls": 1}},
    )
    second_observer = PersistentEventObserver(EventRecorder(store))
    second_observer.record_run_event(
        context,
        LegacyType.BUDGET_SNAPSHOT,
        message="after observer restart",
        payload={"usage": {"tool_calls": 2}},
    )
    second_observer.record_run_event(context, LegacyType.RUN_COMPLETED, message="done")
    events = store.list(EventQuery(context.run_id, limit=100)).items
    assert [event.sequence_no for event in events] == [1, 2, 3, 4]
    assert events[2].causation_event_id == events[1].event_id
    assert events[-1].event_type == EventType.RUN_COMPLETED
    store.integrity_check()
    store.close()
