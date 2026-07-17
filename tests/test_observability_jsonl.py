from __future__ import annotations

import asyncio
import json
import tempfile
import uuid
from pathlib import Path

import pytest

import core.graph as graph_module
from agents.planner import PlannerRunResult
from core.observability import JsonlObserver, NoopObserver, set_observer
from schemas.state import PlannerState
from console_app.service import ResearchConsoleService
from schemas.console import ResearchCreateRequest
from tests.fixtures.offline_research_inputs import MOCK_REPORT_OUTLINE, MOCK_SECTION_GOALS, build_initial_graph_state


@pytest.mark.asyncio
async def test_console_run_writes_jsonl_observability_events(monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    runtime_dir = Path(tempfile.gettempdir()) / "deep-researcher-jsonl-tests" / uuid.uuid4().hex
    service = ResearchConsoleService(runtime_dir=str(runtime_dir))
    try:
        created = await service.create_run(
            ResearchCreateRequest(
                query="Compare AI chip market claims",
                instructions="Use offline sources",
                depth="standard",
            )
        )

        summary = None
        for _ in range(80):
            summary = await service.get_console_summary(created.research_id)
            if summary.status in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)

        assert summary is not None
        assert summary.status == "completed"

        event_path = runtime_dir / "events" / f"{created.research_id}.jsonl"
        assert event_path.exists()

        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert events
        assert all(event.get("event_id") for event in events)
        assert all(event.get("run_id") for event in events)
        assert all(event.get("research_id") == created.research_id for event in events)
        assert all("metrics" in event for event in events)
        assert all("error" in event for event in events)
        assert all("state_version" in event for event in events)

        event_types = [event["event_type"] for event in events]
        assert "run.started" in event_types
        assert "run.completed" in event_types

        started_nodes: dict[str, int] = {}
        completed_nodes: dict[str, int] = {}
        for event in events:
            if event["event_type"] == "node.started":
                node = event.get("node")
                started_nodes[node] = started_nodes.get(node, 0) + 1
            if event["event_type"] == "node.completed":
                node = event.get("node")
                completed_nodes[node] = completed_nodes.get(node, 0) + 1

        assert started_nodes
        assert started_nodes == completed_nodes
    finally:
        await service.aclose()


@pytest.mark.asyncio
async def test_graph_failure_writes_node_failed_and_run_failed_events(tmp_path, monkeypatch):
    async def fake_planner_agent(state, context):
        return PlannerRunResult(
            planner_state=PlannerState(action="start_writing"),
            report_outline=MOCK_REPORT_OUTLINE,
            section_goals=MOCK_SECTION_GOALS,
        )

    async def failing_writer_agent(state, context):
        raise RuntimeError("writer failed for observability test")

    monkeypatch.setattr(graph_module, "_call_planner_agent", fake_planner_agent)
    monkeypatch.setattr(graph_module, "_call_writer_agent", failing_writer_agent)

    set_observer(JsonlObserver(tmp_path))
    try:
        graph = graph_module.create_research_graph(None)
        state = build_initial_graph_state()
        with pytest.raises(RuntimeError, match="writer failed"):
            await graph.ainvoke(
                state,
                {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
            )

        event_path = tmp_path / "offline-research.jsonl"
        events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        event_types = [event["event_type"] for event in events]

        assert "node.failed" in event_types
        assert "run.failed" in event_types
        assert "run.completed" not in event_types

        node_failed = next(event for event in events if event["event_type"] == "node.failed")
        run_failed = next(event for event in events if event["event_type"] == "run.failed")

        assert node_failed["node"] == "writer"
        assert node_failed["level"] == "error"
        assert node_failed["error"]["type"] == "RuntimeError"
        assert node_failed["error"]["message"] == "writer failed for observability test"
        assert run_failed["payload"]["failed_node"] == "writer"
        assert run_failed["error"] == node_failed["error"]
    finally:
        set_observer(NoopObserver())
