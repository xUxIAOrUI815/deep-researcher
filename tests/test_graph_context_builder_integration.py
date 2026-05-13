from __future__ import annotations

import pytest

import core.graph as graph_module
from core.run_context import RunContext
from schemas.state import FinalReport, PlannerState, ResearcherOutputs
from agents.planner import PlannerRunResult


class _FakeBuiltContext:
    def __init__(self, payload: dict):
        self.payload = payload

    def model_dump(self) -> dict:
        return dict(self.payload)


class _FakePlannerBuilder:
    def __init__(self):
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeBuiltContext({"coverage_summary": {"avg_section_coverage": 0.7}, "section_packs": []})


class _FakeResearcherBuilder:
    def __init__(self):
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeBuiltContext({"already_seen_source_ids": ["source-1"], "unresolved_gaps": []})


class _FakeWriterBuilder:
    def __init__(self):
        self.calls = []

    def build(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeBuiltContext({"context_source": "session", "section_evidence_packs": [{"pack_id": "pack-1", "section_id": "sec_summary"}]})


@pytest.mark.asyncio
async def test_planner_node_adapter_uses_planner_context_builder(monkeypatch):
    builder = _FakePlannerBuilder()
    captured: dict = {}

    async def _fake_run_planner(**kwargs):
        captured.update(kwargs)
        return PlannerRunResult(planner_state=PlannerState(action="continue_research"))

    monkeypatch.setattr(graph_module, "PLANNER_CONTEXT_BUILDER", builder)
    monkeypatch.setattr(graph_module, "run_planner", _fake_run_planner)

    await graph_module._call_planner_agent(
        {"user_query": "query", "normalized_query": "query", "task_tree": {}, "section_goals": [], "section_evidence_packs": []},
        RunContext(research_id="research-1", thread_id="thread-1", session_id="session-a", root_query="query"),
    )

    assert builder.calls[0]["research_id"] == "research-1"
    assert captured["planner_context"]["coverage_summary"]["avg_section_coverage"] == 0.7


@pytest.mark.asyncio
async def test_researcher_node_adapter_uses_research_context_builder(monkeypatch):
    builder = _FakeResearcherBuilder()
    captured: dict = {}

    async def _fake_run_researcher(**kwargs):
        captured.update(kwargs)
        return ResearcherOutputs(task_id="task-1")

    monkeypatch.setattr(graph_module, "RESEARCHER_CONTEXT_BUILDER", builder)
    monkeypatch.setattr(graph_module, "run_researcher", _fake_run_researcher)

    await graph_module._call_researcher_agent(
        {"user_query": "query", "normalized_query": "query", "task_tree": {"task-1": {"id": "task-1"}}, "active_task_id": "task-1"},
        RunContext(research_id="research-1", thread_id="thread-1", session_id="session-a", root_query="query"),
    )

    assert builder.calls[0]["task_id"] == "task-1"
    assert captured["research_context"]["already_seen_source_ids"] == ["source-1"]


@pytest.mark.asyncio
async def test_writer_node_adapter_uses_writer_context_builder(monkeypatch):
    builder = _FakeWriterBuilder()
    captured: dict = {}

    async def _fake_run_writer(**kwargs):
        captured.update(kwargs)
        return FinalReport(markdown="# report")

    monkeypatch.setattr(graph_module, "WRITER_CONTEXT_BUILDER", builder)
    monkeypatch.setattr(graph_module, "run_writer", _fake_run_writer)

    await graph_module._call_writer_agent(
        {"report_outline": {"sections": []}, "section_goals": [], "section_evidence_packs": []},
        RunContext(research_id="research-1", thread_id="thread-1", session_id="session-a", root_query="query"),
    )

    assert builder.calls[0]["research_id"] == "research-1"
    assert captured["writer_context"]["context_source"] == "session"
