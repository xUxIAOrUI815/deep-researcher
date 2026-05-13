from __future__ import annotations
import hashlib
import importlib
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.writer import run_writer
from core.session_knowledge import KnowledgeManager
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_SECTION_GOALS,
    MOCK_TASK,
    ROOT_QUERY,
    build_initial_graph_state,
)
from tests.fixtures.session_knowledge_fixtures import StubKnowledgeManager, build_distiller_outputs


def _source_id(url: str) -> str:
    return f"src_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}"


@pytest.fixture
def workspace_tmp_dir():
    path = Path("tests") / ".tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def knowledge_manager(workspace_tmp_dir):
    km = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=":memory:")
    try:
        yield km
    finally:
        km.close()


def _root_only_task_tree() -> dict:
    return {
        "root": {
            "id": "root",
            "query": ROOT_QUERY,
            "title": "Root research task",
            "status": "pending",
            "depth": 0,
            "priority": 1.0,
            "parent_id": None,
            "parent_task_id": None,
            "children_ids": [],
            "node_type": "question",
        }
    }


@pytest.mark.asyncio
async def test_stage2_knowledge_manager_persists_snapshot_and_dedupes_repeated_distiller_outputs(knowledge_manager):
    outputs = await build_distiller_outputs()

    first = knowledge_manager.process_distiller_output(
        outputs,
        research_id="research-1",
        session_id="session-a",
        task_id="task-1",
    )
    second = knowledge_manager.process_distiller_output(
        outputs.model_dump(),
        research_id="research-1",
        session_id="session-a",
        task_id="task-1",
    )
    snapshot = knowledge_manager.get_session_snapshot("research-1", "session-a")

    assert snapshot["knowledge_refs"]["fact_ids"] == second.knowledge_refs["fact_ids"]
    assert len(snapshot["facts"]) == len(first.atomic_facts)
    assert len(snapshot["claims"]) == len(first.claims)
    assert len(snapshot["evidence"]) == len(first.evidence)
    assert len(snapshot["conflicts"]) == len(first.conflicts)
    assert len(snapshot["section_evidence_packs"]) == len(first.section_evidence_packs)

    fact_ids = {item["id"] for item in snapshot["facts"]}
    claim_ids = {item["id"] for item in snapshot["claims"]}
    evidence_ids = {item["id"] for item in snapshot["evidence"]}
    conflict_ids = {item["id"] for item in snapshot["conflicts"]}

    for pack in snapshot["section_evidence_packs"]:
        assert set(pack.get("fact_ids", [])).issubset(fact_ids)
        assert set(pack.get("claim_ids", [])).issubset(claim_ids)
        assert set(pack.get("evidence_ids", [])).issubset(evidence_ids)
        assert set(pack.get("conflict_ids", [])).issubset(conflict_ids)


@pytest.mark.asyncio
async def test_stage2_knowledge_manager_clears_research_scope_snapshot(knowledge_manager):
    outputs_a = await build_distiller_outputs(task_id="task-a")

    knowledge_manager.process_distiller_output(outputs_a, research_id="research-1", session_id="session-a", task_id="task-1")
    assert len(knowledge_manager.reload_session("research-1", "session-a")["facts"]) > 0

    knowledge_manager.clear_session("research-1", "session-a")

    assert knowledge_manager.get_session_snapshot("research-1", "session-a")["knowledge_refs"]["fact_ids"] == []
    assert knowledge_manager.get_session_snapshot("research-1", "session-a")["facts"] == []


@pytest.mark.asyncio
async def test_stage3_planner_prefers_session_snapshot_over_empty_incoming_refs(workspace_tmp_dir):
    distiller_outputs = await build_distiller_outputs()
    knowledge_manager = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=":memory:")
    try:
        knowledge_manager.process_distiller_output(
            distiller_outputs,
            research_id="research-1",
            session_id="session-a",
            task_id="task-1",
        )

        result = await run_planner(
            user_query=ROOT_QUERY,
            normalized_query=ROOT_QUERY,
            task_tree=_root_only_task_tree(),
            active_task_id=None,
            distiller_outputs={},
            knowledge_refs={"collection_name": "empty"},
            report_outline=MOCK_REPORT_OUTLINE,
            section_goals=MOCK_SECTION_GOALS,
            section_evidence_packs=[],
            knowledge_manager=knowledge_manager,
            research_id="research-1",
            session_id="session-a",
        )
    finally:
        knowledge_manager.close()

    assert "session_facts=" in result.planner_state.convergence_summary
    assert "session_evidence=" in result.planner_state.convergence_summary
    assert result.planner_state.action == "continue_research"


@pytest.mark.asyncio
async def test_stage3_researcher_filters_sources_already_present_in_session_snapshot():
    baseline = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )
    duplicate_source_ids = [source["source_id"] for source in baseline.sources]
    knowledge_manager = StubKnowledgeManager(
        {
            "facts": [{"id": "fact-1"}],
            "evidence": [{"id": "evidence-1"}],
            "section_evidence_packs": [],
            "knowledge_refs": {"source_ids": duplicate_source_ids},
        }
    )

    filtered = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )

    assert knowledge_manager.calls == [("research-1", "session-a")]
    assert filtered.metadata["session_knowledge"]["source_count"] == len(duplicate_source_ids)
    assert filtered.sources == []
    assert filtered.passages == []
    assert filtered.metadata["stop_reason"] == "marginal_gain_stop"


@pytest.mark.asyncio
async def test_stage3_writer_prefers_session_snapshot_packs_over_stale_graph_inputs():
    snapshot_pack = {
        "pack_id": "pack-session",
        "section_id": "sec_summary",
        "claim_ids": ["claim-session"],
        "fact_ids": ["fact-session"],
        "evidence_ids": ["evidence-session"],
        "conflict_ids": [],
        "coverage_score": 0.8,
        "notes": "Session-backed pack note.",
    }
    stale_pack = {
        "pack_id": "pack-stale",
        "section_id": "sec_summary",
        "claim_ids": ["claim-stale"],
        "fact_ids": ["fact-stale"],
        "evidence_ids": ["evidence-stale"],
        "conflict_ids": [],
        "coverage_score": 0.1,
        "notes": "Stale pack note.",
    }
    knowledge_manager = StubKnowledgeManager({"section_evidence_packs": [snapshot_pack]})

    report = await run_writer(
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[stale_pack],
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
    )

    assert knowledge_manager.calls
    assert set(knowledge_manager.calls) == {("research-1", "session-a")}
    assert report.evidence_pack_ids == ["pack-session"]
    assert "Session-backed pack note." in report.markdown
    assert "Stale pack note." not in report.markdown
    assert "claim-session" in report.citation_map["sec_summary"]


@pytest.mark.asyncio
async def test_stage4_graph_offline_run_persists_session_knowledge_end_to_end(workspace_tmp_dir, monkeypatch):
    import core.session_knowledge as knowledge_module

    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    original_knowledge_manager_cls = knowledge_module.KnowledgeManager

    class InMemoryKnowledgeManager(original_knowledge_manager_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(base_storage_path=str(workspace_tmp_dir), sqlite_filename=":memory:")

    knowledge_module.KnowledgeManager = InMemoryKnowledgeManager
    sys.modules.pop("core.graph", None)
    graph_module = importlib.import_module("core.graph")
    saver = await graph_module.init_sqlite_saver(":memory:")
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir),
            sqlite_filename=":memory:",
        )
        graph = graph_module.create_research_graph(saver)
        state = build_initial_graph_state()
        result = await graph.ainvoke(
            state,
            {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
        )
        snapshot = graph_module.SESSION_KNOWLEDGE_MANAGER.get_session_snapshot(
            "offline-research",
            "session_offline-research",
        )
    finally:
        await saver.conn.close()
        graph_module.SESSION_KNOWLEDGE_MANAGER.close()
        knowledge_module.KnowledgeManager = original_knowledge_manager_cls
        sys.modules.pop("core.graph", None)

    assert result["final_report"] is not None
    assert len(result["section_evidence_packs"]) > 0
    assert len(snapshot["facts"]) > 0
    assert len(snapshot["evidence"]) > 0
    assert len(snapshot["section_evidence_packs"]) == len(snapshot["knowledge_refs"]["section_pack_ids"])
    assert set(result["knowledge_refs"]["fact_ids"]) == set(snapshot["knowledge_refs"]["fact_ids"])


@pytest.mark.asyncio
async def test_stage5_graph_checkpoint_recovery_keeps_session_knowledge_available(workspace_tmp_dir, monkeypatch):
    import core.session_knowledge as knowledge_module

    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    original_knowledge_manager_cls = knowledge_module.KnowledgeManager

    class InMemoryKnowledgeManager(original_knowledge_manager_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(base_storage_path=str(workspace_tmp_dir / "session-km"), sqlite_filename=":memory:")

    knowledge_module.KnowledgeManager = InMemoryKnowledgeManager
    sys.modules.pop("core.graph", None)
    graph_module = importlib.import_module("core.graph")
    saver = await graph_module.init_sqlite_saver(":memory:")
    config = {
        "configurable": {
            "thread_id": "resume-thread",
            "research_id": "resume-research",
        }
    }
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir / "session-km"),
            sqlite_filename=":memory:",
        )
        graph = graph_module.create_research_graph(saver)
        initial_state = build_initial_graph_state()
        initial_state["run_metadata"]["thread_id"] = "resume-thread"
        initial_state["run_metadata"]["research_id"] = "resume-research"

        result = await graph.ainvoke(initial_state, config)
        recovered = await graph.aget_state(config)
        snapshot = graph_module.SESSION_KNOWLEDGE_MANAGER.reload_session(
            "resume-research",
            "session_resume-research",
        )
    finally:
        await saver.conn.close()
        graph_module.SESSION_KNOWLEDGE_MANAGER.close()
        knowledge_module.KnowledgeManager = original_knowledge_manager_cls
        sys.modules.pop("core.graph", None)

    assert recovered is not None
    assert recovered.values["final_report"]["markdown"] == result["final_report"]["markdown"]
    assert recovered.values["knowledge_refs"]["fact_ids"] == result["knowledge_refs"]["fact_ids"]
    assert len(snapshot) > 0
