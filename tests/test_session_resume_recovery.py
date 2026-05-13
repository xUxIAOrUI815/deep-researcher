from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

from agents.planner import run_planner
from agents.writer import run_writer
from core.session_knowledge import KnowledgeManager
from tests.fixtures.offline_research_inputs import build_initial_graph_state
from tests.fixtures.session_knowledge_fixtures import build_distiller_outputs


RESEARCH_ID = "resume-research"
SESSION_ID = "session_resume-research"
THREAD_ID = "resume-thread"
GRAPH_DB = "graph_resume.sqlite3"
KNOWLEDGE_DB = "session_knowledge.sqlite3"


@pytest.fixture
def workspace_tmp_dir():
    path = Path(tempfile.gettempdir()) / "mini-deep-research-tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _load_graph_module_with_persistent_manager(workspace_dir: Path):
    import core.session_knowledge as knowledge_module

    original_knowledge_manager_cls = knowledge_module.KnowledgeManager

    class PersistentKnowledgeManager(original_knowledge_manager_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(base_storage_path=str(workspace_dir), sqlite_filename=KNOWLEDGE_DB)

    knowledge_module.KnowledgeManager = PersistentKnowledgeManager
    sys.modules.pop("core.graph", None)
    graph_module = importlib.import_module("core.graph")
    return graph_module, knowledge_module, original_knowledge_manager_cls


async def _run_graph_phase(workspace_dir: Path, monkeypatch) -> dict:
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    graph_module, knowledge_module, original_knowledge_manager_cls = _load_graph_module_with_persistent_manager(workspace_dir)
    saver = await graph_module.init_sqlite_saver(str(workspace_dir / GRAPH_DB))
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_dir),
            sqlite_filename=KNOWLEDGE_DB,
        )
        graph = graph_module.create_research_graph(saver)
        initial_state = build_initial_graph_state()
        initial_state["run_metadata"]["thread_id"] = THREAD_ID
        initial_state["run_metadata"]["research_id"] = RESEARCH_ID

        result = await graph.ainvoke(
            initial_state,
            {"configurable": {"thread_id": THREAD_ID, "research_id": RESEARCH_ID}},
        )
    finally:
        await saver.conn.close()
        graph_module.SESSION_KNOWLEDGE_MANAGER.close()
        knowledge_module.KnowledgeManager = original_knowledge_manager_cls
        sys.modules.pop("core.graph", None)

    return result


@pytest.mark.asyncio
async def test_session_snapshot_available_after_store_reconstruction(workspace_tmp_dir):
    manager_1 = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=KNOWLEDGE_DB)
    try:
        outputs = await build_distiller_outputs(task_id="resume-task-1")
        first = manager_1.process_distiller_output(
            outputs,
            research_id=RESEARCH_ID,
            session_id=SESSION_ID,
            task_id="resume-task-1",
        )
    finally:
        manager_1.close()

    manager_2 = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=KNOWLEDGE_DB)
    try:
        snapshot = manager_2.get_session_snapshot(RESEARCH_ID, SESSION_ID)
    finally:
        manager_2.close()

    assert len(snapshot["facts"]) == len(first.atomic_facts)
    assert len(snapshot["claims"]) == len(first.claims)
    assert len(snapshot["evidence"]) == len(first.evidence)
    assert len(snapshot["conflicts"]) == len(first.conflicts)
    assert len(snapshot["section_evidence_packs"]) == len(first.section_evidence_packs)
    assert set(snapshot["knowledge_refs"]["fact_ids"]) == {item["id"] for item in snapshot["facts"]}


@pytest.mark.asyncio
async def test_graph_resume_uses_persisted_session_snapshot(workspace_tmp_dir, monkeypatch):
    phase_a_result = await _run_graph_phase(workspace_tmp_dir, monkeypatch)

    graph_module, knowledge_module, original_knowledge_manager_cls = _load_graph_module_with_persistent_manager(workspace_tmp_dir)
    saver = await graph_module.init_sqlite_saver(str(workspace_tmp_dir / GRAPH_DB))
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir),
            sqlite_filename=KNOWLEDGE_DB,
        )
        graph = graph_module.create_research_graph(saver)
        recovered = await graph.aget_state({"configurable": {"thread_id": THREAD_ID, "research_id": RESEARCH_ID}})
        snapshot = graph_module.SESSION_KNOWLEDGE_MANAGER.get_session_snapshot(RESEARCH_ID, SESSION_ID)

        resume_state = dict(recovered.values)
        resume_state["final_report"] = None
        resume_state["section_evidence_packs"] = []
        resume_state["knowledge_refs"] = {"collection_name": phase_a_result["knowledge_refs"]["collection_name"]}
        resume_state["distiller_outputs"] = {}
        resume_state["planner_state"] = {}
        resume_state["atomic_facts"] = []

        resumed = await graph.ainvoke(
            resume_state,
            {"configurable": {"thread_id": THREAD_ID, "research_id": RESEARCH_ID}},
        )
    finally:
        await saver.conn.close()
        graph_module.SESSION_KNOWLEDGE_MANAGER.close()
        knowledge_module.KnowledgeManager = original_knowledge_manager_cls
        sys.modules.pop("core.graph", None)

    assert recovered is not None
    assert len(snapshot["facts"]) > 0
    assert resumed["final_report"] is not None
    assert resumed["planner_state"]["action"] == "start_writing"
    assert "session_facts=" in resumed["planner_state"]["convergence_summary"]
    assert set(resumed["final_report"]["evidence_pack_ids"]).issubset({pack["id"] for pack in snapshot["section_evidence_packs"]})


@pytest.mark.asyncio
async def test_writer_after_resume_reads_persisted_session_packs(workspace_tmp_dir, monkeypatch):
    phase_a_result = await _run_graph_phase(workspace_tmp_dir, monkeypatch)

    manager = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=KNOWLEDGE_DB)
    try:
        snapshot = manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)
        report = await run_writer(
            report_outline=phase_a_result["report_outline"],
            section_goals=phase_a_result["section_goals"],
            section_evidence_packs=[
                {
                    "pack_id": "stale-pack",
                    "section_id": "sec_summary",
                    "claim_ids": ["stale-claim"],
                    "fact_ids": ["stale-fact"],
                    "evidence_ids": [],
                    "conflict_ids": [],
                    "coverage_score": 0.01,
                    "notes": "stale fallback pack",
                }
            ],
            knowledge_manager=manager,
            research_id=RESEARCH_ID,
            session_id=SESSION_ID,
        )
    finally:
        manager.close()

    assert len(snapshot["section_evidence_packs"]) > 0
    assert set(report.evidence_pack_ids).issubset({pack["id"] for pack in snapshot["section_evidence_packs"]})
    assert "stale fallback pack" not in report.markdown
    assert "stale-claim" not in str(report.citation_map)


@pytest.mark.asyncio
async def test_planner_after_resume_bases_decision_on_restored_snapshot(workspace_tmp_dir, monkeypatch):
    phase_a_result = await _run_graph_phase(workspace_tmp_dir, monkeypatch)

    manager = KnowledgeManager(base_storage_path=str(workspace_tmp_dir), sqlite_filename=KNOWLEDGE_DB)
    try:
        result = await run_planner(
            user_query=phase_a_result["user_query"],
            normalized_query=phase_a_result["normalized_query"],
            task_tree=phase_a_result["task_tree"],
            active_task_id=None,
            distiller_outputs={},
            knowledge_refs={"collection_name": phase_a_result["knowledge_refs"]["collection_name"]},
            report_outline=phase_a_result["report_outline"],
            section_goals=phase_a_result["section_goals"],
            section_evidence_packs=[],
            knowledge_manager=manager,
            research_id=RESEARCH_ID,
            session_id=SESSION_ID,
        )
    finally:
        manager.close()

    assert result.planner_state.action == "start_writing"
    assert "session_facts=" in result.planner_state.convergence_summary
    assert "session_evidence=" in result.planner_state.convergence_summary
    assert "session_packs=" in result.planner_state.convergence_summary
