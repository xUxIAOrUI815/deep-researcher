from __future__ import annotations

import importlib
import shutil
import sys
import uuid
from pathlib import Path

import pytest

from tests.fixtures.offline_research_inputs import build_initial_graph_state


def _load_graph_module_with_in_memory_manager(workspace_dir: Path):
    import core.session_knowledge as knowledge_module

    original_knowledge_manager_cls = knowledge_module.KnowledgeManager

    class InMemoryKnowledgeManager(original_knowledge_manager_cls):
        def __init__(self, *args, **kwargs):
            super().__init__(base_storage_path=str(workspace_dir), sqlite_filename=":memory:")

    knowledge_module.KnowledgeManager = InMemoryKnowledgeManager
    sys.modules.pop("core.graph", None)
    graph_module = importlib.import_module("core.graph")
    return graph_module, knowledge_module, original_knowledge_manager_cls


@pytest.fixture
def workspace_tmp_dir():
    path = Path("tests") / ".tmp" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.mark.asyncio
async def test_graph_offline_session_snapshot_accumulates_across_rounds(workspace_tmp_dir, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    graph_module, knowledge_module, original_knowledge_manager_cls = _load_graph_module_with_in_memory_manager(workspace_tmp_dir)
    saver = await graph_module.init_sqlite_saver(":memory:")
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir),
            sqlite_filename=":memory:",
        )
        graph = graph_module.create_research_graph(saver)
        initial_state = build_initial_graph_state()

        result = await graph.ainvoke(
            initial_state,
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
    assert result["planner_state"]["action"] == "start_writing"
    assert len(result["completed_tasks"]) >= 2
    assert len(snapshot["facts"]) >= 2
    assert len(snapshot["claims"]) >= 2
    assert len(snapshot["evidence"]) >= 2
    assert len(snapshot["section_evidence_packs"]) >= 2
    assert len(snapshot["facts"]) == len(snapshot["knowledge_refs"]["fact_ids"])
    assert len(snapshot["section_evidence_packs"]) == len(snapshot["knowledge_refs"]["section_pack_ids"])
    assert snapshot["session"]["status"] == "completed"
    assert snapshot["session"]["current_active_task_id"] is None


@pytest.mark.asyncio
async def test_graph_writer_consumes_session_snapshot_packs(workspace_tmp_dir, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    graph_module, knowledge_module, original_knowledge_manager_cls = _load_graph_module_with_in_memory_manager(workspace_tmp_dir)
    saver = await graph_module.init_sqlite_saver(":memory:")
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir),
            sqlite_filename=":memory:",
        )
        graph = graph_module.create_research_graph(saver)
        initial_state = build_initial_graph_state()

        result = await graph.ainvoke(
            initial_state,
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

    report = result["final_report"]
    snapshot_pack_ids = {pack["id"] for pack in snapshot["section_evidence_packs"]}
    snapshot_refs = set()
    for pack in snapshot["section_evidence_packs"]:
        snapshot_refs.update(pack.get("claim_ids", []))
        snapshot_refs.update(pack.get("fact_ids", []))
        snapshot_refs.update(pack.get("evidence_ids", []))
        snapshot_refs.update(pack.get("conflict_ids", []))

    assert set(report["evidence_pack_ids"]).issubset(snapshot_pack_ids)
    assert set(result["section_evidence_packs"][i]["pack_id"] for i in range(len(result["section_evidence_packs"]))) == snapshot_pack_ids
    assert set().union(*[set(refs) for refs in report["citation_map"].values()]).issubset(snapshot_refs)
    assert len(report["markdown"]) > 500


@pytest.mark.asyncio
async def test_graph_can_write_when_transient_state_packs_are_not_authoritative(workspace_tmp_dir, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    graph_module, knowledge_module, original_knowledge_manager_cls = _load_graph_module_with_in_memory_manager(workspace_tmp_dir)
    saver = await graph_module.init_sqlite_saver(":memory:")
    try:
        graph_module.SESSION_KNOWLEDGE_MANAGER = original_knowledge_manager_cls(
            base_storage_path=str(workspace_tmp_dir),
            sqlite_filename=":memory:",
        )
        graph = graph_module.create_research_graph(saver)
        initial_state = build_initial_graph_state()
        initial_state["section_evidence_packs"] = [
            {
                "pack_id": "stale-pack",
                "section_id": "sec_summary",
                "claim_ids": ["stale-claim"],
                "fact_ids": ["stale-fact"],
                "evidence_ids": [],
                "conflict_ids": [],
                "coverage_score": 0.05,
                "notes": "stale transient pack",
            }
        ]

        result = await graph.ainvoke(
            initial_state,
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

    report = result["final_report"]

    assert report is not None
    assert "stale transient pack" not in report["markdown"]
    assert "stale-claim" not in str(report["citation_map"])
    assert set(report["evidence_pack_ids"]).issubset({pack["id"] for pack in snapshot["section_evidence_packs"]})
