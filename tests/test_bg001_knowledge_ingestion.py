from __future__ import annotations

import pytest

import core.graph as graph_module
from agents.distiller import run_distiller
from deep_researcher.artifacts import ArtifactQuery
from deep_researcher.contracts import ArtifactKind
from deep_researcher.knowledge import build_knowledge_runtime
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_RESEARCHER_OUTPUTS,
    MOCK_SECTION_GOALS,
    MOCK_TASK,
    build_initial_graph_state,
)


@pytest.mark.asyncio
async def test_ingestion_persists_snapshots_full_candidate_graph_and_is_replay_idempotent(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        researcher = runtime.ingestion.ingest_researcher_outputs(
            MOCK_RESEARCHER_OUTPUTS, run_id="run_ingest", task_id="task_ingest"
        )
        outputs = await run_distiller(
            task_id="task-1", task=MOCK_TASK, researcher_outputs=MOCK_RESEARCHER_OUTPUTS,
            report_outline=MOCK_REPORT_OUTLINE, section_goals=MOCK_SECTION_GOALS,
            knowledge_refs=MOCK_KNOWLEDGE_REFS,
        )
        distilled = runtime.ingestion.ingest_distiller_outputs(
            outputs.model_dump(), run_id="run_ingest", task_id="task_ingest"
        )
        before = runtime.projection.build("run_ingest")
        assert len(before["sources"]) == 2
        assert before["stats"]["total_evidence"] > 0
        assert before["stats"]["total_facts"] > 0
        assert before["stats"]["total_claims"] > 0
        snapshots = runtime.artifacts.list(ArtifactQuery("run_ingest", kinds=(ArtifactKind.SOURCE_SNAPSHOT,))).items
        assert len(snapshots) == 2
        assert all(runtime.artifacts.read_bytes(item.artifact_id) for item in snapshots)
        revisions = runtime.storage._connection.execute("SELECT COUNT(*) FROM entity_revisions").fetchone()[0]
        replay_researcher = runtime.ingestion.ingest_researcher_outputs(
            MOCK_RESEARCHER_OUTPUTS, run_id="run_ingest", task_id="task_ingest"
        )
        replay_distilled = runtime.ingestion.ingest_distiller_outputs(
            outputs.model_dump(), run_id="run_ingest", task_id="task_ingest"
        )
        assert replay_researcher.artifact_ids == researcher.artifact_ids
        assert replay_distilled.artifact_ids == distilled.artifact_ids
        assert runtime.storage._connection.execute("SELECT COUNT(*) FROM entity_revisions").fetchone()[0] == revisions
        hits = runtime.retrieval.search(run_id="run_ingest", query="market 120 billion", limit=5)
        assert hits and hits[0].score > 0
        runtime.integrity_check()
    finally:
        runtime.close()


def test_source_content_change_creates_a_new_immutable_version(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        runtime.ingestion.ingest_researcher_outputs(MOCK_RESEARCHER_OUTPUTS, run_id="run_versions", task_id="task_versions")
        changed = {**MOCK_RESEARCHER_OUTPUTS}
        changed["passages"] = [dict(item) for item in MOCK_RESEARCHER_OUTPUTS["passages"]]
        changed["passages"][0]["text"] += " Updated source body."
        runtime.ingestion.ingest_researcher_outputs(changed, run_id="run_versions", task_id="task_versions")
        source = runtime.repository.find_source_by_url("run_versions", MOCK_RESEARCHER_OUTPUTS["sources"][0]["url"])
        snapshots = runtime.repository.source_snapshots(source.source_id)
        assert [item.source_version for item in snapshots] == [1, 2]
        assert len({item.content_hash for item in snapshots}) == 2
    finally:
        runtime.close()


def test_same_draft_identifiers_are_namespaced_per_run(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        runtime.ingestion.ingest_researcher_outputs(MOCK_RESEARCHER_OUTPUTS, run_id="run_alpha", task_id="task_shared")
        runtime.ingestion.ingest_researcher_outputs(MOCK_RESEARCHER_OUTPUTS, run_id="run_beta", task_id="task_shared")
        alpha = runtime.repository.sources.list("run_alpha")
        beta = runtime.repository.sources.list("run_beta")
        assert {item.canonical_url for item in alpha} == {item.canonical_url for item in beta}
        assert {item.source_id for item in alpha}.isdisjoint(item.source_id for item in beta)
        runtime.integrity_check()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_current_offline_graph_dual_writes_without_changing_report_behavior(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    previous = graph_module.set_durable_knowledge_ingestor(runtime.ingestion)
    state = build_initial_graph_state()
    try:
        graph = graph_module.create_research_graph(None)
        result = await graph.ainvoke(
            state,
            {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
        )
        run_id = result["run_metadata"]["run_id"]
        projection = runtime.projection.build(run_id)
        assert result["final_report"]["markdown"].startswith("# ")
        assert projection["stats"]["total_sources"] >= 2
        assert projection["reports"] and projection["section_evidence_packs"]
        runtime.integrity_check()
    finally:
        graph_module.set_durable_knowledge_ingestor(previous)
        runtime.close()
