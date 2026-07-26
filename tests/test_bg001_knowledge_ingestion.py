from __future__ import annotations

from deep_researcher.artifacts import ArtifactQuery
from deep_researcher.contracts import ArtifactKind
from deep_researcher.knowledge import build_knowledge_runtime


RESEARCH = {
    "sources": [
        {
            "source_id": "source_one",
            "url": "https://example.com/source",
            "title": "Primary source",
            "score": 0.95,
            "source_type": "primary",
        }
    ],
    "passages": [
        {
            "source_id": "source_one",
            "url": "https://example.com/source",
            "text": "Verified benchmark is 42.",
            "extraction_method": "test",
        }
    ],
    "scraped_data_cache": [
        {
            "url": "https://example.com/source",
            "title": "Primary source",
            "markdown": "Verified benchmark is 42.",
            "fetch_method": "test",
            "http_status": 200,
        }
    ],
}
CANDIDATES = {
    "evidence": [
        {
            "id": "evidence_one",
            "source_id": "source_one",
            "quote": "Verified benchmark is 42.",
            "summary": "Exact source statement",
            "confidence": 0.95,
            "quality_score": 0.95,
        }
    ],
    "atomic_facts": [
        {
            "id": "fact_one",
            "text": "Verified benchmark is 42.",
            "source_id": "source_one",
            "confidence": 0.95,
        }
    ],
    "claims": [
        {
            "id": "claim_one",
            "text": "Verified benchmark is 42.",
            "fact_ids": ["fact_one"],
            "evidence_ids": ["evidence_one"],
            "confidence": 0.95,
        }
    ],
}


def test_native_ingestion_persists_candidate_graph_citations_and_replays(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        research = runtime.ingestion.ingest_research_observation(
            RESEARCH,
            run_id="run_ingest",
            task_id="task_ingest",
        )
        candidates = runtime.ingestion.ingest_candidate_knowledge(
            CANDIDATES,
            run_id="run_ingest",
            task_id="task_ingest",
        )
        assert len(runtime.repository.sources.list("run_ingest")) == 1
        assert len(runtime.repository.evidence.list("run_ingest")) == 1
        assert len(runtime.repository.facts.list("run_ingest")) == 1
        assert len(runtime.repository.claims.list("run_ingest")) == 1
        citations = runtime.repository.citations.list("run_ingest")
        assert len(citations) == 1
        assert citations[0].quote == "Verified benchmark is 42."
        snapshots = runtime.artifacts.list(
            ArtifactQuery(
                "run_ingest",
                kinds=(ArtifactKind.SOURCE_SNAPSHOT,),
            )
        ).items
        assert len(snapshots) == 1
        revisions = runtime.storage._connection.execute(
            "SELECT COUNT(*) FROM entity_revisions"
        ).fetchone()[0]
        assert runtime.ingestion.ingest_research_observation(
            RESEARCH,
            run_id="run_ingest",
            task_id="task_ingest",
        ).artifact_ids == research.artifact_ids
        assert runtime.ingestion.ingest_candidate_knowledge(
            CANDIDATES,
            run_id="run_ingest",
            task_id="task_ingest",
        ).artifact_ids == candidates.artifact_ids
        assert runtime.storage._connection.execute(
            "SELECT COUNT(*) FROM entity_revisions"
        ).fetchone()[0] == revisions
        assert runtime.retrieval.search(
            run_id="run_ingest",
            query="benchmark 42",
            limit=5,
        )
        runtime.integrity_check()
    finally:
        runtime.close()


def test_source_content_change_creates_a_new_immutable_version(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        runtime.ingestion.ingest_research_observation(
            RESEARCH,
            run_id="run_versions",
            task_id="task_versions",
        )
        changed = {
            **RESEARCH,
            "passages": [dict(item) for item in RESEARCH["passages"]],
            "scraped_data_cache": [
                dict(item) for item in RESEARCH["scraped_data_cache"]
            ],
        }
        changed["passages"][0]["text"] += " Updated."
        changed["scraped_data_cache"][0]["markdown"] += " Updated."
        runtime.ingestion.ingest_research_observation(
            changed,
            run_id="run_versions",
            task_id="task_versions",
        )
        source = runtime.repository.find_source_by_url(
            "run_versions",
            "https://example.com/source",
        )
        assert source is not None
        snapshots = runtime.repository.source_snapshots(source.source_id)
        assert [item.source_version for item in snapshots] == [1, 2]
        assert len({item.content_hash for item in snapshots}) == 2
    finally:
        runtime.close()


def test_identical_candidate_identifiers_are_namespaced_per_run(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        for run_id in ("run_alpha", "run_beta"):
            runtime.ingestion.ingest_research_observation(
                RESEARCH,
                run_id=run_id,
                task_id="task_shared",
            )
            runtime.ingestion.ingest_candidate_knowledge(
                CANDIDATES,
                run_id=run_id,
                task_id="task_shared",
            )
        alpha = runtime.repository.claims.list("run_alpha")
        beta = runtime.repository.claims.list("run_beta")
        assert {item.statement for item in alpha} == {
            item.statement for item in beta
        }
        assert {item.claim_id for item in alpha}.isdisjoint(
            item.claim_id for item in beta
        )
        runtime.integrity_check()
    finally:
        runtime.close()
