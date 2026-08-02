from __future__ import annotations

from copy import deepcopy

from deep_researcher.artifacts import ArtifactQuery
from deep_researcher.contracts import ArtifactKind, SnapshotStatus, SourceStatus
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


def test_source_authority_uses_canonical_identity_and_survives_read_replay(
    tmp_path,
):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    observation = deepcopy(RESEARCH)
    observation["sources"][0].update(
        {
            "url": "https://arxiv.org/abs/2501.12345",
            "source_type": "other",
            "score": 0.01,
        }
    )
    observation["passages"][0]["url"] = observation["sources"][0]["url"]
    observation["scraped_data_cache"][0]["url"] = observation["sources"][0][
        "url"
    ]
    try:
        first = runtime.ingestion.ingest_research_observation(
            observation,
            run_id="run_authority",
            task_id="task_search",
        )
        replay = deepcopy(observation)
        replay["sources"][0].update(
            {
                "title": "Read copy of the paper original",
                "score": None,
            }
        )
        second = runtime.ingestion.ingest_research_observation(
            replay,
            run_id="run_authority",
            task_id="task_read",
        )
        source = runtime.repository.find_source_by_url(
            "run_authority",
            observation["sources"][0]["url"],
        )
        assert source is not None
        assert source.source_type.value == "primary"
        assert source.source_level.value == "primary"
        assert source.authority_score == 0.9
        assert source.metadata["search_relevance_score"] is None
        assert set(source.provenance.source_artifact_ids) == {
            first.artifact_ids[0],
            second.artifact_ids[0],
        }
    finally:
        runtime.close()


def test_ungrounded_candidate_quote_cannot_create_claim_or_citation(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    invalid = deepcopy(CANDIDATES)
    invalid["evidence"][0]["quote"] = "The benchmark was approximately forty-two."
    try:
        runtime.ingestion.ingest_research_observation(
            RESEARCH,
            run_id="run_ungrounded",
            task_id="task_ungrounded",
        )
        result = runtime.ingestion.ingest_candidate_knowledge(
            invalid,
            run_id="run_ungrounded",
            task_id="task_ungrounded",
        )
        assert any("not an exact substring" in item for item in result.issues)
        assert runtime.repository.evidence.list("run_ungrounded") == ()
        assert runtime.repository.facts.list("run_ungrounded") == ()
        assert runtime.repository.claims.list("run_ungrounded") == ()
        assert runtime.repository.citations.list("run_ungrounded") == ()
    finally:
        runtime.close()


def test_access_denied_interstitial_is_preserved_but_cannot_enter_evidence_graph(
    tmp_path,
):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    denied = deepcopy(RESEARCH)
    denied["sources"][0]["title"] = "Are you a robot?"
    denied["passages"][0]["text"] = "Are you a robot?"
    denied["scraped_data_cache"][0].update(
        {
            "title": "Are you a robot?",
            "markdown": "Are you a robot? Complete the CAPTCHA challenge.",
            "http_status": 403,
        }
    )
    candidates = deepcopy(CANDIDATES)
    candidates["evidence"][0]["quote"] = "Are you a robot?"
    try:
        observed = runtime.ingestion.ingest_research_observation(
            denied,
            run_id="run_access_denied",
            task_id="task_access_denied",
        )
        result = runtime.ingestion.ingest_candidate_knowledge(
            candidates,
            run_id="run_access_denied",
            task_id="task_access_denied",
        )
        source = runtime.repository.sources.list("run_access_denied")[0]
        snapshot = runtime.repository.source_snapshots(source.source_id)[0]
        assert source.status == SourceStatus.BLOCKED
        assert snapshot.status == SnapshotStatus.FAILED
        assert source.metadata["content_admission"] == "rejected_non_content"
        assert observed.artifact_ids
        assert any("no persisted passage" in item for item in result.issues)
        assert runtime.repository.passages.list("run_access_denied") == ()
        assert runtime.repository.evidence.list("run_access_denied") == ()
        assert runtime.repository.facts.list("run_access_denied") == ()
        assert runtime.repository.claims.list("run_access_denied") == ()
        assert runtime.repository.citations.list("run_access_denied") == ()
    finally:
        runtime.close()


def test_candidate_quote_resolves_read_passage_before_search_snippet(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    search = deepcopy(RESEARCH)
    search["passages"][0]["text"] = "Search snippet without the paper quote."
    search["scraped_data_cache"][0]["markdown"] = ""
    read = deepcopy(RESEARCH)
    read["sources"][0]["source_id"] = "source_read_0"
    read["passages"][0].update(
        {
            "source_id": "source_read_0",
            "text": "The paper reports a verified benchmark of 42.",
            "extraction_method": "jina",
        }
    )
    read["scraped_data_cache"][0].update(
        {
            "markdown": "The paper reports a verified benchmark of 42.",
            "fetch_method": "jina",
        }
    )
    candidates = deepcopy(CANDIDATES)
    candidates["evidence"][0].update(
        {
            "source_id": "source_one",
            "source_url": RESEARCH["sources"][0]["url"],
            "quote": "The paper reports a verified benchmark of 42.",
        }
    )
    candidates["atomic_facts"][0].update(
        {
            "source_id": "source_one",
            "source_url": RESEARCH["sources"][0]["url"],
            "text": "The paper reports a verified benchmark of 42.",
        }
    )
    candidates["claims"][0]["text"] = (
        "The paper reports a verified benchmark of 42."
    )
    try:
        runtime.ingestion.ingest_research_observation(
            search,
            run_id="run_read_precedence",
            task_id="task_search",
        )
        runtime.ingestion.ingest_research_observation(
            read,
            run_id="run_read_precedence",
            task_id="task_read",
        )
        result = runtime.ingestion.ingest_candidate_knowledge(
            candidates,
            run_id="run_read_precedence",
            task_id="task_extract",
        )
        assert result.issues == ()
        citation = runtime.repository.citations.list("run_read_precedence")[0]
        passage = runtime.repository.passages.require(citation.passage_id)
        assert passage.extraction_method == "jina"
        assert runtime.artifacts.read_bytes(passage.text_artifact_id).decode(
            "utf-8"
        ) == "The paper reports a verified benchmark of 42."
    finally:
        runtime.close()
