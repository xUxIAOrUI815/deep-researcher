from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import sqlite3

import pytest

from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    Citation,
    Claim,
    Conflict,
    EntityProvenance,
    Evidence,
    EvidenceRelation,
    Passage,
    Report,
    Section,
    Source,
    SourceSnapshot,
    SourceStatus,
    SourceType,
    utc_now,
)
from deep_researcher.knowledge import (
    DeduplicationService,
    KnowledgeConflict,
    KnowledgeCorruption,
    KnowledgeQuery,
    KnowledgeRepository,
    KnowledgeRelation,
    SQLiteKnowledgeStorage,
    build_knowledge_runtime,
    canonicalize_url,
    restore_knowledge_runtime,
)


def _source(run_id: str, suffix: str = "one") -> Source:
    return Source(
        source_id=f"source_{run_id}_{suffix}",
        canonical_url=f"https://example.test/{suffix}",
        source_type=SourceType.PRIMARY,
        status=SourceStatus.ACCESSIBLE,
        title=suffix,
        provenance=EntityProvenance(producer_id="producer_test", run_id=run_id),
    )


def test_normalization_deduplication_revisions_and_natural_keys(tmp_path):
    assert canonicalize_url("HTTPS://ExAmple.Test:443/a/?utm_source=x&b=2&a=1#frag") == "https://example.test/a?a=1&b=2"
    assert DeduplicationService().compare("ＡＩ   chip", "AI chip").duplicate
    with build_knowledge_runtime(tmp_path / "runtime") as runtime:
        source = _source("run_one")
        first = runtime.repository.sources.save(source)
        same = runtime.repository.sources.save(source)
        changed = source.model_copy(update={"title": "updated", "updated_at": utc_now() + timedelta(seconds=1)})
        third = runtime.repository.sources.save(changed)
        assert (first.revision, same.revision, third.revision) == (1, 1, 2)
        assert len(runtime.repository.sources.history(source.source_id)) == 2
        duplicate = source.model_copy(update={"source_id": "source_duplicate"})
        with pytest.raises(KnowledgeConflict, match="natural key"):
            runtime.repository.sources.save(duplicate)


def test_run_pagination_concurrency_restart_backup_restore_and_migration(tmp_path):
    root = tmp_path / "runtime"
    runtime = build_knowledge_runtime(root)
    for index in range(5):
        runtime.repository.sources.save(_source("run_page", str(index)))
    first = runtime.storage.list_latest(KnowledgeQuery("run_page", limit=2))
    second = runtime.storage.list_latest(KnowledgeQuery("run_page", after_created_at=first.next_cursor[0], after_entity_id=first.next_cursor[1], limit=3))
    assert len(first.items) + len(second.items) == 5
    runtime.backup_to(tmp_path / "backup")
    runtime.close()
    connection = sqlite3.connect(root / "knowledge.sqlite3")
    for name in ("entities_no_update", "entities_no_delete", "revisions_no_update", "revisions_no_delete", "relationships_no_update", "relationships_no_delete", "natural_keys_no_update", "natural_keys_no_delete"):
        connection.execute(f"DROP TRIGGER {name}")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()
    restarted = build_knowledge_runtime(root)
    assert restarted.storage._connection.execute("PRAGMA user_version").fetchone()[0] == 2
    restarted.close()
    restored = restore_knowledge_runtime(tmp_path / "backup", tmp_path / "restored")
    assert len(restored.repository.sources.list("run_page")) == 5
    restored.close()


def test_concurrent_storage_connections_keep_runs_isolated(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    stores = [SQLiteKnowledgeStorage(database) for _ in range(6)]
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            list(executor.map(lambda pair: pair[1].save_batch((_source(f"run_{pair[0]}"),)), enumerate(stores)))
        for index, store in enumerate(stores):
            assert len(store.list_latest(KnowledgeQuery(f"run_{index}")).items) == 1
        stores[0].integrity_check()
    finally:
        for store in stores:
            store.close()


def test_payload_corruption_is_detected(tmp_path):
    database = tmp_path / "knowledge.sqlite3"
    store = SQLiteKnowledgeStorage(database)
    source = _source("run_corrupt")
    store.save_batch((source,))
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER revisions_no_update")
    connection.execute("UPDATE entity_revisions SET payload_json='{}'")
    connection.commit()
    connection.close()
    reopened = SQLiteKnowledgeStorage(database)
    try:
        with pytest.raises(KnowledgeCorruption, match="checksum"):
            reopened.get_latest(source.source_id)
        with pytest.raises(KnowledgeCorruption, match="checksum"):
            reopened.integrity_check()
    finally:
        reopened.close()


def test_complete_typed_evidence_graph_and_optional_vector_boundary(tmp_path):
    runtime = build_knowledge_runtime(tmp_path / "runtime")
    try:
        run_id = "run_graph"
        provenance = EntityProvenance(producer_id="producer_test", run_id=run_id)
        snapshot_artifact = runtime.artifacts.put_text(
            "grounded quote", kind=ArtifactKind.SOURCE_SNAPSHOT,
            producer_id="producer_test", run_id=run_id,
        )
        passage_artifact = runtime.artifacts.put_text(
            "grounded quote", kind=ArtifactKind.CLEANED_CONTENT,
            producer_id="producer_test", run_id=run_id,
            source_artifact_ids=(snapshot_artifact.artifact_id,),
        )
        report_artifact = runtime.artifacts.put_text(
            "# Report", kind=ArtifactKind.REPORT, producer_id="producer_test", run_id=run_id,
        )
        source = _source(run_id, "graph")
        snapshot = SourceSnapshot(
            snapshot_id="snapshot_graph", source_id=source.source_id,
            artifact_id=snapshot_artifact.artifact_id, source_version=1,
            content_hash=snapshot_artifact.content_hash, final_url=source.canonical_url,
            provenance=provenance,
        )
        passage = Passage(
            passage_id="passage_graph", snapshot_id=snapshot.snapshot_id,
            text_artifact_id=passage_artifact.artifact_id, ordinal=0, locator="chars:0-14",
            content_hash=passage_artifact.content_hash, provenance=provenance,
        )
        evidence = Evidence(
            evidence_id="evidence_graph", passage_ids=(passage.passage_id,),
            relation=EvidenceRelation.SUPPORTS, summary="Grounded support", confidence=0.9,
            relevance=0.9, source_quality=0.9, provenance=provenance,
        )
        fact = AtomicFact(
            fact_id="fact_graph", statement="The quote is grounded.",
            evidence_ids=(evidence.evidence_id,), confidence=0.9, provenance=provenance,
        )
        claim = Claim(
            claim_id="claim_graph", statement="A grounded claim", fact_ids=(fact.fact_id,),
            evidence_ids=(evidence.evidence_id,), confidence=0.9, provenance=provenance,
        )
        counterclaim = Claim(
            claim_id="claim_counter", statement="A counterclaim", fact_ids=(fact.fact_id,),
            confidence=0.4, provenance=provenance,
        )
        citation = Citation(
            citation_id="citation_graph", claim_id=claim.claim_id, evidence_id=evidence.evidence_id,
            passage_id=passage.passage_id, snapshot_id=snapshot.snapshot_id,
            source_id=source.source_id, locator=passage.locator, quote="grounded quote",
            provenance=provenance,
        )
        conflict = Conflict(
            conflict_id="conflict_graph", claim_ids=(claim.claim_id, counterclaim.claim_id),
            fact_ids=(fact.fact_id,), summary="Candidate disagreement", provenance=provenance,
        )
        section = Section(
            section_id="section_graph", report_id="report_graph", title="Findings", goal="Explain",
            order=1, claim_ids=(claim.claim_id,), citation_ids=(citation.citation_id,),
            content_artifact_id=report_artifact.artifact_id, provenance=provenance,
        )
        report = Report(
            report_id="report_graph", thread_id="thread_graph", run_id=run_id, title="Report",
            research_question="What is grounded?", section_ids=(section.section_id,),
            content_artifact_id=report_artifact.artifact_id, provenance=provenance,
        )
        runtime.repository.save_graph(
            source, snapshot, passage, evidence, fact, claim, counterclaim,
            citation, conflict, report, section,
        )
        assert runtime.repository.related(claim.claim_id, KnowledgeRelation.CLAIM_FACT) == (fact,)
        assert runtime.repository.related(source.source_id, KnowledgeRelation.CITATION_SOURCE, incoming=True) == (citation,)
        assert runtime.repository.related(report.report_id, KnowledgeRelation.SECTION_REPORT, incoming=True) == (section,)
        revised_claim = claim.model_copy(update={"fact_ids": (), "updated_at": utc_now() + timedelta(seconds=1)})
        runtime.repository.claims.save(revised_claim)
        assert claim.claim_id not in {
            item.claim_id for item in runtime.repository.related(fact.fact_id, KnowledgeRelation.CLAIM_FACT, incoming=True)
        }
        with pytest.raises(KnowledgeConflict, match="requires AtomicFact"):
            runtime.repository.claims.save(
                Claim(claim_id="claim_wrong_type", statement="Bad edge", fact_ids=(passage.passage_id,), confidence=0.1, provenance=provenance)
            )
        with pytest.raises(KnowledgeConflict, match="cross runs"):
            runtime.repository.citations.save(
                citation.model_copy(update={
                    "citation_id": "citation_cross_run",
                    "provenance": EntityProvenance(producer_id="producer_test", run_id="run_other"),
                })
            )

        class Vector:
            def score(self, query, entities):
                return {getattr(item, "claim_id", ""): 1.0 for item in entities}

        vector_runtime = runtime.retrieval.__class__(runtime.repository, vector_adapter=Vector(), vector_weight=1.0)
        hits = vector_runtime.search(run_id=run_id, query="unmatched", entity_types=("Claim",))
        assert {hit.entity.claim_id for hit in hits} == {claim.claim_id, counterclaim.claim_id}
        assert all(hit.vector_score == 1.0 for hit in hits)
        runtime.integrity_check()
    finally:
        runtime.close()
