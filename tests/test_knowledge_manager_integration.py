from __future__ import annotations

from typing import Any, Dict

import pytest

from core.session_knowledge import KnowledgeManager
from schemas.state import AtomicFact, Claim, ConflictRecord, DistillerOutputs, Evidence


RESEARCH_ID = "research-km-integration"
SESSION_ID = "session-km-integration"


@pytest.fixture
def knowledge_manager():
    manager = KnowledgeManager(base_storage_path=".", sqlite_filename=":memory:")
    try:
        yield manager
    finally:
        manager.close()


def _fact(*, fact_id: str, text: str, source_url: str, confidence: float = 0.6, task_id: str = "task-1") -> AtomicFact:
    return AtomicFact(
        id=fact_id,
        text=text,
        source_url=source_url,
        confidence=confidence,
        task_id=task_id,
        snippet=f"{text} snippet",
        source_level="A",
        verified_count=1,
    )


def _claim(
    *,
    claim_id: str,
    text: str,
    fact_ids: list[str],
    evidence_ids: list[str],
    confidence: float = 0.7,
    task_id: str = "task-1",
) -> Claim:
    return Claim(
        id=claim_id,
        text=text,
        fact_ids=fact_ids,
        evidence_ids=evidence_ids,
        confidence=confidence,
        task_id=task_id,
    )


def _evidence(
    *,
    evidence_id: str,
    source_url: str,
    quote: str,
    fact_ids: list[str],
    claim_ids: list[str],
    quality_score: float = 0.75,
    task_id: str = "task-1",
) -> Evidence:
    return Evidence(
        id=evidence_id,
        source_id=source_url,
        source_url=source_url,
        quote=quote,
        summary=f"Summary for {quote}",
        fact_ids=fact_ids,
        claim_ids=claim_ids,
        quality_score=quality_score,
        task_id=task_id,
    )


def _conflict(
    *,
    conflict_id: str,
    fact_ids: list[str],
    claim_ids: list[str],
    evidence_ids: list[str],
    description: str,
) -> ConflictRecord:
    return ConflictRecord(
        id=conflict_id,
        fact_ids=fact_ids,
        claim_ids=claim_ids,
        evidence_ids=evidence_ids,
        description=description,
        severity="medium",
    )


def _round_1_output() -> DistillerOutputs:
    source_url = "https://example.com/source-a"
    fact = _fact(
        fact_id="fact-raw-r1",
        text="AI accelerator revenue reached 100 billion dollars.",
        source_url=source_url,
        task_id="task-1",
    )
    claim = _claim(
        claim_id="claim-raw-r1",
        text="AI accelerator revenue reached 100 billion dollars.",
        fact_ids=["fact-raw-r1"],
        evidence_ids=["evidence-raw-r1"],
        task_id="task-1",
    )
    evidence = _evidence(
        evidence_id="evidence-raw-r1",
        source_url=source_url,
        quote="AI accelerator revenue reached 100 billion dollars.",
        fact_ids=["fact-raw-r1"],
        claim_ids=["claim-raw-r1"],
        task_id="task-1",
    )
    conflict = _conflict(
        conflict_id="conflict-raw-r1",
        fact_ids=["fact-raw-r1"],
        claim_ids=["claim-raw-r1"],
        evidence_ids=["evidence-raw-r1"],
        description="Reports differ on whether this total includes edge devices.",
    )
    return DistillerOutputs(
        task_id="task-1",
        atomic_facts=[fact],
        claims=[claim],
        evidence=[evidence],
        conflicts=[conflict],
        fact_ids=["fact-raw-r1", "fake-fact-id"],
        claim_ids=["claim-raw-r1", "fake-claim-id"],
        evidence_ids=["evidence-raw-r1", "fake-evidence-id"],
        conflict_ids=["conflict-raw-r1", "fake-conflict-id"],
        knowledge_refs={
            "collection_name": "spoofed-collection",
            "fact_ids": ["fake-fact-id"],
            "claim_ids": ["fake-claim-id"],
            "evidence_ids": ["fake-evidence-id"],
            "conflict_ids": ["fake-conflict-id"],
            "section_pack_ids": ["fake-pack-id"],
            "source_ids": ["fake-source-id"],
        },
        section_evidence_packs=[
            {
                "pack_id": "pack-raw-r1",
                "section_id": "sec-market",
                "goal": "Summarize market size evidence.",
                "claim_ids": ["claim-raw-r1"],
                "evidence_ids": ["evidence-raw-r1"],
                "fact_ids": ["fact-raw-r1"],
                "conflict_ids": ["conflict-raw-r1"],
                "coverage_score": 0.65,
                "notes": "Round 1 pack",
            }
        ],
        coverage_summary={"avg_section_coverage": 0.65, "sufficiency_level": "partial"},
        unresolved_gaps=["Need one more primary source"],
        summary="Round 1 distilled output",
    )


def _round_2_output() -> DistillerOutputs:
    source_url_a = "https://example.com/source-a"
    source_url_b = "https://example.com/source-b"
    overlap_fact = _fact(
        fact_id="fact-raw-r2-overlap",
        text="AI accelerator revenue reached 100 billion dollars.",
        source_url=source_url_a,
        confidence=0.9,
        task_id="task-2",
    )
    new_fact = _fact(
        fact_id="fact-raw-r2-new",
        text="Cloud GPU spending increased 40 percent year over year.",
        source_url=source_url_b,
        confidence=0.8,
        task_id="task-2",
    )
    overlap_claim = _claim(
        claim_id="claim-raw-r2-overlap",
        text="AI accelerator revenue reached 100 billion dollars.",
        fact_ids=["fact-raw-r2-overlap"],
        evidence_ids=["evidence-raw-r2-overlap"],
        confidence=0.85,
        task_id="task-2",
    )
    new_claim = _claim(
        claim_id="claim-raw-r2-new",
        text="Cloud GPU spending increased 40 percent year over year.",
        fact_ids=["fact-raw-r2-new"],
        evidence_ids=["evidence-raw-r2-new"],
        confidence=0.8,
        task_id="task-2",
    )
    overlap_evidence = _evidence(
        evidence_id="evidence-raw-r2-overlap",
        source_url=source_url_a,
        quote="AI accelerator revenue reached 100 billion dollars.",
        fact_ids=["fact-raw-r2-overlap"],
        claim_ids=["claim-raw-r2-overlap"],
        quality_score=0.8,
        task_id="task-2",
    )
    new_evidence = _evidence(
        evidence_id="evidence-raw-r2-new",
        source_url=source_url_b,
        quote="Cloud GPU spending increased 40 percent year over year.",
        fact_ids=["fact-raw-r2-new"],
        claim_ids=["claim-raw-r2-new"],
        quality_score=0.78,
        task_id="task-2",
    )
    overlap_conflict = _conflict(
        conflict_id="conflict-raw-r2-overlap",
        fact_ids=["fact-raw-r2-overlap"],
        claim_ids=["claim-raw-r2-overlap"],
        evidence_ids=["evidence-raw-r2-overlap"],
        description="Reports differ on whether this total includes edge devices.",
    )
    return DistillerOutputs(
        task_id="task-2",
        atomic_facts=[overlap_fact, new_fact],
        claims=[overlap_claim, new_claim],
        evidence=[overlap_evidence, new_evidence],
        conflicts=[overlap_conflict],
        knowledge_refs={"collection_name": "untrusted-round-2", "fact_ids": ["bogus-round-2"]},
        section_evidence_packs=[
            {
                "pack_id": "pack-raw-r2-overlap",
                "section_id": "sec-market",
                "goal": "Summarize market size evidence.",
                "claim_ids": ["claim-raw-r2-overlap"],
                "evidence_ids": ["evidence-raw-r2-overlap"],
                "fact_ids": ["fact-raw-r2-overlap"],
                "conflict_ids": ["conflict-raw-r2-overlap"],
                "coverage_score": 0.8,
                "notes": "Round 2 updated market pack",
            },
            {
                "pack_id": "pack-raw-r2-new",
                "section_id": "sec-demand",
                "goal": "Summarize demand growth evidence.",
                "claim_ids": ["claim-raw-r2-new"],
                "evidence_ids": ["evidence-raw-r2-new"],
                "fact_ids": ["fact-raw-r2-new"],
                "conflict_ids": [],
                "coverage_score": 0.7,
                "notes": "Round 2 new demand pack",
            },
        ],
        coverage_summary={"avg_section_coverage": 0.8, "sufficiency_level": "sufficient_for_writing"},
        unresolved_gaps=["Need enterprise adoption breakdown"],
        summary="Round 2 distilled output",
    )


def _claim_by_text(result: DistillerOutputs, text: str) -> Claim:
    return next(item for item in result.claims if item.text == text)


def _evidence_by_quote(result: DistillerOutputs, quote: str) -> Evidence:
    return next(item for item in result.evidence if item.quote == quote)


def _pack_by_section(result: DistillerOutputs, section_id: str) -> Dict[str, Any]:
    return next(item for item in result.section_evidence_packs if item["section_id"] == section_id)


def test_manager_ingests_distiller_outputs_into_session_store(knowledge_manager: KnowledgeManager):
    result = knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)

    assert len(result.atomic_facts) == 1
    assert len(result.claims) == 1
    assert len(result.evidence) == 1
    assert len(result.conflicts) == 1
    assert len(result.section_evidence_packs) == 1
    assert len(snapshot["facts"]) == 1
    assert len(snapshot["claims"]) == 1
    assert len(snapshot["evidence"]) == 1
    assert len(snapshot["conflicts"]) == 1
    assert len(snapshot["section_evidence_packs"]) == 1
    assert snapshot["knowledge_refs"]["fact_ids"] == result.knowledge_refs["fact_ids"]


def test_manager_allows_same_source_id_in_different_researches(knowledge_manager: KnowledgeManager):
    source_url = "https://example.com/shared-source"

    def output_for(research_suffix: str) -> DistillerOutputs:
        fact_id = f"fact-{research_suffix}"
        claim_id = f"claim-{research_suffix}"
        evidence_id = f"evidence-{research_suffix}"
        return DistillerOutputs(
            task_id=f"task-{research_suffix}",
            atomic_facts=[
                _fact(
                    fact_id=fact_id,
                    text=f"Shared source supports research {research_suffix}.",
                    source_url=source_url,
                    task_id=f"task-{research_suffix}",
                )
            ],
            claims=[
                _claim(
                    claim_id=claim_id,
                    text=f"Shared source supports research {research_suffix}.",
                    fact_ids=[fact_id],
                    evidence_ids=[evidence_id],
                    task_id=f"task-{research_suffix}",
                )
            ],
            evidence=[
                _evidence(
                    evidence_id=evidence_id,
                    source_url=source_url,
                    quote=f"Shared source supports research {research_suffix}.",
                    fact_ids=[fact_id],
                    claim_ids=[claim_id],
                    task_id=f"task-{research_suffix}",
                )
            ],
        )

    first = knowledge_manager.process_distiller_output(
        output_for("one"),
        research_id="research-one",
        session_id="session-research-one",
        task_id="task-one",
    )
    second = knowledge_manager.process_distiller_output(
        output_for("two"),
        research_id="research-two",
        session_id="session-research-two",
        task_id="task-two",
    )

    first_sources = knowledge_manager.get_session_snapshot("research-one", "session-research-one")["sources"]
    second_sources = knowledge_manager.get_session_snapshot("research-two", "session-research-two")["sources"]

    assert first.knowledge_refs["source_ids"] == [source_url]
    assert second.knowledge_refs["source_ids"] == [source_url]
    assert [row["source_id"] for row in first_sources] == [source_url]
    assert [row["source_id"] for row in second_sources] == [source_url]


def test_manager_remaps_claim_fact_and_evidence_references_consistently(knowledge_manager: KnowledgeManager):
    round_1 = knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    round_2 = knowledge_manager.process_distiller_output(
        _round_2_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-2",
    )

    canonical_fact_id = round_1.atomic_facts[0].id
    canonical_claim_id = round_1.claims[0].id
    canonical_evidence_id = round_1.evidence[0].id
    canonical_conflict_id = round_1.conflicts[0].id

    overlap_claim = _claim_by_text(round_2, "AI accelerator revenue reached 100 billion dollars.")
    overlap_evidence = _evidence_by_quote(round_2, "AI accelerator revenue reached 100 billion dollars.")
    overlap_pack = _pack_by_section(round_2, "sec-market")

    assert overlap_claim.fact_ids == [canonical_fact_id]
    assert overlap_evidence.claim_ids == [canonical_claim_id]
    assert overlap_evidence.fact_ids == [canonical_fact_id]
    assert overlap_pack["fact_ids"] == [canonical_fact_id]
    assert overlap_pack["claim_ids"] == [canonical_claim_id]
    assert overlap_pack["evidence_ids"] == [canonical_evidence_id]
    assert overlap_pack["conflict_ids"] == [canonical_conflict_id]


def test_manager_generates_authoritative_knowledge_refs(knowledge_manager: KnowledgeManager):
    result = knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)

    assert "fake-fact-id" not in result.knowledge_refs["fact_ids"]
    assert "fake-claim-id" not in result.knowledge_refs["claim_ids"]
    assert "fake-evidence-id" not in result.knowledge_refs["evidence_ids"]
    assert "fake-conflict-id" not in result.knowledge_refs["conflict_ids"]
    assert "fake-pack-id" not in result.knowledge_refs["section_pack_ids"]
    assert set(result.knowledge_refs["fact_ids"]) == {item["id"] for item in snapshot["facts"]}
    assert set(result.knowledge_refs["claim_ids"]) == {item["id"] for item in snapshot["claims"]}
    assert set(result.knowledge_refs["evidence_ids"]) == {item["id"] for item in snapshot["evidence"]}
    assert set(result.knowledge_refs["conflict_ids"]) == {item["id"] for item in snapshot["conflicts"]}
    assert set(result.knowledge_refs["section_pack_ids"]) == {item["id"] for item in snapshot["section_evidence_packs"]}
    assert "https://example.com/source-a" in result.knowledge_refs["source_ids"]


def test_manager_persists_novelty_coverage_and_gaps_into_snapshot(knowledge_manager: KnowledgeManager):
    knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)

    assert snapshot["latest_coverage_snapshot"]["sufficiency_level"] == "partial"
    assert snapshot["latest_novelty_snapshot"] is not None
    assert [gap["gap_text"] for gap in snapshot["open_gaps"]] == ["Need one more primary source"]


def test_manager_preserves_original_root_query_and_rolls_back_partial_failures(knowledge_manager: KnowledgeManager, monkeypatch):
    knowledge_manager.store.create_or_get_session(
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        root_query="Original user query",
    )

    result = knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    session = knowledge_manager.store.get_session(RESEARCH_ID)

    assert result.summary == "Round 1 distilled output"
    assert session is not None
    assert session.root_query == "Original user query"

    manager = KnowledgeManager(base_storage_path=".", sqlite_filename=":memory:")
    try:
        def fail_upsert_claims(_rows):
            raise RuntimeError("claim write failed")

        monkeypatch.setattr(manager.store, "upsert_claims", fail_upsert_claims)
        with pytest.raises(RuntimeError, match="claim write failed"):
            manager.process_distiller_output(
                _round_1_output(),
                research_id="research-rollback",
                session_id="session-rollback",
                task_id="task-1",
            )

        assert manager.store.get_session("research-rollback") is None
        assert manager.get_session_snapshot("research-rollback", "session-rollback")["facts"] == []
    finally:
        manager.close()


def test_manager_uses_root_query_payload_when_session_is_created_by_distiller(knowledge_manager: KnowledgeManager):
    payload = _round_1_output().model_dump()
    payload["root_query"] = "Original user query from graph state"

    knowledge_manager.process_distiller_output(
        payload,
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    session = knowledge_manager.store.get_session(RESEARCH_ID)

    assert session is not None
    assert session.root_query == "Original user query from graph state"


def test_manager_backfills_section_ids_and_keeps_source_ids_consistent(knowledge_manager: KnowledgeManager):
    payload = _round_1_output().model_dump()
    payload["sources"] = []
    payload["atomic_facts"][0]["source_id"] = "src-from-fact"
    payload["evidence"][0]["source_id"] = "src-from-evidence"

    knowledge_manager.process_distiller_output(
        payload,
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)

    assert {row["section_id"] for row in snapshot["claims"]} == {"sec-market"}
    assert {row["section_id"] for row in snapshot["facts"]} == {"sec-market"}
    assert {row["section_id"] for row in snapshot["evidence"]} == {"sec-market"}
    assert {row["section_id"] for row in snapshot["conflicts"]} == {"sec-market"}
    assert [row["source_id"] for row in snapshot["sources"]] == ["src-from-evidence"]


def test_manager_preserves_nonempty_pack_when_later_pack_is_empty(knowledge_manager: KnowledgeManager):
    knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    knowledge_manager.process_distiller_output(
        {
            "summary": "Empty distiller output",
            "section_evidence_packs": [
                {
                    "pack_id": "pack-empty-r2",
                    "section_id": "sec-market",
                    "goal": "Summarize market size evidence.",
                    "claim_ids": [],
                    "evidence_ids": [],
                    "fact_ids": [],
                    "conflict_ids": [],
                    "coverage_score": 0.0,
                    "notes": "Empty pack should not erase prior support.",
                }
            ],
            "coverage_summary": {"avg_section_coverage": 0.0, "sufficiency_level": "insufficient"},
        },
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-2",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)
    pack = _pack_by_section(DistillerOutputs(section_evidence_packs=snapshot["section_evidence_packs"]), "sec-market")
    raw_pack = knowledge_manager.store.conn.execute(
        """
        SELECT claim_count, fact_count, evidence_count, coverage_score, metadata_json
        FROM session_section_packs
        WHERE research_id = ? AND section_id = ?
        """,
        (RESEARCH_ID, "sec-market"),
    ).fetchone()
    latest_coverage = snapshot["latest_coverage_snapshot"]

    assert pack["coverage_score"] == pytest.approx(0.65)
    assert pack["claim_ids"]
    assert pack["fact_ids"]
    assert pack["evidence_ids"]
    assert pack["notes"] == "Round 1 pack"
    assert raw_pack["claim_count"] == 1
    assert raw_pack["fact_count"] == 1
    assert raw_pack["evidence_count"] == 1
    assert latest_coverage["round_no"] == 1
    assert latest_coverage["sufficiency_level"] == "partial"


def test_manager_handles_cross_round_overlap_and_new_knowledge_correctly(knowledge_manager: KnowledgeManager):
    round_1 = knowledge_manager.process_distiller_output(
        _round_1_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    round_2 = knowledge_manager.process_distiller_output(
        _round_2_output(),
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-2",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)

    assert len(snapshot["facts"]) == 2
    assert len(snapshot["claims"]) == 2
    assert len(snapshot["evidence"]) == 2
    assert len(snapshot["conflicts"]) == 1
    assert len(snapshot["section_evidence_packs"]) == 2

    overlap_claim = _claim_by_text(round_2, "AI accelerator revenue reached 100 billion dollars.")
    new_claim = _claim_by_text(round_2, "Cloud GPU spending increased 40 percent year over year.")
    new_pack = _pack_by_section(round_2, "sec-demand")

    assert overlap_claim.id == round_1.claims[0].id
    assert new_claim.id != round_1.claims[0].id
    assert new_pack["fact_ids"] == [new_claim.fact_ids[0]]
    assert set(round_2.knowledge_refs["claim_ids"]) == {item["id"] for item in snapshot["claims"]}
    assert set(round_2.knowledge_refs["fact_ids"]) == {item["id"] for item in snapshot["facts"]}
    assert set(round_2.knowledge_refs["section_pack_ids"]) == {item["id"] for item in snapshot["section_evidence_packs"]}


def test_manager_replaces_pack_links_for_updated_section_pack(knowledge_manager: KnowledgeManager):
    round_1 = _round_1_output()
    round_2 = _round_2_output()
    round_2.section_evidence_packs = [round_2.section_evidence_packs[1] | {"section_id": "sec-market"}]

    knowledge_manager.process_distiller_output(
        round_1,
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-1",
    )
    knowledge_manager.process_distiller_output(
        round_2,
        research_id=RESEARCH_ID,
        session_id=SESSION_ID,
        task_id="task-2",
    )
    snapshot = knowledge_manager.get_session_snapshot(RESEARCH_ID, SESSION_ID)
    pack = _pack_by_section(DistillerOutputs(section_evidence_packs=snapshot["section_evidence_packs"]), "sec-market")

    assert pack["claim_ids"] == [round_2.claims[1].id]
    assert pack["fact_ids"] == [round_2.atomic_facts[1].id]
    assert pack["evidence_ids"] == [round_2.evidence[1].id]
