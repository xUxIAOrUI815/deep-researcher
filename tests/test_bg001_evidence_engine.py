from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from deep_researcher.artifacts import ArtifactQuery
from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    BudgetUsage,
    Citation,
    CitationStatus,
    Claim,
    ClaimStatus,
    ComponentKind,
    ComponentVersionSet,
    Conflict,
    ConflictResolutionKind,
    ConflictSeverity,
    ConflictStatus,
    EntityProvenance,
    EventType,
    Evidence,
    EvidenceQuote,
    EvidenceRelation,
    EvidenceStatus,
    FactStatus,
    Passage,
    PassageStatus,
    Report,
    RunEvent,
    RunStatus,
    Section,
    SectionCoverageStatus,
    SnapshotStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    SourceType,
    SpanKind,
    VersionRef,
    utc_now,
)
from deep_researcher.evidence import (
    AgentSpecSemanticVerificationAdapter,
    DeterministicSemanticVerificationAdapter,
    EventRecorderEvidenceSink,
    EvidenceDomainEvent,
    EvidenceTraceContext,
    EvidenceVerificationError,
    RecordingEvidenceEventSink,
    SemanticLabel,
    VerificationPolicy,
    build_evidence_runtime,
    build_evidence_verifier_spec,
)
from deep_researcher.events import EventQuery, EventRecorder, SQLiteEventStore
from deep_researcher.kernel import ModelRequest, ModelResponse


@dataclass(frozen=True)
class SeededGraph:
    run_id: str
    claim: Claim
    fact: AtomicFact
    evidence: tuple[Evidence, ...]
    citations: tuple[Citation, ...]
    sources: tuple[Source, ...]
    snapshots: tuple[SourceSnapshot, ...]
    passages: tuple[Passage, ...]
    report: Report | None
    section: Section | None


def _policy(**updates: Any) -> VerificationPolicy:
    values: dict[str, Any] = {
        "policy_version_id": "policy_evidence_test_1",
        "freshness_days": 365,
        "max_repair_rounds": 2,
    }
    values.update(updates)
    return VerificationPolicy(**values)


def _provenance(
    run_id: str,
    *artifact_ids: str,
    producer_id: str = "agent_distiller",
) -> EntityProvenance:
    return EntityProvenance(
        producer_id=producer_id,
        run_id=run_id,
        task_id=f"task_{run_id}",
        source_artifact_ids=tuple(artifact_ids),
    )


def _seed_claim(
    runtime,
    *,
    run_id: str,
    statement: str,
    relations_and_text: tuple[tuple[EvidenceRelation, str], ...],
    high_impact: bool = False,
    importance: float = 0.6,
    stale: bool = False,
    same_publisher: bool = False,
    corrupt_passage_hash: bool = False,
    include_section: bool = True,
) -> SeededGraph:
    now = utc_now()
    sources: list[Source] = []
    snapshots: list[SourceSnapshot] = []
    passages: list[Passage] = []
    evidence_items: list[Evidence] = []
    citations: list[Citation] = []
    input_artifact_ids: list[str] = []
    entities: list[Any] = []

    for index, (relation, text) in enumerate(relations_and_text):
        suffix = f"{run_id}_{index}"
        snapshot_artifact = runtime.knowledge.artifacts.put_text(
            text,
            kind=ArtifactKind.SOURCE_SNAPSHOT,
            producer_id="tool_scraper",
            run_id=run_id,
            task_id=f"task_{run_id}",
            content_schema="SourceSnapshotText@1",
        )
        passage_artifact = runtime.knowledge.artifacts.put_text(
            text,
            kind=ArtifactKind.CLEANED_CONTENT,
            producer_id="agent_worker",
            run_id=run_id,
            task_id=f"task_{run_id}",
            content_schema="CleanedPassage@1",
            source_artifact_ids=(snapshot_artifact.artifact_id,),
        )
        input_artifact_ids.append(passage_artifact.artifact_id)
        source = Source(
            source_id=f"source_{suffix}",
            canonical_url=f"https://source-{index}.{run_id}.example/evidence",
            source_type=(
                SourceType.OFFICIAL_DOCUMENTATION if index == 0 else SourceType.NEWS
            ),
            source_level=SourceLevel.PRIMARY if index == 0 else SourceLevel.SECONDARY,
            status=SourceStatus.ACCESSIBLE,
            title=f"Source {index}",
            publisher="Shared Publisher" if same_publisher else f"Publisher {index}",
            authority_score=0.92,
            published_at=now - timedelta(days=900 if stale else 10),
            provenance=_provenance(
                run_id, snapshot_artifact.artifact_id, producer_id="tool_scraper"
            ),
        )
        snapshot = SourceSnapshot(
            snapshot_id=f"snapshot_{suffix}",
            source_id=source.source_id,
            artifact_id=snapshot_artifact.artifact_id,
            status=SnapshotStatus.NORMALIZED,
            source_level=source.source_level,
            source_version=1,
            content_hash=snapshot_artifact.content_hash,
            final_url=source.canonical_url,
            media_type="text/plain",
            http_status=200,
            fetched_at=now - timedelta(days=1),
            capture_method="governed_scraper",
            provenance=_provenance(
                run_id, snapshot_artifact.artifact_id, producer_id="tool_scraper"
            ),
        )
        passage_hash = (
            "0" * 64
            if corrupt_passage_hash and index == 0
            else passage_artifact.content_hash
        )
        passage = Passage(
            passage_id=f"passage_{suffix}",
            snapshot_id=snapshot.snapshot_id,
            text_artifact_id=passage_artifact.artifact_id,
            ordinal=0,
            locator=f"chars:0-{len(text)}",
            content_hash=passage_hash,
            extraction_method="dom_text_v1",
            extracted_at=now,
            char_start=0,
            char_end=len(text),
            language="en",
            status=PassageStatus.ACCEPTED,
            provenance=_provenance(
                run_id, passage_artifact.artifact_id, producer_id="agent_worker"
            ),
        )
        quote = EvidenceQuote(
            passage_id=passage.passage_id,
            quote=text,
            char_start=0,
            char_end=len(text),
            passage_content_hash=passage_hash,
            extraction_method=passage.extraction_method,
            extracted_at=passage.extracted_at,
        )
        evidence = Evidence(
            evidence_id=f"evidence_{suffix}",
            passage_ids=(passage.passage_id,),
            relation=relation,
            summary=f"{relation.value} candidate for {statement}",
            confidence=0.95,
            relevance=0.95,
            source_quality=0.92,
            quotes=(quote,),
            provenance=_provenance(run_id, passage_artifact.artifact_id),
        )
        citation = Citation(
            citation_id=f"citation_{suffix}",
            claim_id=f"claim_{run_id}",
            evidence_id=evidence.evidence_id,
            passage_id=passage.passage_id,
            snapshot_id=snapshot.snapshot_id,
            source_id=source.source_id,
            locator=passage.locator,
            quote=text,
            extraction_method=passage.extraction_method,
            provenance=_provenance(run_id, passage_artifact.artifact_id),
        )
        sources.append(source)
        snapshots.append(snapshot)
        passages.append(passage)
        evidence_items.append(evidence)
        citations.append(citation)
        entities.extend((source, snapshot, passage, evidence, citation))

    fact = AtomicFact(
        fact_id=f"fact_{run_id}",
        statement=statement,
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
        confidence=0.9,
        provenance=_provenance(run_id, *input_artifact_ids),
    )
    claim = Claim(
        claim_id=f"claim_{run_id}",
        statement=statement,
        fact_ids=(fact.fact_id,),
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
        confidence=0.9,
        importance=importance,
        high_impact=high_impact,
        provenance=_provenance(run_id, *input_artifact_ids),
    )
    report: Report | None = None
    section: Section | None = None
    if include_section:
        section = Section(
            section_id=f"section_{run_id}",
            report_id=f"report_{run_id}",
            title="Verified findings",
            goal="Present only independently verified findings.",
            order=0,
            claim_ids=(claim.claim_id,),
            required_claim_ids=(claim.claim_id,),
            citation_ids=tuple(item.citation_id for item in citations),
            provenance=_provenance(run_id, *input_artifact_ids),
        )
        report = Report(
            report_id=f"report_{run_id}",
            thread_id=f"thread_{run_id}",
            run_id=run_id,
            title="Evidence report",
            research_question=statement,
            section_ids=(section.section_id,),
            provenance=_provenance(run_id, *input_artifact_ids),
        )
        entities.extend((report, section))
    runtime.knowledge.repository.save_graph(*entities, fact, claim)
    return SeededGraph(
        run_id=run_id,
        claim=claim,
        fact=fact,
        evidence=tuple(evidence_items),
        citations=tuple(citations),
        sources=tuple(sources),
        snapshots=tuple(snapshots),
        passages=tuple(passages),
        report=report,
        section=section,
    )


@pytest.mark.asyncio
async def test_supported_high_impact_graph_is_traceable_separated_and_replay_idempotent(
    tmp_path,
):
    sink = RecordingEvidenceEventSink()
    root = tmp_path / "evidence"
    runtime = build_evidence_runtime(
        root,
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=sink,
        policy=_policy(),
    )
    try:
        statement = "The verified system stores immutable source snapshots."
        seeded = _seed_claim(
            runtime,
            run_id="run_supported",
            statement=statement,
            relations_and_text=(
                (EvidenceRelation.SUPPORTS, statement),
                (EvidenceRelation.SUPPORTS, statement),
            ),
            high_impact=True,
            importance=0.95,
        )
        assert runtime.candidates.claims(seeded.run_id) == (seeded.claim,)
        before_snapshot_history = {
            item.snapshot_id: len(
                runtime.knowledge.repository.snapshots.history(item.snapshot_id)
            )
            for item in seeded.snapshots
        }

        result = await runtime.engine.verify_claim(seeded.claim.claim_id)
        section = await runtime.engine.assess_section(seeded.section.section_id)

        assert result.status == ClaimStatus.SUPPORTED
        assert result.verification_result.passed is True
        assert result.independent_source_count == 2
        assert set(result.verified_evidence_ids) == {
            item.evidence_id for item in seeded.evidence
        }
        assert set(result.verified_citation_ids) == {
            item.citation_id for item in seeded.citations
        }
        assert runtime.candidates.claims(seeded.run_id) == ()
        assert (
            runtime.verified.claims_for_writing(seeded.run_id)[0].claim_id
            == seeded.claim.claim_id
        )
        assert runtime.verified.blocked_high_impact_claims(seeded.run_id) == ()
        assert all(
            item.status == EvidenceStatus.VERIFIED
            for item in runtime.verified.evidence(seeded.run_id)
        )
        assert (
            runtime.knowledge.repository.facts.require(seeded.fact.fact_id).status
            == FactStatus.VERIFIED
        )
        assert all(
            runtime.knowledge.repository.citations.require(item.citation_id).quote_start
            == 0
            for item in seeded.citations
        )
        assert section.blocked is False
        assert section.coverage_score == 1.0
        assert section.citation_score == 1.0
        assert (
            runtime.knowledge.repository.sections.require(
                seeded.section.section_id
            ).coverage_status
            == SectionCoverageStatus.COMPLETE
        )
        assert {
            item.snapshot_id: len(
                runtime.knowledge.repository.snapshots.history(item.snapshot_id)
            )
            for item in seeded.snapshots
        } == before_snapshot_history
        replayed_candidates = tuple(
            runtime.knowledge.ingestion._stable_entity(item)
            for item in (
                *seeded.evidence,
                seeded.fact,
                seeded.claim,
                seeded.section,
            )
        )
        assert [item.status for item in replayed_candidates] == [
            EvidenceStatus.VERIFIED,
            EvidenceStatus.VERIFIED,
            FactStatus.VERIFIED,
            ClaimStatus.SUPPORTED,
            seeded.section.status,
        ]
        runtime.knowledge.repository.save_graph(*replayed_candidates)
        revision_counts = {
            item: len(runtime.knowledge.storage.get_history(item))
            for item in (
                seeded.claim.claim_id,
                seeded.fact.fact_id,
                *(value.evidence_id for value in seeded.evidence),
                *(value.citation_id for value in seeded.citations),
                seeded.section.section_id,
            )
        }
        artifact_count = len(
            runtime.knowledge.artifacts.list(
                ArtifactQuery(run_id=seeded.run_id, limit=1000)
            ).items
        )
        event_count = len(sink.events)

        replayed = await runtime.engine.verify_claim(seeded.claim.claim_id)
        replayed_section = await runtime.engine.assess_section(
            seeded.section.section_id
        )
        run_summary = await runtime.engine.verify_run(seeded.run_id)

        assert replayed == result
        assert replayed_section.result_artifact_id == section.result_artifact_id
        assert run_summary.blocked_high_impact_claim_ids == ()
        assert run_summary.open_severe_conflict_ids == ()
        assert run_summary.claim_results == (result,)
        assert run_summary.section_results == (section,)
        assert {
            item: len(runtime.knowledge.storage.get_history(item))
            for item in revision_counts
        } == revision_counts
        assert (
            len(
                runtime.knowledge.artifacts.list(
                    ArtifactQuery(run_id=seeded.run_id, limit=1000)
                ).items
            )
            == artifact_count
        )
        assert len(sink.events) == event_count
        assert {item.event_type for item in sink.events} == {
            EventType.EVIDENCE_CHANGED,
            EventType.VERIFICATION_COMPLETED,
        }
        runtime.integrity_check()
    finally:
        runtime.close()

    reopened_sink = RecordingEvidenceEventSink()
    reopened = build_evidence_runtime(
        root,
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=reopened_sink,
        policy=_policy(),
    )
    try:
        replayed = await reopened.engine.verify_claim("claim_run_supported")
        assert replayed.status == ClaimStatus.SUPPORTED
        assert reopened.verified.claims_for_writing("run_supported")
        assert len(reopened_sink.events) == 2
    finally:
        reopened.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("suffix", "relations_and_text", "expected", "options"),
    (
        (
            "partial",
            ((EvidenceRelation.SUPPORTS, "A high impact candidate is grounded."),),
            ClaimStatus.PARTIALLY_SUPPORTED,
            {"statement": "A high impact candidate is grounded.", "high_impact": True},
        ),
        (
            "contradicted",
            ((EvidenceRelation.REFUTES, "The metric equals 12."),),
            ClaimStatus.CONTRADICTED,
            {"statement": "The metric equals 12."},
        ),
        (
            "conflicted",
            (
                (EvidenceRelation.SUPPORTS, "The result is 42."),
                (EvidenceRelation.REFUTES, "The result is 42."),
            ),
            ClaimStatus.CONFLICTED,
            {"statement": "The result is 42."},
        ),
        (
            "unsupported",
            (
                (
                    EvidenceRelation.SUPPORTS,
                    "This passage discusses an unrelated topic.",
                ),
            ),
            ClaimStatus.UNSUPPORTED,
            {"statement": "The system guarantees exactly 99 percent accuracy."},
        ),
        (
            "stale",
            ((EvidenceRelation.SUPPORTS, "The historical policy is active."),),
            ClaimStatus.STALE,
            {"statement": "The historical policy is active.", "stale": True},
        ),
    ),
)
async def test_exact_claim_verification_states_and_bounded_feedback(
    tmp_path,
    suffix,
    relations_and_text,
    expected,
    options,
):
    sink = RecordingEvidenceEventSink()
    runtime = build_evidence_runtime(
        tmp_path / suffix,
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=sink,
        policy=_policy(),
    )
    try:
        statement = options["statement"]
        seeded = _seed_claim(
            runtime,
            run_id=f"run_{suffix}",
            statement=statement,
            relations_and_text=relations_and_text,
            high_impact=options.get("high_impact", False),
            stale=options.get("stale", False),
        )
        result = await runtime.engine.verify_claim(seeded.claim.claim_id)
        section = await runtime.engine.assess_section(seeded.section.section_id)
        assert result.status == expected
        assert result.verification_result.passed is False
        assert result.verification_result.repair_requests
        assert all(item.repairable for item in result.verification_result.issues)
        assert runtime.verified.claims_for_writing(seeded.run_id) == ()
        assert section.unsupported_claim_ids == (seeded.claim.claim_id,)
        if options.get("high_impact"):
            assert result.high_impact_blocked is True
            assert section.blocked is True
            assert runtime.verified.blocked_high_impact_claims(seeded.run_id)
        if expected == ClaimStatus.CONFLICTED:
            assert section.conflicted_claim_ids == (seeded.claim.claim_id,)
            assert section.blocked is True
        if expected == ClaimStatus.STALE:
            assert result.stale_source_ids == (seeded.sources[0].source_id,)
        repair_artifacts = runtime.knowledge.artifacts.list(
            ArtifactQuery(
                run_id=seeded.run_id,
                kinds=(ArtifactKind.REPAIR_FEEDBACK,),
                limit=100,
            )
        ).items
        assert len(repair_artifacts) == 1
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_rejected_extra_evidence_does_not_poison_valid_support_path(
    tmp_path,
):
    runtime = build_evidence_runtime(
        tmp_path / "mixed-support",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_policy(),
    )
    try:
        statement = "The verified method retrieves relevant records."
        seeded = _seed_claim(
            runtime,
            run_id="run_mixed_support",
            statement=statement,
            relations_and_text=(
                (EvidenceRelation.SUPPORTS, statement),
                (
                    EvidenceRelation.SUPPORTS,
                    "This candidate passage discusses an unrelated topic.",
                ),
            ),
        )
        result = await runtime.engine.verify_claim(seeded.claim.claim_id)

        assert result.status == ClaimStatus.SUPPORTED
        assert result.verification_result.passed is True
        assert result.verified_evidence_ids == (
            seeded.evidence[0].evidence_id,
        )
        assert result.verified_citation_ids == (
            seeded.citations[0].citation_id,
        )
        assert (
            runtime.knowledge.repository.evidence.require(
                seeded.evidence[1].evidence_id
            ).status
            == EvidenceStatus.REJECTED
        )
        assert "evidence_relation_not_semantically_grounded" in {
            item.code for item in result.verification_result.issues
        }
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_integrity_and_citation_failures_reject_candidates_and_stop_repairs_at_bound(
    tmp_path,
):
    runtime = build_evidence_runtime(
        tmp_path / "integrity",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_policy(),
    )
    try:
        statement = "This statement has a persisted quote."
        seeded = _seed_claim(
            runtime,
            run_id="run_integrity",
            statement=statement,
            relations_and_text=((EvidenceRelation.SUPPORTS, statement),),
            corrupt_passage_hash=True,
        )
        bad_citation = seeded.citations[0].model_copy(
            update={"locator": "wrong-locator"}
        )
        runtime.knowledge.repository.citations.save(bad_citation)
        result = await runtime.engine.verify_claim(
            seeded.claim.claim_id,
            repair_round=runtime.engine.policy.max_repair_rounds,
        )
        codes = {item.code for item in result.verification_result.issues}
        assert result.status == ClaimStatus.UNSUPPORTED
        assert "passage_integrity_failed" in codes
        assert "citation_location_mismatch" in codes
        assert result.verification_result.repair_requests == ()
        assert all(
            item.repairable is False for item in result.verification_result.issues
        )
        assert (
            runtime.knowledge.repository.evidence.require(
                seeded.evidence[0].evidence_id
            ).status
            == EvidenceStatus.REJECTED
        )
        assert (
            runtime.knowledge.repository.citations.require(
                seeded.citations[0].citation_id
            ).status
            == CitationStatus.REJECTED
        )
        assert runtime.knowledge.repository.snapshots.history(
            seeded.snapshots[0].snapshot_id
        ) == (seeded.snapshots[0],)
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_result_artifact_recovers_projection_and_missing_repair_feedback_after_crash(
    tmp_path,
):
    root = tmp_path / "crash_recovery"
    runtime = build_evidence_runtime(
        root,
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_policy(),
    )
    statement = "A high-impact claim initially has only one source."
    seeded = _seed_claim(
        runtime,
        run_id="run_crash_recovery",
        statement=statement,
        relations_and_text=((EvidenceRelation.SUPPORTS, statement),),
        high_impact=True,
    )

    class CrashBeforeRepairArtifact:
        def __init__(self, delegate) -> None:
            self.delegate = delegate
            self.failed = False

        def __getattr__(self, name):
            return getattr(self.delegate, name)

        def put_json(self, value, **kwargs):
            if kwargs.get("kind") == ArtifactKind.REPAIR_FEEDBACK and not self.failed:
                self.failed = True
                raise RuntimeError("simulated crash before repair feedback commit")
            return self.delegate.put_json(value, **kwargs)

    runtime.engine.artifact_store = CrashBeforeRepairArtifact(
        runtime.knowledge.artifacts
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        await runtime.engine.verify_claim(seeded.claim.claim_id)
    assert (
        runtime.knowledge.repository.claims.require(seeded.claim.claim_id).status
        == ClaimStatus.DRAFT
    )
    assert (
        len(
            runtime.knowledge.artifacts.list(
                ArtifactQuery(
                    run_id=seeded.run_id,
                    kinds=(ArtifactKind.VERIFICATION_RESULT,),
                    limit=100,
                )
            ).items
        )
        == 1
    )
    assert (
        runtime.knowledge.artifacts.list(
            ArtifactQuery(
                run_id=seeded.run_id,
                kinds=(ArtifactKind.REPAIR_FEEDBACK,),
                limit=100,
            )
        ).items
        == ()
    )
    runtime.close()

    sink = RecordingEvidenceEventSink()
    recovered = build_evidence_runtime(
        root,
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=sink,
        policy=_policy(),
    )
    try:
        result = await recovered.engine.verify_claim(seeded.claim.claim_id)
        assert result.status == ClaimStatus.PARTIALLY_SUPPORTED
        assert (
            recovered.knowledge.repository.claims.require(seeded.claim.claim_id).status
            == result.status
        )
        assert recovered.knowledge.artifacts.list(
            ArtifactQuery(
                run_id=seeded.run_id,
                kinds=(ArtifactKind.REPAIR_FEEDBACK,),
                limit=100,
            )
        ).items
        assert len(sink.events) == 2
        recovered.integrity_check()
    finally:
        recovered.close()


@pytest.mark.asyncio
async def test_source_independence_uses_publisher_identity(tmp_path):
    runtime = build_evidence_runtime(
        tmp_path / "independence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_policy(),
    )
    try:
        statement = "Two pages from one publisher are not independent."
        seeded = _seed_claim(
            runtime,
            run_id="run_independence",
            statement=statement,
            relations_and_text=(
                (EvidenceRelation.SUPPORTS, statement),
                (EvidenceRelation.SUPPORTS, statement),
            ),
            high_impact=True,
            same_publisher=True,
        )
        result = await runtime.engine.verify_claim(seeded.claim.claim_id)
        assert result.status == ClaimStatus.PARTIALLY_SUPPORTED
        assert result.independent_source_count == 1
        assert "insufficient_independent_sources" in {
            item.code for item in result.verification_result.issues
        }
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_changed_graph_input_invalidates_old_result_and_reverifies(tmp_path):
    sink = RecordingEvidenceEventSink()
    runtime = build_evidence_runtime(
        tmp_path / "input_revision",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=sink,
        policy=_policy(),
    )
    try:
        statement = "The current policy requires immutable evidence."
        seeded = _seed_claim(
            runtime,
            run_id="run_input_revision",
            statement=statement,
            relations_and_text=(
                (EvidenceRelation.SUPPORTS, statement),
                (EvidenceRelation.SUPPORTS, statement),
            ),
            high_impact=True,
            importance=0.95,
        )
        first = await runtime.engine.verify_claim(seeded.claim.claim_id)
        assert first.status == ClaimStatus.SUPPORTED

        source = runtime.knowledge.repository.sources.require(
            seeded.sources[1].source_id
        )
        runtime.knowledge.repository.sources.save(
            source.model_copy(
                update={
                    "authority_score": 0.2,
                    "updated_at": utc_now(),
                    "metadata": {
                        **source.metadata,
                        "revision_reason": "authority reassessment",
                    },
                }
            )
        )
        second = await runtime.engine.verify_claim(seeded.claim.claim_id)
        section = await runtime.engine.assess_section(seeded.section.section_id)

        assert second.verification_result.verification_id != (
            first.verification_result.verification_id
        )
        assert second.status == ClaimStatus.PARTIALLY_SUPPORTED
        assert second.high_impact_blocked is True
        assert "source_authority_below_policy" in {
            item.code for item in second.verification_result.issues
        }
        assert runtime.verified.claims_for_writing(seeded.run_id) == ()
        assert section.blocked is True
        assert (
            len(
                [
                    item
                    for item in sink.events
                    if item.event_type == EventType.VERIFICATION_COMPLETED
                ]
            )
            == 2
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_citation_quote_backfills_evidence_without_breaking_idempotency(tmp_path):
    runtime = build_evidence_runtime(
        tmp_path / "citation_backfill",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_policy(),
    )
    try:
        statement = "A legacy candidate can carry its quote on the citation."
        seeded = _seed_claim(
            runtime,
            run_id="run_citation_backfill",
            statement=statement,
            relations_and_text=((EvidenceRelation.SUPPORTS, statement),),
        )
        evidence = runtime.knowledge.repository.evidence.require(
            seeded.evidence[0].evidence_id
        )
        runtime.knowledge.repository.evidence.save(
            evidence.model_copy(update={"quotes": (), "updated_at": utc_now()})
        )

        result = await runtime.engine.verify_claim(seeded.claim.claim_id)
        history_count = len(
            runtime.knowledge.repository.evidence.history(evidence.evidence_id)
        )
        replayed = await runtime.engine.verify_claim(seeded.claim.claim_id)

        assert result.status == ClaimStatus.SUPPORTED
        assert replayed == result
        assert runtime.knowledge.repository.evidence.require(
            evidence.evidence_id
        ).quotes
        assert (
            len(runtime.knowledge.repository.evidence.history(evidence.evidence_id))
            == history_count
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_severe_conflict_stays_visible_and_resolution_requires_verified_evidence(
    tmp_path,
):
    sink = RecordingEvidenceEventSink()
    runtime = build_evidence_runtime(
        tmp_path / "conflict",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=sink,
        policy=_policy(),
    )
    try:
        statement = "The release date is July 24."
        seeded = _seed_claim(
            runtime,
            run_id="run_conflict",
            statement=statement,
            relations_and_text=(
                (EvidenceRelation.SUPPORTS, statement),
                (EvidenceRelation.SUPPORTS, statement),
            ),
            high_impact=True,
            importance=0.95,
        )
        counterclaim = Claim(
            claim_id="claim_run_conflict_counter",
            statement="The release date is July 25.",
            confidence=0.7,
            importance=0.95,
            high_impact=True,
            provenance=_provenance(seeded.run_id),
        )
        conflict = Conflict(
            conflict_id="conflict_run_conflict",
            claim_ids=(seeded.claim.claim_id, counterclaim.claim_id),
            summary="Two candidate dates disagree.",
            provenance=_provenance(seeded.run_id),
        )
        runtime.knowledge.repository.save_graph(counterclaim, conflict)
        with pytest.raises(EvidenceVerificationError, match="verified evidence"):
            runtime.engine.resolve_conflict(
                conflict.conflict_id,
                resolution="Prefer the primary record.",
                resolution_kind=ConflictResolutionKind.SOURCE_PRECEDENCE,
                resolution_evidence_ids=(seeded.evidence[0].evidence_id,),
            )

        result = await runtime.engine.verify_claim(seeded.claim.claim_id)
        visible = runtime.knowledge.repository.conflicts.require(conflict.conflict_id)
        assert result.status == ClaimStatus.CONFLICTED
        assert visible.severity == ConflictSeverity.CRITICAL
        assert visible.status == ConflictStatus.OPEN

        resolved = runtime.engine.resolve_conflict(
            conflict.conflict_id,
            resolution="The independently verified primary record controls.",
            resolution_kind=ConflictResolutionKind.SOURCE_PRECEDENCE,
            resolution_evidence_ids=(seeded.evidence[0].evidence_id,),
        )
        assert resolved.status == ConflictStatus.RESOLVED
        assert resolved.resolution_kind == ConflictResolutionKind.SOURCE_PRECEDENCE
        assert resolved.resolution_evidence_ids == (seeded.evidence[0].evidence_id,)
        assert (
            runtime.knowledge.repository.claims.require(seeded.claim.claim_id).status
            == ClaimStatus.CONTESTED
        )
        reverified = await runtime.engine.verify_claim(seeded.claim.claim_id)
        assert reverified.status == ClaimStatus.SUPPORTED
        assert any(
            item.payload.get("change") == "conflict_resolved" for item in sink.events
        )

        accepted_conflict = Conflict(
            conflict_id="conflict_run_conflict_accepted",
            claim_ids=(seeded.claim.claim_id, counterclaim.claim_id),
            summary="The remaining scope disagreement is retained for readers.",
            severity=ConflictSeverity.HIGH,
            high_impact=True,
            provenance=_provenance(seeded.run_id),
        )
        runtime.knowledge.repository.conflicts.save(accepted_conflict)
        accepted = runtime.engine.accept_conflict_unresolved(
            accepted_conflict.conflict_id,
            reason="Available verified sources do not resolve the scope difference.",
        )
        assert accepted.status == ConflictStatus.ACCEPTED_UNRESOLVED
        assert accepted.resolution_kind == ConflictResolutionKind.ACCEPTED_UNRESOLVED
        assert any(
            item.payload.get("change") == "conflict_accepted_unresolved"
            for item in sink.events
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_verifier_agent_spec_and_model_schema_repair_are_bounded():
    spec = build_evidence_verifier_spec()
    assert spec.allowed_commands == tuple(
        command
        for command in spec.allowed_commands
        if command.value in {"review", "stop"}
    )
    assert spec.tool_grants == ()
    assert spec.supports_delegation is False
    assert "search" in spec.metadata["forbidden_capabilities"]

    class RepairingModel:
        def __init__(self) -> None:
            self.complete_calls = 0
            self.repair_calls = 0
            self.requests: list[ModelRequest] = []

        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.complete_calls += 1
            self.requests.append(request)
            return ModelResponse(
                structured={"label": "invalid"},
                usage=BudgetUsage(input_tokens=10, model_calls=1),
            )

        async def repair(
            self,
            request: ModelRequest,
            invalid_response: ModelResponse,
            errors: tuple[str, ...],
        ) -> ModelResponse:
            del request, invalid_response
            assert errors
            self.repair_calls += 1
            return ModelResponse(
                structured={
                    "label": "supports",
                    "score": 0.93,
                    "decision_summary": "The persisted quote directly states the claim.",
                },
                usage=BudgetUsage(output_tokens=12, model_calls=1, retries=1),
            )

    model = RepairingModel()
    adapter = AgentSpecSemanticVerificationAdapter(agent_spec=spec, model=model)
    judgment = await adapter.judge(
        run_id="run_semantic",
        task_id="task_semantic",
        subject_id="claim_semantic",
        statement="The source states the result.",
        evidence_id="evidence_semantic",
        relation=EvidenceRelation.SUPPORTS,
        passages=("The source states the result.",),
    )
    assert judgment.label == SemanticLabel.SUPPORTS
    assert judgment.score == 0.93
    assert judgment.usage.input_tokens == 10
    assert judgment.usage.output_tokens == 12
    assert judgment.usage.model_calls == 2
    assert judgment.usage.retries == 1
    assert model.complete_calls == 1
    assert model.repair_calls == 1
    assert model.requests[0].metadata["agent_spec_id"] == spec.agent_spec_id


def test_verified_entity_state_transitions_clear_verification_identity():
    now = utc_now()
    provenance = _provenance("run_transition")
    evidence = Evidence(
        evidence_id="evidence_transition",
        passage_ids=("passage_transition",),
        relation=EvidenceRelation.SUPPORTS,
        status=EvidenceStatus.VERIFIED,
        summary="Verified evidence",
        confidence=0.9,
        relevance=0.9,
        source_quality=0.9,
        verification_id="verification_transition",
        verified_at=now,
        provenance=provenance,
    )
    fact = AtomicFact(
        fact_id="fact_transition",
        statement="A verified fact.",
        evidence_ids=(evidence.evidence_id,),
        status=FactStatus.VERIFIED,
        confidence=0.9,
        verification_id="verification_transition",
        verified_at=now,
        provenance=provenance,
    )
    claim = Claim(
        claim_id="claim_transition",
        statement="A verified claim.",
        fact_ids=(fact.fact_id,),
        evidence_ids=(evidence.evidence_id,),
        status=ClaimStatus.SUPPORTED,
        confidence=0.9,
        verification_id="verification_transition",
        verified_at=now,
        provenance=provenance,
    )
    citation = Citation(
        citation_id="citation_transition",
        claim_id=claim.claim_id,
        evidence_id=evidence.evidence_id,
        passage_id="passage_transition",
        snapshot_id="snapshot_transition",
        source_id="source_transition",
        locator="chars:0-8",
        quote="grounded",
        quote_start=0,
        quote_end=8,
        status=CitationStatus.VERIFIED,
        verification_id="verification_transition",
        verified_at=now,
        provenance=provenance,
    )
    assert evidence.transition(EvidenceStatus.REJECTED).verification_id is None
    assert fact.transition(FactStatus.DISPUTED).verification_id is None
    assert claim.transition(ClaimStatus.CONTESTED).verification_id is None
    assert citation.transition(CitationStatus.REJECTED).verification_id is None


def _version(kind: ComponentKind, name: str) -> VersionRef:
    return VersionRef(kind=kind, name=name, version="1.0.0")


def test_durable_evidence_event_sink_is_idempotent(tmp_path):
    run_id = "run_evidence_events"
    versions = ComponentVersionSet(
        runtime=_version(ComponentKind.RUNTIME, "evidence-runtime"),
        scheduler=_version(ComponentKind.SCHEDULER, "evidence-scheduler"),
    )
    with SQLiteEventStore(tmp_path / "events.sqlite3") as store:
        store.append(
            RunEvent(
                sequence_no=1,
                event_type=EventType.RUN_STARTED,
                status=RunStatus.RUNNING,
                trace_id="trace_evidence_events",
                span_id="span_evidence_events",
                span_kind=SpanKind.RUN,
                correlation_id="correlation_evidence_events",
                run_id=run_id,
                thread_id="thread_evidence_events",
                actor_id="actor_evidence_events",
                producer_id="runtime_evidence_events",
                component_versions=versions,
            )
        )
        store.append(
            RunEvent(
                sequence_no=2,
                event_type=EventType.SPAN_STARTED,
                status=RunStatus.RUNNING,
                trace_id="trace_evidence_events",
                span_id="span_verification_events",
                parent_span_id="span_evidence_events",
                span_kind=SpanKind.VERIFICATION,
                correlation_id="correlation_evidence_events",
                run_id=run_id,
                thread_id="thread_evidence_events",
                actor_id="agent_evidence_verifier",
                producer_id="runtime_evidence_events",
                component_versions=versions,
            )
        )
        sink = EventRecorderEvidenceSink(
            EventRecorder(store),
            EvidenceTraceContext(
                thread_id="thread_evidence_events",
                trace_id="trace_evidence_events",
                span_id="span_verification_events",
                parent_span_id="span_evidence_events",
                correlation_id="correlation_evidence_events",
                component_versions=versions,
            ),
        )
        event = EvidenceDomainEvent(
            event_id="event_evidence_verified",
            event_type=EventType.VERIFICATION_COMPLETED,
            run_id=run_id,
            task_id="task_evidence_events",
            subject_id="claim_evidence_events",
            actor_id="agent_evidence_verifier",
            output_artifact_ids=("artifact_evidence_result",),
            payload={"passed": True},
        )
        sink.emit(event)
        sink.emit(event)
        page = store.list(EventQuery(run_id=run_id, limit=100))
        assert [item.event_type for item in page.items] == [
            EventType.RUN_STARTED,
            EventType.SPAN_STARTED,
            EventType.VERIFICATION_COMPLETED,
        ]
        assert page.items[-1].span_kind == SpanKind.VERIFICATION
        assert page.items[-1].payload["subject_id"] == event.subject_id
