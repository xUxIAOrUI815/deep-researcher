from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    Budget,
    BudgetUsage,
    Citation,
    Claim,
    Command,
    CommandKind,
    Conflict,
    ConflictSeverity,
    EntityProvenance,
    Evidence,
    EvidenceQuote,
    EvidenceRelation,
    Passage,
    PassageStatus,
    Report,
    ReportStatus,
    Section,
    SectionStatus,
    SnapshotStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    SourceType,
    TaskKind,
    TaskStatus,
    utc_now,
)
from deep_researcher.evidence import (
    DeterministicSemanticVerificationAdapter,
    RecordingEvidenceEventSink,
    VerificationPolicy,
    build_evidence_runtime,
)
from deep_researcher.kernel import (
    CancellationToken,
    KernelEvent,
    ModelRequest,
    ModelResponse,
)
from deep_researcher.orchestration import (
    NativeEventSourcedScheduler,
    SQLiteSchedulerStore,
)
from deep_researcher.reporting import (
    DraftStatement,
    FindingSeverity,
    LoopStatus,
    ReportLoopPolicy,
    ReportRepairAction,
    ReportingStoreConflict,
    ReportingStoreCorruption,
    ReviewActionKind,
    ReviewDimension,
    RubricScore,
    SQLiteReportingStore,
    SchedulerTargetedResearchDispatcher,
    SectionDraftProposal,
    StatementCertainty,
    TargetedResearchApprovalRequired,
    WriterDraftProposal,
    WriterEvidencePacket,
    build_report_reviewer_spec,
    build_reporting_runtime,
    build_synthesis_writer_spec,
)


def _budget(**updates: Any) -> Budget:
    values: dict[str, Any] = {
        "max_tokens": 100_000,
        "max_cost_usd": 50.0,
        "max_wall_time_seconds": 1200.0,
        "max_model_calls": 50,
        "max_tool_calls": 50,
        "max_search_calls": 20,
        "max_retries": 10,
        "max_errors": 10,
    }
    values.update(updates)
    return Budget(**values)


def _evidence_policy() -> VerificationPolicy:
    return VerificationPolicy(
        policy_version_id="policy_reporting_evidence_1",
        freshness_days=365,
        max_repair_rounds=3,
    )


def _report_policy(**updates: Any) -> ReportLoopPolicy:
    values: dict[str, Any] = {
        "max_revisions": 4,
        "max_targeted_research_rounds": 2,
        "minimum_score": 0.8,
        "minimum_support_score": 1.0,
        "minimum_citation_score": 1.0,
        "minimum_sources_per_statement": 1,
        "high_impact_minimum_sources": 2,
        "run_budget": _budget(),
    }
    values.update(updates)
    return ReportLoopPolicy(**values)


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


@dataclass(frozen=True)
class Seed:
    run_id: str
    statement: str
    claim_id: str
    report_id: str
    section_id: str
    citation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


def _seed(
    evidence,
    *,
    run_id: str,
    high_impact: bool = True,
) -> Seed:
    statement = "The durable report stores every verified citation."
    now = utc_now()
    claim_id = f"claim_{run_id}"
    fact_id = f"fact_{run_id}"
    entities: list[Any] = []
    evidence_items: list[Evidence] = []
    citations: list[Citation] = []
    artifacts: list[str] = []
    source_ids: list[str] = []
    for index in range(2):
        suffix = f"{run_id}_{index}"
        snapshot_artifact = evidence.knowledge.artifacts.put_text(
            statement,
            kind=ArtifactKind.SOURCE_SNAPSHOT,
            producer_id="tool_scraper",
            run_id=run_id,
            task_id=f"task_{run_id}",
            content_schema="SourceSnapshotText@1",
        )
        passage_artifact = evidence.knowledge.artifacts.put_text(
            statement,
            kind=ArtifactKind.CLEANED_CONTENT,
            producer_id="agent_worker",
            run_id=run_id,
            task_id=f"task_{run_id}",
            content_schema="CleanedPassage@1",
            source_artifact_ids=(snapshot_artifact.artifact_id,),
        )
        artifacts.append(passage_artifact.artifact_id)
        source = Source(
            source_id=f"source_{suffix}",
            canonical_url=f"https://source-{index}.{run_id}.example/report",
            source_type=(
                SourceType.OFFICIAL_DOCUMENTATION
                if index == 0
                else SourceType.NEWS
            ),
            source_level=(
                SourceLevel.PRIMARY
                if index == 0
                else SourceLevel.SECONDARY
            ),
            status=SourceStatus.ACCESSIBLE,
            title=f"Reporting source {index}",
            publisher=f"Independent publisher {index}",
            authority_score=0.95,
            published_at=now - timedelta(days=10),
            provenance=_provenance(
                run_id,
                snapshot_artifact.artifact_id,
                producer_id="tool_scraper",
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
                run_id,
                snapshot_artifact.artifact_id,
                producer_id="tool_scraper",
            ),
        )
        passage = Passage(
            passage_id=f"passage_{suffix}",
            snapshot_id=snapshot.snapshot_id,
            text_artifact_id=passage_artifact.artifact_id,
            ordinal=0,
            locator=f"chars:0-{len(statement)}",
            content_hash=passage_artifact.content_hash,
            extraction_method="dom_text_v1",
            extracted_at=now,
            char_start=0,
            char_end=len(statement),
            language="en",
            status=PassageStatus.ACCEPTED,
            provenance=_provenance(run_id, passage_artifact.artifact_id),
        )
        quote = EvidenceQuote(
            passage_id=passage.passage_id,
            quote=statement,
            char_start=0,
            char_end=len(statement),
            passage_content_hash=passage.content_hash,
            extraction_method=passage.extraction_method,
            extracted_at=passage.extracted_at,
        )
        evidence_item = Evidence(
            evidence_id=f"evidence_{suffix}",
            passage_ids=(passage.passage_id,),
            relation=EvidenceRelation.SUPPORTS,
            summary=f"Independent support {index}",
            confidence=0.95,
            relevance=0.95,
            source_quality=0.95,
            quotes=(quote,),
            provenance=_provenance(run_id, passage_artifact.artifact_id),
        )
        citation = Citation(
            citation_id=f"citation_{suffix}",
            claim_id=claim_id,
            evidence_id=evidence_item.evidence_id,
            passage_id=passage.passage_id,
            snapshot_id=snapshot.snapshot_id,
            source_id=source.source_id,
            locator=passage.locator,
            quote=statement,
            extraction_method=passage.extraction_method,
            provenance=_provenance(run_id, passage_artifact.artifact_id),
        )
        entities.extend((source, snapshot, passage, evidence_item, citation))
        evidence_items.append(evidence_item)
        citations.append(citation)
        source_ids.append(source.source_id)
    fact = AtomicFact(
        fact_id=fact_id,
        statement=statement,
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
        confidence=0.95,
        provenance=_provenance(run_id, *artifacts),
    )
    claim = Claim(
        claim_id=claim_id,
        statement=statement,
        fact_ids=(fact_id,),
        evidence_ids=tuple(item.evidence_id for item in evidence_items),
        confidence=0.95,
        importance=0.95 if high_impact else 0.6,
        high_impact=high_impact,
        provenance=_provenance(run_id, *artifacts),
    )
    section_id = f"section_{run_id}"
    report_id = f"report_{run_id}"
    section = Section(
        section_id=section_id,
        report_id=report_id,
        title="Verified findings",
        goal="Present every verified result and all uncertainty.",
        order=0,
        claim_ids=(claim_id,),
        required_claim_ids=(claim_id,),
        citation_ids=tuple(item.citation_id for item in citations),
        provenance=_provenance(run_id, *artifacts),
    )
    report = Report(
        report_id=report_id,
        thread_id=f"thread_{run_id}",
        run_id=run_id,
        title="Verified reporting",
        research_question=statement,
        section_ids=(section_id,),
        provenance=_provenance(run_id, *artifacts),
    )
    evidence.knowledge.repository.save_graph(
        *entities,
        fact,
        claim,
        report,
        section,
    )
    return Seed(
        run_id=run_id,
        statement=statement,
        claim_id=claim_id,
        report_id=report_id,
        section_id=section_id,
        citation_ids=tuple(item.citation_id for item in citations),
        source_ids=tuple(source_ids),
    )


class QueueModel:
    def __init__(
        self,
        responses: list[ModelResponse],
        repairs: list[ModelResponse] | None = None,
    ) -> None:
        self.responses = deque(responses)
        self.repairs = deque(repairs or [])
        self.requests: list[ModelRequest] = []
        self.repair_requests: list[tuple[str, ...]] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return self.responses.popleft()

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        del request, invalid_response
        self.repair_requests.append(errors)
        return self.repairs.popleft()


class EventCollector:
    def __init__(self) -> None:
        self.events: list[KernelEvent] = []

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)


def _writer_response(
    seed: Seed,
    *,
    revision: int,
    certainty: StatementCertainty = StatementCertainty.DEFINITIVE,
) -> ModelResponse:
    return ModelResponse(
        structured={
            "proposal": {
                "title": "Verified reporting",
                "revision": revision,
                "decision_summary": "Synthesized from verified evidence.",
                "sections": [
                    {
                        "section_id": seed.section_id,
                        "title": "Verified findings",
                        "statements": [
                            {
                                "statement_id": (
                                    f"statement_{seed.run_id}_{revision}"
                                ),
                                "text": seed.statement,
                                "claim_ids": [seed.claim_id],
                                "citation_ids": list(seed.citation_ids),
                                "certainty": certainty.value,
                            }
                        ],
                        "gap_disclosures": [],
                        "conflict_disclosures": [],
                    }
                ],
            }
        },
        usage=BudgetUsage(input_tokens=100, output_tokens=80),
        response_id=f"writer_response_{revision}",
    )


def _scores(value: float = 1.0) -> list[dict[str, Any]]:
    return [
        {
            "dimension": item.value,
            "score": value,
            "rationale": f"Reviewed {item.value}.",
        }
        for item in ReviewDimension
    ]


def _review_response(
    *,
    decision: ReviewActionKind,
    repair_kind: ReviewActionKind | None = None,
) -> ModelResponse:
    repairs: list[dict[str, Any]] = []
    if repair_kind is not None:
        repairs.append(
            {
                "kind": repair_kind.value,
                "reason": "Apply the bounded reviewer repair.",
            }
        )
    return ModelResponse(
        structured={
            "decision": {
                "decision": decision.value,
                "scores": _scores(),
                "findings": [],
                "repair_actions": repairs,
                "decision_summary": f"Reviewer selected {decision.value}.",
            }
        },
        usage=BudgetUsage(input_tokens=80, output_tokens=50),
        response_id=f"review_{decision.value}",
    )


@pytest.fixture
def evidence_runtime(tmp_path):
    runtime = build_evidence_runtime(
        tmp_path / "evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_evidence_policy(),
    )
    try:
        yield runtime
    finally:
        runtime.close()


def test_writer_and_reviewer_specs_enforce_role_boundaries():
    writer = build_synthesis_writer_spec()
    reviewer = build_report_reviewer_spec()

    assert writer.allowed_commands == (
        CommandKind.SYNTHESIZE,
        CommandKind.STOP,
    )
    assert reviewer.allowed_commands == (
        CommandKind.REVIEW,
        CommandKind.STOP,
    )
    assert writer.tool_grants == reviewer.tool_grants == ()
    assert writer.metadata["verified_only"] is True
    assert writer.metadata["search"] is False
    assert reviewer.metadata["evidence_mutation"] is False
    assert set(reviewer.metadata["rubric_dimensions"]) == {
        item.value for item in ReviewDimension
    }


@pytest.mark.asyncio
async def test_verified_packet_and_complete_report_loop_are_traceable(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(evidence_runtime, run_id="run_reporting_accept")
    await evidence_runtime.engine.verify_run(seed.run_id)
    writer_model = QueueModel([_writer_response(seed, revision=1)])
    reviewer_model = QueueModel(
        [_review_response(decision=ReviewActionKind.ACCEPT)]
    )
    events = EventCollector()
    runtime = build_reporting_runtime(
        tmp_path / "reporting",
        evidence=evidence_runtime,
        writer_model=writer_model,
        reviewer_model=reviewer_model,
        event_sink=events,
        policy=_report_policy(),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        assert tuple(item.claim_id for item in packet.claims) == (
            seed.claim_id,
        )
        assert set(packet.citations[0].citation_id for _ in (0,)) <= set(
            seed.citation_ids
        )
        assert packet.gaps == ()
        outcome = await runtime.loop.run(seed.report_id)

        assert outcome.status == LoopStatus.ACCEPTED
        revision = runtime.store.revisions(
            seed.run_id,
            seed.report_id,
        )[0]
        citation_map = runtime.store.citation_map(
            seed.run_id,
            seed.report_id,
            1,
        )
        assert citation_map is not None
        assert tuple(item.marker for item in citation_map.entries) == (
            "[1]",
            "[2]",
        )
        assert revision.markdown.count(f"{seed.statement}[1][2]") == 1
        assert set(item.citation_id for item in citation_map.entries) == set(
            seed.citation_ids
        )
        assert all(
            item.claim_id == seed.claim_id for item in citation_map.entries
        )
        assert evidence_runtime.knowledge.repository.reports.require(
            seed.report_id
        ).status == ReportStatus.APPROVED
        assert evidence_runtime.knowledge.repository.sections.require(
            seed.section_id
        ).status == SectionStatus.APPROVED
        assert runtime.loop.run is not None
        assert events.events
        runtime.integrity_check()
        assert await runtime.loop.run(seed.report_id) == outcome
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_packet_turns_unverified_required_claim_into_explicit_gap(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(
        evidence_runtime,
        run_id="run_reporting_gap",
        high_impact=False,
    )
    await evidence_runtime.engine.verify_run(seed.run_id)
    gap_claim = Claim(
        claim_id=f"claim_gap_{seed.run_id}",
        statement="An unsupported draft statement must never become prose.",
        confidence=0.3,
        importance=0.9,
        high_impact=True,
        provenance=_provenance(seed.run_id),
    )
    repository = evidence_runtime.knowledge.repository
    repository.claims.save(gap_claim)
    section = repository.sections.require(seed.section_id)
    repository.sections.save(
        section.model_copy(
            update={
                "claim_ids": (*section.claim_ids, gap_claim.claim_id),
                "required_claim_ids": (
                    *section.required_claim_ids,
                    gap_claim.claim_id,
                ),
            }
        )
    )
    runtime = build_reporting_runtime(
        tmp_path / "reporting-gap",
        evidence=evidence_runtime,
        writer_model=QueueModel([]),
        reviewer_model=QueueModel([]),
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        assert tuple(item.claim_id for item in packet.claims) == (
            seed.claim_id,
        )
        assert tuple(item.claim_id for item in packet.gaps) == (
            gap_claim.claim_id,
        )
        payload = evidence_runtime.knowledge.artifacts.read_bytes(
            packet.packet_artifact_id
        ).decode()
        assert '"candidate_claim_count":0' in payload
        assert gap_claim.statement in payload
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_writer_presents_conflicts_and_uncertainty_with_citations(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(
        evidence_runtime,
        run_id="run_reporting_conflict",
        high_impact=False,
    )
    await evidence_runtime.engine.verify_run(seed.run_id)
    other_id = f"claim_other_{seed.run_id}"
    repository = evidence_runtime.knowledge.repository
    repository.claims.save(
        Claim(
            claim_id=other_id,
            statement="A competing interpretation remains unverified.",
            confidence=0.4,
            importance=0.4,
            provenance=_provenance(seed.run_id),
        )
    )
    conflict = Conflict(
        conflict_id=f"conflict_{seed.run_id}",
        claim_ids=(seed.claim_id, other_id),
        summary="The verified claim has an unresolved competing interpretation.",
        severity=ConflictSeverity.HIGH,
        high_impact=False,
        provenance=_provenance(seed.run_id),
    )
    repository.conflicts.save(conflict)
    section = repository.sections.require(seed.section_id)
    repository.sections.save(
        section.model_copy(
            update={
                "claim_ids": (*section.claim_ids, other_id),
                "required_claim_ids": (*section.required_claim_ids, other_id),
            }
        )
    )
    response = _writer_response(
        seed,
        revision=1,
        certainty=StatementCertainty.CONFLICTED,
    )
    section_payload = response.structured["proposal"]["sections"][0]
    section_payload["gap_disclosures"] = [
        {"claim_id": other_id, "text": "Evidence is incomplete."}
    ]
    section_payload["conflict_disclosures"] = [
        {
            "conflict_id": conflict.conflict_id,
            "text": "Present the unresolved conflict.",
            "citation_ids": list(seed.citation_ids),
        }
    ]
    runtime = build_reporting_runtime(
        tmp_path / "reporting-conflict",
        evidence=evidence_runtime,
        writer_model=QueueModel([response]),
        reviewer_model=QueueModel([]),
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        written = await runtime.writer.write(
            packet=packet,
            title="Verified reporting",
            revision=1,
            budget=_budget(),
        )
        markdown = written.revision.markdown
        assert "### 证据缺口与不确定性" in markdown
        assert "### 证据冲突" in markdown
        assert conflict.summary in markdown
        assert "当前证据不足以消解该冲突" in markdown
        assert markdown.count("[1]") >= 2
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_writer_rejects_unverified_or_insufficiently_cited_statement(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(evidence_runtime, run_id="run_writer_reject")
    await evidence_runtime.engine.verify_run(seed.run_id)
    bad = _writer_response(seed, revision=1)
    bad.structured["proposal"]["sections"][0]["statements"][0][
        "citation_ids"
    ] = [seed.citation_ids[0]]
    runtime = build_reporting_runtime(
        tmp_path / "reporting-reject",
        evidence=evidence_runtime,
        writer_model=QueueModel([bad]),
        reviewer_model=QueueModel([]),
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        with pytest.raises(RuntimeError, match="Writer failed"):
            await runtime.writer.write(
                packet=packet,
                title="Verified reporting",
                revision=1,
                budget=_budget(),
            )
        assert runtime.store.revisions(seed.run_id, seed.report_id) == ()
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "repair_kind",
    (
        ReviewActionKind.CITATION_REPAIR,
        ReviewActionKind.LOCAL_REWRITE,
        ReviewActionKind.STRUCTURAL_REWRITE,
    ),
)
async def test_reviewer_rewrite_commands_are_bounded_and_then_accept(
    tmp_path,
    evidence_runtime,
    repair_kind,
):
    seed = _seed(evidence_runtime, run_id="run_reporting_rewrite")
    await evidence_runtime.engine.verify_run(seed.run_id)
    runtime = build_reporting_runtime(
        tmp_path / "reporting-rewrite",
        evidence=evidence_runtime,
        writer_model=QueueModel(
            [
                _writer_response(seed, revision=1),
                _writer_response(seed, revision=2),
            ]
        ),
        reviewer_model=QueueModel(
                [
                    _review_response(
                        decision=repair_kind,
                        repair_kind=repair_kind,
                    ),
                _review_response(decision=ReviewActionKind.ACCEPT),
            ]
        ),
        event_sink=EventCollector(),
        policy=_report_policy(max_revisions=2),
    )
    try:
        outcome = await runtime.loop.run(seed.report_id)
        assert outcome.status == LoopStatus.ACCEPTED
        revisions = runtime.store.revisions(seed.run_id, seed.report_id)
        reviews = runtime.store.reviews(seed.run_id, seed.report_id)
        assert [item.revision for item in revisions] == [1, 2]
        assert [item.decision for item in reviews] == [
            repair_kind,
            ReviewActionKind.ACCEPT,
        ]
        assert revisions[1].parent_revision_id == revisions[0].revision_id
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_reviewer_false_accept_cannot_override_high_impact_gap(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(
        evidence_runtime,
        run_id="run_reviewer_override",
        high_impact=False,
    )
    await evidence_runtime.engine.verify_run(seed.run_id)
    gap_id = f"claim_gap_{seed.run_id}"
    gap = Claim(
        claim_id=gap_id,
        statement="A high impact missing fact.",
        confidence=0.2,
        importance=0.95,
        high_impact=True,
        provenance=_provenance(seed.run_id),
    )
    repository = evidence_runtime.knowledge.repository
    repository.claims.save(gap)
    section = repository.sections.require(seed.section_id)
    repository.sections.save(
        section.model_copy(
            update={
                "claim_ids": (*section.claim_ids, gap_id),
                "required_claim_ids": (*section.required_claim_ids, gap_id),
            }
        )
    )
    response = _writer_response(seed, revision=1)
    response.structured["proposal"]["sections"][0]["gap_disclosures"] = [
        {"claim_id": gap_id, "text": "This remains unverified."}
    ]
    runtime = build_reporting_runtime(
        tmp_path / "reporting-override",
        evidence=evidence_runtime,
        writer_model=QueueModel([response]),
        reviewer_model=QueueModel(
            [_review_response(decision=ReviewActionKind.ACCEPT)]
        ),
        event_sink=EventCollector(),
        policy=_report_policy(max_revisions=1),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        written = await runtime.writer.write(
            packet=packet,
            title="Verified reporting",
            revision=1,
            budget=_budget(),
        )
        reviewed = await runtime.reviewer.review(
            revision=written.revision,
            budget=_budget(),
        )
        assert reviewed.decision.decision == (
            ReviewActionKind.TARGETED_RESEARCH
        )
        assert reviewed.decision.repair_actions[0].kind == (
            ReviewActionKind.TARGETED_RESEARCH
        )
        assert any(
            finding.dimension == ReviewDimension.COMPLETENESS
            and finding.severity == FindingSeverity.HIGH
            for finding in reviewed.decision.findings
        )
        assert repository.claims.require(gap_id).status.value == "draft"
        assert repository.claims.require(seed.claim_id).status.value == (
            "supported"
        )
    finally:
        runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("review_response", "max_revisions", "expected"),
    (
        (
            _review_response(decision=ReviewActionKind.REJECT),
            2,
            LoopStatus.REJECTED,
        ),
        (
            _review_response(
                decision=ReviewActionKind.LOCAL_REWRITE,
                repair_kind=ReviewActionKind.LOCAL_REWRITE,
            ),
            1,
            LoopStatus.REVISION_EXHAUSTED,
        ),
    ),
)
async def test_report_loop_reject_and_revision_bounds_are_terminal(
    tmp_path,
    evidence_runtime,
    review_response,
    max_revisions,
    expected,
):
    seed = _seed(
        evidence_runtime,
        run_id=f"run_terminal_{expected.value}",
    )
    await evidence_runtime.engine.verify_run(seed.run_id)
    runtime = build_reporting_runtime(
        tmp_path / f"reporting-{expected.value}",
        evidence=evidence_runtime,
        writer_model=QueueModel([_writer_response(seed, revision=1)]),
        reviewer_model=QueueModel([review_response]),
        event_sink=EventCollector(),
        policy=_report_policy(max_revisions=max_revisions),
    )
    try:
        outcome = await runtime.loop.run(seed.report_id)
        assert outcome.status == expected
        assert runtime.store.outcome(seed.run_id, seed.report_id) == outcome
        assert evidence_runtime.knowledge.repository.reports.require(
            seed.report_id
        ).status == ReportStatus.FAILED
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_report_loop_cancellation_is_persisted_before_model_calls(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(evidence_runtime, run_id="run_reporting_cancel")
    await evidence_runtime.engine.verify_run(seed.run_id)
    writer_model = QueueModel([])
    reviewer_model = QueueModel([])
    runtime = build_reporting_runtime(
        tmp_path / "reporting-cancel",
        evidence=evidence_runtime,
        writer_model=writer_model,
        reviewer_model=reviewer_model,
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    token = CancellationToken()
    token.cancel()
    try:
        outcome = await runtime.loop.run(
            seed.report_id,
            cancellation=token,
        )
        assert outcome.status == LoopStatus.CANCELLED
        assert outcome.revisions == 0
        assert not writer_model.requests
        assert not reviewer_model.requests
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_writer_structured_output_uses_bounded_model_repair(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(evidence_runtime, run_id="run_writer_model_repair")
    await evidence_runtime.engine.verify_run(seed.run_id)
    writer_model = QueueModel(
        [
            ModelResponse(
                structured={"proposal": {"title": "Broken"}},
                response_id="writer_invalid",
            )
        ],
        repairs=[_writer_response(seed, revision=1)],
    )
    runtime = build_reporting_runtime(
        tmp_path / "reporting-model-repair",
        evidence=evidence_runtime,
        writer_model=writer_model,
        reviewer_model=QueueModel([]),
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    try:
        packet = runtime.packet_builder.build(seed.report_id)
        result = await runtime.writer.write(
            packet=packet,
            title="Verified reporting",
            revision=1,
            budget=_budget(),
        )
        assert result.revision.revision == 1
        assert len(writer_model.repair_requests) == 1
    finally:
        runtime.close()


def test_reporting_store_restart_backup_concurrency_and_corruption(
    tmp_path,
    evidence_runtime,
):
    path = tmp_path / "reporting-store.sqlite3"
    store = SQLiteReportingStore(path)
    # Use a deliberately minimal journal object; artifact existence belongs to
    # the runtime verifier, while the store guarantees immutable persistence.
    from deep_researcher.reporting import ReportRevision

    revision = ReportRevision(
        revision_id="report_revision_store_test",
        run_id="run_store_test",
        report_id="report_store_test",
        revision=1,
        title="Stored report",
        markdown="# Stored report\n",
        section_ids=("section_store_test",),
        statement_ids=("statement_store_test",),
        report_artifact_id="artifact_store_report",
        draft_artifact_id="artifact_store_draft",
        citation_map_artifact_id="artifact_store_citations",
        evidence_packet_artifact_id="artifact_store_packet",
    )
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda _: store.save_revision(revision), range(12)))
        assert store.revisions("run_store_test", "report_store_test") == (
            revision,
        )
        backup = store.backup_to(tmp_path / "reporting-backup.sqlite3")
        assert backup.exists()
    finally:
        store.close()

    reopened = SQLiteReportingStore(path)
    try:
        assert reopened.revision(revision.revision_id) == revision
        changed = revision.model_copy(update={"title": "Changed"})
        with pytest.raises(ReportingStoreConflict):
            reopened.save_revision(changed)
        with reopened.transaction() as connection:
            connection.execute(
                "UPDATE report_revisions SET checksum='bad' "
                "WHERE revision_id=?",
                (revision.revision_id,),
            )
        with pytest.raises(ReportingStoreCorruption):
            reopened.integrity_check()
    finally:
        reopened.close()

    backup_store = SQLiteReportingStore(tmp_path / "reporting-backup.sqlite3")
    try:
        assert backup_store.revision(revision.revision_id) == revision
        backup_store.integrity_check()
    finally:
        backup_store.close()


class EmptyWorkerPool:
    def __init__(self) -> None:
        self.calls = 0

    async def drain(self, run_id: str):
        del run_id
        self.calls += 1
        return ()

    def cancel_active(self):
        return ()


@pytest.mark.asyncio
async def test_scheduler_targeted_research_dispatch_is_real_and_bounded(
    tmp_path,
    evidence_runtime,
):
    seed = _seed(
        evidence_runtime,
        run_id="run_targeted_dispatch",
        high_impact=False,
    )
    await evidence_runtime.engine.verify_run(seed.run_id)
    gap_id = f"claim_gap_{seed.run_id}"
    repository = evidence_runtime.knowledge.repository
    repository.claims.save(
        Claim(
            claim_id=gap_id,
            statement="Research this missing claim.",
            confidence=0.2,
            importance=0.95,
            high_impact=True,
            provenance=_provenance(seed.run_id),
        )
    )
    section = repository.sections.require(seed.section_id)
    repository.sections.save(
        section.model_copy(
            update={
                "claim_ids": (*section.claim_ids, gap_id),
                "required_claim_ids": (*section.required_claim_ids, gap_id),
            }
        )
    )
    reporting = build_reporting_runtime(
        tmp_path / "reporting-targeted",
        evidence=evidence_runtime,
        writer_model=QueueModel([]),
        reviewer_model=QueueModel([]),
        event_sink=EventCollector(),
        policy=_report_policy(),
    )
    scheduler_store = SQLiteSchedulerStore(
        tmp_path / "scheduler-targeted.sqlite3"
    )
    scheduler = NativeEventSourcedScheduler(scheduler_store)
    pool = EmptyWorkerPool()
    try:
        await scheduler.create_run(
            seed.run_id,
            max_concurrency=2,
            actor_id="agent_test",
            mutation_id="mutation_create_targeted",
        )
        packet = reporting.packet_builder.build(seed.report_id)
        revision_id = "report_revision_targeted"
        action = ReportRepairAction(
            action_id="repair_action_targeted",
            kind=ReviewActionKind.TARGETED_RESEARCH,
            reason="Resolve the high-impact gap.",
            claim_ids=(gap_id,),
        )
        from deep_researcher.reporting import ReviewerDecision

        decision = ReviewerDecision(
            review_id="report_review_targeted",
            run_id=seed.run_id,
            report_id=seed.report_id,
            revision_id=revision_id,
            decision=ReviewActionKind.TARGETED_RESEARCH,
            scores=tuple(
                RubricScore(
                    dimension=item,
                    score=0.5 if item == ReviewDimension.COMPLETENESS else 1.0,
                    rationale="Targeted test score.",
                )
                for item in ReviewDimension
            ),
            repair_actions=(action,),
            decision_summary="Targeted research is required.",
        )
        dispatcher = SchedulerTargetedResearchDispatcher(
            scheduler=scheduler,
            worker_pool=pool,
            evidence=evidence_runtime,
            packet_builder=reporting.packet_builder,
            research_task_budget=_budget(),
        )
        result = await dispatcher.dispatch(
            report_id=seed.report_id,
            packet=packet,
            decision=decision,
            round_no=1,
            cancellation=CancellationToken(),
        )
        assert len(result.task_ids) == 1
        snapshot = await scheduler.snapshot(seed.run_id)
        assert snapshot.by_id[result.task_ids[0]].envelope.kind == TaskKind.GAP
        assert snapshot.by_id[result.task_ids[0]].envelope.status in {
            TaskStatus.READY,
            TaskStatus.RUNNING,
        }
        assert pool.calls == 1

        await scheduler.cancel_run(
            seed.run_id,
            actor_id="agent_test",
            reason="Finish targeted dispatcher test.",
            mutation_id="mutation_complete_targeted",
        )
        with pytest.raises(TargetedResearchApprovalRequired):
            await dispatcher.dispatch(
                report_id=seed.report_id,
                packet=packet,
                decision=decision,
                round_no=2,
                cancellation=CancellationToken(),
            )
    finally:
        reporting.close()
        scheduler_store.close()
