from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
from typing import Any, Iterable
from urllib.parse import urlsplit

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    AtomicFact,
    BudgetUsage,
    Citation,
    CitationStatus,
    Claim,
    ClaimStatus,
    Conflict,
    ConflictResolutionKind,
    ConflictSeverity,
    ConflictStatus,
    EntityProvenance,
    Evidence,
    EvidenceQuote,
    EvidenceRelation,
    EvidenceStatus,
    EventType,
    FactStatus,
    Passage,
    PassageStatus,
    RepairAction,
    RepairRequest,
    Section,
    SectionCoverageStatus,
    Source,
    SourceLevel,
    SourceSnapshot,
    SourceStatus,
    VerificationCategory,
    VerificationIssue,
    VerificationResult,
    VerificationSeverity,
    canonical_contract_json,
    utc_now,
)
from deep_researcher.knowledge import KnowledgeRepository
from deep_researcher.knowledge.storage import KnowledgeEntity

from .events import EvidenceDomainEvent, EvidenceEventSink
from .graph import ClaimGraph, EvidenceGraphResolver
from .models import (
    ClaimVerificationSummary,
    EvidenceAssessment,
    RunVerificationSummary,
    SectionCoverageAssessment,
    SemanticJudgment,
    SemanticLabel,
    SemanticVerificationAdapter,
    VerificationPolicy,
)


_ENTITY_MODELS: dict[str, type[KnowledgeEntity]] = {
    model.__name__: model
    for model in (
        Source,
        SourceSnapshot,
        Passage,
        Evidence,
        AtomicFact,
        Claim,
        Citation,
        Conflict,
        Section,
    )
}


class EvidenceVerificationError(RuntimeError):
    pass


class RepairBudgetExhausted(EvidenceVerificationError):
    pass


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _combine_usage(items: Iterable[BudgetUsage]) -> BudgetUsage:
    combined = BudgetUsage()
    for item in items:
        combined = combined.plus(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cost_usd=item.cost_usd,
            wall_time_seconds=item.wall_time_seconds,
            model_calls=item.model_calls,
            tool_calls=item.tool_calls,
            search_calls=item.search_calls,
            retries=item.retries,
            errors=item.errors,
        )
    return combined


class EvidenceVerificationEngine:
    """Independent, artifact-backed verifier for the complete evidence graph."""

    CHECKS = (
        VerificationCategory.SOURCE_ACCESS,
        VerificationCategory.SOURCE_AUTHORITY,
        VerificationCategory.SOURCE_FRESHNESS,
        VerificationCategory.SOURCE_INDEPENDENCE,
        VerificationCategory.PASSAGE_INTEGRITY,
        VerificationCategory.QUOTE_GROUNDING,
        VerificationCategory.EVIDENCE_SUPPORT,
        VerificationCategory.CLAIM_SUPPORT,
        VerificationCategory.CLAIM_OVERREACH,
        VerificationCategory.CITATION_ACCURACY,
        VerificationCategory.CITATION_COMPLETENESS,
        VerificationCategory.CONFLICT,
        VerificationCategory.COVERAGE,
    )

    def __init__(
        self,
        *,
        repository: KnowledgeRepository,
        artifact_store: ArtifactStore,
        semantic_adapter: SemanticVerificationAdapter,
        event_sink: EvidenceEventSink,
        policy: VerificationPolicy,
        verifier_id: str = "agent_spec_evidence_verifier_1_0_0",
        clock=utc_now,
    ) -> None:
        self.repository = repository
        self.artifact_store = artifact_store
        self.semantic_adapter = semantic_adapter
        self.event_sink = event_sink
        self.policy = policy
        self.verifier_id = verifier_id
        self.clock = clock
        self.resolver = EvidenceGraphResolver(repository)

    async def verify_claim(
        self,
        claim_id: str,
        *,
        task_id: str | None = None,
        repair_round: int = 0,
    ) -> ClaimVerificationSummary:
        if repair_round < 0 or repair_round > self.policy.max_repair_rounds:
            raise RepairBudgetExhausted(
                f"repair round {repair_round} exceeds policy maximum {self.policy.max_repair_rounds}"
            )
        graph = self.resolver.claim_graph(claim_id)
        claim = graph.claim
        if claim.status in {ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}:
            raise EvidenceVerificationError(
                f"terminal claim cannot be verified: {claim_id}"
            )
        existing = self._result_for_existing_claim(claim, graph)
        if existing is not None:
            self._emit_claim_events(existing)
            return existing

        graph_fingerprint = self._graph_fingerprint(graph, repair_round)
        verification_id = _stable_id(
            "verification",
            claim.provenance.run_id,
            claim.claim_id,
            self.policy.policy_version_id,
            graph_fingerprint,
        )
        result_artifact_id = _stable_id("artifact", verification_id)
        persisted = self.artifact_store.get(result_artifact_id)
        if persisted is not None:
            summary = self._replay_verification_artifact(result_artifact_id)
            self._emit_claim_events(summary)
            return summary

        started_at = self.clock()
        now = started_at
        passages_by_id = {item.passage_id: item for item in graph.passages}
        snapshots_by_id = {item.snapshot_id: item for item in graph.snapshots}
        sources_by_id = {item.source_id: item for item in graph.sources}
        citations_by_evidence: dict[str, list[Citation]] = {}
        for citation in graph.citations:
            citations_by_evidence.setdefault(citation.evidence_id, []).append(citation)

        passage_text: dict[str, str] = {}
        passage_integrity: dict[str, bool] = {}
        for passage in graph.passages:
            text, valid = self._read_passage(passage, snapshots_by_id)
            passage_text[passage.passage_id] = text
            passage_integrity[passage.passage_id] = valid

        issues: list[VerificationIssue] = []
        usage_items: list[BudgetUsage] = []
        assessments: list[EvidenceAssessment] = []
        evidence_revisions: list[Evidence] = []
        quotes_by_evidence: dict[str, tuple[EvidenceQuote, ...]] = {}
        source_ids_by_evidence: dict[str, tuple[str, ...]] = {}
        snapshot_ids_by_evidence: dict[str, tuple[str, ...]] = {}

        for evidence in graph.evidence:
            evidence_citations = tuple(
                citations_by_evidence.get(evidence.evidence_id, ())
            )
            valid_quotes, quote_issues = self._ground_quotes(
                evidence,
                evidence_citations,
                passages_by_id,
                passage_text,
                passage_integrity,
            )
            quotes_by_evidence[evidence.evidence_id] = valid_quotes
            for code, message, subject_id in quote_issues:
                issues.append(
                    self._issue(
                        verification_id,
                        category=(
                            VerificationCategory.PASSAGE_INTEGRITY
                            if code.startswith("passage_")
                            else VerificationCategory.QUOTE_GROUNDING
                        ),
                        severity=(
                            VerificationSeverity.CRITICAL
                            if code.startswith("passage_")
                            else VerificationSeverity.ERROR
                        ),
                        code=code,
                        message=message,
                        subject_id=subject_id,
                        evidence_ids=(evidence.evidence_id,),
                        repair_round=repair_round,
                    )
                )
            evidence_passages = tuple(
                passage_text[item]
                for item in evidence.passage_ids
                if item in passage_text
            )
            semantic_input = (
                tuple(item.quote for item in valid_quotes) or evidence_passages
            )
            semantic = await self.semantic_adapter.judge(
                run_id=claim.provenance.run_id,
                task_id=task_id or claim.provenance.task_id,
                subject_id=claim.claim_id,
                statement=claim.statement,
                evidence_id=evidence.evidence_id,
                relation=evidence.relation,
                passages=semantic_input,
            )
            usage_items.append(semantic.usage)
            relation_valid = self._relation_matches(evidence.relation, semantic)
            integrity_valid = all(
                passage_integrity.get(passage_id, False)
                for passage_id in evidence.passage_ids
            )
            grounded = bool(valid_quotes) and not quote_issues
            if not relation_valid:
                issues.append(
                    self._issue(
                        verification_id,
                        category=VerificationCategory.EVIDENCE_SUPPORT,
                        severity=VerificationSeverity.ERROR,
                        code="evidence_relation_not_semantically_grounded",
                        message=(
                            f"Evidence {evidence.evidence_id} does not semantically establish "
                            f"its declared {evidence.relation.value} relation."
                        ),
                        subject_id=evidence.evidence_id,
                        evidence_ids=(evidence.evidence_id,),
                        repair_round=repair_round,
                    )
                )
            evidence_verified = grounded and integrity_valid and relation_valid
            evidence_revisions.append(
                self._revise_evidence(
                    evidence,
                    status=(
                        EvidenceStatus.VERIFIED
                        if evidence_verified
                        else EvidenceStatus.REJECTED
                    ),
                    verification_id=verification_id,
                    quotes=valid_quotes or evidence.quotes,
                    now=now,
                )
            )
            path_snapshots = tuple(
                dict.fromkeys(
                    passages_by_id[passage_id].snapshot_id
                    for passage_id in evidence.passage_ids
                    if passage_id in passages_by_id
                )
            )
            path_sources = tuple(
                dict.fromkeys(
                    snapshots_by_id[snapshot_id].source_id
                    for snapshot_id in path_snapshots
                    if snapshot_id in snapshots_by_id
                )
            )
            source_ids_by_evidence[evidence.evidence_id] = path_sources
            snapshot_ids_by_evidence[evidence.evidence_id] = path_snapshots
            assessment_issues = tuple(
                item.code
                for item in issues
                if evidence.evidence_id in item.evidence_ids
            )
            assessments.append(
                EvidenceAssessment(
                    evidence_id=evidence.evidence_id,
                    relation=evidence.relation,
                    grounded=grounded,
                    passage_integrity=integrity_valid,
                    source_ids=path_sources,
                    snapshot_ids=path_snapshots,
                    source_authority=min(
                        (sources_by_id[item].authority_score for item in path_sources),
                        default=0.0,
                    ),
                    source_fresh=all(
                        self._source_fresh(
                            sources_by_id[item],
                            tuple(
                                snapshots_by_id[snapshot_id]
                                for snapshot_id in path_snapshots
                                if snapshots_by_id[snapshot_id].source_id == item
                            ),
                            now,
                        )
                        for item in path_sources
                    ),
                    semantic=semantic,
                    citation_ids=tuple(item.citation_id for item in evidence_citations),
                    issues=assessment_issues,
                )
            )

        evidence_revision_by_id = {
            item.evidence_id: item for item in evidence_revisions
        }
        citation_revisions: list[Citation] = []
        verified_citations_by_evidence: dict[str, list[str]] = {}
        for citation in graph.citations:
            revised, citation_issues = self._verify_citation(
                citation,
                claim=claim,
                evidence=evidence_revision_by_id.get(citation.evidence_id),
                passages_by_id=passages_by_id,
                snapshots_by_id=snapshots_by_id,
                passage_text=passage_text,
                valid_quotes=quotes_by_evidence.get(citation.evidence_id, ()),
                verification_id=verification_id,
                now=now,
            )
            citation_revisions.append(revised)
            if revised.status == CitationStatus.VERIFIED:
                verified_citations_by_evidence.setdefault(
                    revised.evidence_id, []
                ).append(revised.citation_id)
            for code, message in citation_issues:
                issues.append(
                    self._issue(
                        verification_id,
                        category=VerificationCategory.CITATION_ACCURACY,
                        severity=VerificationSeverity.ERROR,
                        code=code,
                        message=message,
                        subject_id=citation.citation_id,
                        evidence_ids=(citation.evidence_id,),
                        repair_round=repair_round,
                    )
                )

        fact_revisions: list[AtomicFact] = []
        for fact in graph.facts:
            fact_judgments: list[tuple[Evidence, SemanticJudgment]] = []
            for evidence_id in fact.evidence_ids:
                evidence = next(
                    item for item in graph.evidence if item.evidence_id == evidence_id
                )
                semantic_input = tuple(
                    item.quote for item in quotes_by_evidence.get(evidence_id, ())
                )
                if not semantic_input:
                    semantic_input = tuple(
                        passage_text[item]
                        for item in evidence.passage_ids
                        if item in passage_text
                    )
                judgment = await self.semantic_adapter.judge(
                    run_id=claim.provenance.run_id,
                    task_id=task_id or claim.provenance.task_id,
                    subject_id=fact.fact_id,
                    statement=fact.statement,
                    evidence_id=evidence_id,
                    relation=evidence.relation,
                    passages=semantic_input,
                )
                usage_items.append(judgment.usage)
                fact_judgments.append((evidence, judgment))
            supporting = [
                judgment.score
                for evidence, judgment in fact_judgments
                if evidence_revision_by_id[evidence.evidence_id].status
                == EvidenceStatus.VERIFIED
                and evidence.relation == EvidenceRelation.SUPPORTS
                and judgment.label == SemanticLabel.SUPPORTS
            ]
            refuting = [
                judgment.score
                for evidence, judgment in fact_judgments
                if evidence_revision_by_id[evidence.evidence_id].status
                == EvidenceStatus.VERIFIED
                and evidence.relation == EvidenceRelation.REFUTES
                and judgment.label == SemanticLabel.REFUTES
            ]
            if refuting and max(refuting) >= self.policy.contradiction_threshold:
                fact_status = FactStatus.DISPUTED
            elif supporting and max(supporting) >= self.policy.support_threshold:
                fact_status = FactStatus.VERIFIED
            else:
                fact_status = FactStatus.REJECTED
            fact_revisions.append(
                self._revise_fact(
                    fact,
                    status=fact_status,
                    verification_id=verification_id,
                    now=now,
                )
            )

        support_assessments = [
            item
            for item in assessments
            if evidence_revision_by_id[item.evidence_id].status
            == EvidenceStatus.VERIFIED
            and item.relation == EvidenceRelation.SUPPORTS
            and item.semantic.label == SemanticLabel.SUPPORTS
        ]
        refute_assessments = [
            item
            for item in assessments
            if evidence_revision_by_id[item.evidence_id].status
            == EvidenceStatus.VERIFIED
            and item.relation == EvidenceRelation.REFUTES
            and item.semantic.label == SemanticLabel.REFUTES
        ]
        support_score = max(
            (item.semantic.score for item in support_assessments), default=0.0
        )
        contradiction_score = max(
            (item.semantic.score for item in refute_assessments), default=0.0
        )
        support_evidence_ids = tuple(item.evidence_id for item in support_assessments)
        verified_citation_ids = tuple(
            citation_id
            for evidence_id in support_evidence_ids
            for citation_id in verified_citations_by_evidence.get(evidence_id, ())
        )
        citation_coverage = (
            sum(
                evidence_id in verified_citations_by_evidence
                for evidence_id in support_evidence_ids
            )
            / len(support_evidence_ids)
            if support_evidence_ids
            else 0.0
        )
        if (
            support_evidence_ids
            and citation_coverage < self.policy.citation_coverage_threshold
        ):
            uncited = tuple(
                item
                for item in support_evidence_ids
                if item not in verified_citations_by_evidence
            )
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.CITATION_COMPLETENESS,
                    severity=VerificationSeverity.ERROR,
                    code="supporting_evidence_missing_verified_citation",
                    message=f"Supporting evidence lacks verified citations: {list(uncited)}",
                    subject_id=claim.claim_id,
                    evidence_ids=uncited,
                    repair_round=repair_round,
                )
            )

        support_source_ids = tuple(
            dict.fromkeys(
                source_id
                for item in support_assessments
                for source_id in item.source_ids
            )
        )
        inaccessible_source_ids = tuple(
            source_id
            for source_id in support_source_ids
            if sources_by_id[source_id].status != SourceStatus.ACCESSIBLE
        )
        if inaccessible_source_ids:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.SOURCE_ACCESS,
                    severity=VerificationSeverity.ERROR,
                    code="supporting_source_not_accessible",
                    message=f"Supporting sources are not accessible: {list(inaccessible_source_ids)}",
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )
        independent_keys = {
            self._independence_key(sources_by_id[item]) for item in support_source_ids
        }
        independent_count = len(independent_keys)
        high_impact = (
            claim.high_impact or claim.importance >= self.policy.high_impact_threshold
        )
        required_independence = (
            self.policy.high_impact_minimum_independent_sources
            if high_impact
            else self.policy.minimum_independent_sources
        )
        if independent_count < required_independence:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.SOURCE_INDEPENDENCE,
                    severity=(
                        VerificationSeverity.CRITICAL
                        if high_impact
                        else VerificationSeverity.ERROR
                    ),
                    code="insufficient_independent_sources",
                    message=(
                        f"Claim has {independent_count} independent source(s); "
                        f"{required_independence} required."
                    ),
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )

        authority_threshold = (
            self.policy.high_impact_minimum_authority
            if high_impact
            else self.policy.minimum_authority
        )
        authority_values = [
            sources_by_id[item].authority_score for item in support_source_ids
        ]
        authority_score = min(authority_values, default=0.0)
        if authority_score < authority_threshold:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.SOURCE_AUTHORITY,
                    severity=(
                        VerificationSeverity.ERROR
                        if high_impact
                        else VerificationSeverity.WARNING
                    ),
                    code="source_authority_below_policy",
                    message=(
                        f"Minimum supporting-source authority {authority_score:.3f} "
                        f"is below {authority_threshold:.3f}."
                    ),
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )
        if (
            high_impact
            and support_source_ids
            and not any(
                self._source_level(sources_by_id[item]) == SourceLevel.PRIMARY
                for item in support_source_ids
            )
        ):
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.SOURCE_AUTHORITY,
                    severity=VerificationSeverity.ERROR,
                    code="high_impact_primary_source_missing",
                    message="High-impact claim lacks a verified primary-level source.",
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )

        stale_source_ids = tuple(
            source_id
            for source_id in support_source_ids
            if not self._source_fresh(
                sources_by_id[source_id],
                tuple(
                    snapshot
                    for snapshot in graph.snapshots
                    if snapshot.source_id == source_id
                ),
                now,
            )
        )
        if stale_source_ids:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.SOURCE_FRESHNESS,
                    severity=VerificationSeverity.WARNING,
                    code="supporting_source_stale",
                    message=f"Supporting sources are older than policy: {list(stale_source_ids)}",
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )

        overreach = tuple(
            dict.fromkeys(
                fragment
                for item in support_assessments
                for fragment in item.semantic.overreach_fragments
            )
        )
        if overreach:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.CLAIM_OVERREACH,
                    severity=VerificationSeverity.ERROR,
                    code="claim_contains_unsupported_specificity",
                    message=f"Claim contains unsupported fragments: {list(overreach)}",
                    subject_id=claim.claim_id,
                    evidence_ids=support_evidence_ids,
                    repair_round=repair_round,
                )
            )

        conflict_revisions: list[Conflict] = []
        severe_open_conflicts: list[str] = []
        for conflict in graph.conflicts:
            severity = self._conflict_severity(conflict, graph, high_impact)
            conflict_revisions.append(
                self._revise_conflict(
                    conflict, severity=severity, high_impact=high_impact, now=now
                )
            )
            if conflict.status != ConflictStatus.RESOLVED and severity in {
                ConflictSeverity.HIGH,
                ConflictSeverity.CRITICAL,
            }:
                severe_open_conflicts.append(conflict.conflict_id)
                issues.append(
                    self._issue(
                        verification_id,
                        category=VerificationCategory.CONFLICT,
                        severity=(
                            VerificationSeverity.CRITICAL
                            if severity == ConflictSeverity.CRITICAL
                            else VerificationSeverity.ERROR
                        ),
                        code="severe_claim_conflict_unresolved",
                        message=f"Claim participates in unresolved {severity.value} conflict {conflict.conflict_id}.",
                        subject_id=conflict.conflict_id,
                        evidence_ids=conflict.resolution_evidence_ids,
                        repair_round=repair_round,
                    )
                )

        if (
            support_score >= self.policy.support_threshold
            and contradiction_score >= self.policy.contradiction_threshold
        ):
            claim_status = ClaimStatus.CONFLICTED
        elif severe_open_conflicts:
            claim_status = ClaimStatus.CONFLICTED
        elif (
            contradiction_score >= self.policy.contradiction_threshold
            and support_score < self.policy.partial_support_threshold
        ):
            claim_status = ClaimStatus.CONTRADICTED
        elif support_score < self.policy.partial_support_threshold:
            claim_status = ClaimStatus.UNSUPPORTED
        elif support_source_ids and len(stale_source_ids) == len(support_source_ids):
            claim_status = ClaimStatus.STALE
        else:
            blocking = any(
                item.severity
                in {VerificationSeverity.ERROR, VerificationSeverity.CRITICAL}
                for item in issues
            )
            complete_support = (
                support_score >= self.policy.support_threshold
                and citation_coverage >= self.policy.citation_coverage_threshold
                and independent_count >= required_independence
                and authority_score >= authority_threshold
                and not overreach
                and not blocking
            )
            claim_status = (
                ClaimStatus.SUPPORTED
                if complete_support
                else ClaimStatus.PARTIALLY_SUPPORTED
            )

        if claim_status in {
            ClaimStatus.UNSUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.CONFLICTED,
        }:
            issues.append(
                self._issue(
                    verification_id,
                    category=VerificationCategory.CLAIM_SUPPORT,
                    severity=(
                        VerificationSeverity.CRITICAL
                        if high_impact
                        else VerificationSeverity.ERROR
                    ),
                    code=f"claim_{claim_status.value}",
                    message=f"Claim verification concluded {claim_status.value}.",
                    subject_id=claim.claim_id,
                    evidence_ids=tuple(item.evidence_id for item in assessments),
                    repair_round=repair_round,
                )
            )

        grounding_score = (
            sum(item.grounded and item.passage_integrity for item in assessments)
            / len(assessments)
            if assessments
            else 0.0
        )
        independence_score = min(
            1.0,
            independent_count / required_independence if required_independence else 1.0,
        )
        freshness_score = (
            1.0 - (len(stale_source_ids) / len(support_source_ids))
            if support_source_ids
            else 0.0
        )
        raw_score = (
            sum(
                (
                    support_score,
                    1.0 - contradiction_score,
                    grounding_score,
                    citation_coverage,
                    authority_score,
                    independence_score,
                    freshness_score,
                    0.0 if overreach else 1.0,
                )
            )
            / 8.0
        )
        passed = claim_status == ClaimStatus.SUPPORTED
        score = (
            max(raw_score, self.policy.verification_threshold)
            if passed
            else min(raw_score, max(0.0, self.policy.verification_threshold - 0.001))
        )
        repairs = self._repair_requests(
            verification_id,
            tuple(issues),
            repair_round=repair_round,
        )
        usage = _combine_usage(usage_items)
        input_artifact_ids = self._input_artifacts(graph)
        completed_at = self.clock()
        result = VerificationResult(
            verification_id=verification_id,
            run_id=claim.provenance.run_id,
            task_id=task_id or claim.provenance.task_id,
            verifier_id=self.verifier_id,
            subject_id=claim.claim_id,
            subject_type="Claim",
            passed=passed,
            score=score,
            threshold=self.policy.verification_threshold,
            checks_performed=self.CHECKS,
            issues=tuple(issues),
            repair_requests=repairs,
            usage=usage,
            policy_version_id=self.policy.policy_version_id,
            input_artifact_ids=input_artifact_ids,
            result_artifact_id=result_artifact_id,
            started_at=started_at,
            completed_at=completed_at,
        )
        claim_revision = self._revise_claim(
            claim,
            status=claim_status,
            verification_id=verification_id,
            support_score=support_score,
            high_impact=high_impact,
            now=now,
        )
        summary = ClaimVerificationSummary(
            claim_id=claim.claim_id,
            status=claim_status,
            support_score=support_score,
            contradiction_score=contradiction_score,
            independent_source_count=independent_count,
            verified_evidence_ids=tuple(
                item.evidence_id
                for item in evidence_revisions
                if item.status == EvidenceStatus.VERIFIED
            ),
            verified_citation_ids=verified_citation_ids,
            stale_source_ids=stale_source_ids,
            high_impact_blocked=high_impact and claim_status != ClaimStatus.SUPPORTED,
            verification_result=result,
        )

        revisions: list[KnowledgeEntity] = [
            *evidence_revisions,
            *fact_revisions,
            *citation_revisions,
            *conflict_revisions,
            claim_revision,
        ]
        wrapper = {
            "schema": "ClaimVerificationArtifact@1",
            "graph_fingerprint": graph_fingerprint,
            "repair_round": repair_round,
            "summary": summary.model_dump(mode="json"),
            "entity_revisions": [
                {
                    "entity_type": type(item).__name__,
                    "value": item.model_dump(mode="json"),
                }
                for item in revisions
            ],
        }
        self.artifact_store.put_json(
            wrapper,
            redact=False,
            kind=ArtifactKind.VERIFICATION_RESULT,
            producer_id=self.verifier_id,
            run_id=claim.provenance.run_id,
            task_id=task_id or claim.provenance.task_id,
            content_schema="ClaimVerificationArtifact@1",
            source_artifact_ids=input_artifact_ids,
            artifact_id=result_artifact_id,
            idempotency_key=f"verification:{verification_id}",
            metadata={
                "claim_id": claim.claim_id,
                "claim_status": claim_status.value,
                "policy_version_id": self.policy.policy_version_id,
                "repair_round": repair_round,
            },
        )
        feedback_artifact_id = self._persist_repair_feedback(
            summary,
            repair_round=repair_round,
        )
        provenance_artifacts = tuple(
            item
            for item in (result_artifact_id, feedback_artifact_id)
            if item is not None
        )
        revisions = [
            self._with_verifier_provenance(
                item,
                task_id=task_id or claim.provenance.task_id,
                artifact_ids=provenance_artifacts,
            )
            for item in revisions
        ]
        self.repository.save_graph(*revisions)
        self._emit_claim_events(summary, feedback_artifact_id=feedback_artifact_id)
        return summary

    async def verify_run(
        self,
        run_id: str,
        *,
        task_id: str | None = None,
        repair_round: int = 0,
    ) -> RunVerificationSummary:
        claim_results: list[ClaimVerificationSummary] = []
        for claim in self.repository.claims.list(run_id):
            if claim.status in {ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}:
                continue
            claim_results.append(
                await self.verify_claim(
                    claim.claim_id,
                    task_id=task_id,
                    repair_round=repair_round,
                )
            )
        section_results: list[SectionCoverageAssessment] = []
        for section in self.repository.sections.list(run_id):
            section_results.append(
                await self.assess_section(section.section_id, task_id=task_id)
            )
        severe_conflicts = tuple(
            conflict.conflict_id
            for conflict in self.repository.conflicts.list(run_id)
            if conflict.status != ConflictStatus.RESOLVED
            and conflict.severity in {ConflictSeverity.HIGH, ConflictSeverity.CRITICAL}
        )
        return RunVerificationSummary(
            run_id=run_id,
            claim_results=tuple(claim_results),
            section_results=tuple(section_results),
            verification_artifact_ids=tuple(
                [
                    item.verification_result.result_artifact_id
                    for item in claim_results
                    if item.verification_result.result_artifact_id is not None
                ]
                + [item.result_artifact_id for item in section_results]
            ),
            blocked_high_impact_claim_ids=tuple(
                item.claim_id for item in claim_results if item.high_impact_blocked
            ),
            open_severe_conflict_ids=severe_conflicts,
        )

    async def assess_section(
        self,
        section_id: str,
        *,
        task_id: str | None = None,
    ) -> SectionCoverageAssessment:
        section = self.repository.sections.require(section_id)
        run_id = section.provenance.run_id
        required_ids = section.required_claim_ids or section.claim_ids
        claims = tuple(self.repository.claims.require(item) for item in required_ids)
        supported = tuple(
            item.claim_id for item in claims if item.status == ClaimStatus.SUPPORTED
        )
        conflicted = tuple(
            item.claim_id
            for item in claims
            if item.status in {ClaimStatus.CONFLICTED, ClaimStatus.CONTRADICTED}
        )
        stale = tuple(
            item.claim_id for item in claims if item.status == ClaimStatus.STALE
        )
        unsupported = tuple(
            item.claim_id for item in claims if item.status != ClaimStatus.SUPPORTED
        )
        citations = self.repository.citations.list(run_id)
        cited_claim_ids = {
            item.claim_id
            for item in citations
            if item.status == CitationStatus.VERIFIED
        }
        uncited = tuple(item for item in supported if item not in cited_claim_ids)
        weights = {
            claim.claim_id: (
                2.0
                if claim.high_impact
                or claim.importance >= self.policy.high_impact_threshold
                else 1.0
            )
            for claim in claims
        }
        total_weight = sum(weights.values())
        coverage_score = (
            sum(weights[item] for item in supported) / total_weight
            if total_weight
            else 0.0
        )
        citation_score = (
            (len(supported) - len(uncited)) / len(supported) if supported else 0.0
        )
        blocked = bool(conflicted) or any(
            claim.claim_id in unsupported and weights[claim.claim_id] > 1.0
            for claim in claims
        )
        if blocked:
            coverage_status = SectionCoverageStatus.BLOCKED
        elif (
            coverage_score >= self.policy.section_coverage_threshold
            and citation_score >= self.policy.citation_coverage_threshold
        ):
            coverage_status = SectionCoverageStatus.COMPLETE
        elif coverage_score > 0.0:
            coverage_status = SectionCoverageStatus.PARTIAL
        else:
            coverage_status = SectionCoverageStatus.INSUFFICIENT
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "section": {
                        "section_id": section.section_id,
                        "report_id": section.report_id,
                        "claim_ids": list(section.claim_ids),
                        "required_claim_ids": list(section.required_claim_ids),
                    },
                    "claims": [item.model_dump(mode="json") for item in claims],
                    "citations": [
                        item.model_dump(mode="json")
                        for item in citations
                        if item.claim_id in required_ids
                    ],
                    "policy": self.policy.model_dump(mode="json"),
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result_artifact_id = _stable_id(
            "artifact",
            section.section_id,
            self.policy.policy_version_id,
            fingerprint,
        )
        existing_artifact = self.artifact_store.get(result_artifact_id)
        if existing_artifact is not None:
            assessment = SectionCoverageAssessment.model_validate_json(
                self.artifact_store.read_bytes(result_artifact_id),
                strict=False,
            )
            self._save_section_assessment(
                section,
                assessment,
                coverage_status=self._coverage_status(assessment),
                task_id=task_id,
            )
            self._emit_section_assessment(section, assessment, task_id=task_id)
            return assessment
        assessment = SectionCoverageAssessment(
            section_id=section.section_id,
            result_artifact_id=result_artifact_id,
            required_claim_ids=required_ids,
            supported_claim_ids=supported,
            unsupported_claim_ids=unsupported,
            conflicted_claim_ids=conflicted,
            stale_claim_ids=stale,
            uncited_claim_ids=uncited,
            coverage_score=coverage_score,
            citation_score=citation_score,
            blocked=blocked,
            assessed_at=self.clock(),
        )
        self.artifact_store.put_json(
            assessment.model_dump(mode="json"),
            redact=False,
            kind=ArtifactKind.VERIFICATION_RESULT,
            producer_id=self.verifier_id,
            run_id=run_id,
            task_id=task_id or section.provenance.task_id,
            content_schema="SectionCoverageAssessment@1",
            source_artifact_ids=tuple(
                dict.fromkeys(
                    (
                        *section.provenance.source_artifact_ids,
                        *(
                            artifact_id
                            for claim in claims
                            for artifact_id in claim.provenance.source_artifact_ids
                        ),
                    )
                )
            ),
            artifact_id=result_artifact_id,
            idempotency_key=f"section-coverage:{section.section_id}:{fingerprint}",
            metadata={
                "section_id": section.section_id,
                "policy_version_id": self.policy.policy_version_id,
                "fingerprint": fingerprint,
            },
        )
        self._save_section_assessment(
            section,
            assessment,
            coverage_status=coverage_status,
            task_id=task_id,
        )
        self._emit_section_assessment(section, assessment, task_id=task_id)
        return assessment

    def resolve_conflict(
        self,
        conflict_id: str,
        *,
        resolution: str,
        resolution_kind: ConflictResolutionKind,
        resolution_evidence_ids: tuple[str, ...],
        task_id: str | None = None,
    ) -> Conflict:
        conflict = self.repository.conflicts.require(conflict_id)
        if conflict.status == ConflictStatus.RESOLVED:
            return conflict
        if resolution_kind == ConflictResolutionKind.ACCEPTED_UNRESOLVED:
            raise ValueError(
                "accepted_unresolved is not a definitive conflict resolution"
            )
        evidence = tuple(
            self.repository.evidence.require(item) for item in resolution_evidence_ids
        )
        if not evidence or any(
            item.status != EvidenceStatus.VERIFIED for item in evidence
        ):
            raise EvidenceVerificationError(
                "conflict resolution requires verified evidence"
            )
        resolved = conflict.transition(
            ConflictStatus.RESOLVED,
            resolution=resolution,
            resolution_kind=resolution_kind,
            resolution_evidence_ids=resolution_evidence_ids,
        )
        artifact_id = _stable_id(
            "artifact",
            conflict_id,
            resolution_kind.value,
            *resolution_evidence_ids,
        )
        self.artifact_store.put_json(
            {
                "schema": "ConflictResolution@1",
                "conflict": resolved.model_dump(mode="json"),
            },
            redact=False,
            kind=ArtifactKind.REPAIR_FEEDBACK,
            producer_id=self.verifier_id,
            run_id=conflict.provenance.run_id,
            task_id=task_id or conflict.provenance.task_id,
            content_schema="ConflictResolution@1",
            source_artifact_ids=tuple(
                dict.fromkeys(
                    artifact_id
                    for item in evidence
                    for artifact_id in item.provenance.source_artifact_ids
                )
            ),
            artifact_id=artifact_id,
            idempotency_key=f"conflict-resolution:{conflict_id}:{resolution_kind.value}",
        )
        resolved = self._with_verifier_provenance(
            resolved,
            task_id=task_id or conflict.provenance.task_id,
            artifact_ids=(artifact_id,),
        )
        invalidated_claims: list[Claim] = []
        evaluated_statuses = {
            ClaimStatus.SUPPORTED,
            ClaimStatus.PARTIALLY_SUPPORTED,
            ClaimStatus.CONTRADICTED,
            ClaimStatus.CONFLICTED,
            ClaimStatus.UNSUPPORTED,
            ClaimStatus.STALE,
        }
        for claim_id in conflict.claim_ids:
            claim = self.repository.claims.require(claim_id)
            if claim.status not in evaluated_statuses:
                continue
            candidate = claim.transition(ClaimStatus.CONTESTED)
            invalidated_claims.append(
                self._with_verifier_provenance(
                    candidate,
                    task_id=task_id or conflict.provenance.task_id,
                    artifact_ids=(artifact_id,),
                )
            )
        self.repository.save_graph(resolved, *invalidated_claims)
        self.event_sink.emit(
            EvidenceDomainEvent(
                event_id=_stable_id("event", artifact_id, "conflict_resolved"),
                event_type=EventType.EVIDENCE_CHANGED,
                run_id=conflict.provenance.run_id,
                task_id=task_id or conflict.provenance.task_id,
                subject_id=conflict_id,
                actor_id=self.verifier_id,
                output_artifact_ids=(artifact_id,),
                payload={
                    "change": "conflict_resolved",
                    "resolution_kind": resolution_kind.value,
                    "resolution_evidence_ids": list(resolution_evidence_ids),
                    "invalidated_claim_ids": [
                        item.claim_id for item in invalidated_claims
                    ],
                },
            )
        )
        return resolved

    def accept_conflict_unresolved(
        self,
        conflict_id: str,
        *,
        reason: str,
        task_id: str | None = None,
    ) -> Conflict:
        conflict = self.repository.conflicts.require(conflict_id)
        if conflict.status == ConflictStatus.ACCEPTED_UNRESOLVED:
            return conflict
        accepted = conflict.transition(ConflictStatus.ACCEPTED_UNRESOLVED)
        values = accepted.model_dump(mode="python")
        values.update(
            resolution=reason,
            resolution_kind=ConflictResolutionKind.ACCEPTED_UNRESOLVED,
        )
        accepted = Conflict.model_validate(values, strict=False)
        artifact_id = _stable_id("artifact", conflict_id, "accepted_unresolved", reason)
        self.artifact_store.put_json(
            {
                "schema": "AcceptedUnresolvedConflict@1",
                "conflict": accepted.model_dump(mode="json"),
            },
            redact=False,
            kind=ArtifactKind.REPAIR_FEEDBACK,
            producer_id=self.verifier_id,
            run_id=conflict.provenance.run_id,
            task_id=task_id or conflict.provenance.task_id,
            content_schema="AcceptedUnresolvedConflict@1",
            source_artifact_ids=conflict.provenance.source_artifact_ids,
            artifact_id=artifact_id,
            idempotency_key=f"accepted-unresolved:{conflict_id}:{hashlib.sha256(reason.encode()).hexdigest()}",
        )
        accepted = self._with_verifier_provenance(
            accepted,
            task_id=task_id or conflict.provenance.task_id,
            artifact_ids=(artifact_id,),
        )
        self.repository.conflicts.save(accepted)
        self.event_sink.emit(
            EvidenceDomainEvent(
                event_id=_stable_id(
                    "event", artifact_id, "conflict_accepted_unresolved"
                ),
                event_type=EventType.EVIDENCE_CHANGED,
                run_id=conflict.provenance.run_id,
                task_id=task_id or conflict.provenance.task_id,
                subject_id=conflict_id,
                actor_id=self.verifier_id,
                output_artifact_ids=(artifact_id,),
                payload={
                    "change": "conflict_accepted_unresolved",
                    "severity": conflict.severity.value,
                },
            )
        )
        return accepted

    def _result_for_existing_claim(
        self,
        claim: Claim,
        graph: ClaimGraph,
    ) -> ClaimVerificationSummary | None:
        if claim.verification_id is None:
            return None
        artifact_id = _stable_id("artifact", claim.verification_id)
        if self.artifact_store.get(artifact_id) is None:
            raise EvidenceVerificationError(
                f"claim references missing verification artifact: {claim.claim_id}"
            )
        payload = json.loads(
            self.artifact_store.read_bytes(artifact_id).decode("utf-8")
        )
        repair_round = int(payload.get("repair_round", 0))
        if payload.get("graph_fingerprint") != self._graph_fingerprint(
            graph,
            repair_round,
        ):
            return None
        return ClaimVerificationSummary.model_validate(
            payload["summary"],
            strict=False,
        )

    def _replay_verification_artifact(
        self,
        artifact_id: str,
    ) -> ClaimVerificationSummary:
        payload = json.loads(
            self.artifact_store.read_bytes(artifact_id).decode("utf-8")
        )
        summary = ClaimVerificationSummary.model_validate(
            payload["summary"], strict=False
        )
        feedback_artifact_id = self._persist_repair_feedback(
            summary,
            repair_round=int(payload.get("repair_round", 0)),
        )
        provenance_artifacts = tuple(
            item for item in (artifact_id, feedback_artifact_id) if item is not None
        )
        revisions: list[KnowledgeEntity] = []
        for item in payload.get("entity_revisions", []):
            model = _ENTITY_MODELS.get(str(item.get("entity_type")))
            if model is None:
                raise EvidenceVerificationError(
                    "verification artifact contains an unknown entity type"
                )
            revisions.append(model.model_validate(item.get("value"), strict=False))
        if revisions:
            revisions = [
                self._with_verifier_provenance(
                    item,
                    task_id=getattr(item, "provenance").task_id,
                    artifact_ids=provenance_artifacts,
                )
                for item in revisions
            ]
            self.repository.save_graph(*revisions)
        return summary

    def _read_summary_artifact(self, artifact_id: str) -> ClaimVerificationSummary:
        payload = json.loads(
            self.artifact_store.read_bytes(artifact_id).decode("utf-8")
        )
        return ClaimVerificationSummary.model_validate(payload["summary"], strict=False)

    def _persist_repair_feedback(
        self,
        summary: ClaimVerificationSummary,
        *,
        repair_round: int,
    ) -> str | None:
        result = summary.verification_result
        if not result.repair_requests:
            return None
        feedback_artifact_id = _stable_id(
            "artifact",
            result.verification_id,
            "repair",
        )
        self.artifact_store.put_json(
            {
                "schema": "VerificationRepairFeedback@1",
                "verification_id": result.verification_id,
                "repair_round": repair_round,
                "remaining_rounds": max(
                    0,
                    self.policy.max_repair_rounds - repair_round,
                ),
                "requests": [
                    item.model_dump(mode="json") for item in result.repair_requests
                ],
            },
            redact=False,
            kind=ArtifactKind.REPAIR_FEEDBACK,
            producer_id=self.verifier_id,
            run_id=result.run_id,
            task_id=result.task_id,
            content_schema="VerificationRepairFeedback@1",
            source_artifact_ids=(result.result_artifact_id,),
            artifact_id=feedback_artifact_id,
            idempotency_key=f"verification-repair:{result.verification_id}",
        )
        return feedback_artifact_id

    def _graph_fingerprint(self, graph: ClaimGraph, repair_round: int) -> str:
        entities: tuple[Any, ...] = (
            graph.claim,
            *graph.facts,
            *graph.evidence,
            *graph.passages,
            *graph.snapshots,
            *graph.sources,
            *graph.citations,
            *graph.conflicts,
        )
        value = "\n".join(
            [
                *(
                    json.dumps(
                        self._verification_input(item, graph),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for item in entities
                ),
                canonical_contract_json(self.policy),
                str(repair_round),
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _verification_input(
        self,
        entity: KnowledgeEntity,
        graph: ClaimGraph,
    ) -> dict[str, Any]:
        value = entity.model_dump(mode="json")
        value.pop("updated_at", None)
        provenance = value.pop("provenance", None)
        if isinstance(provenance, dict):
            source_artifact_ids = []
            for artifact_id in provenance.get("source_artifact_ids", ()):
                artifact = self.artifact_store.get(str(artifact_id))
                if artifact is not None and artifact.kind in {
                    ArtifactKind.VERIFICATION_RESULT,
                    ArtifactKind.REPAIR_FEEDBACK,
                }:
                    continue
                source_artifact_ids.append(str(artifact_id))
            value["input_provenance"] = {
                "run_id": provenance.get("run_id"),
                "task_id": provenance.get("task_id"),
                "causation_event_id": provenance.get("causation_event_id"),
                "source_artifact_ids": source_artifact_ids,
            }
        if isinstance(entity, Evidence):
            if not entity.quotes:
                passage_by_id = {item.passage_id: item for item in graph.passages}
                effective_quotes: list[dict[str, Any]] = []
                for citation in graph.citations:
                    if citation.evidence_id != entity.evidence_id:
                        continue
                    passage = passage_by_id.get(citation.passage_id)
                    if passage is None:
                        continue
                    try:
                        text = self.artifact_store.read_bytes(
                            passage.text_artifact_id
                        ).decode("utf-8")
                    except (UnicodeDecodeError, OSError, KeyError):
                        continue
                    start = citation.quote_start
                    if start is None:
                        start = text.find(citation.quote)
                    if start < 0:
                        continue
                    effective_quotes.append(
                        EvidenceQuote(
                            passage_id=passage.passage_id,
                            quote=citation.quote,
                            char_start=start,
                            char_end=start + len(citation.quote),
                            passage_content_hash=passage.content_hash,
                            extraction_method=citation.extraction_method,
                            extracted_at=citation.created_at,
                        ).model_dump(mode="json")
                    )
                value["quotes"] = effective_quotes
            for name in ("status", "verification_id", "verified_at"):
                value.pop(name, None)
        elif isinstance(entity, AtomicFact):
            for name in ("status", "verification_id", "verified_at"):
                value.pop(name, None)
        elif isinstance(entity, Claim):
            value["effective_high_impact"] = bool(
                entity.high_impact
                or entity.importance >= self.policy.high_impact_threshold
            )
            for name in (
                "status",
                "high_impact",
                "support_score",
                "verification_id",
                "verified_at",
            ):
                value.pop(name, None)
        elif isinstance(entity, Citation):
            for name in (
                "status",
                "quote_start",
                "quote_end",
                "verification_id",
                "verified_at",
            ):
                value.pop(name, None)
        elif isinstance(entity, Conflict):
            for name in ("severity", "high_impact"):
                value.pop(name, None)
        return value

    def _read_passage(
        self,
        passage: Passage,
        snapshots_by_id: dict[str, SourceSnapshot],
    ) -> tuple[str, bool]:
        artifact = self.artifact_store.get(passage.text_artifact_id)
        snapshot = snapshots_by_id.get(passage.snapshot_id)
        if artifact is None or snapshot is None:
            return "", False
        snapshot_artifact = (
            self.artifact_store.get(snapshot.artifact_id)
            if snapshot.artifact_id is not None
            else None
        )
        try:
            text = self.artifact_store.read_bytes(passage.text_artifact_id).decode(
                "utf-8"
            )
        except (UnicodeDecodeError, OSError, KeyError):
            return "", False
        valid = (
            passage.status == PassageStatus.ACCEPTED
            and artifact.content_hash == passage.content_hash
            and snapshot_artifact is not None
            and snapshot.content_hash == snapshot_artifact.content_hash
        )
        return text, valid

    def _ground_quotes(
        self,
        evidence: Evidence,
        citations: tuple[Citation, ...],
        passages_by_id: dict[str, Passage],
        passage_text: dict[str, str],
        passage_integrity: dict[str, bool],
    ) -> tuple[tuple[EvidenceQuote, ...], tuple[tuple[str, str, str], ...]]:
        candidates = list(evidence.quotes)
        if not candidates:
            for citation in citations:
                passage = passages_by_id.get(citation.passage_id)
                text = passage_text.get(citation.passage_id, "")
                if passage is None or not text:
                    continue
                start = citation.quote_start
                if start is None:
                    start = text.find(citation.quote)
                if start < 0:
                    continue
                candidates.append(
                    EvidenceQuote(
                        passage_id=passage.passage_id,
                        quote=citation.quote,
                        char_start=start,
                        char_end=start + len(citation.quote),
                        passage_content_hash=passage.content_hash,
                        extraction_method=citation.extraction_method,
                        extracted_at=citation.created_at,
                    )
                )
        issues: list[tuple[str, str, str]] = []
        valid: list[EvidenceQuote] = []
        if len(candidates) > self.policy.max_quotes_per_evidence:
            issues.append(
                (
                    "quote_count_exceeds_policy",
                    f"Evidence has {len(candidates)} quotes; policy allows {self.policy.max_quotes_per_evidence}.",
                    evidence.evidence_id,
                )
            )
            candidates = candidates[: self.policy.max_quotes_per_evidence]
        for quote in candidates:
            passage = passages_by_id.get(quote.passage_id)
            text = passage_text.get(quote.passage_id, "")
            if passage is None or quote.passage_id not in evidence.passage_ids:
                issues.append(
                    (
                        "quote_passage_mismatch",
                        "Evidence quote refers to a passage outside the evidence.",
                        quote.passage_id,
                    )
                )
                continue
            if not passage_integrity.get(quote.passage_id, False):
                issues.append(
                    (
                        "passage_integrity_failed",
                        "Passage or source snapshot content hash does not match its immutable artifact.",
                        quote.passage_id,
                    )
                )
                continue
            if quote.passage_content_hash != passage.content_hash:
                issues.append(
                    (
                        "quote_content_hash_mismatch",
                        "Evidence quote was extracted from a different passage content hash.",
                        quote.passage_id,
                    )
                )
                continue
            if (
                quote.char_end > len(text)
                or text[quote.char_start : quote.char_end] != quote.quote
            ):
                issues.append(
                    (
                        "quote_offset_not_grounded",
                        "Evidence quote text does not match its persisted character offsets.",
                        quote.passage_id,
                    )
                )
                continue
            valid.append(quote)
        if not candidates:
            issues.append(
                (
                    "grounded_quote_missing",
                    "Evidence has no persisted quote and character offsets.",
                    evidence.evidence_id,
                )
            )
        return tuple(valid), tuple(issues)

    def _verify_citation(
        self,
        citation: Citation,
        *,
        claim: Claim,
        evidence: Evidence | None,
        passages_by_id: dict[str, Passage],
        snapshots_by_id: dict[str, SourceSnapshot],
        passage_text: dict[str, str],
        valid_quotes: tuple[EvidenceQuote, ...],
        verification_id: str,
        now: datetime,
    ) -> tuple[Citation, tuple[tuple[str, str], ...]]:
        errors: list[tuple[str, str]] = []
        passage = passages_by_id.get(citation.passage_id)
        snapshot = snapshots_by_id.get(citation.snapshot_id)
        if citation.claim_id != claim.claim_id:
            errors.append(
                ("citation_claim_mismatch", "Citation points to a different claim.")
            )
        if evidence is None or citation.passage_id not in evidence.passage_ids:
            errors.append(
                (
                    "citation_evidence_path_mismatch",
                    "Citation passage is outside its evidence.",
                )
            )
        if (
            passage is None
            or snapshot is None
            or passage.snapshot_id != citation.snapshot_id
        ):
            errors.append(
                (
                    "citation_snapshot_path_mismatch",
                    "Citation passage/snapshot path is inconsistent.",
                )
            )
        elif snapshot.source_id != citation.source_id:
            errors.append(
                (
                    "citation_source_path_mismatch",
                    "Citation snapshot/source path is inconsistent.",
                )
            )
        if passage is not None and citation.locator != passage.locator:
            errors.append(
                (
                    "citation_location_mismatch",
                    "Citation locator does not match the persisted passage locator.",
                )
            )
        text = passage_text.get(citation.passage_id, "")
        start = citation.quote_start
        if start is None:
            start = text.find(citation.quote)
        end = start + len(citation.quote) if start is not None and start >= 0 else None
        if (
            start is None
            or start < 0
            or end is None
            or end > len(text)
            or text[start:end] != citation.quote
        ):
            errors.append(
                (
                    "citation_quote_not_grounded",
                    "Citation quote is not grounded at its passage offsets.",
                )
            )
        if valid_quotes and citation.quote not in {item.quote for item in valid_quotes}:
            errors.append(
                (
                    "citation_quote_not_in_evidence",
                    "Citation quote is not part of the verified evidence.",
                )
            )
        values = citation.model_dump(mode="python")
        if errors:
            values.update(
                status=CitationStatus.REJECTED,
                verification_id=None,
                verified_at=None,
                quote_start=start if start is not None and start >= 0 else None,
                quote_end=end,
                updated_at=now,
            )
        else:
            values.update(
                status=CitationStatus.VERIFIED,
                verification_id=verification_id,
                verified_at=now,
                quote_start=start,
                quote_end=end,
                updated_at=now,
            )
        return Citation.model_validate(values, strict=False), tuple(errors)

    def _repair_requests(
        self,
        verification_id: str,
        issues: tuple[VerificationIssue, ...],
        *,
        repair_round: int,
    ) -> tuple[RepairRequest, ...]:
        if repair_round >= self.policy.max_repair_rounds:
            return ()
        mapping = {
            VerificationCategory.PASSAGE_INTEGRITY: RepairAction.REEXTRACT_PASSAGE,
            VerificationCategory.QUOTE_GROUNDING: RepairAction.REEXTRACT_PASSAGE,
            VerificationCategory.EVIDENCE_SUPPORT: RepairAction.REVISE_CLAIM,
            VerificationCategory.CLAIM_SUPPORT: RepairAction.RESEARCH,
            VerificationCategory.CLAIM_OVERREACH: RepairAction.REVISE_CLAIM,
            VerificationCategory.SOURCE_ACCESS: RepairAction.REFETCH_SOURCE,
            VerificationCategory.SOURCE_AUTHORITY: RepairAction.RESEARCH,
            VerificationCategory.SOURCE_FRESHNESS: RepairAction.RESEARCH,
            VerificationCategory.SOURCE_INDEPENDENCE: RepairAction.RESEARCH,
            VerificationCategory.CITATION_ACCURACY: RepairAction.REPLACE_CITATION,
            VerificationCategory.CITATION_COMPLETENESS: RepairAction.REPLACE_CITATION,
            VerificationCategory.CONFLICT: RepairAction.RESOLVE_CONFLICT,
            VerificationCategory.COVERAGE: RepairAction.RESEARCH,
        }
        requests: list[RepairRequest] = []
        for issue in issues:
            if not issue.repairable or issue.severity == VerificationSeverity.INFO:
                continue
            action = mapping.get(issue.category)
            if action is None:
                continue
            requests.append(
                RepairRequest(
                    repair_id=_stable_id(
                        "repair", verification_id, issue.issue_id, action.value
                    ),
                    verification_id=verification_id,
                    issue_ids=(issue.issue_id,),
                    action=action,
                    target_id=issue.subject_id,
                    instructions=(
                        f"Repair {issue.code}: {issue.message} Re-run independent verification "
                        f"after producing a new candidate revision. Remaining repair rounds: "
                        f"{self.policy.max_repair_rounds - repair_round - 1}."
                    )[:5000],
                    priority={
                        VerificationSeverity.CRITICAL: 1.0,
                        VerificationSeverity.ERROR: 0.85,
                        VerificationSeverity.WARNING: 0.55,
                        VerificationSeverity.INFO: 0.2,
                    }[issue.severity],
                )
            )
            if len(requests) >= self.policy.max_repairs_per_verification:
                break
        return tuple(requests)

    def _issue(
        self,
        verification_id: str,
        *,
        category: VerificationCategory,
        severity: VerificationSeverity,
        code: str,
        message: str,
        subject_id: str,
        evidence_ids: tuple[str, ...],
        repair_round: int,
    ) -> VerificationIssue:
        return VerificationIssue(
            issue_id=_stable_id(
                "verification_issue",
                verification_id,
                category.value,
                code,
                subject_id,
                *evidence_ids,
            ),
            category=category,
            severity=severity,
            code=code,
            message=message[:4000],
            subject_id=subject_id,
            evidence_ids=tuple(dict.fromkeys(evidence_ids)),
            repairable=repair_round < self.policy.max_repair_rounds,
            metadata={
                "repair_round": repair_round,
                "max_repair_rounds": self.policy.max_repair_rounds,
            },
        )

    def _input_artifacts(self, graph: ClaimGraph) -> tuple[str, ...]:
        values: list[str] = []
        for item in (
            graph.claim,
            *graph.facts,
            *graph.evidence,
            *graph.passages,
            *graph.snapshots,
            *graph.sources,
            *graph.citations,
            *graph.conflicts,
        ):
            provenance = getattr(item, "provenance", None)
            if provenance is not None:
                values.extend(provenance.source_artifact_ids)
            for name in ("artifact_id", "text_artifact_id", "content_artifact_id"):
                value = getattr(item, name, None)
                if value:
                    values.append(value)
        return tuple(dict.fromkeys(values))

    def _emit_claim_events(
        self,
        summary: ClaimVerificationSummary,
        *,
        feedback_artifact_id: str | None = None,
    ) -> None:
        result = summary.verification_result
        if feedback_artifact_id is None and result.repair_requests:
            candidate = _stable_id("artifact", result.verification_id, "repair")
            if self.artifact_store.get(candidate) is not None:
                feedback_artifact_id = candidate
        outputs = tuple(
            item
            for item in (result.result_artifact_id, feedback_artifact_id)
            if item is not None
        )
        common = {
            "run_id": result.run_id,
            "task_id": result.task_id,
            "subject_id": summary.claim_id,
            "actor_id": self.verifier_id,
            "input_artifact_ids": result.input_artifact_ids,
            "output_artifact_ids": outputs,
            "usage": result.usage,
        }
        self.event_sink.emit(
            EvidenceDomainEvent(
                event_id=_stable_id(
                    "event", result.verification_id, "evidence_changed"
                ),
                event_type=EventType.EVIDENCE_CHANGED,
                payload={
                    "change": "claim_evidence_verified",
                    "claim_status": summary.status.value,
                    "verified_evidence_ids": list(summary.verified_evidence_ids),
                    "verified_citation_ids": list(summary.verified_citation_ids),
                    "high_impact_blocked": summary.high_impact_blocked,
                },
                **common,
            )
        )
        self.event_sink.emit(
            EvidenceDomainEvent(
                event_id=_stable_id(
                    "event", result.verification_id, "verification_completed"
                ),
                event_type=EventType.VERIFICATION_COMPLETED,
                payload={
                    "verification_id": result.verification_id,
                    "passed": result.passed,
                    "score": result.score,
                    "threshold": result.threshold,
                    "issue_ids": [item.issue_id for item in result.issues],
                    "repair_ids": [item.repair_id for item in result.repair_requests],
                },
                **common,
            )
        )

    def _coverage_status(
        self,
        assessment: SectionCoverageAssessment,
    ) -> SectionCoverageStatus:
        if assessment.blocked:
            return SectionCoverageStatus.BLOCKED
        if (
            assessment.coverage_score >= self.policy.section_coverage_threshold
            and assessment.citation_score >= self.policy.citation_coverage_threshold
        ):
            return SectionCoverageStatus.COMPLETE
        if assessment.coverage_score > 0.0:
            return SectionCoverageStatus.PARTIAL
        return SectionCoverageStatus.INSUFFICIENT

    def _save_section_assessment(
        self,
        section: Section,
        assessment: SectionCoverageAssessment,
        *,
        coverage_status: SectionCoverageStatus,
        task_id: str | None,
    ) -> None:
        values = section.model_dump(mode="python")
        values.update(
            coverage_score=assessment.coverage_score,
            citation_score=assessment.citation_score,
            coverage_status=coverage_status,
            unsupported_claim_ids=assessment.unsupported_claim_ids,
            conflicted_claim_ids=assessment.conflicted_claim_ids,
            updated_at=assessment.assessed_at,
        )
        revised = Section.model_validate(values, strict=False)
        revised = self._with_verifier_provenance(
            revised,
            task_id=task_id or section.provenance.task_id,
            artifact_ids=(assessment.result_artifact_id,),
        )
        self.repository.sections.save(revised)

    def _emit_section_assessment(
        self,
        section: Section,
        assessment: SectionCoverageAssessment,
        *,
        task_id: str | None,
    ) -> None:
        coverage_status = self._coverage_status(assessment)
        self.event_sink.emit(
            EvidenceDomainEvent(
                event_id=_stable_id(
                    "event",
                    assessment.result_artifact_id,
                    "section_coverage",
                ),
                event_type=EventType.EVIDENCE_CHANGED,
                run_id=section.provenance.run_id,
                task_id=task_id or section.provenance.task_id,
                subject_id=section.section_id,
                actor_id=self.verifier_id,
                input_artifact_ids=tuple(
                    item
                    for item in section.provenance.source_artifact_ids
                    if item != assessment.result_artifact_id
                ),
                output_artifact_ids=(assessment.result_artifact_id,),
                payload={
                    "change": "section_coverage_assessed",
                    "coverage_status": coverage_status.value,
                    "coverage_score": assessment.coverage_score,
                    "citation_score": assessment.citation_score,
                    "blocked": assessment.blocked,
                },
            )
        )

    def _source_fresh(
        self,
        source: Source,
        snapshots: tuple[SourceSnapshot, ...],
        now: datetime,
    ) -> bool:
        timestamp = source.published_at or max(
            (item.fetched_at for item in snapshots),
            default=source.discovered_at,
        )
        return now - timestamp <= timedelta(days=self.policy.freshness_days)

    @staticmethod
    def _independence_key(source: Source) -> str:
        publisher = (source.publisher or "").strip().casefold()
        host = (urlsplit(source.canonical_url).hostname or source.source_id).casefold()
        return publisher or host

    @staticmethod
    def _source_level(source: Source) -> SourceLevel:
        if source.source_level != SourceLevel.UNKNOWN:
            return source.source_level
        if source.source_type.value in {"primary", "official_documentation", "dataset"}:
            return SourceLevel.PRIMARY
        if source.source_type.value in {"secondary", "academic", "news"}:
            return SourceLevel.SECONDARY
        if source.source_type.value in {"tertiary", "community"}:
            return SourceLevel.TERTIARY
        return SourceLevel.UNKNOWN

    def _relation_matches(
        self,
        relation: EvidenceRelation,
        semantic: SemanticJudgment,
    ) -> bool:
        if relation == EvidenceRelation.CONTEXTUALIZES:
            return semantic.score >= self.policy.partial_support_threshold
        expected = {
            EvidenceRelation.SUPPORTS: SemanticLabel.SUPPORTS,
            EvidenceRelation.REFUTES: SemanticLabel.REFUTES,
        }[relation]
        return (
            semantic.label == expected
            and semantic.score >= self.policy.partial_support_threshold
        )

    def _conflict_severity(
        self,
        conflict: Conflict,
        graph: ClaimGraph,
        high_impact: bool,
    ) -> ConflictSeverity:
        if conflict.status == ConflictStatus.RESOLVED:
            return conflict.severity
        importance = max(
            (
                self.repository.claims.require(item).importance
                for item in conflict.claim_ids
            ),
            default=graph.claim.importance,
        )
        if high_impact and importance >= 0.8:
            return ConflictSeverity.CRITICAL
        if high_impact or importance >= 0.8:
            return ConflictSeverity.HIGH
        if importance >= 0.5:
            return ConflictSeverity.MEDIUM
        return ConflictSeverity.LOW

    @staticmethod
    def _revise_evidence(
        evidence: Evidence,
        *,
        status: EvidenceStatus,
        verification_id: str,
        quotes: tuple[EvidenceQuote, ...],
        now: datetime,
    ) -> Evidence:
        values = evidence.model_dump(mode="python")
        values.update(
            status=status,
            quotes=quotes,
            verification_id=(
                verification_id if status == EvidenceStatus.VERIFIED else None
            ),
            verified_at=now if status == EvidenceStatus.VERIFIED else None,
            updated_at=now,
        )
        return Evidence.model_validate(values, strict=False)

    @staticmethod
    def _revise_fact(
        fact: AtomicFact,
        *,
        status: FactStatus,
        verification_id: str,
        now: datetime,
    ) -> AtomicFact:
        values = fact.model_dump(mode="python")
        values.update(
            status=status,
            verification_id=verification_id if status == FactStatus.VERIFIED else None,
            verified_at=now if status == FactStatus.VERIFIED else None,
            updated_at=now,
        )
        return AtomicFact.model_validate(values, strict=False)

    @staticmethod
    def _revise_claim(
        claim: Claim,
        *,
        status: ClaimStatus,
        verification_id: str,
        support_score: float,
        high_impact: bool,
        now: datetime,
    ) -> Claim:
        values = claim.model_dump(mode="python")
        values.update(
            status=status,
            support_score=support_score,
            high_impact=high_impact,
            verification_id=verification_id,
            verified_at=now,
            updated_at=now,
        )
        return Claim.model_validate(values, strict=False)

    @staticmethod
    def _revise_conflict(
        conflict: Conflict,
        *,
        severity: ConflictSeverity,
        high_impact: bool,
        now: datetime,
    ) -> Conflict:
        values = conflict.model_dump(mode="python")
        values.update(severity=severity, high_impact=high_impact, updated_at=now)
        return Conflict.model_validate(values, strict=False)

    def _with_verifier_provenance(
        self,
        entity: KnowledgeEntity,
        *,
        task_id: str | None,
        artifact_ids: tuple[str, ...],
    ) -> KnowledgeEntity:
        provenance = getattr(entity, "provenance")
        values = entity.model_dump(mode="python")
        values["provenance"] = EntityProvenance(
            producer_id=self.verifier_id,
            run_id=provenance.run_id,
            task_id=task_id,
            causation_event_id=provenance.causation_event_id,
            source_artifact_ids=tuple(
                dict.fromkeys((*provenance.source_artifact_ids, *artifact_ids))
            ),
        )
        return type(entity).model_validate(values, strict=False)
