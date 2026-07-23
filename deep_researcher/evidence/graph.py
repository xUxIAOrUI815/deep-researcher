from __future__ import annotations

from dataclasses import dataclass

from deep_researcher.contracts import (
    AtomicFact,
    Citation,
    CitationStatus,
    Claim,
    ClaimStatus,
    Conflict,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    FactStatus,
    Passage,
    PassageStatus,
    Source,
    SourceSnapshot,
)
from deep_researcher.knowledge import KnowledgeRepository
from deep_researcher.knowledge.sqlite_storage import entity_run_id


class EvidenceGraphError(RuntimeError):
    pass


@dataclass(frozen=True)
class ClaimGraph:
    claim: Claim
    facts: tuple[AtomicFact, ...]
    evidence: tuple[Evidence, ...]
    passages: tuple[Passage, ...]
    snapshots: tuple[SourceSnapshot, ...]
    sources: tuple[Source, ...]
    citations: tuple[Citation, ...]
    conflicts: tuple[Conflict, ...]

    @property
    def support_evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            item.evidence_id
            for item in self.evidence
            if item.relation == EvidenceRelation.SUPPORTS
        )


class EvidenceGraphResolver:
    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    def claim_graph(
        self, claim_id: str, *, strict_citations: bool = False
    ) -> ClaimGraph:
        claim = self.repository.claims.require(claim_id)
        run_id = claim.provenance.run_id
        facts = tuple(self.repository.facts.require(item) for item in claim.fact_ids)
        evidence_ids = list(claim.evidence_ids)
        for fact in facts:
            evidence_ids.extend(fact.evidence_ids)
        evidence = tuple(
            self.repository.evidence.require(item)
            for item in dict.fromkeys(evidence_ids)
        )
        passages = tuple(
            self.repository.passages.require(item)
            for item in dict.fromkeys(
                passage_id
                for evidence_item in evidence
                for passage_id in evidence_item.passage_ids
            )
        )
        snapshots = tuple(
            self.repository.snapshots.require(item)
            for item in dict.fromkeys(passage.snapshot_id for passage in passages)
        )
        sources = tuple(
            self.repository.sources.require(item)
            for item in dict.fromkeys(snapshot.source_id for snapshot in snapshots)
        )
        citations = tuple(
            item
            for item in self.repository.citations.list(run_id)
            if item.claim_id == claim_id
        )
        conflicts = tuple(
            item
            for item in self.repository.conflicts.list(run_id)
            if claim_id in item.claim_ids
        )
        entities = (
            *facts,
            *evidence,
            *passages,
            *snapshots,
            *sources,
            *citations,
            *conflicts,
        )
        cross_run = [
            type(item).__name__ for item in entities if entity_run_id(item) != run_id
        ]
        if cross_run:
            raise EvidenceGraphError(f"claim graph crosses run boundary: {cross_run}")
        if strict_citations:
            self._validate_citation_paths(
                claim,
                evidence,
                passages,
                snapshots,
                sources,
                citations,
            )
        return ClaimGraph(
            claim=claim,
            facts=facts,
            evidence=evidence,
            passages=passages,
            snapshots=snapshots,
            sources=sources,
            citations=citations,
            conflicts=conflicts,
        )

    @staticmethod
    def _validate_citation_paths(
        claim: Claim,
        evidence: tuple[Evidence, ...],
        passages: tuple[Passage, ...],
        snapshots: tuple[SourceSnapshot, ...],
        sources: tuple[Source, ...],
        citations: tuple[Citation, ...],
    ) -> None:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        passage_by_id = {item.passage_id: item for item in passages}
        snapshot_by_id = {item.snapshot_id: item for item in snapshots}
        source_ids = {item.source_id for item in sources}
        for citation in citations:
            evidence_item = evidence_by_id.get(citation.evidence_id)
            passage = passage_by_id.get(citation.passage_id)
            snapshot = snapshot_by_id.get(citation.snapshot_id)
            if citation.claim_id != claim.claim_id or evidence_item is None:
                raise EvidenceGraphError(
                    f"citation does not resolve to claim evidence: {citation.citation_id}"
                )
            if passage is None or citation.passage_id not in evidence_item.passage_ids:
                raise EvidenceGraphError(
                    f"citation passage is outside its evidence: {citation.citation_id}"
                )
            if snapshot is None or passage.snapshot_id != citation.snapshot_id:
                raise EvidenceGraphError(
                    f"citation snapshot path is inconsistent: {citation.citation_id}"
                )
            if (
                snapshot.source_id != citation.source_id
                or citation.source_id not in source_ids
            ):
                raise EvidenceGraphError(
                    f"citation source path is inconsistent: {citation.citation_id}"
                )


class CandidateKnowledgeView:
    """Read boundary exposing only entities that still require verification."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    def evidence(self, run_id: str) -> tuple[Evidence, ...]:
        return self.repository.evidence.list(
            run_id,
            statuses=(EvidenceStatus.PROPOSED.value,),
        )

    def facts(self, run_id: str) -> tuple[AtomicFact, ...]:
        return self.repository.facts.list(
            run_id,
            statuses=(FactStatus.PROPOSED.value, FactStatus.DISPUTED.value),
        )

    def claims(self, run_id: str) -> tuple[Claim, ...]:
        return self.repository.claims.list(
            run_id,
            statuses=(ClaimStatus.DRAFT.value, ClaimStatus.CONTESTED.value),
        )

    def citations(self, run_id: str) -> tuple[Citation, ...]:
        return self.repository.citations.list(
            run_id,
            statuses=(CitationStatus.PROPOSED.value,),
        )


class VerifiedKnowledgeView:
    """Read boundary that cannot return candidate knowledge to downstream writers."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository
        self.resolver = EvidenceGraphResolver(repository)

    def evidence(self, run_id: str) -> tuple[Evidence, ...]:
        return self.repository.evidence.list(
            run_id,
            statuses=(EvidenceStatus.VERIFIED.value,),
        )

    def facts(self, run_id: str) -> tuple[AtomicFact, ...]:
        return self.repository.facts.list(
            run_id,
            statuses=(FactStatus.VERIFIED.value,),
        )

    def citations(self, run_id: str) -> tuple[Citation, ...]:
        return self.repository.citations.list(
            run_id,
            statuses=(CitationStatus.VERIFIED.value,),
        )

    def claims_for_writing(self, run_id: str) -> tuple[Claim, ...]:
        output: list[Claim] = []
        claims = self.repository.claims.list(
            run_id,
            statuses=(ClaimStatus.SUPPORTED.value,),
        )
        for claim in claims:
            graph = self.resolver.claim_graph(claim.claim_id, strict_citations=True)
            if any(item.status != FactStatus.VERIFIED for item in graph.facts):
                continue
            support_evidence = tuple(
                item
                for item in graph.evidence
                if item.relation == EvidenceRelation.SUPPORTS
            )
            if not support_evidence or any(
                item.status != EvidenceStatus.VERIFIED for item in support_evidence
            ):
                continue
            verified_citation_evidence = {
                item.evidence_id
                for item in graph.citations
                if item.status == CitationStatus.VERIFIED
            }
            if any(
                item.evidence_id not in verified_citation_evidence
                for item in support_evidence
            ):
                continue
            output.append(claim)
        return tuple(output)

    def blocked_high_impact_claims(self, run_id: str) -> tuple[Claim, ...]:
        return tuple(
            claim
            for claim in self.repository.claims.list(run_id)
            if (claim.high_impact or claim.importance >= 0.8)
            and claim.status != ClaimStatus.SUPPORTED
        )

    def accepted_passages(self, run_id: str) -> tuple[Passage, ...]:
        return self.repository.passages.list(
            run_id,
            statuses=(PassageStatus.ACCEPTED.value,),
        )
