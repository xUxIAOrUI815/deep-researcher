from __future__ import annotations

import hashlib
import json
from typing import Any

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    CitationStatus,
    ConflictStatus,
    utc_now,
)
from deep_researcher.evidence import EvidenceRuntime

from .models import (
    ConflictPacket,
    EvidenceGapPacket,
    VerifiedCitationPacket,
    VerifiedClaimPacket,
    WriterEvidencePacket,
    WriterSectionPacket,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def read_json_artifact(
    artifact_store: ArtifactStore,
    artifact_id: str,
) -> dict[str, Any]:
    try:
        value = json.loads(
            artifact_store.read_bytes(artifact_id).decode("utf-8")
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact is not valid JSON: {artifact_id}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"artifact JSON is not an object: {artifact_id}")
    return value


class VerifiedWriterPacketBuilder:
    """Builds the sole Writer input boundary from VerifiedKnowledgeView."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        artifact_store: ArtifactStore | None = None,
        producer_id: str = "runtime_verified_writer_packet_builder",
        clock=utc_now,
    ) -> None:
        self.evidence = evidence
        self.artifact_store = (
            artifact_store or evidence.knowledge.artifacts
        )
        self.producer_id = producer_id
        self.clock = clock

    def build(self, report_id: str) -> WriterEvidencePacket:
        repository = self.evidence.knowledge.repository
        report = repository.reports.require(report_id)
        run_id = report.run_id
        verified = {
            item.claim_id: item
            for item in self.evidence.verified.claims_for_writing(run_id)
        }
        section_entities = tuple(
            repository.sections.require(item)
            for item in report.section_ids
        )
        citation_packets: dict[str, VerifiedCitationPacket] = {}
        claim_packets: dict[str, VerifiedClaimPacket] = {}
        gaps: list[EvidenceGapPacket] = []
        conflict_packets: dict[str, ConflictPacket] = {}
        section_packets: list[WriterSectionPacket] = []
        all_conflicts = repository.conflicts.list(run_id)

        for section in sorted(
            section_entities,
            key=lambda item: (item.order, item.section_id),
        ):
            required = section.required_claim_ids or section.claim_ids
            verified_ids: list[str] = []
            gap_ids: list[str] = []
            for claim_id in required:
                claim = verified.get(claim_id)
                if claim is None:
                    candidate = repository.claims.require(claim_id)
                    gaps.append(
                        EvidenceGapPacket(
                            claim_id=claim_id,
                            section_id=section.section_id,
                            statement=candidate.statement,
                            reason=(
                                "The claim is not available through the "
                                "verified-only writing boundary."
                            ),
                            high_impact=(
                                candidate.high_impact
                                or candidate.importance >= 0.8
                            ),
                        )
                    )
                    gap_ids.append(claim_id)
                    continue
                graph = self.evidence.verified.resolver.claim_graph(
                    claim_id,
                    strict_citations=True,
                )
                verified_citations = tuple(
                    item
                    for item in graph.citations
                    if item.status == CitationStatus.VERIFIED
                )
                source_by_id = {
                    item.source_id: item for item in graph.sources
                }
                for citation in verified_citations:
                    source = source_by_id[citation.source_id]
                    citation_packets[citation.citation_id] = (
                        VerifiedCitationPacket(
                            citation_id=citation.citation_id,
                            claim_id=claim_id,
                            evidence_id=citation.evidence_id,
                            passage_id=citation.passage_id,
                            snapshot_id=citation.snapshot_id,
                            source_id=citation.source_id,
                            source_title=source.title,
                            canonical_url=source.canonical_url,
                            publisher=source.publisher,
                            locator=citation.locator,
                            quote=citation.quote,
                        )
                    )
                claim_packets[claim_id] = VerifiedClaimPacket(
                    claim_id=claim_id,
                    statement=claim.statement,
                    importance=claim.importance,
                    high_impact=(
                        claim.high_impact or claim.importance >= 0.8
                    ),
                    citation_ids=tuple(
                        item.citation_id for item in verified_citations
                    ),
                    source_ids=tuple(
                        dict.fromkeys(
                            item.source_id for item in verified_citations
                        )
                    ),
                )
                verified_ids.append(claim_id)

            section_claim_ids = set(section.claim_ids)
            section_conflicts = tuple(
                item
                for item in all_conflicts
                if section_claim_ids.intersection(item.claim_ids)
            )
            for conflict in section_conflicts:
                conflict_packets[conflict.conflict_id] = ConflictPacket(
                    conflict_id=conflict.conflict_id,
                    claim_ids=conflict.claim_ids,
                    summary=conflict.summary,
                    status=conflict.status,
                    severity=conflict.severity,
                    high_impact=conflict.high_impact,
                    resolution=conflict.resolution,
                )
            section_packets.append(
                WriterSectionPacket(
                    section_id=section.section_id,
                    title=section.title,
                    goal=section.goal,
                    order=section.order,
                    required_claim_ids=required,
                    verified_claim_ids=tuple(verified_ids),
                    gap_claim_ids=tuple(gap_ids),
                    conflict_ids=tuple(
                        item.conflict_id for item in section_conflicts
                    ),
                )
            )

        material = {
            "run_id": run_id,
            "report_id": report_id,
            "research_question": report.research_question,
            "sections": [
                item.model_dump(mode="json") for item in section_packets
            ],
            "claims": [
                item.model_dump(mode="json")
                for item in sorted(
                    claim_packets.values(),
                    key=lambda value: value.claim_id,
                )
            ],
            "citations": [
                item.model_dump(mode="json")
                for item in sorted(
                    citation_packets.values(),
                    key=lambda value: value.citation_id,
                )
            ],
            "gaps": [item.model_dump(mode="json") for item in gaps],
            "conflicts": [
                item.model_dump(mode="json")
                for item in sorted(
                    conflict_packets.values(),
                    key=lambda value: value.conflict_id,
                )
            ],
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        packet_id = _stable_id("writer_packet", report_id, fingerprint)
        artifact_id = _stable_id("artifact", packet_id)
        existing = self.artifact_store.get(artifact_id)
        if existing is not None:
            payload = read_json_artifact(self.artifact_store, artifact_id)
            return WriterEvidencePacket.model_validate(
                payload["packet"],
                strict=False,
            )
        packet = WriterEvidencePacket(
            packet_id=packet_id,
            run_id=run_id,
            report_id=report_id,
            research_question=report.research_question,
            sections=tuple(section_packets),
            claims=tuple(
                sorted(
                    claim_packets.values(),
                    key=lambda value: value.claim_id,
                )
            ),
            citations=tuple(
                sorted(
                    citation_packets.values(),
                    key=lambda value: value.citation_id,
                )
            ),
            gaps=tuple(gaps),
            conflicts=tuple(
                sorted(
                    conflict_packets.values(),
                    key=lambda value: value.conflict_id,
                )
            ),
            packet_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        source_artifacts = tuple(
            dict.fromkeys(
                artifact_id
                for claim_id in claim_packets
                for claim in (repository.claims.require(claim_id),)
                for artifact_id in claim.provenance.source_artifact_ids
            )
        )
        self.artifact_store.put_json(
            {
                "schema": "WriterEvidencePacket@1",
                "packet": packet.model_dump(mode="json"),
                "boundary": {
                    "verified_only": True,
                    "candidate_claim_count": 0,
                    "unverified_claims_are_gaps": True,
                },
            },
            redact=False,
            kind=ArtifactKind.EVIDENCE_PACK,
            producer_id=self.producer_id,
            run_id=run_id,
            content_schema="WriterEvidencePacket@1",
            source_artifact_ids=source_artifacts,
            artifact_id=artifact_id,
            idempotency_key=f"writer-evidence-packet:{packet_id}",
        )
        return packet
