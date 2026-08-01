from __future__ import annotations

from deep_researcher.contracts import (
    Command,
    CommandKind,
    Observation,
    ObservationStatus,
    SectionStatus,
)
from deep_researcher.gateway import ProtocolToolGateway
from deep_researcher.knowledge import KnowledgeIngestionService, KnowledgeRepository


class KnowledgeIngestingCommandBoundary:
    """Crosses Tool -> candidate knowledge exactly once after governed success."""

    def __init__(
        self,
        *,
        gateway: ProtocolToolGateway,
        ingestion: KnowledgeIngestionService,
        repository: KnowledgeRepository,
    ) -> None:
        self.gateway = gateway
        self.ingestion = ingestion
        self.repository = repository

    async def execute(self, command: Command) -> Observation:
        observation = await self.gateway.execute(command)
        if observation.status != ObservationStatus.SUCCEEDED:
            return observation
        data = dict(observation.normalized_data)
        artifact_ids = list(observation.output_artifact_ids)
        entity_ids: list[str] = []
        issues: list[str] = []
        candidate_entity_ids: tuple[str, ...] = ()

        if any(
            data.get(name)
            for name in ("sources", "passages", "scraped_data_cache")
        ):
            research = self.ingestion.ingest_research_observation(
                data,
                run_id=command.run_id,
                task_id=command.task_id,
            )
            artifact_ids.extend(research.artifact_ids)
            entity_ids.extend(research.entity_ids)
            issues.extend(research.issues)

        if command.kind in {
            CommandKind.EXTRACT,
            CommandKind.COMPARE,
            CommandKind.VERIFY_SOURCE,
        } and any(
            data.get(name)
            for name in (
                "evidence",
                "atomic_facts",
                "claims",
                "conflicts",
                "section_evidence_packs",
            )
        ):
            candidate = self.ingestion.ingest_candidate_knowledge(
                data,
                run_id=command.run_id,
                task_id=command.task_id,
            )
            artifact_ids.extend(candidate.artifact_ids)
            entity_ids.extend(candidate.entity_ids)
            issues.extend(candidate.issues)
            candidate_entity_ids = candidate.entity_ids
            self._attach_claims_to_sections(
                run_id=command.run_id,
                preferred_section_id=str(data.get("section_id") or "") or None,
                entity_ids=candidate.entity_ids,
            )

        persisted_counts = {
            prefix.removesuffix("_"): sum(
                item.startswith(prefix) for item in candidate_entity_ids
            )
            for prefix in (
                "evidence_",
                "fact_",
                "claim_",
                "citation_",
                "conflict_",
            )
        }
        strict_extract_complete = bool(
            command.kind != CommandKind.EXTRACT
            or (
                not issues
                and persisted_counts["evidence"] > 0
                and persisted_counts["fact"] > 0
                and persisted_counts["claim"] > 0
                and persisted_counts["citation"]
                >= persisted_counts["claim"]
            )
        )
        if command.kind == CommandKind.EXTRACT:
            data["semantic_complete"] = bool(
                data.get("semantic_complete", False)
                and strict_extract_complete
            )
        data["ingestion"] = {
            "artifact_ids": list(dict.fromkeys(artifact_ids)),
            "entity_ids": list(dict.fromkeys(entity_ids)),
            "issues": issues,
            "persisted_counts": persisted_counts,
            "strict_extract_complete": strict_extract_complete,
        }
        if data.get("atomic_facts") and "facts" not in data:
            data["facts"] = data["atomic_facts"]
        return observation.model_copy(
            update={
                "normalized_data": data,
                "output_artifact_ids": tuple(dict.fromkeys(artifact_ids)),
            }
        )

    def _attach_claims_to_sections(
        self,
        *,
        run_id: str,
        preferred_section_id: str | None,
        entity_ids: tuple[str, ...],
    ) -> None:
        claim_ids = tuple(
            item
            for item in entity_ids
            if item.startswith("claim_")
            and self.repository.claims.get(item) is not None
        )
        if not claim_ids:
            return
        sections = list(self.repository.sections.list(run_id))
        if not sections:
            raise RuntimeError(
                "candidate claims cannot be attached without a report scaffold"
            )
        selected = next(
            (
                item
                for item in sections
                if preferred_section_id
                and item.section_id == preferred_section_id
            ),
            sections[0],
        )
        citation_ids = tuple(
            item.citation_id
            for item in self.repository.citations.list(run_id)
            if item.claim_id in claim_ids
        )
        updated = selected.model_copy(
            update={
                "claim_ids": tuple(
                    dict.fromkeys((*selected.claim_ids, *claim_ids))
                ),
                "required_claim_ids": tuple(
                    dict.fromkeys((*selected.required_claim_ids, *claim_ids))
                ),
                "citation_ids": tuple(
                    dict.fromkeys((*selected.citation_ids, *citation_ids))
                ),
                "status": (
                    selected.status
                    if selected.status != SectionStatus.PLANNED
                    else SectionStatus.DRAFTING
                ),
            }
        )
        self.repository.sections.save(updated)
