from __future__ import annotations

from typing import Any

from deep_researcher.contracts import Conflict, Evidence, Source

from .repository import KnowledgeRepository


class LegacySessionProjectionAdapter:
    """Rebuildable compatibility read model backed only by domain repositories."""

    def __init__(self, repository: KnowledgeRepository) -> None:
        self.repository = repository

    @staticmethod
    def _dump(values: tuple[Any, ...]) -> list[dict[str, Any]]:
        return [value.model_dump(mode="json") for value in values]

    def build(self, run_id: str) -> dict[str, Any]:
        sources = self.repository.sources.list(run_id)
        claims = self.repository.claims.list(run_id)
        facts = self.repository.facts.list(run_id)
        evidence = self.repository.evidence.list(run_id)
        conflicts = self.repository.conflicts.list(run_id)
        sections = self.repository.sections.list(run_id)
        reports = self.repository.reports.list(run_id)
        return {
            "session": {"research_id": run_id, "status": "projected"},
            "knowledge_refs": {
                "source_ids": [item.source_id for item in sources],
                "claim_ids": [item.claim_id for item in claims],
                "fact_ids": [item.fact_id for item in facts],
                "evidence_ids": [item.evidence_id for item in evidence],
                "conflict_ids": [item.conflict_id for item in conflicts],
            },
            "sources": self._dump(sources),
            "claims": self._dump(claims),
            "facts": self._dump(facts),
            "evidence": self._dump(evidence),
            "conflicts": self._dump(conflicts),
            "section_evidence_packs": self._dump(sections),
            "reports": self._dump(reports),
            "latest_coverage_snapshot": None,
            "open_gaps": [],
            "latest_novelty_snapshot": None,
            "stats": {
                "total_sources": len(sources),
                "accessible_sources": sum(isinstance(item, Source) and item.status.value == "accessible" for item in sources),
                "total_claims": len(claims),
                "total_facts": len(facts),
                "total_evidence": len(evidence),
                "verified_evidence": sum(isinstance(item, Evidence) and item.status.value == "verified" for item in evidence),
                "open_conflicts": sum(isinstance(item, Conflict) and item.status.value != "resolved" for item in conflicts),
            },
        }
