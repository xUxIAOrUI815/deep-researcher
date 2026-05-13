from __future__ import annotations

from core.session_knowledge import KnowledgeManager
from core.session_retrieval import SessionRetrievalService
from schemas.retrieval import SessionRetrievalQuery


def _payload(
    *,
    fact_id: str,
    claim_id: str,
    evidence_id: str,
    pack_id: str,
    section_id: str,
    section_title: str,
    fact_text: str,
    claim_text: str,
    evidence_summary: str,
    source_id: str,
    source_url: str,
    authority_score: float,
    task_id: str,
    gap_text: str = "",
    coverage_score: float = 0.5,
) -> dict:
    unresolved_gaps = [gap_text] if gap_text else []
    return {
        "summary": "session retrieval fixture",
        "sources": [
            {
                "source_id": source_id,
                "url": source_url,
                "title": f"Source {source_id}",
                "authority_score": authority_score,
                "freshness_score": 0.4,
                "task_id": task_id,
            }
        ],
        "atomic_facts": [
            {
                "id": fact_id,
                "text": fact_text,
                "source_url": source_url,
                "confidence": 0.82,
                "task_id": task_id,
                "snippet": fact_text,
                "source_level": "A",
                "verified_count": 1,
            }
        ],
        "claims": [
            {
                "id": claim_id,
                "text": claim_text,
                "fact_ids": [fact_id],
                "evidence_ids": [evidence_id],
                "confidence": 0.78,
                "task_id": task_id,
            }
        ],
        "evidence": [
            {
                "id": evidence_id,
                "source_id": source_id,
                "source_url": source_url,
                "quote": evidence_summary,
                "summary": evidence_summary,
                "fact_ids": [fact_id],
                "claim_ids": [claim_id],
                "quality_score": 0.74,
                "task_id": task_id,
            }
        ],
        "conflicts": [],
        "section_evidence_packs": [
            {
                "pack_id": pack_id,
                "section_id": section_id,
                "section_title": section_title,
                "goal": f"Support {section_title}",
                "claim_ids": [claim_id],
                "fact_ids": [fact_id],
                "evidence_ids": [evidence_id],
                "conflict_ids": [],
                "coverage_score": coverage_score,
                "notes": f"Notes for {section_title}",
            }
        ],
        "coverage_summary": {
            "avg_section_coverage": coverage_score,
            "sufficiency_level": "partial" if coverage_score < 0.7 else "sufficient_for_writing",
            "covered_sections": [section_id] if coverage_score >= 0.7 else [],
            "uncovered_sections": [] if coverage_score >= 0.7 else [section_id],
            "section_status": [{"section_id": section_id, "goal": f"Support {section_title}"}],
        },
        "unresolved_gaps": unresolved_gaps,
    }


def test_retrieval_is_scoped_to_session_research_and_cumulative_across_rounds():
    km = KnowledgeManager(sqlite_filename=":memory:")
    service = SessionRetrievalService(km)

    km.process_distiller_output(
        _payload(
            fact_id="fact-r1-a",
            claim_id="claim-r1-a",
            evidence_id="evidence-r1-a",
            pack_id="pack-r1-summary",
            section_id="sec_summary",
            section_title="Summary",
            fact_text="Alpha launched the first platform.",
            claim_text="Alpha currently leads launch timing.",
            evidence_summary="Official source confirms launch timing.",
            source_id="source-r1-a",
            source_url="https://alpha.example.com/launch",
            authority_score=0.95,
            task_id="task-summary",
            gap_text="Need supply chain confirmation",
            coverage_score=0.62,
        ),
        research_id="research-1",
        session_id="session-a",
        task_id="task-summary",
    )
    km.process_distiller_output(
        _payload(
            fact_id="fact-r1-b",
            claim_id="claim-r1-b",
            evidence_id="evidence-r1-b",
            pack_id="pack-r1-summary-new",
            section_id="sec_summary",
            section_title="Summary",
            fact_text="Alpha also expanded manufacturing.",
            claim_text="Manufacturing expansion strengthens the lead.",
            evidence_summary="Manufacturing source confirms expansion.",
            source_id="source-r1-b",
            source_url="https://beta.example.com/factory",
            authority_score=0.65,
            task_id="task-summary",
            coverage_score=0.81,
        ),
        research_id="research-1",
        session_id="session-a",
        task_id="task-summary",
    )
    km.process_distiller_output(
        _payload(
            fact_id="fact-r2-a",
            claim_id="claim-r2-a",
            evidence_id="evidence-r2-a",
            pack_id="pack-r2-other",
            section_id="sec_other",
            section_title="Other",
            fact_text="Another company unrelated to research-1.",
            claim_text="Other research line.",
            evidence_summary="Separate source.",
            source_id="source-r2-a",
            source_url="https://other.example.com/doc",
            authority_score=0.5,
            task_id="task-other",
            coverage_score=0.45,
        ),
        research_id="research-2",
        session_id="session-b",
        task_id="task-other",
    )

    result = service.retrieve(
        research_id="research-1",
        session_id="session-a",
        query=SessionRetrievalQuery(section_id="sec_summary", limit_per_type=10),
    )

    assert [pack["section_id"] for pack in result.section_packs] == ["sec_summary"]
    assert {row["id"] for row in result.facts} == {"fact-r1-a", "fact-r1-b"}
    assert {row["id"] for row in result.claims} == {"claim-r1-a", "claim-r1-b"}
    assert {row["id"] for row in result.evidence} == {"evidence-r1-a", "evidence-r1-b"}
    assert all("r2" not in row["id"] for row in result.facts)
    assert result.section_packs[0]["coverage_score"] == 0.81


def test_retrieval_filters_and_sorts_sources_and_supports_claim_lookup():
    km = KnowledgeManager(sqlite_filename=":memory:")
    service = SessionRetrievalService(km)
    km.process_distiller_output(
        _payload(
            fact_id="fact-1",
            claim_id="claim-1",
            evidence_id="evidence-1",
            pack_id="pack-1",
            section_id="sec_market",
            section_title="Market",
            fact_text="Battery shipments grew quickly.",
            claim_text="Battery growth is accelerating.",
            evidence_summary="Primary filing on battery shipments.",
            source_id="source-primary",
            source_url="https://filing.example.com/battery",
            authority_score=0.95,
            task_id="task-market",
            coverage_score=0.72,
        ),
        research_id="research-1",
        session_id="session-a",
        task_id="task-market",
    )
    km.process_distiller_output(
        _payload(
            fact_id="fact-2",
            claim_id="claim-2",
            evidence_id="evidence-2",
            pack_id="pack-2",
            section_id="sec_market",
            section_title="Market",
            fact_text="Commentary discussed battery demand.",
            claim_text="Battery demand remains strong.",
            evidence_summary="Secondary commentary on battery demand.",
            source_id="source-secondary",
            source_url="https://blog.example.com/battery",
            authority_score=0.35,
            task_id="task-market",
            coverage_score=0.55,
        ),
        research_id="research-1",
        session_id="session-a",
        task_id="task-market",
    )

    by_claim = service.retrieve(
        research_id="research-1",
        session_id="session-a",
        query=SessionRetrievalQuery(claim_id="claim-1", limit_per_type=10),
    )
    ranked_sources = service.retrieve(
        research_id="research-1",
        session_id="session-a",
        query=SessionRetrievalQuery(
            section_id="sec_market",
            sort_by="authority",
            limit_per_type=10,
            semantic_query="battery shipment growth",
            semantic_weight=0.25,
        ),
    )

    assert [row["id"] for row in by_claim.claims] == ["claim-1"]
    assert {row["id"] for row in by_claim.facts} == {"fact-1"}
    assert {row["id"] for row in by_claim.evidence} == {"evidence-1"}
    assert ranked_sources.sources[0]["source_id"] == "source-primary"
    assert ranked_sources.retrieval_meta["semantic_enabled"] is True
