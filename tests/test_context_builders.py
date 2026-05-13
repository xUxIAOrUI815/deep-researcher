from __future__ import annotations

from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
from core.session_retrieval import SessionRetrievalService
from tests.fixtures.offline_research_inputs import MOCK_REPORT_OUTLINE, MOCK_SECTION_GOALS, ROOT_QUERY
from tests.fixtures.session_knowledge_fixtures import StubKnowledgeManager


def _snapshot() -> dict:
    return {
        "knowledge_refs": {
            "fact_ids": ["fact-1", "fact-2"],
            "claim_ids": ["claim-1"],
            "evidence_ids": ["evidence-1", "evidence-2"],
            "source_ids": ["source-1", "source-2"],
        },
        "sources": [
            {
                "source_id": "source-1",
                "url": "https://primary.example.com/report",
                "authority_score": 0.95,
                "freshness_score": 0.4,
                "task_id": "task-gap",
            },
            {
                "source_id": "source-2",
                "url": "https://secondary.example.com/blog",
                "authority_score": 0.3,
                "freshness_score": 0.2,
                "task_id": "task-gap",
            },
        ],
        "source_registry": {
            "source-1": {"source_id": "source-1", "url": "https://primary.example.com/report"},
            "source-2": {"source_id": "source-2", "url": "https://secondary.example.com/blog"},
        },
        "claims": [
            {
                "id": "claim-1",
                "text": "Capacity remains constrained.",
                "task_id": "task-gap",
                "section_id": "sec_summary",
                "confidence": 0.8,
                "evidence_ids": ["evidence-1"],
            }
        ],
        "facts": [
            {"id": "fact-1", "text": "Capacity is constrained.", "task_id": "task-gap", "section_id": "sec_summary", "confidence": 0.82},
            {"id": "fact-2", "text": "A supplier ramp is pending.", "task_id": "task-gap", "section_id": "sec_summary", "confidence": 0.72},
        ],
        "evidence": [
            {"id": "evidence-1", "summary": "Primary source on capacity.", "source_id": "source-1", "task_id": "task-gap", "section_id": "sec_summary", "confidence": 0.8},
            {"id": "evidence-2", "summary": "Secondary source on supplier ramp.", "source_id": "source-2", "task_id": "task-gap", "section_id": "sec_summary", "confidence": 0.55},
        ],
        "conflicts": [
            {"id": "conflict-1", "description": "Suppliers disagree on timing.", "severity": "high", "task_id": "task-gap", "section_id": "sec_summary"}
        ],
        "section_evidence_packs": [
            {
                "pack_id": "pack-session",
                "section_id": "sec_summary",
                "section_title": "Summary",
                "goal": "Explain the current supply picture",
                "claim_ids": ["claim-1"],
                "fact_ids": ["fact-1", "fact-2"],
                "evidence_ids": ["evidence-1", "evidence-2"],
                "conflict_ids": ["conflict-1"],
                "coverage_score": 0.76,
                "notes": "Session-backed writer note.",
            }
        ],
        "latest_coverage_snapshot": {
            "avg_section_coverage": 0.76,
            "sufficiency_level": "partial",
            "completed_section_count": 0,
            "partial_section_count": 1,
            "uncovered_section_count": 0,
        },
        "latest_novelty_snapshot": {
            "new_fact_count": 1,
            "new_claim_count": 1,
        },
        "open_gaps": [
            {
                "gap_id": "gap-1",
                "gap_text": "Need one primary supplier confirmation",
                "section_id": "sec_summary",
                "task_id": "task-gap",
                "severity": "high",
            }
        ],
    }


def test_planner_context_builder_exposes_coverage_gaps_conflicts_and_maturity():
    builder = PlannerContextBuilder(SessionRetrievalService(StubKnowledgeManager(_snapshot())))
    context = builder.build(
        research_id="research-1",
        session_id="session-a",
        user_query=ROOT_QUERY,
        task_tree={"task-gap": {"id": "task-gap", "title": "Close supplier gap", "query": ROOT_QUERY}},
        active_task_id="task-gap",
    )

    assert context.coverage_summary["sufficiency_level"] == "partial"
    assert context.unresolved_gaps[0]["gap_text"] == "Need one primary supplier confirmation"
    assert context.conflict_hotspots[0]["id"] == "conflict-1"
    assert context.section_readiness[0]["maturity"] == "developing"
    assert context.writing_ready_sections == ["sec_summary"]


def test_researcher_context_builder_exposes_seen_sources_and_gap_focus():
    builder = ResearcherContextBuilder(SessionRetrievalService(StubKnowledgeManager(_snapshot())))
    context = builder.build(
        research_id="research-1",
        session_id="session-a",
        root_user_query=ROOT_QUERY,
        task_id="task-gap",
        task={"id": "task-gap", "title": "Close supplier gap", "query": ROOT_QUERY},
    )

    assert context.already_seen_source_ids == ["source-1", "source-2"]
    assert context.unresolved_gaps[0]["gap_id"] == "gap-1"
    assert context.focus_sections == ["sec_summary"]
    assert context.search_dedup_hints["source_ids"] == ["source-1", "source-2"]


def test_writer_context_builder_prefers_session_packs_over_fallback():
    builder = WriterContextBuilder(SessionRetrievalService(StubKnowledgeManager(_snapshot())))
    context = builder.build(
        research_id="research-1",
        session_id="session-a",
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        fallback_section_packs=[
            {
                "pack_id": "pack-fallback",
                "section_id": "sec_summary",
                "claim_ids": ["claim-fallback"],
                "fact_ids": ["fact-fallback"],
                "evidence_ids": [],
                "conflict_ids": [],
                "coverage_score": 0.2,
                "notes": "Fallback note.",
            }
        ],
    )

    assert context.context_source == "session"
    assert context.section_evidence_packs[0]["pack_id"] == "pack-session"
    assert context.section_contexts[0]["pack"]["pack_id"] == "pack-session"
