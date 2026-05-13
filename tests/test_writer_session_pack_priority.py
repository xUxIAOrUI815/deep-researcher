from __future__ import annotations

import pytest

from agents.writer import run_writer
from tests.fixtures.offline_research_inputs import MOCK_REPORT_OUTLINE, MOCK_SECTION_GOALS
from tests.fixtures.session_knowledge_fixtures import StubKnowledgeManager


@pytest.mark.asyncio
async def test_writer_prefers_session_packs_over_state_fallback_packs():
    session_pack = {
        "pack_id": "pack-session",
        "section_id": "sec_summary",
        "claim_ids": ["claim-session-1"],
        "fact_ids": ["fact-session-1"],
        "evidence_ids": ["evidence-session-1", "evidence-session-2"],
        "conflict_ids": [],
        "coverage_score": 0.82,
        "notes": "Session-backed note.",
    }
    fallback_pack = {
        "pack_id": "pack-fallback",
        "section_id": "sec_summary",
        "claim_ids": ["claim-fallback-1"],
        "fact_ids": ["fact-fallback-1"],
        "evidence_ids": [],
        "conflict_ids": [],
        "coverage_score": 0.2,
        "notes": "Fallback-only note.",
    }
    knowledge_manager = StubKnowledgeManager({"section_evidence_packs": [session_pack]})

    report = await run_writer(
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[fallback_pack],
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
    )

    assert knowledge_manager.calls
    assert set(knowledge_manager.calls) == {("research-1", "session-a")}
    assert report.evidence_pack_ids == ["pack-session"]
    assert "Session-backed note." in report.markdown
    assert "Fallback-only note." not in report.markdown
    assert report.citation_map["sec_summary"] == [
        "claim-session-1",
        "fact-session-1",
        "evidence-session-1",
        "evidence-session-2",
    ]


@pytest.mark.asyncio
async def test_writer_falls_back_to_state_packs_only_when_session_packs_absent():
    fallback_pack = {
        "pack_id": "pack-fallback",
        "section_id": "sec_summary",
        "claim_ids": ["claim-fallback-1"],
        "fact_ids": ["fact-fallback-1"],
        "evidence_ids": ["evidence-fallback-1"],
        "conflict_ids": [],
        "coverage_score": 0.68,
        "notes": "Fallback-only note.",
    }
    knowledge_manager = StubKnowledgeManager({})

    report = await run_writer(
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[fallback_pack],
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
    )

    assert knowledge_manager.calls
    assert set(knowledge_manager.calls) == {("research-1", "session-a")}
    assert report.evidence_pack_ids == ["pack-fallback"]
    assert "Fallback-only note." in report.markdown
    assert report.citation_map["sec_summary"] == [
        "claim-fallback-1",
        "fact-fallback-1",
        "evidence-fallback-1",
    ]


@pytest.mark.asyncio
async def test_writer_recovers_section_context_when_session_pack_is_empty():
    empty_session_pack = {
        "pack_id": "pack-empty",
        "section_id": "sec_summary",
        "claim_ids": [],
        "fact_ids": [],
        "evidence_ids": [],
        "conflict_ids": [],
        "coverage_score": 0.0,
        "notes": "",
    }
    knowledge_manager = StubKnowledgeManager(
        {
            "section_evidence_packs": [empty_session_pack],
            "claims": [{"id": "claim-context-1", "section_id": "sec_summary", "text": "Context-backed claim."}],
            "facts": [{"id": "fact-context-1", "section_id": "sec_summary", "text": "Context-backed fact."}],
            "evidence": [
                {
                    "id": "evidence-context-1",
                    "section_id": "sec_summary",
                    "summary": "Context-backed evidence.",
                }
            ],
            "conflicts": [],
        }
    )

    report = await run_writer(
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[],
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
    )

    assert report.evidence_pack_ids[0] == "pack-empty"
    assert report.citation_map["sec_summary"] == [
        "claim-context-1",
        "fact-context-1",
        "evidence-context-1",
    ]
    assert "Context-backed claim." in report.markdown
