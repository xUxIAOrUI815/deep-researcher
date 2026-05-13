from __future__ import annotations

import pytest

from agents.planner import run_planner
from tests.fixtures.offline_research_inputs import MOCK_REPORT_OUTLINE, MOCK_SECTION_GOALS, ROOT_QUERY
from tests.fixtures.session_knowledge_fixtures import StubKnowledgeManager


def _completed_task_tree() -> dict:
    return {
        "root": {
            "id": "root",
            "query": ROOT_QUERY,
            "title": "Root research task",
            "status": "completed",
            "depth": 0,
            "priority": 1.0,
            "parent_id": None,
            "parent_task_id": None,
            "children_ids": [],
            "node_type": "question",
        }
    }


@pytest.mark.asyncio
async def test_planner_prefers_session_snapshot_over_transient_distiller_outputs():
    knowledge_manager = StubKnowledgeManager(
        {
            "knowledge_refs": {
                "fact_ids": ["fact-1", "fact-2", "fact-3"],
                "evidence_ids": ["evidence-1", "evidence-2", "evidence-3"],
            },
            "facts": [{"id": "fact-1"}, {"id": "fact-2"}, {"id": "fact-3"}],
            "claims": [{"id": "claim-1"}, {"id": "claim-2"}],
            "evidence": [{"id": "evidence-1"}, {"id": "evidence-2"}, {"id": "evidence-3"}],
            "conflicts": [],
            "section_evidence_packs": [
                {
                    "pack_id": "pack-session",
                    "section_id": "sec_summary",
                    "coverage_score": 0.75,
                    "claim_ids": ["claim-1"],
                    "fact_ids": ["fact-1"],
                    "evidence_ids": ["evidence-1"],
                    "conflict_ids": [],
                }
            ],
        }
    )

    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_completed_task_tree(),
        active_task_id=None,
        distiller_outputs={
            "coverage_summary": {"avg_section_coverage": 0.0, "sufficiency_level": "insufficient"},
            "fact_ids": [],
            "claim_ids": [],
            "conflict_ids": [],
            "unresolved_gaps": [],
        },
        knowledge_refs={"collection_name": "transient-only", "fact_ids": [], "evidence_ids": []},
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[],
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
    )

    assert knowledge_manager.calls == [("research-1", "session-a")]
    assert result.planner_state.action == "start_writing"
    assert "coverage=0.75" in result.planner_state.convergence_summary
    assert "session_facts=3" in result.planner_state.convergence_summary
    assert "session_evidence=3" in result.planner_state.convergence_summary
    assert "session_packs=1" in result.planner_state.convergence_summary

