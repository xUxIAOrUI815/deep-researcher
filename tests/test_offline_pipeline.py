import pytest

from agents.distiller import run_distiller
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.writer import run_writer
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_TASK,
    ROOT_QUERY,
)


def _task_tree_with_root() -> dict:
    return {
        "root": {
            "id": "root",
            "query": ROOT_QUERY,
            "title": "Root research task",
            "status": "pending",
            "depth": 0,
            "priority": 1.0,
            "parent_id": None,
            "parent_task_id": None,
            "children_ids": [],
            "node_type": "question",
        }
    }


@pytest.mark.asyncio
async def test_offline_agent_pipeline_runs_without_network_dependencies():
    first_planner = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_task_tree_with_root(),
        active_task_id=None,
        distiller_outputs={},
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        report_outline={},
        section_goals=[],
        section_evidence_packs=[],
    )

    assert first_planner.planner_state.action in {"continue_research", "start_writing", "stop"}
    assert first_planner.active_task_id
    assert first_planner.report_outline
    assert first_planner.section_goals

    researcher_outputs = await run_researcher(
        task_id=first_planner.active_task_id,
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )

    assert len(researcher_outputs.sources) > 0
    assert len(researcher_outputs.passages) > 0

    distiller_outputs = await run_distiller(
        task_id=first_planner.active_task_id,
        task=MOCK_TASK,
        researcher_outputs=researcher_outputs.model_dump(),
        report_outline=first_planner.report_outline,
        section_goals=first_planner.section_goals,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
    )

    assert len(distiller_outputs.section_evidence_packs) > 0

    task_tree_after_first_loop = {
        "root": {
            "id": "root",
            "query": ROOT_QUERY,
            "title": "Root research task",
            "status": "deferred",
            "depth": 0,
            "priority": 1.0,
            "parent_id": None,
            "parent_task_id": None,
            "children_ids": ["task-1"],
            "node_type": "question",
        },
        "task-1": {
            "id": "task-1",
            "query": MOCK_TASK["query"],
            "title": MOCK_TASK["title"],
            "status": "completed",
            "depth": 1,
            "priority": MOCK_TASK["priority"],
            "parent_id": "root",
            "parent_task_id": "root",
            "children_ids": [],
            "node_type": MOCK_TASK["node_type"],
        },
    }

    second_planner = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=task_tree_after_first_loop,
        active_task_id=None,
        distiller_outputs=distiller_outputs.model_dump(),
        knowledge_refs=distiller_outputs.knowledge_refs,
        report_outline=first_planner.report_outline,
        section_goals=first_planner.section_goals,
        section_evidence_packs=distiller_outputs.section_evidence_packs,
    )

    assert second_planner.planner_state.action in {"continue_research", "start_writing", "stop"}

    report = await run_writer(
        report_outline=first_planner.report_outline,
        section_goals=first_planner.section_goals,
        section_evidence_packs=distiller_outputs.section_evidence_packs,
    )

    assert len(report.markdown) > 800
