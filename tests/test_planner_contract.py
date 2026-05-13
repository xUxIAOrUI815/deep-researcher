import pytest

from agents.distiller import run_distiller
from agents.planner import run_planner
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_RESEARCHER_OUTPUTS,
    MOCK_SECTION_GOALS,
    ROOT_QUERY,
)


def _root_only_task_tree() -> dict:
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


def _completed_task_tree() -> dict:
    return {
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
            "query": "AI chip market 2026",
            "title": "Current AI chip market",
            "status": "completed",
            "depth": 1,
            "priority": 0.9,
            "parent_id": "root",
            "parent_task_id": "root",
            "children_ids": [],
            "node_type": "question",
        },
    }


def _budget_ready_task_tree() -> dict:
    tree = {
        "root": {
            "id": "root",
            "query": ROOT_QUERY,
            "title": "Root research task",
            "status": "deferred",
            "depth": 0,
            "priority": 1.0,
            "parent_id": None,
            "parent_task_id": None,
            "children_ids": [],
            "node_type": "question",
        }
    }
    for index in range(6):
        task_id = f"completed-{index}"
        tree[task_id] = {
            "id": task_id,
            "query": ROOT_QUERY,
            "title": f"Completed task {index}",
            "status": "completed",
            "depth": 1,
            "priority": 0.8,
            "parent_id": "root",
            "parent_task_id": "root",
            "children_ids": [],
            "node_type": "question",
        }
        tree["root"]["children_ids"].append(task_id)
    tree["pending-extra"] = {
        "id": "pending-extra",
        "query": ROOT_QUERY,
        "title": "Another follow-up search",
        "status": "pending",
        "depth": 1,
        "priority": 0.9,
        "parent_id": "root",
        "parent_task_id": "root",
        "children_ids": [],
        "node_type": "gap",
    }
    tree["root"]["children_ids"].append("pending-extra")
    return tree


@pytest.mark.asyncio
async def test_planner_initial_decomposition_outputs_task_tree_patches_and_writing_contract():
    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_root_only_task_tree(),
        active_task_id=None,
        distiller_outputs={},
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        report_outline={},
        section_goals=[],
        section_evidence_packs=[],
    )

    planner_state = result.planner_state
    operations = [patch.operation for patch in planner_state.task_updates]

    assert planner_state.action == "continue_research"
    assert planner_state.active_task_id is not None
    assert "defer" in operations
    assert operations.count("attach") >= 3
    assert len(planner_state.new_task_ids) >= 3
    assert result.report_outline
    assert result.report_outline["sections"]
    assert result.section_goals
    assert planner_state.writing_constraints["must_use_evidence_packs"] is True
    assert planner_state.writing_constraints["do_not_research_in_writer"] is True


@pytest.mark.asyncio
async def test_planner_uses_structured_distiller_signals_for_replanning_and_start_writing():
    distiller_outputs = await run_distiller(
        task_id="task-1",
        task={"id": "task-1", "query": "AI chip market 2026"},
        researcher_outputs=MOCK_RESEARCHER_OUTPUTS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
    )

    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_completed_task_tree(),
        active_task_id=None,
        distiller_outputs=distiller_outputs.model_dump(),
        knowledge_refs=distiller_outputs.knowledge_refs,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=distiller_outputs.section_evidence_packs,
    )

    planner_state = result.planner_state
    operations = [patch.operation for patch in planner_state.task_updates]

    assert operations.count("attach") >= 2
    assert any("Resolve gap:" in (patch.task.title if patch.task else "") for patch in planner_state.task_updates if patch.operation == "attach")
    assert any(
        "Verify conflicting evidence" == (patch.task.title if patch.task else "")
        for patch in planner_state.task_updates
        if patch.operation == "attach"
    )
    assert "sufficiency=" in planner_state.convergence_summary
    assert "covered_sections=" in planner_state.convergence_summary
    assert planner_state.action in {"continue_research", "start_writing"}


@pytest.mark.asyncio
async def test_planner_starts_writing_when_standard_depth_budget_is_reached():
    planner_context = {
        "coverage_summary": {
            "avg_section_coverage": 0.62,
            "sufficiency_level": "partial",
            "covered_sections": ["sec-1", "sec-2", "sec-3"],
            "uncovered_sections": ["sec-4"],
        },
        "section_packs": [
            {"section_id": "sec-1", "coverage_score": 0.7},
            {"section_id": "sec-2", "coverage_score": 0.7},
            {"section_id": "sec-3", "coverage_score": 0.7},
            {"section_id": "sec-4", "coverage_score": 0.4},
        ],
        "writing_ready_sections": ["sec-1", "sec-2", "sec-3"],
        "conflict_hotspots": [],
        "unresolved_gaps": [],
        "latest_novelty_snapshot": {"new_fact_count": 0, "new_claim_count": 0},
    }

    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_budget_ready_task_tree(),
        active_task_id=None,
        distiller_outputs={},
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[],
        planner_context=planner_context,
        research_depth="standard",
    )

    assert result.planner_state.action == "start_writing"
    assert result.planner_state.active_task_id is None
    assert result.planner_state.next_task_id is None
    assert "Research budget for standard depth reached" in result.planner_state.rationale


@pytest.mark.asyncio
async def test_planner_filters_low_quality_gap_hints():
    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree=_completed_task_tree(),
        active_task_id=None,
        distiller_outputs={
            "unresolved_gaps": [
                "Coverage gap around hinted topic: verification reasoning using conflicting crave",
                "Coverage gap around hinted topic: without alan 无需检索的事实核查 what into",
                "Low evidence coverage for section sec-context: Frame the background and scope.",
            ],
            "coverage_summary": {},
            "conflict_ids": [],
        },
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=[],
    )

    attached_titles = [
        patch.task.title
        for patch in result.planner_state.task_updates
        if patch.operation == "attach" and patch.task
    ]

    assert attached_titles == [
        "Resolve gap: Low evidence coverage for section sec-context: Frame the background and scope."
    ]


@pytest.mark.asyncio
async def test_planner_returns_stop_for_empty_task_tree():
    result = await run_planner(
        user_query=ROOT_QUERY,
        normalized_query=ROOT_QUERY,
        task_tree={},
        active_task_id=None,
        distiller_outputs={},
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        report_outline={},
        section_goals=[],
        section_evidence_packs=[],
    )

    assert result.planner_state.action == "stop"
    assert result.planner_state.stop_reason == "empty_task_tree"
