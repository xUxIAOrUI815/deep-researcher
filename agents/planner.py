from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.config import ResearchConfig


@dataclass
class PlannerRunResult:
    planner_state: "PlannerState"
    active_task_id: Optional[str] = None
    report_outline: Optional[Dict[str, Any]] = None
    section_goals: Optional[List[Dict[str, Any]]] = None


def _pending_task_ids(task_tree: Dict[str, Any]) -> List[str]:
    return [
        task_id
        for task_id, task in task_tree.items()
        if task.get("status") == "pending"
    ]


def _select_next_task_id(task_tree: Dict[str, Any]) -> Optional[str]:
    pending = _pending_task_ids(task_tree)
    if not pending:
        return None
    return sorted(
        pending,
        key=lambda tid: (
            -float(task_tree.get(tid, {}).get("priority", 0.0)),
            int(task_tree.get(tid, {}).get("depth", 0)),
            task_tree.get(tid, {}).get("created_at", ""),
        ),
    )[0]


def _find_root_task_id(task_tree: Dict[str, Any]) -> Optional[str]:
    for task_id, task in task_tree.items():
        if task.get("parent_id") is None and task.get("parent_task_id") is None:
            return task_id
    return next(iter(task_tree), None)


def _task_children(task_tree: Dict[str, Any], task_id: str) -> List[str]:
    task = task_tree.get(task_id, {})
    children = list(task.get("children_ids", []))
    if children:
        return children
    return [
        child_id
        for child_id, child in task_tree.items()
        if child.get("parent_id") == task_id or child.get("parent_task_id") == task_id
    ]


def _is_initial_decomposition_needed(task_tree: Dict[str, Any]) -> bool:
    root_id = _find_root_task_id(task_tree)
    if not root_id:
        return False
    root = task_tree[root_id]
    return (
        root.get("depth", 0) == 0
        and root.get("status") == "pending"
        and len(_task_children(task_tree, root_id)) == 0
        and len(task_tree) == 1
    )


def _make_default_outline(user_query: str, report_outline: Dict[str, Any]) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if report_outline and report_outline.get("sections"):
        return report_outline, []

    from schemas.state import ReportOutline, ReportSection, SectionGoal

    sections = [
        ReportSection(
            title="Context",
            goal=f"Frame the background and scope for: {user_query}",
            order=1,
        ),
        ReportSection(
            title="Key Findings",
            goal="Synthesize the strongest supported findings and cite evidence.",
            order=2,
        ),
        ReportSection(
            title="Conflicts And Gaps",
            goal="Explain unresolved conflicts, weak evidence, and open questions.",
            order=3,
        ),
        ReportSection(
            title="Conclusion",
            goal="State what the evidence supports and what remains uncertain.",
            order=4,
        ),
    ]
    outline = ReportOutline(
        title=user_query or "Research Report",
        sections=sections,
        rationale="Default outline prepared by planner for evidence-bound writing.",
    )
    goals = [
        SectionGoal(
            section_id=section.section_id,
            goal=section.goal,
            required_claim_types=["evidence_backed"],
            priority=1.0 if section.order <= 2 else 0.8,
        ).model_dump()
        for section in sections
    ]
    return outline.model_dump(), goals


def _build_initial_decomposition(
    user_query: str,
    task_tree: Dict[str, Any],
) -> tuple[List["TaskTreePatch"], List[str]]:
    from schemas.task_tree import TaskNode, TaskTreePatch

    root_id = _find_root_task_id(task_tree)
    if not root_id:
        return [], []

    root_depth = int(task_tree[root_id].get("depth", 0))
    specs = [
        (
            "source_discovery",
            "Map the evidence base",
            "Identify authoritative and recent sources that can support the research question.",
            0.95,
        ),
        (
            "question",
            "Establish current state",
            "Collect facts that directly answer the main research question.",
            0.9,
        ),
        (
            "gap",
            "Check gaps and conflicting claims",
            "Look for missing coverage, disagreements, and claims that need verification.",
            0.82,
        ),
    ]

    patches: List[TaskTreePatch] = [
        TaskTreePatch(
            operation="defer",
            task_id=root_id,
            updates={"status": "deferred", "defer_reason": "Root task decomposed into executable subtasks."},
            rationale="Use root as planning container after initial decomposition.",
            created_by="planner",
        )
    ]
    new_task_ids: List[str] = []

    for node_type, title, rationale, priority in specs:
        task = TaskNode(
            query=user_query,
            title=title,
            rationale=rationale,
            node_type=node_type,
            status="pending",
            depth=root_depth + 1,
            priority=priority,
            parent_id=root_id,
            parent_task_id=root_id,
            created_by="planner",
            updated_by="planner",
        )
        new_task_ids.append(task.id)
        patches.append(
            TaskTreePatch(
                operation="attach",
                parent_task_id=root_id,
                task=task,
                rationale=rationale,
                created_by="planner",
            )
        )

    return patches, new_task_ids


def _build_replanning_patches(
    user_query: str,
    task_tree: Dict[str, Any],
    distiller_outputs: Dict[str, Any],
) -> tuple[List["TaskTreePatch"], List[str]]:
    from schemas.task_tree import TaskNode, TaskTreePatch

    root_id = _find_root_task_id(task_tree)
    if not root_id:
        return [], []

    existing_titles = {str(task.get("title", "")).lower() for task in task_tree.values()}
    patches: List[TaskTreePatch] = []
    new_task_ids: List[str] = []
    coverage_summary = distiller_outputs.get("coverage_summary", {}) or {}

    for gap in distiller_outputs.get("unresolved_gaps", [])[:2]:
        title = f"Resolve gap: {gap}"[:120]
        if title.lower() in existing_titles:
            continue
        task = TaskNode(
            query=user_query,
            title=title,
            rationale=str(gap),
            node_type="gap",
            status="pending",
            depth=1,
            priority=0.78,
            parent_id=root_id,
            parent_task_id=root_id,
            created_by="planner",
            updated_by="planner",
        )
        patches.append(TaskTreePatch(operation="attach", parent_task_id=root_id, task=task, rationale=str(gap), created_by="planner"))
        new_task_ids.append(task.id)

    conflict_ids = distiller_outputs.get("conflict_ids", [])
    if conflict_ids and "verify conflicting evidence" not in existing_titles:
        task = TaskNode(
            query=user_query,
            title="Verify conflicting evidence",
            rationale=f"Distiller reported conflicts: {conflict_ids[:5]}",
            node_type="conflict",
            status="pending",
            depth=1,
            priority=0.88,
            parent_id=root_id,
            parent_task_id=root_id,
            related_evidence_ids=distiller_outputs.get("evidence_ids", []),
            related_fact_ids=distiller_outputs.get("fact_ids", []),
            created_by="planner",
            updated_by="planner",
        )
        patches.append(TaskTreePatch(operation="attach", parent_task_id=root_id, task=task, rationale=task.rationale, created_by="planner"))
        new_task_ids.append(task.id)

    for section_id in coverage_summary.get("uncovered_sections", [])[:1]:
        title = f"Strengthen section coverage: {section_id}"[:120]
        if title.lower() in existing_titles:
            continue
        section_goal = next(
            (
                item.get("goal", "")
                for item in coverage_summary.get("section_status", [])
                if item.get("section_id") == section_id
            ),
            "",
        )
        task = TaskNode(
            query=user_query,
            title=title,
            rationale=f"Coverage summary marked section {section_id} as under-covered. {section_goal}".strip(),
            node_type="section_support",
            status="pending",
            depth=1,
            priority=0.74,
            parent_id=root_id,
            parent_task_id=root_id,
            created_by="planner",
            updated_by="planner",
        )
        patches.append(TaskTreePatch(operation="attach", parent_task_id=root_id, task=task, rationale=task.rationale, created_by="planner"))
        new_task_ids.append(task.id)

    return patches, new_task_ids


def _build_maintenance_patches(task_tree: Dict[str, Any]) -> List["TaskTreePatch"]:
    from schemas.task_tree import TaskTreePatch

    patches: List[TaskTreePatch] = []
    seen_pending: Dict[str, str] = {}

    for task_id, task in task_tree.items():
        if task.get("status") != "pending":
            continue

        if int(task.get("depth", 0)) > ResearchConfig.MAX_DEPTH:
            patches.append(
                TaskTreePatch(
                    operation="prune",
                    task_id=task_id,
                    updates={
                        "status": "pruned",
                        "prune_reason": f"Depth exceeds max depth {ResearchConfig.MAX_DEPTH}.",
                    },
                    rationale="Prune task beyond max depth budget.",
                    created_by="planner",
                )
            )
            continue

        signature = (task.get("title") or task.get("query") or "").strip().lower()
        if not signature:
            continue
        existing_id = seen_pending.get(signature)
        if existing_id:
            patches.append(
                TaskTreePatch(
                    operation="merge",
                    task_id=task_id,
                    target_task_id=existing_id,
                    updates={"merge_into": existing_id},
                    rationale="Merge duplicate pending task.",
                    created_by="planner",
                )
            )
        else:
            seen_pending[signature] = task_id

    return patches


def _apply_patches_to_draft(task_tree: Dict[str, Any], patches: List[Any]) -> Dict[str, Any]:
    draft = {task_id: dict(task) for task_id, task in task_tree.items()}
    for patch in patches:
        patch_data = patch if isinstance(patch, dict) else patch.model_dump()
        operation = patch_data.get("operation")
        task_id = patch_data.get("task_id")
        parent_task_id = patch_data.get("parent_task_id")
        task = patch_data.get("task")
        updates = patch_data.get("updates", {})

        if operation == "attach" and task is not None:
            task_data = task.model_dump() if hasattr(task, "model_dump") else dict(task)
            draft[task_data["id"]] = task_data
            parent_id = parent_task_id or task_data.get("parent_task_id") or task_data.get("parent_id")
            if parent_id and parent_id in draft:
                children = draft[parent_id].setdefault("children_ids", [])
                if task_data["id"] not in children:
                    children.append(task_data["id"])
        elif operation in {"update", "defer", "prune"} and task_id in draft:
            draft[task_id].update(updates)
        elif operation == "merge" and task_id in draft:
            draft[task_id].update(updates)
            draft[task_id]["status"] = "merged"
    return draft


def _coverage_score(
    knowledge_refs: Dict[str, Any],
    section_evidence_packs: List[Dict[str, Any]],
    distiller_outputs: Dict[str, Any],
) -> float:
    coverage_summary = distiller_outputs.get("coverage_summary", {}) or {}
    fact_count = len(knowledge_refs.get("fact_ids", [])) + len(distiller_outputs.get("fact_ids", []))
    evidence_count = len(knowledge_refs.get("evidence_ids", [])) + len(distiller_outputs.get("evidence_ids", []))
    pack_score = 0.0
    if section_evidence_packs:
        pack_score = sum(float(pack.get("coverage_score", 0.0)) for pack in section_evidence_packs) / len(section_evidence_packs)
    count_score = min(1.0, (fact_count * 0.08) + (evidence_count * 0.12))
    summary_score = float(coverage_summary.get("avg_section_coverage", 0.0) or 0.0)
    if coverage_summary.get("sufficiency_level") == "sufficient_for_writing":
        summary_score = max(summary_score, 0.7)
    elif coverage_summary.get("sufficiency_level") == "partial":
        summary_score = max(summary_score, 0.35)
    return max(pack_score, count_score, summary_score)


async def run_planner(
    *,
    user_query: str,
    normalized_query: str,
    task_tree: Dict[str, Any],
    active_task_id: Optional[str],
    distiller_outputs: Dict[str, Any],
    knowledge_refs: Dict[str, Any],
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    section_evidence_packs: List[Dict[str, Any]],
    current_convergence_status: Optional[Dict[str, Any]] = None,
    run_context: Any = None,
) -> PlannerRunResult:
    """Graph-facing planner entry for the task-tree contract.

    The current implementation is rule-based and deliberately replaceable. It
    makes task-level decisions from task_tree plus distiller signals; detailed
    search query construction remains the researcher's job.
    """
    from core.observability import EventType, get_observer
    from schemas.state import PlannerAction, PlannerState

    observer = get_observer()
    query = normalized_query or user_query
    patches = []
    new_task_ids: List[str] = []
    rationale_parts: List[str] = []

    if _is_initial_decomposition_needed(task_tree):
        initial_patches, initial_task_ids = _build_initial_decomposition(query, task_tree)
        patches.extend(initial_patches)
        new_task_ids.extend(initial_task_ids)
        rationale_parts.append("Initial root task decomposed into executable subtasks.")
        if run_context:
            for task_id in initial_task_ids:
                observer.record_task_event(
                    run_context,
                    EventType.TASK_CREATED,
                    task_id,
                    message="Planner proposed initial subtask.",
                )

    replanning_patches, replanned_task_ids = _build_replanning_patches(query, task_tree, distiller_outputs)
    patches.extend(replanning_patches)
    new_task_ids.extend(replanned_task_ids)
    if replanned_task_ids:
        rationale_parts.append(f"Added {len(replanned_task_ids)} follow-up task(s) from distiller signals.")
        if run_context:
            for task_id in replanned_task_ids:
                observer.record_task_event(
                    run_context,
                    EventType.TASK_CREATED,
                    task_id,
                    message="Planner proposed dynamic follow-up task.",
                )

    maintenance_patches = _build_maintenance_patches(task_tree)
    patches.extend(maintenance_patches)
    if maintenance_patches:
        rationale_parts.append(f"Prepared {len(maintenance_patches)} task tree maintenance patch(es).")

    draft_tree = _apply_patches_to_draft(task_tree, patches)
    outline, generated_goals = _make_default_outline(query, report_outline)
    effective_section_goals = section_goals or generated_goals

    coverage_summary = distiller_outputs.get("coverage_summary", {}) or {}
    coverage = _coverage_score(knowledge_refs, section_evidence_packs, distiller_outputs)
    conflict_count = len(distiller_outputs.get("conflict_ids", []))
    gap_count = len(distiller_outputs.get("unresolved_gaps", []))
    novelty_count = len(distiller_outputs.get("fact_ids", [])) + len(distiller_outputs.get("claim_ids", []))
    pending_count = len(_pending_task_ids(draft_tree))
    completed_count = sum(1 for task in draft_tree.values() if task.get("status") == "completed")
    max_depth = max((int(task.get("depth", 0)) for task in draft_tree.values()), default=0)
    depth_penalty = min(0.4, max_depth * 0.08)

    convergence_summary = (
        f"coverage={coverage:.2f}; conflicts={conflict_count}; gaps={gap_count}; "
        f"novelty={novelty_count}; pending={pending_count}; completed={completed_count}; "
        f"depth_penalty={depth_penalty:.2f}"
    )
    if coverage_summary:
        convergence_summary += (
            f"; sufficiency={coverage_summary.get('sufficiency_level', 'unknown')}; "
            f"covered_sections={len(coverage_summary.get('covered_sections', []))}; "
            f"uncovered_sections={len(coverage_summary.get('uncovered_sections', []))}"
        )
    if current_convergence_status:
        convergence_summary += f"; external={current_convergence_status}"

    next_task_id = _select_next_task_id(draft_tree)
    action = PlannerAction.CONTINUE_RESEARCH.value
    stop_reason = None

    if next_task_id:
        rationale_parts.append(f"Selected active task: {next_task_id}.")
    elif coverage >= 0.35 or completed_count > 0 or section_evidence_packs:
        action = PlannerAction.START_WRITING.value
        rationale_parts.append("No pending tasks remain and evidence is sufficient for a draft.")
    elif draft_tree:
        action = PlannerAction.START_WRITING.value
        rationale_parts.append("No pending tasks remain; start writing with available material and explicit caveats.")
    else:
        action = PlannerAction.STOP.value
        stop_reason = "empty_task_tree"
        rationale_parts.append("No task tree is available.")

    writing_constraints = {
        "must_use_evidence_packs": True,
        "do_not_research_in_writer": True,
        "highlight_conflicts": conflict_count > 0,
        "prioritize_sections": [goal.get("section_id") for goal in effective_section_goals[:3]],
    }

    planner_state = PlannerState(
        action=action,
        rationale=" ".join(rationale_parts) or "Planner completed rule-based task-tree pass.",
        convergence_summary=convergence_summary,
        writing_constraints=writing_constraints,
        task_updates=patches,
        new_task_ids=new_task_ids,
        active_task_id=next_task_id,
        next_task_id=next_task_id,
        stop_reason=stop_reason,
    )

    if run_context:
        observer.record_task_event(
            run_context,
            EventType.TASK_UPDATED,
            next_task_id or "",
            message="Planner selected next action.",
            payload={
                "action": action,
                "next_task_id": next_task_id,
                "convergence_summary": convergence_summary,
                "new_task_ids": new_task_ids,
            },
        )

    return PlannerRunResult(
        planner_state=planner_state,
        active_task_id=next_task_id,
        report_outline=outline,
        section_goals=effective_section_goals,
    )
