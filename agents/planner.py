import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from core.config import ResearchConfig
from core.context_builders import PlannerContextBuilder
from core.session_retrieval import SessionRetrievalService
from schemas.retrieval import PlannerContext
from schemas.state import PlannerState
from schemas.task_tree import TaskTreePatch


@dataclass
class PlannerRunResult:
    planner_state: "PlannerState"
    active_task_id: Optional[str] = None
    report_outline: Optional[Dict[str, Any]] = None
    section_goals: Optional[List[Dict[str, Any]]] = None


def _pending_task_ids(task_tree: Dict[str, Any]) -> List[str]:
    """返回当前任务树中状态仍为 pending 的任务 ID 列表。"""
    return [
        task_id
        for task_id, task in task_tree.items()
        if task.get("status") == "pending"
    ]


def _select_next_task_id(task_tree: Dict[str, Any]) -> Optional[str]:
    """按优先级、层级深度和创建顺序选择下一个待执行任务。"""
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
    """查找根任务 ID；若结构不完整，则退化为返回第一个任务。"""
    for task_id, task in task_tree.items():
        if task.get("parent_id") is None and task.get("parent_task_id") is None:
            return task_id
    return next(iter(task_tree), None)


def _task_children(task_tree: Dict[str, Any], task_id: str) -> List[str]:
    """获取指定任务的子任务 ID，优先使用 children_ids，否则扫描父子关系。"""
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
    """判断当前任务树是否只包含一个尚未拆解的根任务。"""
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
    """在缺少大纲时生成默认报告结构和对应的小节目标。"""
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
    """将根任务拆解为首批可执行的研究子任务。"""
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
    """根据缺口、冲突和覆盖不足情况生成后续补充任务。"""
    from schemas.task_tree import TaskNode, TaskTreePatch

    root_id = _find_root_task_id(task_tree)
    if not root_id:
        return [], []

    existing_titles = {str(task.get("title", "")).lower() for task in task_tree.values()}
    patches: List[TaskTreePatch] = []
    new_task_ids: List[str] = []
    coverage_summary = distiller_outputs.get("coverage_summary", {}) or {}

    added_gap_tasks = 0
    for gap in distiller_outputs.get("unresolved_gaps", []):
        if added_gap_tasks >= 2:
            break
        if not _is_actionable_gap(gap, user_query):
            continue
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
        added_gap_tasks += 1

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


def _tokenize_planning_text(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]+", (text or "").lower())


def _is_actionable_gap(gap: Any, user_query: str) -> bool:
    text = str(gap or "").strip()
    if len(text) < 12:
        return False
    lower = text.lower()
    if "low evidence coverage for section" in lower:
        return True
    noisy_markers = (
        "seek high_authority_primary_source",
        "without alan",
        "verification reasoning using conflicting crave",
        "what into",
    )
    if any(marker in lower for marker in noisy_markers):
        return False

    hint = lower
    marker = "coverage gap around hinted topic:"
    if marker in hint:
        hint = hint.split(marker, 1)[1].strip()
    tokens = _tokenize_planning_text(hint)
    if len(tokens) < 2:
        return False

    query_tokens = set(_tokenize_planning_text(user_query))
    overlap_count = len(set(tokens) & query_tokens)
    if overlap_count == 0:
        return False
    if len(tokens) > 8 and (overlap_count / max(1, len(set(tokens)))) < 0.2:
        return False
    return True


def _writing_budget_reached(
    *,
    research_depth: str,
    completed_count: int,
    ready_section_count: int,
    pack_count: int,
    coverage: float,
) -> bool:
    thresholds = {
        "quick": {"completed": 3, "ready": 2, "packs": 2, "coverage": 0.45},
        "standard": {"completed": 6, "ready": 3, "packs": 4, "coverage": 0.55},
        "deep": {"completed": 10, "ready": 4, "packs": 4, "coverage": 0.75},
    }
    threshold = thresholds.get((research_depth or "standard").lower(), thresholds["standard"])
    return (
        completed_count >= threshold["completed"]
        and ready_section_count >= threshold["ready"]
        and pack_count >= threshold["packs"]
        and coverage >= threshold["coverage"]
    )


def _build_maintenance_patches(task_tree: Dict[str, Any]) -> List["TaskTreePatch"]:
    """为过深任务和重复待办任务生成维护性修补操作。"""
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
    """将补丁应用到内存中的任务树草稿，供后续决策使用。"""
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
    session_snapshot: Optional[Dict[str, Any]] = None,
    planner_context: Optional[Dict[str, Any]] = None,
) -> float:
    """基于证据包、事实数量和覆盖摘要估算当前研究覆盖度。"""
    coverage_summary = dict(distiller_outputs.get("coverage_summary", {}) or {})
    if planner_context:
        coverage_summary = dict(planner_context.get("coverage_summary", {}) or coverage_summary)
        effective_refs = dict(session_snapshot.get("knowledge_refs", {}) or {}) if session_snapshot else dict(knowledge_refs or {})
        fact_count = len(session_snapshot.get("facts", [])) if session_snapshot else len(effective_refs.get("fact_ids", []))
        evidence_count = len(session_snapshot.get("evidence", [])) if session_snapshot else len(effective_refs.get("evidence_ids", []))
        effective_packs = list(planner_context.get("section_packs", []))
    elif session_snapshot:
        effective_refs = dict(session_snapshot.get("knowledge_refs", {}) or {})
        fact_count = len(effective_refs.get("fact_ids", []))
        evidence_count = len(effective_refs.get("evidence_ids", []))
        effective_packs = list(session_snapshot.get("section_evidence_packs", []))
    else:
        effective_refs = dict(knowledge_refs or {})
        fact_count = len(effective_refs.get("fact_ids", [])) + len(distiller_outputs.get("fact_ids", []))
        evidence_count = len(effective_refs.get("evidence_ids", [])) + len(distiller_outputs.get("evidence_ids", []))
        effective_packs = list(section_evidence_packs or [])
    pack_score = 0.0
    if effective_packs:
        pack_score = sum(float(pack.get("coverage_score", 0.0)) for pack in effective_packs) / len(effective_packs)
    count_score = min(1.0, (fact_count * 0.08) + (evidence_count * 0.12))
    summary_score = float(coverage_summary.get("avg_section_coverage", 0.0) or 0.0)
    if coverage_summary.get("sufficiency_level") == "sufficient_for_writing":
        summary_score = max(summary_score, 0.7)
    elif coverage_summary.get("sufficiency_level") == "partial":
        summary_score = max(summary_score, 0.35)
    return max(pack_score, count_score, summary_score)


def _planner_context_from_snapshot(
    *,
    knowledge_manager: Any,
    research_id: str,
    session_id: str,
    user_query: str,
    task_tree: Dict[str, Any],
    active_task_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    """从持久化会话快照构建 planner 上下文，失败时返回 None。"""
    try:
        builder = PlannerContextBuilder(SessionRetrievalService(knowledge_manager))
        return builder.build(
            research_id=research_id,
            session_id=session_id,
            user_query=user_query,
            task_tree=task_tree,
            active_task_id=active_task_id,
        ).model_dump()
    except Exception:
        return None


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
    knowledge_manager: Any = None,
    research_id: Optional[str] = None,
    session_id: Optional[str] = None,
    planner_context: Optional[Dict[str, Any]] = None,
    research_depth: str = "standard",
    run_context: Any = None,
) -> PlannerRunResult:
    """执行基于规则的 planner，并产出任务更新与下一步动作决策。

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
    session_snapshot: Optional[Dict[str, Any]] = None

    if planner_context is None and knowledge_manager is not None and research_id and session_id:
        planner_context = _planner_context_from_snapshot(
            knowledge_manager=knowledge_manager,
            research_id=research_id,
            session_id=session_id,
            user_query=query,
            task_tree=task_tree,
            active_task_id=active_task_id,
        )

    if planner_context is None and knowledge_manager is not None and research_id and session_id:
        try:
            session_snapshot = knowledge_manager.get_session_snapshot(research_id, session_id)
        except Exception:
            session_snapshot = None

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

    replanning_signals = dict(distiller_outputs or {})
    if planner_context:
        replanning_signals = {
            "coverage_summary": planner_context.get("coverage_summary", {}),
            "unresolved_gaps": [row.get("gap_text", "") for row in planner_context.get("unresolved_gaps", [])],
            "conflict_ids": [row.get("id", "") for row in planner_context.get("conflict_hotspots", [])],
            "evidence_ids": [row.get("id", "") for row in planner_context.get("relevant_evidence", [])],
            "fact_ids": [row.get("id", "") for row in (session_snapshot or {}).get("facts", [])],
        }
    replanning_patches, replanned_task_ids = _build_replanning_patches(query, task_tree, replanning_signals)
    patches.extend(replanning_patches)
    new_task_ids.extend(replanned_task_ids)
    if replanned_task_ids:
        rationale_parts.append(f"Added {len(replanned_task_ids)} follow-up task(s) from session planning context.")
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

    coverage_summary = dict(
        (planner_context or {}).get("coverage_summary")
        or distiller_outputs.get("coverage_summary", {})
        or {}
    )
    effective_packs = list((planner_context or {}).get("section_packs", []) or section_evidence_packs or [])
    coverage = _coverage_score(
        knowledge_refs,
        effective_packs,
        distiller_outputs,
        session_snapshot,
        planner_context,
    )
    conflict_count = (
        len((planner_context or {}).get("conflict_hotspots", []))
        if planner_context
        else len(session_snapshot.get("conflicts", []))
        if session_snapshot
        else len(distiller_outputs.get("conflict_ids", []))
    )
    gap_count = len((planner_context or {}).get("unresolved_gaps", []) or distiller_outputs.get("unresolved_gaps", []))
    novelty_count = (
        int(((planner_context or {}).get("latest_novelty_snapshot", {}) or {}).get("new_fact_count", 0))
        + int(((planner_context or {}).get("latest_novelty_snapshot", {}) or {}).get("new_claim_count", 0))
        if planner_context and (planner_context.get("latest_novelty_snapshot") or {})
        else len(session_snapshot.get("facts", [])) + len(session_snapshot.get("claims", []))
        if session_snapshot
        else len(distiller_outputs.get("fact_ids", [])) + len(distiller_outputs.get("claim_ids", []))
    )
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
    if session_snapshot:
        convergence_summary += (
            f"; session_facts={len(session_snapshot.get('facts', []))}"
            f"; session_evidence={len(session_snapshot.get('evidence', []))}"
            f"; session_packs={len((planner_context or {}).get('section_packs', []) or session_snapshot.get('section_evidence_packs', []))}"
        )
    elif planner_context:
        counts = dict((planner_context.get("retrieval_meta", {}) or {}).get("counts", {}) or {})
        convergence_summary += (
            f"; session_facts={counts.get('facts', 0)}"
            f"; session_evidence={counts.get('evidence', 0)}"
            f"; session_packs={counts.get('section_packs', 0)}"
        )
    ready_section_count = 0
    if planner_context:
        ready_section_count = len(planner_context.get("writing_ready_sections", []))
        convergence_summary += (
            f"; planner_context=active"
            f"; ready_sections={ready_section_count}"
        )
    if current_convergence_status:
        convergence_summary += f"; external={current_convergence_status}"

    next_task_id = _select_next_task_id(draft_tree)
    action = PlannerAction.CONTINUE_RESEARCH.value
    stop_reason = None

    if _writing_budget_reached(
        research_depth=research_depth,
        completed_count=completed_count,
        ready_section_count=ready_section_count,
        pack_count=len(effective_packs),
        coverage=coverage,
    ):
        action = PlannerAction.START_WRITING.value
        next_task_id = None
        rationale_parts.append(
            f"Research budget for {research_depth or 'standard'} depth reached and evidence is sufficient for a draft."
        )
    elif next_task_id:
        rationale_parts.append(f"Selected active task: {next_task_id}.")
    elif coverage >= 0.35 or completed_count > 0 or effective_packs:
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
