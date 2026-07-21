import asyncio
from datetime import datetime
from typing import Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

# Active graph implementation only.
# Legacy pre-cleanup snapshot has been copied to docs/legacy/graph_v1.py.

async def init_sqlite_saver(db_path: str = "research.db") -> AsyncSqliteSaver:
    """初始化 SQLite checkpointer，并设置 WAL、同步模式和超时参数。"""
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.commit()

    saver = AsyncSqliteSaver(conn)

    return saver

from langchain_core.runnables import RunnableConfig

from agents.distiller import run_distiller
from agents.planner import run_planner
from agents.researcher import run_researcher
from agents.writer import run_writer
from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
from core.session_knowledge import KnowledgeManager
from core.session_retrieval import SessionRetrievalService
from core.observability import EventType, get_observer
from core.run_context import RunContext
from schemas.state import (
    DistillerOutputs,
    FinalReport,
    KnowledgeRefs,
    PlannerAction,
    PlannerState,
    ResearchGraphState,
    ResearcherOutputs,
    RunMetadata,
    TaskNode,
)

try:
    SESSION_KNOWLEDGE_MANAGER = KnowledgeManager()
except Exception:
    SESSION_KNOWLEDGE_MANAGER = KnowledgeManager(sqlite_filename=":memory:")
SESSION_RETRIEVAL_SERVICE = SessionRetrievalService(SESSION_KNOWLEDGE_MANAGER)
PLANNER_CONTEXT_BUILDER = PlannerContextBuilder(SESSION_RETRIEVAL_SERVICE)
RESEARCHER_CONTEXT_BUILDER = ResearcherContextBuilder(SESSION_RETRIEVAL_SERVICE)
WRITER_CONTEXT_BUILDER = WriterContextBuilder(SESSION_RETRIEVAL_SERVICE)
DURABLE_KNOWLEDGE_INGESTOR: Any | None = None


def set_durable_knowledge_ingestor(ingestor: Any | None) -> Any | None:
    """Install the transitional v2 ingestion sink and return the prior sink."""
    global DURABLE_KNOWLEDGE_INGESTOR
    previous = DURABLE_KNOWLEDGE_INGESTOR
    DURABLE_KNOWLEDGE_INGESTOR = ingestor
    return previous


GraphState = ResearchGraphState


def _now_iso() -> str:
    """返回当前时间的 ISO 8601 字符串，用于记录状态事件时间戳。"""
    return datetime.now().isoformat()


def _record_budget_snapshot(state: GraphState, context: RunContext, stage: str) -> None:
    usage = dict(state.get("token_usage", {}) or {})
    get_observer().record_run_event(
        context,
        EventType.BUDGET_SNAPSHOT,
        message=f"Budget usage snapshot after {stage}.",
        payload={
            "stage": stage,
            "usage": {
                "input_tokens": 0,
                "output_tokens": int(usage.get("total_tokens", 0) or 0),
                "cost_usd": float(usage.get("cost_usd", 0.0) or 0.0),
                "wall_time_seconds": float(usage.get("wall_time_seconds", 0.0) or 0.0),
                "model_calls": int(usage.get("model_calls", 0) or 0),
                "tool_calls": int(usage.get("tool_calls", 0) or 0),
                "search_calls": int(usage.get("search_calls", 0) or 0),
                "retries": int(usage.get("retries", 0) or 0),
                "errors": int(usage.get("errors", 0) or 0),
            },
            "legacy_token_usage": usage,
        },
    )


def _get_config_dict(config: Optional[Any]) -> dict:
    """把 RunnableConfig 或兼容对象统一转换成普通 dict。"""
    if config is None:
        return {}
    if isinstance(config, dict):
        return config
    configurable = getattr(config, "configurable", None)
    return {"configurable": configurable or {}}


def _get_run_context(state: GraphState, config: Optional[Any] = None) -> RunContext:
    """优先从状态恢复运行上下文；没有时再根据 config 生成并回填。"""
    metadata = state.get("run_metadata") or {}
    if metadata:
        return RunContext(
            research_id=metadata.get("research_id", "default"),
            thread_id=metadata.get("thread_id", "default-thread"),
            run_id=metadata.get("run_id") or RunContext.from_config(None).run_id,
            trace_id=metadata.get("trace_id") or RunContext.from_config(None).trace_id,
            session_id=metadata.get("session_id"),
            root_query=metadata.get("root_query", state.get("user_query", "")),
            graph_version=metadata.get("graph_version", "graph.v1"),
            prompt_version=metadata.get("prompt_version", "prompt.v1"),
        )

    context = RunContext.from_config(
        _get_config_dict(config),
        root_query=state.get("user_query", "") or state.get("normalized_query", ""),
    )
    state["run_metadata"] = RunMetadata(
        research_id=context.research_id,
        thread_id=context.thread_id,
        run_id=context.run_id,
        trace_id=context.trace_id,
        session_id=context.session_id,
        graph_version=context.graph_version,
        prompt_version=context.prompt_version,
        root_query=context.root_query,
    ).model_dump()
    return context


def _append_state_event(
    state: GraphState,
    event_type: str,
    message: str = "",
    payload: Optional[dict] = None,
) -> None:
    """向状态事件列表追加一条事件，并将历史裁剪到最近 50 条。"""
    events = state.setdefault("state_events", [])
    events.append(
        {
            "event_type": event_type,
            "message": message,
            "payload": payload or {},
            "timestamp": _now_iso(),
        }
    )
    if len(events) > 50:
        del events[:-50]


def _ensure_state_defaults(state: GraphState, config: Optional[Any] = None) -> RunContext:
    """补齐图运行需要的默认字段，并确保知识库集合名已写入状态。"""
    state.setdefault("user_query", state.get("normalized_query", "") or "Mock research query")
    state.setdefault("normalized_query", state["user_query"])
    state.setdefault("task_tree", {})
    state.setdefault("root_task_id", None)
    state.setdefault("active_task_id", None)
    state.setdefault("current_focus", None)
    state.setdefault("completed_tasks", [])
    state.setdefault("failed_tasks", [])
    state.setdefault("researcher_outputs", ResearcherOutputs().model_dump())
    state.setdefault("distiller_outputs", DistillerOutputs().model_dump())
    state.setdefault("planner_state", PlannerState().model_dump())
    state.setdefault("knowledge_refs", KnowledgeRefs().model_dump())
    state.setdefault("report_outline", {})
    state.setdefault("section_goals", [])
    state.setdefault("section_evidence_packs", [])
    state.setdefault("final_report", None)
    state.setdefault("error_state", None)
    state.setdefault("token_usage", {
        "planning_tokens": 0,
        "research_tokens": 0,
        "distillation_tokens": 0,
        "writing_tokens": 0,
        "total_tokens": 0,
    })
    # Legacy working-set fields stay during the transition, but task status is
    # sourced exclusively from task_tree.
    state.setdefault("fact_pool", [])
    state.setdefault("atomic_facts", [])
    state.setdefault("messages", [])
    state.setdefault("raw_scraped_data", [])
    state.setdefault("search_results", [])

    context = _get_run_context(state, config)
    knowledge_refs = state.setdefault("knowledge_refs", KnowledgeRefs().model_dump())
    knowledge_refs.setdefault("collection_name", context.knowledge_collection)
    return context


def _ensure_root_task(state: GraphState, context: RunContext) -> str:
    """确保任务树中存在根任务；若不存在则根据用户问题创建。"""
    task_tree = state.setdefault("task_tree", {})
    root_task_id = state.get("root_task_id")
    if root_task_id and root_task_id in task_tree:
        return root_task_id

    root_task = TaskNode(
        query=state.get("user_query", "") or context.root_query or "Mock research query",
        title="Root research question",
        rationale="Initial user research request",
        node_type="question",
        status="pending",
        depth=0,
        priority=1.0,
        created_by="graph",
        updated_by="graph",
    )
    task_tree[root_task.id] = root_task.model_dump()
    state["root_task_id"] = root_task.id
    get_observer().record_task_event(
        context,
        EventType.TASK_CREATED,
        root_task.id,
        message="Created root task from user_query",
    )
    _append_state_event(state, "task.created", "Created root task", {"task_id": root_task.id})
    return root_task.id


def _pending_task_ids(state: GraphState) -> list[str]:
    """收集当前任务树里所有状态为 pending 的任务 ID。"""
    return [
        task_id
        for task_id, task in state.get("task_tree", {}).items()
        if task.get("status") == "pending"
    ]


def _select_next_task_id(state: GraphState) -> Optional[str]:
    """按优先级、深度和创建时间选择下一条待执行任务。"""
    pending = _pending_task_ids(state)
    if not pending:
        return None
    task_tree = state.get("task_tree", {})
    return sorted(
        pending,
        key=lambda tid: (
            -float(task_tree.get(tid, {}).get("priority", 0.0)),
            int(task_tree.get(tid, {}).get("depth", 0)),
            task_tree.get(tid, {}).get("created_at", ""),
        ),
    )[0]


def _set_task_status(
    state: GraphState,
    context: RunContext,
    task_id: str,
    status: str,
    *,
    updated_by: str,
) -> None:
    """更新任务状态，并同步 completed/failed 列表与观测事件。"""
    task = state.get("task_tree", {}).get(task_id)
    if not task:
        return
    old_status = task.get("status")
    task["status"] = status
    task["updated_by"] = updated_by
    task["updated_at"] = datetime.now()
    if status == "completed" and task_id not in state.setdefault("completed_tasks", []):
        state["completed_tasks"].append(task_id)
    if status == "failed" and task_id not in state.setdefault("failed_tasks", []):
        state["failed_tasks"].append(task_id)

    get_observer().record_task_event(
        context,
        EventType.TASK_STATUS_CHANGED,
        task_id,
        message=f"Task status changed: {old_status} -> {status}",
        payload={"old_status": old_status, "new_status": status},
    )
    _append_state_event(
        state,
        "task.status_changed",
        f"Task status changed: {old_status} -> {status}",
        {"task_id": task_id, "old_status": old_status, "new_status": status},
    )


def _apply_task_tree_patches(
    state: GraphState,
    context: RunContext,
    patches: list[Any],
) -> None:
    """应用 planner 产出的任务树补丁，处理 attach/update/defer/prune/merge。"""
    task_tree = state.setdefault("task_tree", {})

    for patch in patches:
        patch_data = patch if isinstance(patch, dict) else patch.model_dump()
        operation = patch_data.get("operation")
        task_id = patch_data.get("task_id")
        parent_task_id = patch_data.get("parent_task_id")
        target_task_id = patch_data.get("target_task_id")
        task = patch_data.get("task")
        updates = patch_data.get("updates", {})
        rationale = patch_data.get("rationale", "")

        if operation == "attach" and task is not None:
            task_data = task.model_dump() if hasattr(task, "model_dump") else dict(task)
            task_id = task_data["id"]
            task_tree[task_id] = task_data
            parent_id = parent_task_id or task_data.get("parent_task_id") or task_data.get("parent_id")
            if parent_id and parent_id in task_tree:
                children = task_tree[parent_id].setdefault("children_ids", [])
                if task_id not in children:
                    children.append(task_id)
            get_observer().record_task_event(
                context,
                EventType.TASK_CREATED,
                task_id,
                message="Applied planner attach patch",
                payload={"parent_task_id": parent_id, "rationale": rationale},
            )
            _append_state_event(state, "task.created", "Applied planner attach patch", {"task_id": task_id})
            continue

        if operation in {"update", "defer", "prune"} and task_id in task_tree:
            old_status = task_tree[task_id].get("status")
            task_tree[task_id].update(updates)
            task_tree[task_id]["updated_by"] = "planner"
            task_tree[task_id]["updated_at"] = datetime.now()
            event_type = {
                "defer": EventType.TASK_DEFERRED,
                "prune": EventType.TASK_PRUNED,
            }.get(operation, EventType.TASK_UPDATED)
            get_observer().record_task_event(
                context,
                event_type,
                task_id,
                message=f"Applied planner {operation} patch",
                payload={"old_status": old_status, "updates": updates, "rationale": rationale},
            )
            _append_state_event(
                state,
                f"task.{operation}",
                f"Applied planner {operation} patch",
                {"task_id": task_id, "updates": updates},
            )
            continue

        if operation == "merge" and task_id in task_tree:
            task_tree[task_id].update(updates)
            task_tree[task_id]["status"] = "merged"
            task_tree[task_id]["merge_into"] = target_task_id
            task_tree[task_id]["updated_by"] = "planner"
            task_tree[task_id]["updated_at"] = datetime.now()
            get_observer().record_task_event(
                context,
                EventType.TASK_MERGED,
                task_id,
                message="Applied planner merge patch",
                payload={"target_task_id": target_task_id, "rationale": rationale},
            )
            _append_state_event(
                state,
                "task.merged",
                "Applied planner merge patch",
                {"task_id": task_id, "target_task_id": target_task_id},
            )


async def _call_planner_agent(state: GraphState, context: RunContext) -> Any:
    """构造 planner 上下文并调用规划 agent。"""
    # Thin transition adapter only. Planner behavior lives in agents/planner.py.
    session_id = context.session_id or f"session_{context.research_id}"
    planner_context = PLANNER_CONTEXT_BUILDER.build(
        research_id=context.research_id,
        session_id=session_id,
        user_query=state.get("user_query", ""),
        task_tree=state.get("task_tree", {}),
        active_task_id=state.get("active_task_id"),
    ).model_dump()
    return await run_planner(
        user_query=state.get("user_query", ""),
        normalized_query=state.get("normalized_query", ""),
        task_tree=state.get("task_tree", {}),
        active_task_id=state.get("active_task_id"),
        distiller_outputs=state.get("distiller_outputs", {}),
        knowledge_refs=state.get("knowledge_refs", {}),
        report_outline=state.get("report_outline", {}),
        section_goals=state.get("section_goals", []),
        section_evidence_packs=state.get("section_evidence_packs", []),
        current_convergence_status=state.get("convergence_status"),
        knowledge_manager=SESSION_KNOWLEDGE_MANAGER,
        research_id=context.research_id,
        session_id=session_id,
        planner_context=planner_context,
        research_depth=str((state.get("messages") or [{}])[0].get("depth", "standard") or "standard"),
        run_context=context,
    )


async def _call_researcher_agent(state: GraphState, context: RunContext) -> ResearcherOutputs:
    """构造 researcher 上下文并调用研究 agent。"""
    # Thin transition adapter only. Researcher behavior lives in agents/researcher.py.
    task_id = state.get("active_task_id")
    session_id = context.session_id or f"session_{context.research_id}"
    research_context = RESEARCHER_CONTEXT_BUILDER.build(
        research_id=context.research_id,
        session_id=session_id,
        root_user_query=state.get("user_query") or state.get("normalized_query") or context.root_query,
        task_id=task_id,
        task=state.get("task_tree", {}).get(task_id) if task_id else None,
    ).model_dump()
    return await run_researcher(
        task_id=task_id,
        task=state.get("task_tree", {}).get(task_id) if task_id else None,
        root_user_query=state.get("user_query") or state.get("normalized_query") or context.root_query,
        knowledge_refs=state.get("knowledge_refs", {}),
        knowledge_manager=SESSION_KNOWLEDGE_MANAGER,
        research_id=context.research_id,
        session_id=session_id,
        research_context=research_context,
        run_context=context,
    )


async def _call_distiller_agent(state: GraphState, context: RunContext) -> DistillerOutputs:
    """调用 distiller agent，把 researcher 输出转成结构化知识。"""
    # Thin transition adapter only. Distiller behavior lives in agents/distiller.py.
    task_id = state.get("active_task_id")
    return await run_distiller(
        task_id=task_id,
        task=state.get("task_tree", {}).get(task_id) if task_id else None,
        researcher_outputs=state.get("researcher_outputs", {}),
        report_outline=state.get("report_outline", {}),
        section_goals=state.get("section_goals", []),
        knowledge_refs=state.get("knowledge_refs", {}),
        run_context=context,
    )


async def _call_writer_agent(state: GraphState, context: RunContext) -> FinalReport:
    """构造 writer 上下文并调用写作 agent 生成最终报告。"""
    # Thin transition adapter only. Writer behavior lives in agents/writer.py.
    session_id = context.session_id or f"session_{context.research_id}"
    writer_context = WRITER_CONTEXT_BUILDER.build(
        research_id=context.research_id,
        session_id=session_id,
        report_outline=state.get("report_outline", {}),
        section_goals=state.get("section_goals", []),
        fallback_section_packs=state.get("section_evidence_packs", []),
    ).model_dump()
    return await run_writer(
        report_outline=state.get("report_outline", {}),
        section_goals=state.get("section_goals", []),
        section_evidence_packs=state.get("section_evidence_packs", []),
        knowledge_manager=SESSION_KNOWLEDGE_MANAGER,
        research_id=context.research_id,
        session_id=session_id,
        writer_context=writer_context,
        run_context=context,
    )


async def planner(state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
    """v2 规划节点：补齐状态、创建根任务、执行规划并写回任务树。"""
    context = _ensure_state_defaults(state, config)
    observer = get_observer()
    observer.record_node_event(context, EventType.NODE_STARTED, "planner")

    if not any(event.get("event_type") == "run.started" for event in state.get("state_events", [])):
        observer.record_run_event(context, EventType.RUN_STARTED, message="Research run started")
        _append_state_event(state, "run.started", "Research run started")

    _ensure_root_task(state, context)

    planner_result = await _call_planner_agent(state, context)
    planner_state = planner_result.planner_state
    _apply_task_tree_patches(state, context, planner_state.task_updates)
    state["planner_state"] = planner_state.model_dump()
    if planner_result.active_task_id:
        state["active_task_id"] = planner_result.active_task_id
        state["current_focus"] = planner_result.active_task_id
    if planner_result.report_outline is not None:
        state["report_outline"] = planner_result.report_outline
    if planner_result.section_goals is not None:
        state["section_goals"] = planner_result.section_goals
    state["token_usage"]["planning_tokens"] += 0

    _record_budget_snapshot(state, context, "planner")

    observer.record_node_event(
        context,
        EventType.NODE_COMPLETED,
        "planner",
        payload={"action": planner_state.action},
    )
    if planner_state.action == PlannerAction.STOP.value:
        observer.record_run_event(
            context,
            EventType.RUN_COMPLETED,
            message="Research run stopped by planner semantic stop decision.",
            payload={"stop_reason": planner_state.stop_reason or "planner_stop"},
        )
    _append_state_event(
        state,
        "planner.decision",
        planner_state.rationale,
        {"action": planner_state.action},
    )
    return state


async def researcher_async(state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
    """v2 研究节点：选择活动任务并执行检索探索。"""
    context = _ensure_state_defaults(state, config)
    observer = get_observer()
    observer.record_node_event(context, EventType.NODE_STARTED, "researcher")

    planner_state = state.get("planner_state") or {}
    task_id = planner_state.get("active_task_id") or planner_state.get("next_task_id")
    if task_id and state.get("task_tree", {}).get(task_id, {}).get("status") != "pending":
        task_id = None
    if not task_id:
        # Transitional fallback only. Planner should own active task selection.
        task_id = _select_next_task_id(state)
    if not task_id:
        state["active_task_id"] = None
        state["current_focus"] = None
        observer.record_node_event(
            context,
            EventType.NODE_COMPLETED,
            "researcher",
            message="No pending task to research",
        )
        return state

    state["active_task_id"] = task_id
    state["current_focus"] = task_id
    _set_task_status(state, context, task_id, "running", updated_by="researcher")

    outputs = await _call_researcher_agent(state, context)
    state["researcher_outputs"] = outputs.model_dump()
    if DURABLE_KNOWLEDGE_INGESTOR is not None:
        DURABLE_KNOWLEDGE_INGESTOR.ingest_researcher_outputs(
            state["researcher_outputs"], run_id=context.run_id, task_id=task_id
        )

    _record_budget_snapshot(state, context, "researcher")

    observer.record_node_event(
        context,
        EventType.NODE_COMPLETED,
        "researcher",
        payload={"task_id": task_id},
    )
    return state


def researcher(state: GraphState) -> GraphState:
    """同步包装器：兼容同步调用方式触发 researcher_async。"""
    return asyncio.get_event_loop().run_until_complete(researcher_async(state))


async def distiller_async(state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
    """v2 蒸馏节点：处理研究结果，更新知识状态并完成当前任务。"""
    context = _ensure_state_defaults(state, config)
    observer = get_observer()
    observer.record_node_event(context, EventType.NODE_STARTED, "distiller")

    outputs = await _call_distiller_agent(state, context)
    session_id = context.session_id or f"session_{context.research_id}"
    distiller_payload = outputs.model_dump()
    distiller_payload["root_query"] = state.get("user_query") or state.get("normalized_query") or context.root_query
    distiller_payload["report_outline"] = state.get("report_outline", {})
    distiller_payload["section_goals"] = state.get("section_goals", [])
    outputs = SESSION_KNOWLEDGE_MANAGER.process_distiller_output(
        distiller_payload,
        research_id=context.research_id,
        session_id=session_id,
        task_id=state.get("active_task_id"),
    )
    state["distiller_outputs"] = outputs.model_dump()
    state["section_evidence_packs"] = outputs.section_evidence_packs
    if outputs.knowledge_refs:
        state["knowledge_refs"] = outputs.knowledge_refs
    if outputs.atomic_facts:
        state["atomic_facts"] = [fact.model_dump() for fact in outputs.atomic_facts]
    if DURABLE_KNOWLEDGE_INGESTOR is not None:
        DURABLE_KNOWLEDGE_INGESTOR.ingest_distiller_outputs(
            state["distiller_outputs"], run_id=context.run_id, task_id=state.get("active_task_id")
        )

    task_id = state.get("active_task_id")
    if task_id:
        _set_task_status(state, context, task_id, "completed", updated_by="distiller")
        state["active_task_id"] = None

    _record_budget_snapshot(state, context, "distiller")

    observer.record_node_event(
        context,
        EventType.NODE_COMPLETED,
        "distiller",
        payload={"task_id": task_id},
    )
    return state


async def writer_async(state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
    """v2 写作节点：基于大纲和证据包生成最终报告。"""
    context = _ensure_state_defaults(state, config)
    observer = get_observer()
    observer.record_node_event(context, EventType.NODE_STARTED, "writer")

    report = await _call_writer_agent(state, context)
    state["final_report"] = report.model_dump()
    if DURABLE_KNOWLEDGE_INGESTOR is not None:
        DURABLE_KNOWLEDGE_INGESTOR.ingest_report(
            state["final_report"],
            report_outline=state.get("report_outline", {}),
            run_id=context.run_id,
            thread_id=context.thread_id,
            research_question=state.get("user_query") or state.get("normalized_query") or context.root_query,
        )
    SESSION_KNOWLEDGE_MANAGER.store.update_session_status(
        context.research_id,
        status="completed",
        current_active_task_id=None,
    )

    _record_budget_snapshot(state, context, "writer")

    observer.record_node_event(context, EventType.NODE_COMPLETED, "writer")
    observer.record_run_event(context, EventType.RUN_COMPLETED, message="Research run completed")
    _append_state_event(
        state,
        "report.finalized",
        "Final report produced by writer node",
        {"report_id": report.report_id},
    )
    return state


def writer(state: GraphState) -> GraphState:
    """同步包装器：兼容同步调用方式触发 writer_async。"""
    return asyncio.get_event_loop().run_until_complete(writer_async(state))


def route_after_planner(state: GraphState) -> str:
    """根据 planner 决策结果选择下一跳：researcher、writer 或 end。"""
    planner_state = state.get("planner_state") or {}
    action = planner_state.get("action")

    if action == PlannerAction.CONTINUE_RESEARCH.value:
        return "researcher" if _pending_task_ids(state) else "writer"
    if action == PlannerAction.START_WRITING.value:
        return "writer"
    if action == PlannerAction.STOP.value:
        return "end"

    return "researcher" if _pending_task_ids(state) else "writer"


def should_continue(state: GraphState) -> str:
    """兼容旧接口的路由适配器，返回 researcher/writer/end。"""
    route = route_after_planner(state)
    if route == "writer":
        return "writer"
    if route == "end":
        return "end"
    return "researcher"


def _with_failure_events(node_name: str, node: Any) -> Any:
    async def observed(state: GraphState, config: Optional[RunnableConfig] = None) -> GraphState:
        try:
            return await node(state, config)
        except Exception as exc:
            context = _ensure_state_defaults(state, config)
            observer = get_observer()
            payload = {
                "error": f"{type(exc).__name__}: {str(exc)[:1500]}",
                "error_code": f"{node_name}_failed",
                "retryable": False,
                "fatal": True,
            }
            try:
                observer.record_node_event(
                    context,
                    EventType.NODE_FAILED,
                    node_name,
                    message=f"{node_name} node failed.",
                    payload=payload,
                )
                observer.record_run_event(
                    context,
                    EventType.RUN_FAILED,
                    message="Research run failed.",
                    payload=payload,
                )
            except Exception:
                # Observability must preserve the original application exception.
                pass
            raise

    observed.__name__ = f"observed_{node_name}"
    return observed


def create_research_graph(checkpointer: AsyncSqliteSaver) -> StateGraph:
    """构建 v2 研究图：planner 与 researcher/distiller 循环，最终进入 writer。"""
    workflow = StateGraph(ResearchGraphState)

    workflow.add_node("planner", _with_failure_events("planner", planner))
    workflow.add_node("researcher", _with_failure_events("researcher", researcher_async))
    workflow.add_node("distiller", _with_failure_events("distiller", distiller_async))
    workflow.add_node("writer", _with_failure_events("writer", writer_async))

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "researcher": "researcher",
            "writer": "writer",
            "end": END,
        },
    )
    workflow.add_edge("researcher", "distiller")
    workflow.add_edge("distiller", "planner")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=checkpointer)


async def run_research_cycle(initial_query: str = "Mock research query") -> dict:
    """研究主入口：构造初始状态，执行完整图调度并返回最终状态。"""
    saver = await init_sqlite_saver()
    graph = create_research_graph(saver)

    config = {
        "configurable": {
            "thread_id": "main-research-thread",
            "research_id": "main-research",
        }
    }
    context = RunContext.from_config(config, root_query=initial_query)

    initial_state: GraphState = {
        "user_query": initial_query,
        "normalized_query": initial_query,
        "run_metadata": RunMetadata(
            research_id=context.research_id,
            thread_id=context.thread_id,
            run_id=context.run_id,
            trace_id=context.trace_id,
            session_id=context.session_id,
            graph_version=context.graph_version,
            prompt_version=context.prompt_version,
            root_query=initial_query,
        ).model_dump(),
        "task_tree": {},
        "root_task_id": None,
        "active_task_id": None,
        "planner_state": PlannerState().model_dump(),
        "researcher_outputs": ResearcherOutputs().model_dump(),
        "distiller_outputs": DistillerOutputs().model_dump(),
        "knowledge_refs": KnowledgeRefs(collection_name=context.knowledge_collection).model_dump(),
        "report_outline": {},
        "section_goals": [],
        "section_evidence_packs": [],
        "final_report": None,
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0,
        },
        "state_events": [],
        "error_state": None,
        # Legacy transition fields.
        "fact_pool": [],
        "atomic_facts": [],
        "current_focus": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": [],
        "raw_scraped_data": [],
        "search_results": [],
    }

    try:
        result = await graph.ainvoke(initial_state, config)
    finally:
        await saver.conn.close()

    final_report = result.get("final_report")
    if isinstance(final_report, dict):
        report_preview = final_report.get("markdown", "")[:200]
    elif final_report:
        report_preview = str(final_report)[:200]
    else:
        report_preview = "N/A"

    print("\n" + "=" * 60)
    print("RESEARCH CYCLE COMPLETE")
    print("=" * 60)
    print(f"Tasks completed: {len(result.get('completed_tasks', []))}")
    print(f"Atomic facts extracted: {len(result.get('atomic_facts', []))}")
    print(f"Search results: {len(result.get('search_results', []))}")
    print(f"Scraped pages: {len(result.get('raw_scraped_data', []))}")
    print(f"Final report: {report_preview}...")

    return result
