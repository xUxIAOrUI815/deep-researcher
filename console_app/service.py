from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

import core.graph as graph_module
from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
from core.observability import EventLevel, EventType, NoopObserver, ObservabilityEvent, set_observer
from core.run_context import RunContext
from core.session_knowledge import KnowledgeManager
from core.session_retrieval import SessionRetrievalService
from schemas.console import (
    ActiveAgentSummary,
    ConsoleRunSummary,
    ContextPanelSummary,
    DebugViewResponse,
    KnowledgeSummary,
    ReportViewResponse,
    ResearchCreateRequest,
    ResearchCreateResponse,
    TimelineEventSummary,
)
from schemas.state import DistillerOutputs, KnowledgeRefs, PlannerState, ResearcherOutputs, RunMetadata


class MemoryObserver(NoopObserver):
    def __init__(self) -> None:
        self._events: dict[str, list[ObservabilityEvent]] = {}
        self._lock = asyncio.Lock()

    def emit(self, event: ObservabilityEvent) -> None:
        events = self._events.setdefault(event.research_id, [])
        events.append(event)
        if len(events) > 500:
            del events[:-500]

    def list_events(self, research_id: str) -> list[dict[str, Any]]:
        return [item.to_dict() for item in self._events.get(research_id, [])]


@dataclass
class RunHandle:
    research_id: str
    thread_id: str
    session_id: str
    query: str
    instructions: str = ""
    depth: str = "standard"
    started_at: datetime = field(default_factory=datetime.now)
    status: str = "initializing"
    error: str = ""
    resumed: bool = False


class ResearchConsoleService:
    def __init__(self, runtime_dir: str = ".console_runtime"):
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.graph_db_path = self.runtime_dir / "research_console.sqlite3"
        self.knowledge_dir = self.runtime_dir / "knowledge"
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)
        self.knowledge_manager = KnowledgeManager(
            base_storage_path=str(self.knowledge_dir),
            sqlite_filename="session_knowledge.sqlite3",
        )
        self.retrieval_service = SessionRetrievalService(self.knowledge_manager)
        self.planner_context_builder = PlannerContextBuilder(self.retrieval_service)
        self.researcher_context_builder = ResearcherContextBuilder(self.retrieval_service)
        self.writer_context_builder = WriterContextBuilder(self.retrieval_service)
        self.active_runs: dict[str, RunHandle] = {}
        self._run_tasks: set[asyncio.Task[Any]] = set()
        self._closed = False
        self.observer = MemoryObserver()
        set_observer(self.observer)
        graph_module.SESSION_KNOWLEDGE_MANAGER = self.knowledge_manager
        graph_module.SESSION_RETRIEVAL_SERVICE = self.retrieval_service
        graph_module.PLANNER_CONTEXT_BUILDER = self.planner_context_builder
        graph_module.RESEARCHER_CONTEXT_BUILDER = self.researcher_context_builder
        graph_module.WRITER_CONTEXT_BUILDER = self.writer_context_builder

    async def create_run(self, request: ResearchCreateRequest) -> ResearchCreateResponse:
        research_id = f"research-{uuid.uuid4().hex[:10]}"
        thread_id = research_id
        session_id = f"session_{research_id}"
        handle = RunHandle(
            research_id=research_id,
            thread_id=thread_id,
            session_id=session_id,
            query=request.query,
            instructions=request.instructions,
            depth=request.depth,
        )
        self.active_runs[research_id] = handle
        self.knowledge_manager.create_or_get_session(
            research_id=research_id,
            root_query=request.query,
            session_id=session_id,
            metadata_json={
                "instructions": request.instructions,
                "depth": request.depth,
                "created_via": "research_console",
            },
        )
        task = asyncio.get_running_loop().create_task(self._execute_run(handle))
        self._run_tasks.add(task)
        task.add_done_callback(self._run_tasks.discard)
        return ResearchCreateResponse(
            research_id=research_id,
            thread_id=thread_id,
            session_id=session_id,
            status=handle.status,
            console_url=f"/console/{research_id}",
            report_url=f"/report/{research_id}",
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = [task for task in self._run_tasks if not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._run_tasks.clear()
        self.knowledge_manager.close()

    async def _execute_run(self, handle: RunHandle) -> None:
        config = {"configurable": {"thread_id": handle.thread_id, "research_id": handle.research_id}}
        context = RunContext.from_config(config, root_query=handle.query)
        initial_state = self._build_initial_state(context, handle.query, handle.instructions, handle.depth)
        saver = await graph_module.init_sqlite_saver(str(self.graph_db_path))
        graph = graph_module.create_research_graph(saver)
        handle.status = "running"
        try:
            await graph.ainvoke(initial_state, config)
            handle.status = "completed"
        except asyncio.CancelledError:
            handle.status = "cancelled"
            raise
        except Exception as exc:
            handle.status = "failed"
            handle.error = str(exc)
            self.knowledge_manager.store.update_session_status(
                handle.research_id,
                status="failed",
                current_active_task_id=None,
            )
        finally:
            await saver.conn.close()

    def _build_initial_state(
        self,
        context: RunContext,
        query: str,
        instructions: str,
        depth: str,
    ) -> Dict[str, Any]:
        return {
            "user_query": query,
            "normalized_query": query,
            "run_metadata": RunMetadata(
                research_id=context.research_id,
                thread_id=context.thread_id,
                run_id=context.run_id,
                trace_id=context.trace_id,
                session_id=context.session_id or f"session_{context.research_id}",
                graph_version=context.graph_version,
                prompt_version=context.prompt_version,
                root_query=query,
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
            "fact_pool": [],
            "atomic_facts": [],
            "current_focus": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "messages": [
                {
                    "role": "system",
                    "content": "Research console run initialized.",
                    "instructions": instructions,
                    "depth": depth,
                }
            ],
            "raw_scraped_data": [],
            "search_results": [],
        }

    async def _load_graph_state(self, research_id: str, thread_id: str) -> Dict[str, Any]:
        saver = await graph_module.init_sqlite_saver(str(self.graph_db_path))
        try:
            graph = graph_module.create_research_graph(saver)
            state = await graph.aget_state({"configurable": {"thread_id": thread_id, "research_id": research_id}})
            return dict(state.values) if state and state.values else {}
        except Exception:
            return {}
        finally:
            await saver.conn.close()

    def _get_handle(self, research_id: str) -> Optional[RunHandle]:
        return self.active_runs.get(research_id)

    async def list_runs(self) -> List[Dict[str, Any]]:
        rows = self.knowledge_manager.store.conn.execute(
            "SELECT research_id, session_id, root_query, status, updated_at, current_round FROM research_sessions ORDER BY updated_at DESC LIMIT 20"
        ).fetchall()
        output = []
        for row in rows:
            handle = self._get_handle(str(row["research_id"]))
            output.append(
                {
                    "research_id": str(row["research_id"]),
                    "session_id": str(row["session_id"]),
                    "query": str(row["root_query"]),
                    "status": handle.status if handle else str(row["status"]),
                    "current_round": int(row["current_round"]),
                    "updated_at": str(row["updated_at"]),
                    "console_url": f"/console/{row['research_id']}",
                    "report_url": f"/report/{row['research_id']}",
                }
            )
        return output

    async def get_console_summary(self, research_id: str) -> ConsoleRunSummary:
        session = self.knowledge_manager.store.get_session(research_id)
        if session is None:
            raise KeyError(research_id)
        thread_id = self._get_handle(research_id).thread_id if self._get_handle(research_id) else research_id
        state = await self._load_graph_state(research_id, thread_id)
        snapshot = self.knowledge_manager.get_session_snapshot(research_id, session.session_id)
        planner_context = self.planner_context_builder.build(
            research_id=research_id,
            session_id=session.session_id,
            user_query=session.root_query,
            task_tree=state.get("task_tree", {}),
            active_task_id=state.get("active_task_id"),
        ).model_dump()
        researcher_context = self.researcher_context_builder.build(
            research_id=research_id,
            session_id=session.session_id,
            root_user_query=session.root_query,
            task_id=state.get("active_task_id"),
            task=state.get("task_tree", {}).get(state.get("active_task_id")),
        ).model_dump()
        writer_context = self.writer_context_builder.build(
            research_id=research_id,
            session_id=session.session_id,
            report_outline=state.get("report_outline", {}),
            section_goals=state.get("section_goals", []),
            fallback_section_packs=state.get("section_evidence_packs", []),
        ).model_dump()
        handle = self._get_handle(research_id)
        status = self._derive_status(state, handle)
        current_stage = self._derive_stage(state, handle, self.observer.list_events(research_id))
        elapsed_seconds = max(
            0.0,
            (datetime.now() - (handle.started_at if handle else session.created_at)).total_seconds(),
        )
        return ConsoleRunSummary(
            research_id=research_id,
            thread_id=thread_id,
            session_id=session.session_id,
            query=session.root_query,
            status=status,
            current_stage=current_stage,
            current_round=session.current_round,
            elapsed_seconds=elapsed_seconds,
            resumed=bool(handle and handle.resumed),
            has_report=bool(state.get("final_report")),
            root_task_id=state.get("root_task_id"),
            active_task_id=state.get("active_task_id"),
            planner_state=state.get("planner_state", {}),
            report_outline=state.get("report_outline", {}),
            task_tree=state.get("task_tree", {}),
            timeline=self._build_timeline(research_id, state),
            knowledge_summary=self._build_knowledge_summary(snapshot),
            latest_coverage_snapshot=snapshot.get("latest_coverage_snapshot"),
            open_gaps=snapshot.get("open_gaps", []),
            conflicts=snapshot.get("conflicts", []),
            section_packs=snapshot.get("section_evidence_packs", []),
            sources=snapshot.get("sources", []),
            active_agent=self._build_active_agent_summary(state, current_stage),
            context_summary=ContextPanelSummary(
                planner=self._summarize_planner_context(planner_context),
                researcher=self._summarize_researcher_context(researcher_context),
                writer=self._summarize_writer_context(writer_context),
            ),
            run_metadata=state.get("run_metadata", {}),
        )

    async def get_report_view(self, research_id: str) -> ReportViewResponse:
        summary = await self.get_console_summary(research_id)
        state = await self._load_graph_state(research_id, summary.thread_id)
        report = dict(state.get("final_report", {}) or {})
        return ReportViewResponse(
            research_id=research_id,
            session_id=summary.session_id,
            query=summary.query,
            status=summary.status,
            title=str((summary.report_outline or {}).get("title", "") or summary.query),
            markdown=str(report.get("markdown", "") or ""),
            outline=summary.report_outline,
            report=report,
            knowledge_summary=summary.knowledge_summary,
            latest_coverage_snapshot=summary.latest_coverage_snapshot,
            open_gaps=summary.open_gaps,
            section_packs=summary.section_packs,
            context_summary=summary.context_summary,
        )

    async def get_debug_view(self, research_id: str) -> DebugViewResponse:
        summary = await self.get_console_summary(research_id)
        state = await self._load_graph_state(research_id, summary.thread_id)
        snapshot = self.knowledge_manager.get_session_snapshot(research_id, summary.session_id)
        return DebugViewResponse(
            research_id=research_id,
            session_id=summary.session_id,
            status=summary.status,
            state_summary={
                "active_task_id": state.get("active_task_id"),
                "root_task_id": state.get("root_task_id"),
                "planner_action": (state.get("planner_state", {}) or {}).get("action"),
                "next_task_id": (state.get("planner_state", {}) or {}).get("next_task_id"),
                "completed_tasks": len(state.get("completed_tasks", [])),
                "failed_tasks": len(state.get("failed_tasks", [])),
            },
            context_summary=summary.context_summary,
            trace=summary.timeline,
            raw_state={
                "run_metadata": state.get("run_metadata", {}),
                "planner_state": state.get("planner_state", {}),
                "researcher_outputs": state.get("researcher_outputs", {}),
                "distiller_outputs": state.get("distiller_outputs", {}),
            },
            snapshot_summary={
                "session": snapshot.get("session", {}),
                "knowledge_refs": snapshot.get("knowledge_refs", {}),
                "stats": snapshot.get("stats", {}),
            },
        )

    def _build_knowledge_summary(self, snapshot: Dict[str, Any]) -> KnowledgeSummary:
        refs = snapshot.get("knowledge_refs", {}) or {}
        return KnowledgeSummary(
            source_count=len(refs.get("source_ids", [])),
            claim_count=len(snapshot.get("claims", [])),
            fact_count=len(snapshot.get("facts", [])),
            evidence_count=len(snapshot.get("evidence", [])),
            conflict_count=len(snapshot.get("conflicts", [])),
            open_gap_count=len(snapshot.get("open_gaps", [])),
            section_pack_count=len(snapshot.get("section_evidence_packs", [])),
        )

    def _build_timeline(self, research_id: str, state: Dict[str, Any]) -> List[TimelineEventSummary]:
        events = []
        for item in self.observer.list_events(research_id):
            events.append(
                TimelineEventSummary(
                    event_id=str(item.get("event_id", "")),
                    event_type=str(item.get("event_type", "")),
                    timestamp=str(item.get("timestamp", "")),
                    level=str(item.get("level", "info")),
                    message=str(item.get("message", "")),
                    node_name=item.get("node_name"),
                    agent_name=item.get("agent_name"),
                    task_id=item.get("task_id"),
                    section_id=item.get("section_id"),
                    payload=dict(item.get("payload", {}) or {}),
                )
            )
        if not events:
            for item in state.get("state_events", [])[-50:]:
                events.append(
                    TimelineEventSummary(
                        event_type=str(item.get("event_type", "")),
                        timestamp=str(item.get("timestamp", "")),
                        message=str(item.get("message", "")),
                        payload=dict(item.get("payload", {}) or {}),
                    )
                )
        return events[-50:]

    def _derive_status(self, state: Dict[str, Any], handle: Optional[RunHandle]) -> str:
        if state.get("final_report"):
            return "completed"
        if handle and handle.status == "failed":
            return "failed"
        if state.get("error_state"):
            return "failed"
        if handle:
            return handle.status
        return "idle"

    def _derive_stage(self, state: Dict[str, Any], handle: Optional[RunHandle], observer_events: List[Dict[str, Any]]) -> str:
        if state.get("final_report"):
            return "completed"
        if handle and handle.status == "failed":
            return "failed"
        if observer_events:
            latest = observer_events[-1].get("event_type", "")
            if "writer" in latest or latest == EventType.REPORT_FINALIZED.value:
                return "writing"
            if "distill" in latest or "evidence_pack" in latest:
                return "knowledge_updating"
            if "source" in latest or "query" in latest or "exploration" in latest:
                return "researching"
            if "task" in latest or "planner" in latest:
                return "planning"
        planner_action = (state.get("planner_state", {}) or {}).get("action")
        if planner_action == "start_writing":
            return "writing"
        if state.get("active_task_id"):
            return "researching"
        return "planning" if state.get("task_tree") else "initializing"

    def _build_active_agent_summary(self, state: Dict[str, Any], current_stage: str) -> ActiveAgentSummary:
        task_id = state.get("active_task_id") or (state.get("planner_state", {}) or {}).get("next_task_id")
        task = (state.get("task_tree", {}) or {}).get(task_id, {}) if task_id else {}
        stage_to_agent = {
            "planning": "planner",
            "researching": "researcher",
            "knowledge_updating": "distiller",
            "writing": "writer",
            "completed": "writer",
            "failed": "system",
        }
        return ActiveAgentSummary(
            name=stage_to_agent.get(current_stage, "planner"),
            status=current_stage,
            target=str(task.get("title") or task.get("query") or ""),
            last_output_summary=str((state.get("planner_state", {}) or {}).get("rationale", ""))[:240],
        )

    def _summarize_planner_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "ready_sections": payload.get("writing_ready_sections", []),
            "coverage": (payload.get("coverage_summary", {}) or {}).get("avg_section_coverage", 0.0),
            "gap_count": len(payload.get("unresolved_gaps", [])),
            "conflict_count": len(payload.get("conflict_hotspots", [])),
        }

    def _summarize_researcher_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "seen_sources": len(payload.get("already_seen_source_ids", [])),
            "gap_count": len(payload.get("unresolved_gaps", [])),
            "focus_sections": payload.get("focus_sections", []),
            "authority_gaps": payload.get("authority_gaps", []),
        }

    def _summarize_writer_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "context_source": payload.get("context_source", "fallback"),
            "pack_count": len(payload.get("section_evidence_packs", [])),
            "section_count": len(payload.get("section_contexts", [])),
        }
