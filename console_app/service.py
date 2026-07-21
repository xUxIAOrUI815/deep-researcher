from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import uuid
from typing import Any, Dict, List, Optional

import core.graph as graph_module
from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
from core.observability import get_observer, set_observer
from core.run_context import RunContext
from core.session_knowledge import KnowledgeManager
from core.session_retrieval import SessionRetrievalService
from deep_researcher.events import EventRecorder, PersistentEventObserver, SQLiteEventStore
from deep_researcher.studio import (
    SQLiteStudioProjectionStore,
    StudioProjectionExporter,
    StudioProjector,
    TimelineQuery,
    normalize_projection_id,
)
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
    run_id: str = ""


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
        self.event_store = SQLiteEventStore(self.runtime_dir / "events.sqlite3")
        self.studio_store = SQLiteStudioProjectionStore(self.runtime_dir / "studio_projection.sqlite3")
        self.studio_projector = StudioProjector(self.event_store, self.studio_store)
        self.event_recorder = EventRecorder(
            self.event_store, (StudioProjectionExporter(self.studio_store),)
        )
        self.observer = PersistentEventObserver(self.event_recorder)
        self._previous_observer = get_observer()
        set_observer(self.observer)
        self.studio_projector.sync_all()
        while self.event_store.pending_exports(exporter_name="studio_projection", limit=1000):
            outcomes = self.event_recorder.retry_pending(exporter_name="studio_projection", limit=1000)
            if not any(outcome.exported_to for outcome in outcomes):
                break
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
        if get_observer() is self.observer:
            set_observer(self._previous_observer)
        self.studio_store.close()
        self.event_store.close()

    async def _execute_run(self, handle: RunHandle) -> None:
        config = {"configurable": {"thread_id": handle.thread_id, "research_id": handle.research_id}}
        context = RunContext.from_config(config, root_query=handle.query)
        handle.run_id = context.run_id
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
            self.studio_projector.sync_run(handle.run_id)

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

    def _resolve_run_id(self, research_id: str) -> str | None:
        handle = self._get_handle(research_id)
        if handle and handle.run_id:
            return handle.run_id
        thread_id = normalize_projection_id("thread", research_id)
        page = self.studio_store.list_runs(thread_id=thread_id, limit=1000)
        return str(page.items[-1]["run_id"]) if page.items else None

    def _projected_run(self, research_id: str) -> dict[str, Any] | None:
        run_id = self._resolve_run_id(research_id)
        if not run_id:
            return None
        self.studio_projector.sync_run(run_id)
        return self.studio_store.get_run(run_id)

    def list_studio_threads(
        self, *, after_created_at: str | None = None, after_thread_id: str | None = None, limit: int = 100
    ) -> dict[str, Any]:
        self.studio_projector.sync_all()
        page = self.studio_store.list_threads(
            after_created_at=after_created_at, after_thread_id=after_thread_id, limit=limit
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def list_studio_runs(
        self, *, thread_id: str | None = None, after_started_at: str | None = None,
        after_run_id: str | None = None, limit: int = 100,
    ) -> dict[str, Any]:
        self.studio_projector.sync_all()
        page = self.studio_store.list_runs(
            thread_id=thread_id, after_started_at=after_started_at,
            after_run_id=after_run_id, limit=limit,
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def get_studio_run(self, run_id: str) -> dict[str, Any]:
        self.studio_projector.sync_run(run_id)
        run = self.studio_store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def list_studio_spans(self, run_id: str, *, after_started_sequence: int = 0, limit: int = 100) -> dict[str, Any]:
        self.studio_projector.sync_run(run_id)
        if self.studio_store.get_run(run_id) is None:
            raise KeyError(run_id)
        page = self.studio_store.list_spans(
            run_id, after_started_sequence=after_started_sequence, limit=limit
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def get_timeline_page(self, query: TimelineQuery) -> dict[str, Any]:
        self.studio_projector.sync_run(query.run_id)
        if self.studio_store.get_run(query.run_id) is None:
            raise KeyError(query.run_id)
        page = self.studio_store.timeline(query)
        return {"items": list(page.items), "next_after_sequence": page.next_after_sequence}

    def export_studio_trace(self, run_id: str) -> dict[str, Any]:
        return self.studio_projector.export_trace(run_id)

    async def list_runs(self) -> List[Dict[str, Any]]:
        rows = self.knowledge_manager.store.conn.execute(
            "SELECT research_id, session_id, root_query, status, updated_at, current_round FROM research_sessions ORDER BY updated_at DESC LIMIT 20"
        ).fetchall()
        output = []
        for row in rows:
            handle = self._get_handle(str(row["research_id"]))
            projected = self._projected_run(str(row["research_id"]))
            projected_status = {
                "succeeded": "completed",
                "failed": "failed",
                "cancelled": "cancelled",
                "running": "running",
                "queued": "initializing",
                "waiting": "waiting",
            }.get(str((projected or {}).get("status", "")), "")
            output.append(
                {
                    "research_id": str(row["research_id"]),
                    "session_id": str(row["session_id"]),
                    "query": str(row["root_query"]),
                    "status": (handle.status if handle else projected_status or str(row["status"])),
                    "run_id": (projected or {}).get("run_id", ""),
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
        projected_run = self._projected_run(research_id)
        timeline = self._build_timeline(research_id)
        status = self._derive_status(state, handle, projected_run)
        current_stage = self._derive_stage(state, handle, [item.model_dump() for item in timeline])
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
            timeline=timeline,
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
        snapshot = self.knowledge_manager.get_session_snapshot(research_id, summary.session_id)
        run_id = self._resolve_run_id(research_id)
        run = self.studio_store.get_run(run_id) if run_id else None
        spans = self.list_studio_spans(run_id, limit=1000)["items"] if run_id else []
        planner_action = None
        for event in reversed(summary.timeline):
            if event.payload.get("action"):
                planner_action = event.payload["action"]
                break
        return DebugViewResponse(
            research_id=research_id,
            session_id=summary.session_id,
            status=summary.status,
            state_summary={
                "run_id": run_id,
                "planner_action": planner_action,
                "event_count": (run or {}).get("event_count", 0),
                "span_count": len(spans),
                "terminal_event_id": (run or {}).get("terminal_event_id"),
            },
            context_summary=summary.context_summary,
            trace=summary.timeline,
            raw_state={
                "projection_schema": "StudioProjection@1",
                "run": run or {},
                "spans": spans,
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

    def _build_timeline(self, research_id: str) -> List[TimelineEventSummary]:
        projected = self._projected_run(research_id)
        if not projected:
            return []
        after = max(0, int(projected["event_count"]) - 200)
        page = self.studio_store.timeline(TimelineQuery(projected["run_id"], after_sequence=after, limit=200))
        events = []
        for item in page.items:
            payload = dict(item.get("payload", {}) or {})
            permissions = {
                key: payload[key]
                for key in ("permission", "permissions", "approval", "risk_level", "policy_decision")
                if key in payload
            }
            events.append(TimelineEventSummary(
                event_id=str(item.get("event_id", "")), event_type=str(item.get("event_type", "")),
                timestamp=str(item.get("occurred_at", "")), level=str(item.get("level", "info")),
                message=str(payload.get("message", "")), node_name=payload.get("node_name"),
                agent_name=str(item.get("actor_id", "")), task_id=item.get("task_id"),
                section_id=payload.get("section_id"), payload=payload,
                sequence_no=int(item.get("sequence_no", 0)), run_id=str(item.get("run_id", "")),
                trace_id=str(item.get("trace_id", "")), span_id=str(item.get("span_id", "")),
                parent_span_id=item.get("parent_span_id"), span_kind=str(item.get("span_kind", "")),
                actor_id=str(item.get("actor_id", "")), status=str(item.get("status", "")),
                input_artifact_ids=list(item.get("input_artifact_ids", []) or []),
                output_artifact_ids=list(item.get("output_artifact_ids", []) or []),
                state_artifact_id=item.get("state_artifact_id"), usage=dict(item.get("usage", {}) or {}),
                latency_ms=float(item.get("latency_ms", 0.0)), attempt=int(item.get("attempt", 1)),
                error=item.get("error"), component_versions=dict(item.get("component_versions", {}) or {}),
                permissions=permissions,
            ))
        return events

    def _derive_status(self, state: Dict[str, Any], handle: Optional[RunHandle], projected_run: dict[str, Any] | None = None) -> str:
        if handle and handle.status in {"initializing", "running", "failed", "cancelled", "completed"}:
            return handle.status
        if projected_run:
            projected_status = {
                "succeeded": "completed", "failed": "failed", "cancelled": "cancelled",
                "running": "running", "queued": "initializing", "waiting": "waiting",
            }.get(str(projected_run.get("status")))
            if projected_status:
                return projected_status
        if state.get("final_report"):
            return "completed"
        if state.get("error_state"):
            return "failed"
        return "idle"

    def _derive_stage(self, state: Dict[str, Any], handle: Optional[RunHandle], observer_events: List[Dict[str, Any]]) -> str:
        if state.get("final_report"):
            return "completed"
        if handle and handle.status == "completed":
            return "completed"
        if handle and handle.status == "failed":
            return "failed"
        if observer_events:
            latest = observer_events[-1].get("event_type", "")
            if latest == "run_completed":
                return "completed"
            if latest == "run_failed":
                return "failed"
            if latest == "report_changed":
                return "writing"
            if latest in {"evidence_changed", "verification_completed"}:
                return "knowledge_updating"
            if latest in {"tool_started", "tool_completed", "model_started", "model_completed", "decision_recorded"}:
                return "researching"
            if latest in {"task_created", "task_state_changed", "span_started", "span_completed"}:
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
