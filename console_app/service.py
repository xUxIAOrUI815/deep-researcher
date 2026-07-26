from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from deep_researcher.application import (
    ActiveAgentSummary,
    ApplicationRunRecord,
    ApplicationRuntime,
    ConsoleRunSummary,
    ContextPanelSummary,
    DebugViewResponse,
    KnowledgeSummary,
    ReportViewResponse,
    ResearchCreateRequest,
    ResearchCreateResponse,
    TimelineEventSummary,
    build_live_application_runtime,
)
from deep_researcher.artifacts import ArtifactQuery
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentVersionSet,
    SectionCoverageStatus,
    TaskStatus,
)
from deep_researcher.events import EventQuery
from deep_researcher.studio import (
    ReplayMode,
    ReplayRequestStatus,
    TimelineQuery,
)


class ResearchConsoleService:
    """Projection-only Console/Studio facade over ApplicationRuntime."""

    def __init__(
        self,
        runtime_dir: str = ".console_runtime",
        *,
        runtime: ApplicationRuntime | None = None,
    ) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.runtime = runtime or build_live_application_runtime(
            self.runtime_dir
        )
        self._owns_runtime = runtime is None
        self._run_tasks: dict[str, asyncio.Task[Any]] = {}
        self._closed = False

    @property
    def event_store(self):
        return self.runtime.event_store

    @property
    def studio_store(self):
        return self.runtime.studio_store

    @property
    def studio_projector(self):
        return self.runtime.studio_projector

    @property
    def studio_v2(self):
        return self.runtime.studio_v2

    @property
    def studio_advanced(self):
        return self.runtime.studio_advanced

    @property
    def studio_advanced_store(self):
        return self.runtime.studio_advanced_store

    async def start(self) -> None:
        for record in self.runtime.recoverable_runs():
            self._schedule(record.research_id)

    async def create_run(
        self,
        request: ResearchCreateRequest,
    ) -> ResearchCreateResponse:
        record = self.runtime.new_run(request)
        self._schedule(record.research_id)
        return ResearchCreateResponse(
            research_id=record.research_id,
            thread_id=record.thread_id,
            session_id=record.session_id,
            status=record.status.value,
            console_url=f"/console/{record.research_id}",
            report_url=f"/report/{record.research_id}",
        )

    async def approve_run(
        self,
        research_id: str,
        *,
        approved_by: str,
        note: str,
    ) -> dict[str, Any]:
        record = await self.runtime.approve_run(
            research_id,
            approved_by=approved_by,
            note=note,
        )
        self._schedule(record.research_id)
        return record.model_dump(mode="json")

    async def cancel_run(
        self,
        research_id: str,
        *,
        reason: str,
    ) -> dict[str, Any]:
        record = await self.runtime.cancel_run(research_id, reason=reason)
        return record.model_dump(mode="json")

    def _schedule(self, research_id: str) -> None:
        existing = self._run_tasks.get(research_id)
        if existing is not None and not existing.done():
            return
        task = asyncio.get_running_loop().create_task(
            self.runtime.execute(research_id)
        )
        self._run_tasks[research_id] = task

        def completed(done: asyncio.Task[Any]) -> None:
            self._run_tasks.pop(research_id, None)
            if not done.cancelled():
                done.exception()

        task.add_done_callback(completed)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        pending = [item for item in self._run_tasks.values() if not item.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        self._run_tasks.clear()
        if self._owns_runtime:
            await self.runtime.aclose()

    async def list_runs(self) -> list[dict[str, Any]]:
        return [
            {
                "research_id": item.research_id,
                "session_id": item.session_id,
                "query": item.query,
                "status": item.status.value,
                "run_id": item.run_id,
                "current_round": self._current_round(item.run_id),
                "updated_at": item.updated_at.isoformat(),
                "console_url": f"/console/{item.research_id}",
                "report_url": f"/report/{item.research_id}",
            }
            for item in self.runtime.application_store.list(limit=100)
        ]

    async def get_console_summary(
        self,
        research_id: str,
    ) -> ConsoleRunSummary:
        record = self._record(research_id)
        try:
            snapshot = await self.runtime.scheduler.snapshot(record.run_id)
            task_records = snapshot.tasks
        except KeyError:
            snapshot = None
            task_records = ()
        repository = self.runtime.knowledge_repository
        sources = repository.sources.list(record.run_id)
        claims = repository.claims.list(record.run_id)
        facts = repository.facts.list(record.run_id)
        evidence = repository.evidence.list(record.run_id)
        conflicts = repository.conflicts.list(record.run_id)
        sections = sorted(
            repository.sections.list(record.run_id),
            key=lambda item: (item.order, item.section_id),
        )
        report = repository.reports.get(record.report_id)
        decisions = self.runtime.research_store.convergence_decisions(
            record.run_id
        )
        latest_decision = decisions[-1] if decisions else None
        active = next(
            (
                item
                for item in task_records
                if item.envelope.status
                in {
                    TaskStatus.RUNNING,
                    TaskStatus.WAITING_APPROVAL,
                    TaskStatus.READY,
                }
            ),
            None,
        )
        task_tree = {
            item.task_id: {
                **item.envelope.model_dump(mode="json"),
                "lease_owner": item.lease_owner,
                "lease_expires_at": (
                    item.lease_expires_at.isoformat()
                    if item.lease_expires_at is not None
                    else None
                ),
                "budget_usage": item.budget_usage.model_dump(mode="json"),
                "result_id": item.result_id,
                "output_artifact_ids": list(item.output_artifact_ids),
                "error_ref": item.error_ref,
            }
            for item in task_records
        }
        outline = {
            "report_id": record.report_id,
            "title": report.title if report is not None else record.query,
            "sections": [
                {
                    "section_id": item.section_id,
                    "title": item.title,
                    "goal": item.goal,
                    "order": item.order,
                    "claim_ids": list(item.claim_ids),
                    "required_claim_ids": list(item.required_claim_ids),
                    "coverage_score": item.coverage_score,
                    "citation_score": item.citation_score,
                    "coverage_status": item.coverage_status.value,
                }
                for item in sections
            ],
        }
        gaps = [
            {
                "section_id": item.section_id,
                "title": item.title,
                "coverage_status": item.coverage_status.value,
                "coverage_score": item.coverage_score,
                "citation_score": item.citation_score,
                "unsupported_claim_ids": list(item.unsupported_claim_ids),
            }
            for item in sections
            if item.required_claim_ids
            and item.coverage_status != SectionCoverageStatus.COMPLETE
        ]
        packs = self._artifact_payloads(
            record.run_id,
            ArtifactKind.EVIDENCE_PACK,
        )
        timeline = self._timeline(record.run_id)
        current_round = latest_decision.cycle + 1 if latest_decision else 0
        planner_state = (
            latest_decision.model_dump(mode="json")
            if latest_decision is not None
            else {}
        )
        active_name = self._active_agent(record.current_stage, active)
        latest_coverage = (
            {
                "required_section_ids": list(
                    latest_decision.snapshot.required_section_ids
                ),
                "complete_section_ids": list(
                    latest_decision.snapshot.complete_section_ids
                ),
                "coverage_gap_section_ids": list(
                    latest_decision.snapshot.coverage_gap_section_ids
                ),
                "blocked_high_impact_claim_ids": list(
                    latest_decision.snapshot.blocked_high_impact_claim_ids
                ),
                "severe_conflict_ids": list(
                    latest_decision.snapshot.severe_conflict_ids
                ),
            }
            if latest_decision is not None
            else None
        )
        context = ContextPanelSummary(
            planner={
                "required_sections": (
                    list(latest_decision.snapshot.required_section_ids)
                    if latest_decision
                    else []
                ),
                "gap_count": len(gaps),
                "conflict_count": len(conflicts),
                "decision": (
                    latest_decision.action.value if latest_decision else None
                ),
            },
            researcher={
                "source_count": len(sources),
                "candidate_claim_count": len(claims),
                "task_count": len(task_records),
                "active_task_id": active.task_id if active else None,
            },
            writer={
                "verified_claim_count": len(
                    [
                        item
                        for item in claims
                        if item.status.value == "supported"
                    ]
                ),
                "section_count": len(sections),
                "report_artifact_id": record.report_artifact_id,
            },
        )
        elapsed = max(
            0.0,
            (datetime.now(timezone.utc) - record.created_at).total_seconds(),
        )
        return ConsoleRunSummary(
            research_id=record.research_id,
            thread_id=record.thread_id,
            session_id=record.session_id,
            query=record.query,
            status=record.status.value,
            current_stage=record.current_stage,
            current_round=current_round,
            elapsed_seconds=elapsed,
            resumed=record.resumed,
            has_report=record.report_artifact_id is not None,
            root_task_id=record.root_task_id,
            active_task_id=active.task_id if active else None,
            planner_state=planner_state,
            report_outline=outline,
            task_tree=task_tree,
            timeline=timeline,
            knowledge_summary=KnowledgeSummary(
                source_count=len(sources),
                claim_count=len(claims),
                fact_count=len(facts),
                evidence_count=len(evidence),
                conflict_count=len(conflicts),
                open_gap_count=len(gaps),
                section_pack_count=len(packs),
            ),
            latest_coverage_snapshot=latest_coverage,
            open_gaps=gaps,
            conflicts=[
                item.model_dump(mode="json") for item in conflicts
            ],
            section_packs=packs,
            sources=[item.model_dump(mode="json") for item in sources],
            active_agent=ActiveAgentSummary(
                name=active_name,
                status=record.current_stage,
                target=active.envelope.title if active else "",
                last_output_summary=(
                    "; ".join(latest_decision.reasons)
                    if latest_decision
                    else ""
                )[:240],
            ),
            context_summary=context,
            run_metadata={
                **record.model_dump(mode="json"),
                "scheduler_status": (
                    snapshot.control.status.value if snapshot else "not_created"
                ),
                "projection_revision": (
                    snapshot.control.projection_revision if snapshot else 0
                ),
            },
        )

    async def get_report_view(self, research_id: str) -> ReportViewResponse:
        summary = await self.get_console_summary(research_id)
        record = self._record(research_id)
        markdown = ""
        if record.report_artifact_id is not None:
            markdown = self.runtime.artifact_store.read_bytes(
                record.report_artifact_id
            ).decode("utf-8")
        revisions = self.runtime.reporting_store.revisions(
            record.run_id,
            record.report_id,
        )
        latest = revisions[-1].model_dump(mode="json") if revisions else {}
        return ReportViewResponse(
            research_id=record.research_id,
            session_id=record.session_id,
            query=record.query,
            status=record.status.value,
            title=str(summary.report_outline.get("title") or record.query),
            markdown=markdown,
            outline=summary.report_outline,
            report=latest,
            knowledge_summary=summary.knowledge_summary,
            latest_coverage_snapshot=summary.latest_coverage_snapshot,
            open_gaps=summary.open_gaps,
            section_packs=summary.section_packs,
            context_summary=summary.context_summary,
        )

    async def get_debug_view(self, research_id: str) -> DebugViewResponse:
        summary = await self.get_console_summary(research_id)
        record = self._record(research_id)
        run = self.get_studio_run(record.run_id)
        spans = self.list_studio_spans(
            record.run_id,
            limit=1000,
        )["items"]
        return DebugViewResponse(
            research_id=record.research_id,
            session_id=record.session_id,
            status=record.status.value,
            state_summary={
                "run_id": record.run_id,
                "stage": record.current_stage,
                "event_count": run.get("event_count", 0),
                "span_count": len(spans),
                "terminal_event_id": run.get("terminal_event_id"),
                "application_revision": record.revision,
            },
            context_summary=summary.context_summary,
            trace=summary.timeline,
            raw_state={
                "projection_schema": "StudioProjection@1",
                "run": run,
                "spans": spans,
            },
            snapshot_summary={
                "application": record.model_dump(mode="json"),
                "knowledge": summary.knowledge_summary.model_dump(mode="json"),
                "coverage": summary.latest_coverage_snapshot or {},
            },
        )

    def list_studio_threads(
        self,
        *,
        after_created_at: str | None = None,
        after_thread_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.studio_projector.sync_all()
        page = self.studio_store.list_threads(
            after_created_at=after_created_at,
            after_thread_id=after_thread_id,
            limit=limit,
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def list_studio_runs(
        self,
        *,
        thread_id: str | None = None,
        after_started_at: str | None = None,
        after_run_id: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.studio_projector.sync_all()
        page = self.studio_store.list_runs(
            thread_id=thread_id,
            after_started_at=after_started_at,
            after_run_id=after_run_id,
            limit=limit,
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def get_studio_run(self, run_id: str) -> dict[str, Any]:
        self.studio_projector.sync_run(run_id)
        run = self.studio_store.get_run(run_id)
        if run is None:
            raise KeyError(run_id)
        return run

    def list_studio_spans(
        self,
        run_id: str,
        *,
        after_started_sequence: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        self.studio_projector.sync_run(run_id)
        if self.studio_store.get_run(run_id) is None:
            raise KeyError(run_id)
        page = self.studio_store.list_spans(
            run_id,
            after_started_sequence=after_started_sequence,
            limit=limit,
        )
        return {"items": list(page.items), "next_cursor": page.next_cursor}

    def get_timeline_page(self, query: TimelineQuery) -> dict[str, Any]:
        self.studio_projector.sync_run(query.run_id)
        if self.studio_store.get_run(query.run_id) is None:
            raise KeyError(query.run_id)
        page = self.studio_store.timeline(query)
        return {
            "items": list(page.items),
            "next_after_sequence": page.next_after_sequence,
        }

    def export_studio_trace(self, run_id: str) -> dict[str, Any]:
        return self.studio_projector.export_trace(run_id)

    def get_studio_v2_task_graph(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.studio_v2.task_graph(run_id, **kwargs).model_dump(mode="json")

    def get_studio_v2_evidence_graph(
        self,
        run_id: str,
        *,
        entity_types: tuple[str, ...] = (),
        **kwargs: Any,
    ) -> dict[str, Any]:
        if entity_types:
            kwargs["entity_types"] = entity_types
        return self.studio_v2.evidence_graph(
            run_id,
            **kwargs,
        ).model_dump(mode="json")

    def get_studio_v2_state_diff(
        self,
        run_id: str,
        *,
        domain: str,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if domain == "scheduler":
            page = self.studio_v2.scheduler_state_diffs(
                run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        elif domain == "evidence":
            page = self.studio_v2.evidence_state_diffs(
                run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        else:
            raise ValueError("domain must be scheduler or evidence")
        return page.model_dump(mode="json")

    def get_studio_v2_errors(
        self,
        run_id: str,
        *,
        domain: str = "runtime",
        after_sequence: int = 0,
        limit: int = 100,
    ) -> dict[str, Any]:
        if domain == "runtime":
            page = self.studio_v2.error_retry_chain(
                run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        elif domain == "scheduler":
            page = self.studio_v2.scheduler_error_retry_chain(
                run_id,
                after_sequence=after_sequence,
                limit=limit,
            )
        else:
            raise ValueError("domain must be runtime or scheduler")
        return page.model_dump(mode="json")

    def get_studio_v2_conflicts(self, run_id: str, **kwargs: Any) -> dict[str, Any]:
        return self.studio_v2.conflict_navigation(
            run_id,
            **kwargs,
        ).model_dump(mode="json")

    def get_studio_v2_components(self, run_id: str) -> list[dict[str, Any]]:
        return [
            item.model_dump(mode="json")
            for item in self.studio_v2.component_versions(run_id)
        ]

    def get_studio_v2_metrics(self, run_id: str) -> dict[str, Any]:
        return self.studio_v2.metrics(run_id).model_dump(mode="json")

    def get_studio_component_selection(self, run_id: str) -> dict[str, Any]:
        page = self.event_store.list(EventQuery(run_id, limit=1))
        if not page.items:
            raise KeyError(run_id)
        return page.items[0].component_versions.model_dump(mode="json")

    def get_studio_replay_eligibility(
        self,
        run_id: str,
        span_id: str,
    ) -> dict[str, Any]:
        return self.studio_advanced.replay_eligibility(run_id, span_id)

    def prepare_studio_replay(
        self,
        *,
        run_id: str,
        span_id: str,
        mode: str,
        selected_component_versions: dict[str, Any],
        requested_by: str,
        reason: str,
        restart_failed_span: bool,
        environment_label: str | None,
    ) -> dict[str, Any]:
        return self.studio_advanced.prepare_replay(
            source_run_id=run_id,
            source_span_id=span_id,
            mode=ReplayMode(mode),
            selected_component_versions=ComponentVersionSet.model_validate(
                selected_component_versions,
                strict=False,
            ),
            requested_by=requested_by,
            reason=reason,
            restart_failed_span=restart_failed_span,
            environment_label=environment_label,
        ).model_dump(mode="json")

    def list_studio_replays(
        self,
        *,
        statuses: tuple[str, ...] = (),
        cursor: str | None = None,
        limit: int = 100,
    ) -> dict[str, Any]:
        return self.studio_advanced_store.list(
            statuses=tuple(ReplayRequestStatus(item) for item in statuses),
            cursor=cursor,
            limit=limit,
        ).model_dump(mode="json")

    def get_studio_replay(self, request_id: str) -> dict[str, Any]:
        record = self.studio_advanced_store.get(request_id)
        if record is None:
            raise KeyError(request_id)
        return record.model_dump(mode="json")

    def approve_studio_replay(self, **kwargs: Any) -> dict[str, Any]:
        return self.studio_advanced.approve_replay(
            replay_request_id=kwargs.pop("request_id"),
            **kwargs,
        ).model_dump(mode="json")

    async def execute_studio_replay(self, request_id: str) -> dict[str, Any]:
        return (
            await self.studio_advanced.execute_replay(request_id)
        ).model_dump(mode="json")

    def compare_studio_runs(self, **kwargs: Any) -> dict[str, Any]:
        return self.studio_advanced.compare_runs(**kwargs).model_dump(mode="json")

    def get_studio_comparison(self, comparison_id: str) -> dict[str, Any]:
        value = self.studio_advanced_store.comparison(comparison_id)
        if value is None:
            raise KeyError(comparison_id)
        return value.model_dump(mode="json")

    def get_studio_component_diff(
        self,
        left_version_id: str,
        right_version_id: str,
    ) -> dict[str, Any]:
        return self.studio_advanced.component_diff(
            left_version_id,
            right_version_id,
        ).model_dump(mode="json")

    def create_studio_badcase(self, **kwargs: Any) -> dict[str, Any]:
        return self.studio_advanced.create_badcase(
            **kwargs
        ).model_dump(mode="json")

    def get_studio_badcase(self, badcase_id: str) -> dict[str, Any]:
        value = self.studio_advanced_store.badcase(badcase_id)
        if value is None:
            raise KeyError(badcase_id)
        return value.model_dump(mode="json")

    def _record(self, research_id: str) -> ApplicationRunRecord:
        record = self.runtime.application_store.get(research_id)
        if record is None:
            raise KeyError(research_id)
        return record

    def _current_round(self, run_id: str) -> int:
        decisions = self.runtime.research_store.convergence_decisions(run_id)
        return decisions[-1].cycle + 1 if decisions else 0

    def _artifact_payloads(
        self,
        run_id: str,
        kind: ArtifactKind,
    ) -> list[dict[str, Any]]:
        page = self.runtime.artifact_store.list(
            ArtifactQuery(run_id=run_id, kinds=(kind,), limit=1000)
        )
        output: list[dict[str, Any]] = []
        for item in page.items:
            try:
                value = json.loads(
                    self.runtime.artifact_store.read_bytes(
                        item.artifact_id
                    ).decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            if isinstance(value, dict):
                output.append(value)
        return output

    def _timeline(self, run_id: str) -> list[TimelineEventSummary]:
        run = self.runtime.studio_store.get_run(run_id)
        if run is None:
            self.runtime.studio_projector.sync_run(run_id)
            run = self.runtime.studio_store.get_run(run_id)
        if run is None:
            return []
        after = max(0, int(run["event_count"]) - 500)
        page = self.runtime.studio_store.timeline(
            TimelineQuery(run_id, after_sequence=after, limit=500)
        )
        return [
            TimelineEventSummary(
                event_id=str(item.get("event_id", "")),
                event_type=str(item.get("event_type", "")),
                timestamp=str(item.get("occurred_at", "")),
                level=str(item.get("level", "info")),
                message=str((item.get("payload") or {}).get("message", "")),
                node_name=(item.get("payload") or {}).get("node_name"),
                agent_name=str(item.get("actor_id", "")),
                task_id=item.get("task_id"),
                section_id=(item.get("payload") or {}).get("section_id"),
                payload=dict(item.get("payload") or {}),
                sequence_no=int(item.get("sequence_no", 0)),
                run_id=str(item.get("run_id", "")),
                trace_id=str(item.get("trace_id", "")),
                span_id=str(item.get("span_id", "")),
                parent_span_id=item.get("parent_span_id"),
                span_kind=str(item.get("span_kind", "")),
                actor_id=str(item.get("actor_id", "")),
                status=str(item.get("status", "")),
                input_artifact_ids=list(
                    item.get("input_artifact_ids", []) or []
                ),
                output_artifact_ids=list(
                    item.get("output_artifact_ids", []) or []
                ),
                state_artifact_id=item.get("state_artifact_id"),
                usage=dict(item.get("usage") or {}),
                latency_ms=float(item.get("latency_ms", 0.0)),
                attempt=int(item.get("attempt", 1)),
                error=item.get("error"),
                component_versions=dict(
                    item.get("component_versions") or {}
                ),
                permissions={
                    key: (item.get("payload") or {})[key]
                    for key in (
                        "permission",
                        "permissions",
                        "approval",
                        "risk_level",
                        "policy_decision",
                    )
                    if key in (item.get("payload") or {})
                },
            )
            for item in page.items
        ]

    @staticmethod
    def _active_agent(stage: str, active: Any) -> str:
        if active is not None and active.envelope.assigned_actor_id:
            return str(active.envelope.assigned_actor_id)
        return {
            "initializing": "runtime_application",
            "researching": "research_supervisor",
            "reporting": "synthesis_writer",
            "waiting_approval": "human_approval",
            "completed": "report_reviewer",
            "failed": "runtime_application",
            "cancelled": "runtime_application",
        }.get(stage, "runtime_application")
