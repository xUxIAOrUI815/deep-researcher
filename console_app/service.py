from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

from deep_researcher.application import (
    ApplicationRunRecord,
    ApplicationRuntime,
    ConsoleActionsView,
    ConsoleApprovalView,
    ConsoleCausalErrorView,
    ConsoleCitationView,
    ConsoleConflictView,
    ConsoleCoverageView,
    ConsoleEvidencePacketView,
    ConsoleEvidenceView,
    ConsoleGapView,
    ConsoleIdentityView,
    ConsoleNavigationView,
    ConsoleProgressStep,
    ConsoleRepairActionView,
    ConsoleReportingView,
    ConsoleReportOutcomeView,
    ConsoleRevisionView,
    ConsoleReviewFindingView,
    ConsoleReviewScoreView,
    ConsoleReviewView,
    ConsoleRoleView,
    ConsoleRunListItem,
    ConsoleRuntimeView,
    ConsoleSchedulerView,
    ConsoleSectionView,
    ConsoleSourceView,
    ConsoleTaskView,
    ConsoleVerifiedClaimView,
    ConsoleWorkspaceResponse,
    DebugViewResponse,
    KnowledgeSummary,
    ReportWorkspaceResponse,
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

    def system_status(self) -> dict[str, Any]:
        model_configured = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
        search_providers = {
            "tavily": bool(os.getenv("TAVILY_API_KEY", "").strip()),
            "exa": bool(os.getenv("EXA_API_KEY", "").strip()),
        }
        return {
            "status": "ok",
            "runtime": "Background001NativeRuntime",
            "console_schema_version": "ConsoleWorkspace@2",
            "live_providers": {
                "model_configured": model_configured,
                "search_configured": any(search_providers.values()),
                "search_providers": search_providers,
                "scraper_mode": os.getenv(
                    "RESEARCHER_SCRAPER_MODE",
                    "live",
                ).strip()
                or "live",
            },
        }

    async def list_runs(self) -> list[ConsoleRunListItem]:
        return [
            ConsoleRunListItem(
                research_id=item.research_id,
                run_id=item.run_id,
                session_id=item.session_id,
                query=item.query,
                depth=item.depth,
                status=item.status.value,
                current_stage=item.current_stage,
                current_round=self._current_round(item.run_id),
                has_report=item.report_artifact_id is not None,
                resumed=item.resumed,
                created_at=item.created_at.isoformat(),
                updated_at=item.updated_at.isoformat(),
                console_url=f"/console/{item.research_id}",
                report_url=f"/report/{item.research_id}",
            )
            for item in self.runtime.application_store.list(limit=100)
        ]

    async def get_console_workspace(
        self,
        research_id: str,
    ) -> ConsoleWorkspaceResponse:
        record = self._record(research_id)
        try:
            snapshot = await self.runtime.scheduler.snapshot(record.run_id)
            task_records = tuple(snapshot.tasks)
        except KeyError:
            snapshot = None
            task_records = ()

        repository = self.runtime.knowledge_repository
        sources = tuple(repository.sources.list(record.run_id))
        claims = tuple(repository.claims.list(record.run_id))
        facts = tuple(repository.facts.list(record.run_id))
        evidence_records = tuple(repository.evidence.list(record.run_id))
        conflicts = tuple(repository.conflicts.list(record.run_id))
        sections = tuple(
            sorted(
                repository.sections.list(record.run_id),
                key=lambda item: (item.order, item.section_id),
            )
        )
        report = repository.reports.get(record.report_id)
        decisions = self.runtime.research_store.convergence_decisions(
            record.run_id
        )
        latest_decision = decisions[-1] if decisions else None
        timeline = tuple(self._timeline(record.run_id))
        raw_error_refs = record.metadata.get("research_error_refs", ())
        if not isinstance(raw_error_refs, (list, tuple)):
            raw_error_refs = ()
        causal_errors = self._causal_errors(
            timeline,
            primary_error_refs=tuple(str(item) for item in raw_error_refs),
            primary_code=record.error_code,
            primary_message=record.error_message,
        )
        active = self._active_task(task_records)
        task_views = self._task_views(task_records)
        approval_views = tuple(
            task.approval
            for task in task_views
            if task.approval is not None
            and task.status == TaskStatus.WAITING_APPROVAL.value
        )
        roles, active_role_id = self._role_views(
            record=record,
            active_task=active,
            timeline=timeline,
            task_records=task_records,
            failure_role_id=next(
                (
                    self._role_id(item.actor_id)
                    for item in causal_errors
                    if item.is_primary and self._role_id(item.actor_id)
                ),
                None,
            ),
            decision_summary=(
                "; ".join(latest_decision.reasons)
                if latest_decision is not None
                else ""
            ),
        )
        progress = self._progress_steps(record, roles)
        evidence_view = self._evidence_view(
            record=record,
            sources=sources,
            claims=claims,
            facts=facts,
            evidence_records=evidence_records,
            conflicts=conflicts,
            sections=sections,
            latest_decision=latest_decision,
        )
        reporting_view = self._reporting_view(
            record=record,
            report_title=report.title if report is not None else record.query,
            sections=evidence_view.sections,
        )
        current_round = latest_decision.cycle + 1 if latest_decision else 0
        elapsed = max(
            0.0,
            (datetime.now(timezone.utc) - record.created_at).total_seconds(),
        )
        terminal = record.status.value in {
            "completed",
            "failed",
            "cancelled",
        }
        waiting_ids = tuple(item.task_id for item in approval_views)
        scheduler_counts = Counter(item.status for item in task_views)

        return ConsoleWorkspaceResponse(
            identity=ConsoleIdentityView(
                research_id=record.research_id,
                thread_id=record.thread_id,
                session_id=record.session_id,
                run_id=record.run_id,
                trace_id=record.trace_id,
                root_task_id=record.root_task_id,
                report_id=record.report_id,
                query=record.query,
                instructions=record.instructions,
                depth=record.depth,
                created_at=record.created_at.isoformat(),
                updated_at=record.updated_at.isoformat(),
                resumed=record.resumed,
                has_report=record.report_artifact_id is not None,
            ),
            runtime=ConsoleRuntimeView(
                status=record.status.value,
                current_stage=record.current_stage,
                current_round=current_round,
                elapsed_seconds=elapsed,
                active_role_id=active_role_id,
                active_task_id=active.task_id if active is not None else None,
                decision=(
                    latest_decision.action.value
                    if latest_decision is not None
                    else None
                ),
                decision_reasons=(
                    tuple(latest_decision.reasons)
                    if latest_decision is not None
                    else ()
                ),
                error_code=(
                    record.error_code
                    or (
                        causal_errors[0].code
                        if record.status.value == "failed" and causal_errors
                        else None
                    )
                ),
                error_message=(
                    record.error_message
                    or (
                        causal_errors[0].message
                        if record.status.value == "failed" and causal_errors
                        else None
                    )
                ),
                causal_errors=causal_errors,
                progress=progress,
                roles=roles,
            ),
            actions=ConsoleActionsView(
                terminal=terminal,
                can_approve=(
                    record.status.value == "waiting_approval"
                    and bool(approval_views)
                ),
                can_cancel=record.status.value
                in {"queued", "running", "waiting_approval"},
                waiting_approval_task_ids=waiting_ids,
                approvals=approval_views,
            ),
            scheduler=ConsoleSchedulerView(
                status=(
                    snapshot.control.status.value
                    if snapshot is not None
                    else "not_created"
                ),
                projection_revision=(
                    snapshot.control.projection_revision
                    if snapshot is not None
                    else 0
                ),
                max_concurrency=(
                    snapshot.control.max_concurrency
                    if snapshot is not None
                    else 0
                ),
                cancellation_reason=(
                    snapshot.control.cancellation_reason
                    if snapshot is not None
                    else record.metadata.get("cancellation_reason")
                ),
                task_counts=dict(sorted(scheduler_counts.items())),
                active_task_ids=tuple(
                    item.task_id
                    for item in task_views
                    if item.status == TaskStatus.RUNNING.value
                ),
                ready_task_ids=tuple(
                    item.task_id
                    for item in task_views
                    if item.status == TaskStatus.READY.value
                ),
                waiting_approval_task_ids=waiting_ids,
                tasks=task_views,
            ),
            evidence=evidence_view,
            reporting=reporting_view,
            timeline=timeline,
            navigation=self._navigation(record),
        )

    async def get_report_view(
        self,
        research_id: str,
    ) -> ReportWorkspaceResponse:
        workspace = await self.get_console_workspace(research_id)
        record = self._record(research_id)
        markdown = ""
        if record.report_artifact_id is not None:
            markdown = self.runtime.artifact_store.read_bytes(
                record.report_artifact_id
            ).decode("utf-8")
        return ReportWorkspaceResponse(
            identity=workspace.identity,
            runtime=workspace.runtime,
            evidence=workspace.evidence,
            reporting=workspace.reporting,
            markdown=markdown,
            navigation=workspace.navigation,
        )

    async def get_debug_view(self, research_id: str) -> DebugViewResponse:
        workspace = await self.get_console_workspace(research_id)
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
            context_summary={
                "runtime": workspace.runtime.model_dump(mode="json"),
                "scheduler": {
                    "status": workspace.scheduler.status,
                    "projection_revision": (
                        workspace.scheduler.projection_revision
                    ),
                    "task_counts": workspace.scheduler.task_counts,
                },
                "evidence": (
                    workspace.evidence.knowledge.model_dump(mode="json")
                ),
                "reporting": {
                    "revision_count": workspace.reporting.revision_count,
                    "artifact_ready": workspace.reporting.artifact_ready,
                },
            },
            trace=list(workspace.timeline),
            raw_state={
                "projection_schema": "StudioProjection@1",
                "console_schema": workspace.schema_version,
                "run": run,
                "spans": spans,
            },
            snapshot_summary={
                "application": record.model_dump(mode="json"),
                "knowledge": (
                    workspace.evidence.knowledge.model_dump(mode="json")
                ),
                "coverage": (
                    workspace.evidence.coverage.model_dump(mode="json")
                ),
            },
        )

    @staticmethod
    def _active_task(task_records: tuple[Any, ...]) -> Any | None:
        rank = {
            TaskStatus.RUNNING: 0,
            TaskStatus.WAITING_APPROVAL: 1,
            TaskStatus.READY: 2,
        }
        candidates = [
            item
            for item in task_records
            if item.envelope.status in rank
        ]
        if not candidates:
            return None
        return sorted(
            candidates,
            key=lambda item: (
                rank[item.envelope.status],
                -item.envelope.priority,
                item.envelope.created_at,
                item.task_id,
            ),
        )[0]

    @staticmethod
    def _approval_view(item: Any) -> ConsoleApprovalView | None:
        approval = item.approval
        if approval is None:
            return None
        return ConsoleApprovalView(
            approval_id=approval.approval_id,
            task_id=item.task_id,
            task_title=item.envelope.title,
            requested_by=approval.requested_by,
            reason=approval.reason,
            status=approval.status.value,
            requested_at=approval.requested_at.isoformat(),
            resolved_by=approval.resolved_by,
            resolution_note=approval.resolution_note,
            resolved_at=(
                approval.resolved_at.isoformat()
                if approval.resolved_at is not None
                else None
            ),
        )

    def _task_views(
        self,
        task_records: tuple[Any, ...],
    ) -> tuple[ConsoleTaskView, ...]:
        by_id = {item.task_id: item for item in task_records}
        depth_cache: dict[str, int] = {}

        def depth(task_id: str, lineage: frozenset[str] = frozenset()) -> int:
            if task_id in depth_cache:
                return depth_cache[task_id]
            if task_id in lineage:
                return 0
            item = by_id[task_id]
            parent_id = item.envelope.parent_task_id
            value = (
                depth(parent_id, lineage | {task_id}) + 1
                if parent_id in by_id
                else 0
            )
            depth_cache[task_id] = value
            return value

        views = [
            ConsoleTaskView(
                task_id=item.task_id,
                parent_task_id=item.envelope.parent_task_id,
                dependency_task_ids=tuple(
                    item.envelope.dependency_task_ids
                ),
                depth=depth(item.task_id),
                kind=item.envelope.kind.value,
                status=item.envelope.status.value,
                title=item.envelope.title,
                goal=item.envelope.goal,
                constraints=dict(item.envelope.constraints),
                input_artifact_ids=tuple(
                    item.envelope.input_artifact_ids
                ),
                expected_output_schema=(
                    item.envelope.expected_output_schema
                ),
                priority=item.envelope.priority,
                deadline=(
                    item.envelope.deadline.isoformat()
                    if item.envelope.deadline is not None
                    else None
                ),
                attempt=item.envelope.attempt,
                max_attempts=item.envelope.max_attempts,
                created_by=item.envelope.created_by,
                assigned_actor_id=item.envelope.assigned_actor_id,
                tags=tuple(item.envelope.tags),
                created_at=item.envelope.created_at.isoformat(),
                updated_at=item.updated_at.isoformat(),
                budget=item.envelope.budget.model_dump(mode="json"),
                budget_usage=item.budget_usage.model_dump(mode="json"),
                lease_owner=item.lease_owner,
                lease_expires_at=(
                    item.lease_expires_at.isoformat()
                    if item.lease_expires_at is not None
                    else None
                ),
                result_id=item.result_id,
                output_artifact_ids=tuple(item.output_artifact_ids),
                error_ref=item.error_ref,
                merged_into_task_id=item.merged_into_task_id,
                defer_reason=item.defer_reason,
                pause_reason=item.pause_reason,
                approval=self._approval_view(item),
            )
            for item in task_records
        ]
        return tuple(
            sorted(
                views,
                key=lambda item: (
                    item.depth,
                    -item.priority,
                    item.created_at,
                    item.task_id,
                ),
            )
        )

    @staticmethod
    def _role_id(actor_id: str | None) -> str | None:
        value = str(actor_id or "").casefold()
        if "research_supervisor" in value:
            return "research_supervisor"
        if "research_worker" in value:
            return "research_worker_pool"
        if "evidence_verifier" in value:
            return "evidence_verifier"
        if "synthesis_writer" in value:
            return "synthesis_writer"
        if "report_reviewer" in value:
            return "report_reviewer"
        return None

    @classmethod
    def _task_role(cls, task: Any | None) -> str | None:
        if task is None:
            return None
        assigned = cls._role_id(task.envelope.assigned_actor_id)
        if assigned is not None:
            return assigned
        tags = {str(item).casefold() for item in task.envelope.tags}
        if "research_worker" in tags:
            return "research_worker_pool"
        return cls._role_id(task.envelope.created_by)

    @classmethod
    def _role_views(
        cls,
        *,
        record: ApplicationRunRecord,
        active_task: Any | None,
        timeline: tuple[TimelineEventSummary, ...],
        task_records: tuple[Any, ...],
        failure_role_id: str | None,
        decision_summary: str,
    ) -> tuple[tuple[ConsoleRoleView, ...], str | None]:
        definitions = (
            ("research_supervisor", "Research Supervisor"),
            ("research_worker_pool", "Research Worker Pool"),
            ("evidence_verifier", "Evidence Verifier"),
            ("synthesis_writer", "Synthesis Writer"),
            ("report_reviewer", "Report Reviewer"),
        )
        events: dict[str, list[TimelineEventSummary]] = {
            role_id: [] for role_id, _ in definitions
        }
        latest_terminal: dict[str, str] = {}
        for event in timeline:
            role_id = cls._role_id(event.actor_id)
            if role_id is None:
                continue
            events[role_id].append(event)
            if event.event_type in {"span_completed", "span_failed"}:
                latest_terminal[role_id] = event.event_type

        failed_task_roles = {
            cls._task_role(task)
            for task in task_records
            if task.envelope.status == TaskStatus.FAILED
        }
        cancelled_task_roles = {
            cls._task_role(task)
            for task in task_records
            if task.envelope.status == TaskStatus.CANCELLED
        }

        status = record.status.value
        terminal = status in {"completed", "failed", "cancelled"}
        active_role_id: str | None = None
        if not terminal:
            if status == "waiting_approval":
                active_role_id = cls._task_role(active_task)
            elif record.current_stage == "researching":
                candidates = [
                    event
                    for event in timeline
                    if cls._role_id(event.actor_id)
                    in {
                        "research_supervisor",
                        "research_worker_pool",
                        "evidence_verifier",
                    }
                ]
                active_role_id = (
                    cls._role_id(candidates[-1].actor_id)
                    if candidates
                    else cls._task_role(active_task)
                    or "research_supervisor"
                )
            elif record.current_stage == "reporting":
                candidates = [
                    event
                    for event in timeline
                    if cls._role_id(event.actor_id)
                    in {"synthesis_writer", "report_reviewer"}
                ]
                active_role_id = (
                    cls._role_id(candidates[-1].actor_id)
                    if candidates
                    else "synthesis_writer"
                )

        last_role = None
        observed = [
            event
            for event in timeline
            if cls._role_id(event.actor_id) is not None
        ]
        if observed:
            last_role = cls._role_id(observed[-1].actor_id)

        views: list[ConsoleRoleView] = []
        for role_id, label in definitions:
            role_events = events[role_id]
            last = role_events[-1] if role_events else None
            terminal_event = latest_terminal.get(role_id)
            if (
                terminal_event == "span_failed"
                or role_id in failed_task_roles
            ):
                role_status = "failed"
            elif status == "cancelled" and role_id in cancelled_task_roles:
                role_status = "cancelled"
            elif status == "completed":
                role_status = "completed"
            elif role_id == active_role_id:
                role_status = (
                    "blocked"
                    if status == "waiting_approval"
                    else "active"
                )
            elif status == "failed" and role_id == failure_role_id:
                role_status = "failed"
            elif status == "cancelled" and role_id == last_role:
                role_status = "cancelled"
            elif role_events and (
                terminal_event == "span_completed"
                or active_role_id is not None
                or terminal
            ):
                role_status = "completed"
            else:
                role_status = "waiting"
            views.append(
                ConsoleRoleView(
                    role_id=role_id,
                    label=label,
                    status=role_status,
                    task_id=(
                        active_task.task_id
                        if role_id == active_role_id
                        and active_task is not None
                        else None
                    ),
                    target=(
                        active_task.envelope.title
                        if role_id == active_role_id
                        and active_task is not None
                        else ""
                    ),
                    last_event_sequence=(
                        last.sequence_no if last is not None else 0
                    ),
                    last_event_type=(
                        last.event_type if last is not None else None
                    ),
                    last_output_summary=(
                        decision_summary[:500]
                        if role_id == "research_supervisor"
                        else ""
                    ),
                )
            )
        return tuple(views), active_role_id

    @staticmethod
    def _causal_errors(
        timeline: tuple[TimelineEventSummary, ...],
        *,
        primary_error_refs: tuple[str, ...] = (),
        primary_code: str | None = None,
        primary_message: str | None = None,
    ) -> tuple[ConsoleCausalErrorView, ...]:
        values: list[ConsoleCausalErrorView] = []
        seen: set[tuple[str, str, str | None]] = set()
        for event in sorted(timeline, key=lambda item: item.sequence_no):
            payload = event.payload if isinstance(event.payload, dict) else {}
            candidate = payload.get("error")
            if not isinstance(candidate, dict):
                candidate = event.error if isinstance(event.error, dict) else None
            kernel_type = str(payload.get("kernel_event_type") or "")
            if candidate is None and kernel_type == "command.schema_invalid":
                raw_errors = payload.get("errors", ())
                if isinstance(raw_errors, dict):
                    raw_errors = raw_errors.get("value", ())
                if not isinstance(raw_errors, (list, tuple)):
                    raw_errors = (raw_errors,)
                candidate = {
                    "category": "schema_validation",
                    "code": "command_schema_invalid",
                    "message": "; ".join(
                        str(item) for item in raw_errors if str(item)
                    )
                    or "Worker command schema validation failed.",
                    "retryable": True,
                    "fatal": False,
                    "task_id": event.task_id,
                }
            if candidate is None:
                continue
            code = str(candidate.get("code") or "runtime_error")
            message = str(candidate.get("message") or "Runtime error.")
            task_id = candidate.get("task_id") or event.task_id
            fingerprint = (code, message, task_id)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            values.append(
                ConsoleCausalErrorView(
                    sequence_no=event.sequence_no,
                    error_id=(
                        str(candidate.get("error_id"))
                        if candidate.get("error_id")
                        else None
                    ),
                    event_type=event.event_type,
                    actor_id=str(
                        candidate.get("actor_id") or event.actor_id or ""
                    ),
                    task_id=str(task_id) if task_id is not None else None,
                    category=str(candidate.get("category") or "internal"),
                    code=code,
                    message=message,
                    retryable=bool(candidate.get("retryable", False)),
                    fatal=bool(candidate.get("fatal", False)),
                    is_primary=False,
                )
            )
        if not values:
            return ()
        authoritative_refs = set(primary_error_refs)
        primary_index = next(
            (
                index
                for index, item in enumerate(values)
                if item.error_id in authoritative_refs
            ),
            None,
        )
        if primary_index is None and primary_code and primary_message:
            primary_index = next(
                (
                    index
                    for index, item in enumerate(values)
                    if item.code == primary_code
                    and item.message == primary_message
                ),
                None,
            )
        if primary_index is None and primary_code:
            primary_index = next(
                (
                    index
                    for index, item in enumerate(values)
                    if item.code == primary_code
                ),
                None,
            )
        if (
            primary_index is not None
            and primary_code.startswith("application_")
        ):
            # The terminal application wrapper is useful for machine status,
            # but an earlier non-retryable agent failure explains what the
            # user can actually fix.  Prefer that causal event in the console.
            primary_index = next(
                (
                    index
                    for index in range(len(values) - 1, -1, -1)
                    if not values[index].code.startswith("application_")
                    and values[index].fatal
                    and not values[index].retryable
                ),
                primary_index,
            )
        if primary_index is None:
            primary_index = 0
        primary = values.pop(primary_index).model_copy(
            update={"is_primary": True}
        )
        return (primary, *values)

    @staticmethod
    def _progress_steps(
        record: ApplicationRunRecord,
        roles: tuple[ConsoleRoleView, ...],
    ) -> tuple[ConsoleProgressStep, ...]:
        role_status = {item.role_id: item.status for item in roles}

        def aggregate(role_ids: tuple[str, ...]) -> str:
            values = [role_status[item] for item in role_ids]
            for candidate in ("failed", "cancelled", "blocked", "active"):
                if candidate in values:
                    return candidate
            if values and all(item == "completed" for item in values):
                return "completed"
            return "waiting"

        queued_status = (
            "active"
            if record.status.value == "queued"
            else (
                "cancelled"
                if record.status.value == "cancelled"
                and record.current_stage in {"queued", "initializing"}
                else "completed"
            )
        )
        complete_status = {
            "completed": "completed",
            "failed": "failed",
            "cancelled": "cancelled",
        }.get(record.status.value, "waiting")
        return (
            ConsoleProgressStep(
                step_id="queued",
                label="Runtime queued",
                status=queued_status,
            ),
            ConsoleProgressStep(
                step_id="research",
                label="Plan and research",
                status=aggregate(
                    ("research_supervisor", "research_worker_pool")
                ),
                role_ids=(
                    "research_supervisor",
                    "research_worker_pool",
                ),
            ),
            ConsoleProgressStep(
                step_id="verification",
                label="Independent verification",
                status=aggregate(("evidence_verifier",)),
                role_ids=("evidence_verifier",),
            ),
            ConsoleProgressStep(
                step_id="synthesis",
                label="Evidence-only synthesis",
                status=aggregate(("synthesis_writer",)),
                role_ids=("synthesis_writer",),
            ),
            ConsoleProgressStep(
                step_id="review",
                label="Report review",
                status=aggregate(("report_reviewer",)),
                role_ids=("report_reviewer",),
            ),
            ConsoleProgressStep(
                step_id="complete",
                label="Terminal outcome",
                status=complete_status,
            ),
        )

    def _evidence_view(
        self,
        *,
        record: ApplicationRunRecord,
        sources: tuple[Any, ...],
        claims: tuple[Any, ...],
        facts: tuple[Any, ...],
        evidence_records: tuple[Any, ...],
        conflicts: tuple[Any, ...],
        sections: tuple[Any, ...],
        latest_decision: Any | None,
    ) -> ConsoleEvidenceView:
        section_views = tuple(
            ConsoleSectionView(
                section_id=item.section_id,
                parent_section_id=item.parent_section_id,
                title=item.title,
                goal=item.goal,
                order=item.order,
                status=item.status.value,
                coverage_status=item.coverage_status.value,
                coverage_score=item.coverage_score,
                citation_score=item.citation_score,
                claim_ids=tuple(item.claim_ids),
                required_claim_ids=tuple(item.required_claim_ids),
                unsupported_claim_ids=tuple(item.unsupported_claim_ids),
                conflicted_claim_ids=tuple(item.conflicted_claim_ids),
            )
            for item in sections
        )
        if latest_decision is not None:
            required_ids = tuple(
                latest_decision.snapshot.required_section_ids
            )
            complete_ids = tuple(
                latest_decision.snapshot.complete_section_ids
            )
            gap_ids = tuple(
                latest_decision.snapshot.coverage_gap_section_ids
            )
            blocked_ids = tuple(
                latest_decision.snapshot.blocked_high_impact_claim_ids
            )
            severe_ids = tuple(
                latest_decision.snapshot.severe_conflict_ids
            )
        else:
            required_ids = tuple(
                item.section_id
                for item in sections
                if item.required_claim_ids
            )
            complete_ids = tuple(
                item.section_id
                for item in sections
                if item.required_claim_ids
                and item.coverage_status == SectionCoverageStatus.COMPLETE
            )
            gap_ids = tuple(
                item for item in required_ids if item not in complete_ids
            )
            blocked_ids = tuple(
                claim.claim_id
                for claim in claims
                if claim.high_impact
                and claim.status.value != "supported"
            )
            severe_ids = tuple(
                conflict.conflict_id
                for conflict in conflicts
                if conflict.status.value == "open"
                and conflict.severity.value in {"high", "critical"}
            )
        required_count = len(required_ids)
        complete_count = len(complete_ids)
        ratio = (
            complete_count / required_count
            if required_count
            else (
                1.0
                if record.current_stage in {"reporting", "completed"}
                else 0.0
            )
        )
        gaps = tuple(
            ConsoleGapView(
                section_id=item.section_id,
                section_title=item.title,
                coverage_status=item.coverage_status.value,
                coverage_score=item.coverage_score,
                citation_score=item.citation_score,
                unsupported_claim_ids=tuple(item.unsupported_claim_ids),
            )
            for item in sections
            if item.required_claim_ids
            and item.coverage_status != SectionCoverageStatus.COMPLETE
        )
        conflict_views = tuple(
            ConsoleConflictView(
                conflict_id=item.conflict_id,
                summary=item.summary,
                status=item.status.value,
                severity=item.severity.value,
                high_impact=item.high_impact,
                claim_ids=tuple(item.claim_ids),
                fact_ids=tuple(item.fact_ids),
                resolution=item.resolution,
                resolution_kind=(
                    item.resolution_kind.value
                    if item.resolution_kind is not None
                    else None
                ),
                resolution_evidence_ids=tuple(
                    item.resolution_evidence_ids
                ),
                updated_at=item.updated_at.isoformat(),
            )
            for item in conflicts
        )
        packets = self._evidence_packet_views(
            self._artifact_payloads(
                record.run_id,
                ArtifactKind.EVIDENCE_PACK,
            )
        )
        source_views = tuple(
            ConsoleSourceView(
                source_id=item.source_id,
                canonical_url=item.canonical_url,
                title=item.title,
                publisher=item.publisher,
                source_type=item.source_type.value,
                source_level=item.source_level.value,
                status=item.status.value,
                authority_score=item.authority_score,
                published_at=(
                    item.published_at.isoformat()
                    if item.published_at is not None
                    else None
                ),
                discovered_at=item.discovered_at.isoformat(),
                task_id=item.provenance.task_id,
            )
            for item in sources
        )
        return ConsoleEvidenceView(
            knowledge=KnowledgeSummary(
                source_count=len(sources),
                claim_count=len(claims),
                fact_count=len(facts),
                evidence_count=len(evidence_records),
                conflict_count=len(conflicts),
                open_gap_count=len(gaps),
                section_pack_count=len(packets),
            ),
            coverage=ConsoleCoverageView(
                required_section_ids=required_ids,
                complete_section_ids=complete_ids,
                gap_section_ids=gap_ids,
                blocked_high_impact_claim_ids=blocked_ids,
                severe_conflict_ids=severe_ids,
                required_count=required_count,
                complete_count=complete_count,
                completion_ratio=ratio,
                ready_for_reporting=(
                    complete_count == required_count
                    and not blocked_ids
                    and not severe_ids
                    and (
                        bool(required_ids)
                        or record.current_stage
                        in {"reporting", "completed"}
                    )
                ),
            ),
            sections=section_views,
            gaps=gaps,
            conflicts=conflict_views,
            packets=packets,
            sources=source_views,
        )

    @staticmethod
    def _evidence_packet_views(
        payloads: list[dict[str, Any]],
    ) -> tuple[ConsoleEvidencePacketView, ...]:
        output: list[ConsoleEvidencePacketView] = []
        for payload in payloads:
            packet = payload.get("packet")
            if not isinstance(packet, dict):
                continue
            packet_id = str(packet.get("packet_id") or "")
            report_id = str(packet.get("report_id") or "")
            created_at = str(packet.get("created_at") or "")
            if not packet_id or not report_id or not created_at:
                continue
            claims = tuple(
                ConsoleVerifiedClaimView(
                    claim_id=str(item.get("claim_id") or ""),
                    statement=str(item.get("statement") or ""),
                    importance=float(item.get("importance") or 0.0),
                    high_impact=bool(item.get("high_impact")),
                    citation_ids=tuple(item.get("citation_ids") or ()),
                    source_ids=tuple(item.get("source_ids") or ()),
                )
                for item in packet.get("claims") or ()
                if isinstance(item, dict)
                and item.get("claim_id")
                and item.get("statement")
            )
            citations = tuple(
                ConsoleCitationView(
                    citation_id=str(item.get("citation_id") or ""),
                    claim_id=str(item.get("claim_id") or ""),
                    evidence_id=str(item.get("evidence_id") or ""),
                    source_id=str(item.get("source_id") or ""),
                    source_title=str(item.get("source_title") or ""),
                    canonical_url=str(item.get("canonical_url") or ""),
                    publisher=(
                        str(item["publisher"])
                        if item.get("publisher") is not None
                        else None
                    ),
                    locator=str(item.get("locator") or ""),
                    quote=str(item.get("quote") or ""),
                )
                for item in packet.get("citations") or ()
                if isinstance(item, dict)
                and item.get("citation_id")
                and item.get("canonical_url")
            )
            output.append(
                ConsoleEvidencePacketView(
                    packet_id=packet_id,
                    artifact_id=(
                        str(packet["packet_artifact_id"])
                        if packet.get("packet_artifact_id") is not None
                        else None
                    ),
                    report_id=report_id,
                    created_at=created_at,
                    verified_claims=claims,
                    citations=citations,
                    section_count=len(packet.get("sections") or ()),
                    gap_count=len(packet.get("gaps") or ()),
                    conflict_count=len(packet.get("conflicts") or ()),
                )
            )
        return tuple(
            sorted(output, key=lambda item: (item.created_at, item.packet_id))
        )

    def _reporting_view(
        self,
        *,
        record: ApplicationRunRecord,
        report_title: str,
        sections: tuple[ConsoleSectionView, ...],
    ) -> ConsoleReportingView:
        revisions = self.runtime.reporting_store.revisions(
            record.run_id,
            record.report_id,
        )
        reviews = self.runtime.reporting_store.reviews(
            record.run_id,
            record.report_id,
        )
        outcome = self.runtime.reporting_store.outcome(
            record.run_id,
            record.report_id,
        )
        latest_revision = revisions[-1] if revisions else None
        latest_review = reviews[-1] if reviews else None
        citation_count = 0
        if latest_revision is not None:
            citation_map = self.runtime.reporting_store.citation_map(
                record.run_id,
                record.report_id,
                latest_revision.revision,
            )
            citation_count = (
                len(citation_map.entries)
                if citation_map is not None
                else 0
            )
        revision_view = (
            ConsoleRevisionView(
                revision_id=latest_revision.revision_id,
                revision=latest_revision.revision,
                title=latest_revision.title,
                parent_revision_id=latest_revision.parent_revision_id,
                report_artifact_id=latest_revision.report_artifact_id,
                draft_artifact_id=latest_revision.draft_artifact_id,
                citation_map_artifact_id=(
                    latest_revision.citation_map_artifact_id
                ),
                evidence_packet_artifact_id=(
                    latest_revision.evidence_packet_artifact_id
                ),
                statement_count=len(latest_revision.statement_ids),
                citation_count=citation_count,
                usage=latest_revision.usage.model_dump(mode="json"),
                created_at=latest_revision.created_at.isoformat(),
            )
            if latest_revision is not None
            else None
        )
        review_view = (
            ConsoleReviewView(
                review_id=latest_review.review_id,
                revision_id=latest_review.revision_id,
                decision=latest_review.decision.value,
                decision_summary=latest_review.decision_summary,
                scores=tuple(
                    ConsoleReviewScoreView(
                        dimension=item.dimension.value,
                        score=item.score,
                        rationale=item.rationale,
                    )
                    for item in latest_review.scores
                ),
                findings=tuple(
                    ConsoleReviewFindingView(
                        finding_id=item.finding_id,
                        dimension=item.dimension.value,
                        severity=item.severity.value,
                        message=item.message,
                        section_id=item.section_id,
                        claim_ids=tuple(item.claim_ids),
                        citation_ids=tuple(item.citation_ids),
                    )
                    for item in latest_review.findings
                ),
                repair_actions=tuple(
                    ConsoleRepairActionView(
                        action_id=item.action_id,
                        kind=item.kind.value,
                        reason=item.reason,
                        section_ids=tuple(item.section_ids),
                        claim_ids=tuple(item.claim_ids),
                        citation_ids=tuple(item.citation_ids),
                    )
                    for item in latest_review.repair_actions
                ),
                usage=latest_review.usage.model_dump(mode="json"),
                created_at=latest_review.created_at.isoformat(),
            )
            if latest_review is not None
            else None
        )
        outcome_view = (
            ConsoleReportOutcomeView(
                status=outcome.status.value,
                revisions=outcome.revisions,
                final_revision_id=outcome.final_revision_id,
                final_report_artifact_id=(
                    outcome.final_report_artifact_id
                ),
                citation_map_artifact_id=(
                    outcome.citation_map_artifact_id
                ),
                final_review_id=outcome.final_review_id,
                usage=outcome.usage.model_dump(mode="json"),
                summary=outcome.summary,
                completed_at=outcome.completed_at.isoformat(),
            )
            if outcome is not None
            else None
        )
        return ConsoleReportingView(
            report_id=record.report_id,
            title=(
                latest_revision.title
                if latest_revision is not None
                else report_title
            ),
            artifact_ready=record.report_artifact_id is not None,
            revision_count=len(revisions),
            latest_revision=revision_view,
            latest_review=review_view,
            outcome=outcome_view,
            outline=sections,
        )

    @staticmethod
    def _navigation(record: ApplicationRunRecord) -> ConsoleNavigationView:
        run_id = record.run_id
        return ConsoleNavigationView(
            console_url=f"/console/{record.research_id}",
            report_url=f"/report/{record.research_id}",
            studio_url=f"/studio/{run_id}",
            trace_export_json_url=(
                f"/api/studio/runs/{run_id}/export?format=json"
            ),
            trace_export_ndjson_url=(
                f"/api/studio/runs/{run_id}/export?format=ndjson"
            ),
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
