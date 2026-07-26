from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from deep_researcher.application import (
    ApplicationRunStatus,
    ResearchCreateRequest,
)
from deep_researcher.contracts import (
    BudgetUsage,
    EventType,
    RunStatus,
)
from deep_researcher.events import EventQuery
from deep_researcher.kernel import ModelResponse
from deep_researcher.orchestration import RunControlStatus, TaskStatus
from tests.fixtures.background001_application import (
    STATEMENT,
    DeterministicSupervisorModel,
    build_deterministic_application,
)


async def _close(runtime, tools) -> None:
    await runtime.aclose()
    tools.close()


def _request(query: str = "Verify the benchmark.") -> ResearchCreateRequest:
    return ResearchCreateRequest(
        query=query,
        instructions=(
            "Use governed sources, preserve conflicts, and cite every claim."
        ),
        depth="standard",
    )


@pytest.mark.asyncio
async def test_production_composition_runs_research_to_verified_report(
    tmp_path: Path,
):
    runtime, tools, search = build_deterministic_application(tmp_path)
    try:
        created = runtime.new_run(_request())
        completed = await runtime.execute(created.research_id)

        assert completed.status == ApplicationRunStatus.COMPLETED
        assert completed.report_artifact_id is not None
        assert search.calls >= 1
        scheduler = await runtime.scheduler.snapshot(completed.run_id)
        assert scheduler.control.status == RunControlStatus.COMPLETED
        assert all(
            item.envelope.status
            in {
                TaskStatus.COMPLETED,
                TaskStatus.MERGED,
                TaskStatus.PRUNED,
                TaskStatus.CANCELLED,
            }
            for item in scheduler.tasks
        )

        report = runtime.artifact_store.read_bytes(
            completed.report_artifact_id
        ).decode("utf-8")
        assert STATEMENT in report
        claims = runtime.knowledge_repository.claims.list(completed.run_id)
        citations = runtime.knowledge_repository.citations.list(
            completed.run_id
        )
        assert claims and all(
            item.status.value == "supported" for item in claims
        )
        assert citations

        event_run = runtime.event_store.get_run(completed.run_id)
        assert event_run is not None
        assert event_run.status == RunStatus.SUCCEEDED
        events = runtime.event_store.list(
            EventQuery(completed.run_id, limit=1000)
        ).items
        assert events[0].event_type == EventType.RUN_STARTED
        assert events[-1].event_type == EventType.RUN_COMPLETED
        assert len(
            [item for item in events if item.event_type == EventType.RUN_COMPLETED]
        ) == 1
        assert runtime.studio_v2.task_graph(completed.run_id).nodes
        assert runtime.studio_v2.evidence_graph(completed.run_id).nodes
        metrics = runtime.studio_v2.metrics(completed.run_id)
        assert metrics.run_id == completed.run_id

        history = runtime.application_store.history(completed.research_id)
        assert [item.status.value for item in history] == [
            "queued",
            "running",
            "running",
            "running",
            "completed",
        ]
        runtime.integrity_check()
    finally:
        await _close(runtime, tools)


@pytest.mark.asyncio
async def test_two_runs_execute_concurrently_without_cross_run_state(
    tmp_path: Path,
):
    runtime, tools, search = build_deterministic_application(
        tmp_path,
        search_delay=0.03,
    )
    try:
        left = runtime.new_run(_request("Verify benchmark A."))
        right = runtime.new_run(_request("Verify benchmark B."))
        left_done, right_done = await asyncio.gather(
            runtime.execute(left.research_id),
            runtime.execute(right.research_id),
        )
        assert {
            left_done.status,
            right_done.status,
        } == {ApplicationRunStatus.COMPLETED}
        assert left_done.run_id != right_done.run_id
        assert left_done.report_artifact_id != right_done.report_artifact_id
        left_sources = runtime.knowledge_repository.sources.list(
            left_done.run_id
        )
        right_sources = runtime.knowledge_repository.sources.list(
            right_done.run_id
        )
        assert left_sources and right_sources
        assert {item.provenance.run_id for item in left_sources} == {
            left_done.run_id
        }
        assert {item.provenance.run_id for item in right_sources} == {
            right_done.run_id
        }
        assert {
            item.source_id for item in left_sources
        }.isdisjoint({item.source_id for item in right_sources})
        assert search.peak >= 2
        runtime.integrity_check()
    finally:
        await _close(runtime, tools)


@pytest.mark.asyncio
async def test_restart_recovers_a_durable_in_progress_application_run(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(tmp_path)
    created = runtime.new_run(_request("Recover this research run."))
    runtime.application_store.transition(
        created.research_id,
        status=ApplicationRunStatus.RUNNING,
        current_stage="researching",
    )
    await _close(runtime, tools)

    restarted, restarted_tools, _ = build_deterministic_application(tmp_path)
    try:
        recoverable = {
            item.research_id for item in restarted.recoverable_runs()
        }
        assert created.research_id in recoverable
        completed = await restarted.execute(created.research_id)
        assert completed.status == ApplicationRunStatus.COMPLETED
        assert completed.resumed is True
        history = restarted.application_store.history(created.research_id)
        assert [item.revision for item in history] == list(
            range(len(history))
        )
        assert history[1].current_stage == "researching"
        restarted.integrity_check()
    finally:
        await _close(restarted, restarted_tools)


@pytest.mark.asyncio
async def test_cancellation_fences_late_worker_completion(
    tmp_path: Path,
):
    runtime, tools, search = build_deterministic_application(
        tmp_path,
        search_delay=0.5,
    )
    try:
        created = runtime.new_run(_request("Cancel this run."))
        execution = asyncio.create_task(runtime.execute(created.research_id))
        for _ in range(200):
            if search.active:
                break
            await asyncio.sleep(0.005)
        assert search.active == 1
        cancelled = await runtime.cancel_run(
            created.research_id,
            reason="Explicit user cancellation.",
        )
        outcome = await execution
        assert cancelled.status == ApplicationRunStatus.CANCELLED
        assert outcome.status == ApplicationRunStatus.CANCELLED
        await asyncio.sleep(0.05)
        persisted = runtime.application_store.get(created.research_id)
        assert persisted is not None
        assert persisted.status == ApplicationRunStatus.CANCELLED
        assert (
            await runtime.scheduler.snapshot(created.run_id)
        ).control.status == RunControlStatus.CANCELLED
        event_run = runtime.event_store.get_run(created.run_id)
        assert event_run is not None
        assert event_run.status == RunStatus.CANCELLED
        assert (
            runtime.event_store.list(
                EventQuery(created.run_id, limit=1000)
            ).items[-1].event_type
            == EventType.RUN_CANCELLED
        )
    finally:
        await _close(runtime, tools)


@pytest.mark.asyncio
async def test_queued_cancellation_still_has_a_complete_runtime_trace(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(tmp_path)
    try:
        created = runtime.new_run(_request("Cancel before dispatch."))
        cancelled = await runtime.cancel_run(
            created.research_id,
            reason="Cancelled while queued.",
        )
        assert cancelled.status == ApplicationRunStatus.CANCELLED
        events = runtime.event_store.list(
            EventQuery(created.run_id, limit=100)
        ).items
        assert [item.event_type for item in events] == [
            EventType.RUN_STARTED,
            EventType.RUN_CANCELLED,
        ]
        assert runtime.event_store.get_run(created.run_id).status == (
            RunStatus.CANCELLED
        )
    finally:
        await _close(runtime, tools)


class _FailingModel:
    async def complete(self, request):
        raise RuntimeError(
            "provider failed with sk-12345678901234567890"
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


@pytest.mark.asyncio
async def test_model_failure_is_terminal_structured_and_redacted(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path,
        supervisor_model=_FailingModel(),
    )
    try:
        created = runtime.new_run(_request("Fail safely."))
        failed = await runtime.execute(created.research_id)
        assert failed.status == ApplicationRunStatus.FAILED
        assert failed.error_code == "application_runtimeerror"
        assert failed.error_message is not None
        assert "sk-123" not in failed.error_message
        sanitized = runtime._error(
            RuntimeError("provider failed with sk-12345678901234567890")
        )
        assert "[REDACTED]" in sanitized.message
        event = runtime.event_store.list(
            EventQuery(created.run_id, limit=1000)
        ).items[-1]
        assert event.event_type == EventType.RUN_FAILED
        assert event.error is not None
        assert "sk-123" not in event.error.message
        assert (
            await runtime.scheduler.snapshot(created.run_id)
        ).control.status == RunControlStatus.CANCELLED
    finally:
        await _close(runtime, tools)


class _ApprovalThenResearchModel:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = DeterministicSupervisorModel()

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                structured={
                    "action": "request_approval",
                    "tasks": [],
                    "decision_summary": "Explicit approval is required.",
                    "approval_reason": (
                        "The research owner must approve this source boundary."
                    ),
                },
                usage=BudgetUsage(model_calls=1),
            )
        return await self.delegate.complete(request)

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


@pytest.mark.asyncio
async def test_approval_pause_resume_uses_a_new_balanced_evidence_span(
    tmp_path: Path,
):
    supervisor = _ApprovalThenResearchModel()
    runtime, tools, _ = build_deterministic_application(
        tmp_path,
        supervisor_model=supervisor,
    )
    try:
        created = runtime.new_run(_request("Require approval."))
        waiting = await runtime.execute(created.research_id)
        assert waiting.status == ApplicationRunStatus.WAITING_APPROVAL
        snapshot = await runtime.scheduler.snapshot(created.run_id)
        approval_tasks = [
            item
            for item in snapshot.tasks
            if item.envelope.status == TaskStatus.WAITING_APPROVAL
        ]
        assert len(approval_tasks) == 1

        queued = await runtime.approve_run(
            created.research_id,
            approved_by="user_research_owner",
            note="Approved for the bounded source scope.",
        )
        assert queued.status == ApplicationRunStatus.QUEUED
        completed = await runtime.execute(created.research_id)
        assert completed.status == ApplicationRunStatus.COMPLETED
        assert completed.resumed is True

        events = runtime.event_store.list(
            EventQuery(created.run_id, limit=1000)
        ).items
        evidence_starts = [
            item
            for item in events
            if item.event_type == EventType.SPAN_STARTED
            and item.payload.get("stage") == "evidence_verification"
        ]
        evidence_ends = [
            item
            for item in events
            if item.event_type == EventType.SPAN_COMPLETED
            and item.payload.get("stage") == "evidence_verification"
        ]
        assert len(evidence_starts) == len(evidence_ends) == 2
        assert len({item.span_id for item in evidence_starts}) == 2
        assert events[-1].event_type == EventType.RUN_COMPLETED
        runtime.integrity_check()
    finally:
        await _close(runtime, tools)
