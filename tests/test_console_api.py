from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from console_app.service import ResearchConsoleService
from deep_researcher.application import (
    ApplicationRunRecord,
    ApplicationRunStatus,
    ResearchCreateRequest,
    TimelineEventSummary,
)
from deep_researcher.contracts import BudgetUsage
from deep_researcher.kernel import ModelResponse
from tests.fixtures.background001_application import (
    DeterministicSupervisorModel,
    STATEMENT,
    build_deterministic_application,
)


def test_console_causal_chain_promotes_authoritative_terminal_error():
    timeline = tuple(
        TimelineEventSummary(
            event_type="kernel_event",
            timestamp="2026-08-01T00:00:00+00:00",
            sequence_no=sequence,
            actor_id="agent:worker",
            task_id="task:extract",
            payload={"error": error},
        )
        for sequence, error in (
            (
                10,
                {
                    "error_id": "error_recovered_schema",
                    "category": "schema_validation",
                    "code": "command_schema_invalid",
                    "message": "The command was repaired.",
                    "retryable": True,
                },
            ),
            (
                22,
                {
                    "error_id": "error_terminal_action",
                    "category": "internal",
                    "code": "action_execution_failed",
                    "message": "The extraction action failed.",
                    "fatal": True,
                },
            ),
        )
    )

    errors = ResearchConsoleService._causal_errors(
        timeline,
        primary_error_refs=("error_terminal_action",),
        primary_code="action_execution_failed",
        primary_message="The extraction action failed.",
    )

    assert [item.error_id for item in errors] == [
        "error_terminal_action",
        "error_recovered_schema",
    ]
    assert errors[0].is_primary is True
    assert errors[0].sequence_no == 22
    assert errors[1].is_primary is False


def test_console_prefers_writer_cause_over_application_terminal_wrapper():
    timeline = (
        TimelineEventSummary(
            event_type="model_failed",
            timestamp="2026-08-01T00:00:00+00:00",
            sequence_no=10,
            actor_id="agent_synthesis_writer",
            task_id="task_writer",
            payload={
                "error": {
                    "category": "verification",
                    "code": "model_invocation_failed",
                    "message": "Writer proposal remained invalid.",
                    "fatal": True,
                }
            },
        ),
        TimelineEventSummary(
            event_type="run_failed",
            timestamp="2026-08-01T00:01:00+00:00",
            sequence_no=11,
            actor_id="runtime_application",
            task_id=None,
            payload={
                "error": {
                    "category": "internal",
                    "code": "application_runtimeerror",
                    "message": "Writer failed: The model could not produce a decision.",
                    "fatal": True,
                }
            },
        ),
    )
    errors = ResearchConsoleService._causal_errors(
        timeline,
        primary_code="application_runtimeerror",
        primary_message="Writer failed: The model could not produce a decision.",
    )
    assert errors[0].code == "model_invocation_failed"
    assert errors[0].is_primary is True


def test_budget_terminal_failure_does_not_relabel_completed_evidence_role():
    record = ApplicationRunRecord(
        research_id="research_budget_role",
        thread_id="thread_budget_role",
        session_id="session_budget_role",
        run_id="run_budget_role",
        trace_id="trace_budget_role",
        root_task_id="task_budget_role",
        report_id="report_budget_role",
        query="Budget projection test",
        status=ApplicationRunStatus.FAILED,
        current_stage="failed",
        error_code="research_budget_exhausted",
    )
    timeline = (
        TimelineEventSummary(
            event_type="span_completed",
            timestamp="2026-08-01T00:00:00+00:00",
            sequence_no=1,
            actor_id="agent_spec_evidence_verifier_1_0_0",
            task_id=None,
            payload={"stage": "evidence_verification"},
        ),
        TimelineEventSummary(
            event_type="run_failed",
            timestamp="2026-08-01T00:01:00+00:00",
            sequence_no=2,
            actor_id="runtime_application",
            task_id=None,
            payload={"error_code": "research_budget_exhausted"},
        ),
    )
    roles, _ = ResearchConsoleService._role_views(
        record=record,
        active_task=None,
        timeline=timeline,
        task_records=(),
        failure_role_id=None,
        decision_summary="The authoritative run budget is exhausted.",
    )
    status_by_role = {item.role_id: item.status for item in roles}
    assert status_by_role["evidence_verifier"] == "completed"
    assert "failed" not in status_by_role.values()


async def _wait_for_status(
    service: ResearchConsoleService,
    research_id: str,
    statuses: set[str],
):
    workspace = None
    for _ in range(300):
        workspace = await service.get_console_workspace(research_id)
        if workspace.runtime.status in statuses:
            return workspace
        await asyncio.sleep(0.01)
    raise AssertionError(
        f"run did not reach {sorted(statuses)}; "
        f"last status={workspace.runtime.status if workspace else None}"
    )


def test_console_shell_routes_assets_health_and_security_headers(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path / "runtime"
    )
    app = create_app(
        runtime_dir=str(tmp_path / "runtime"),
        runtime=runtime,
    )
    try:
        with TestClient(app) as client:
            for route in ["/", "/console/demo-run", "/report/demo-run"]:
                response = client.get(route)
                assert response.status_code == 200
                assert "DeepResearcher · 研究运行工作台" in response.text
                assert "/static/styles.css" in response.text
                assert "/static/app.js" in response.text
                assert "default-src 'self'" in response.headers[
                    "content-security-policy"
                ]
                assert (
                    response.headers["x-content-type-options"]
                    == "nosniff"
                )

            assert client.get("/static/console_models.js").status_code == 200
            health = client.get("/api/health")
            assert health.status_code == 200
            assert health.json()["runtime"] == (
                "Background001NativeRuntime"
            )
            assert health.json()["console_schema_version"] == (
                "ConsoleWorkspace@2"
            )
            assert "live_providers" in health.json()

            missing = client.get("/api/runs/missing/console")
            assert missing.status_code == 404
            assert missing.json()["detail"] == "Run not found"

            invalid_approval = client.post(
                "/api/runs/missing/approve",
                json={
                    "approved_by": "local-operator",
                    "note": "This actor ID is not namespaced.",
                },
            )
            assert invalid_approval.status_code == 422
            assert invalid_approval.json()["detail"][0]["loc"][-1] == (
                "approved_by"
            )

            blank_cancellation = client.post(
                "/api/runs/missing/cancel",
                json={"reason": "   "},
            )
            assert blank_cancellation.status_code == 422
            assert blank_cancellation.json()["detail"][0]["loc"][-1] == (
                "reason"
            )
    finally:
        asyncio.run(runtime.aclose())
        tools.close()


def test_console_run_catalog_serializes_typed_projection_items(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path / "runtime"
    )
    record = runtime.new_run(
        ResearchCreateRequest(query="Serialize the typed run catalog.")
    )
    app = create_app(
        runtime_dir=str(tmp_path / "runtime"),
        runtime=runtime,
    )
    try:
        with TestClient(app) as client:
            response = client.get("/api/runs")
            assert response.status_code == 200
            payload = response.json()
            item = next(
                entry
                for entry in payload
                if entry["research_id"] == record.research_id
            )
            assert item["schema_version"] == (
                "ConsoleRunListItem@2"
            )
            assert item["console_url"] == (
                f"/console/{record.research_id}"
            )
    finally:
        asyncio.run(runtime.aclose())
        tools.close()


@pytest.mark.asyncio
async def test_console_workspace_projects_complete_background001_runtime(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path / "runtime"
    )
    service = ResearchConsoleService(
        runtime_dir=str(tmp_path / "runtime"),
        runtime=runtime,
    )
    try:
        created = await service.create_run(
            ResearchCreateRequest(
                query=(
                    "Compare verified benchmark claims across vendors"
                ),
                instructions=(
                    "Preserve conflicts and cite verified sources."
                ),
                depth="standard",
            )
        )
        workspace = await _wait_for_status(
            service,
            created.research_id,
            {"completed", "failed", "cancelled"},
        )

        assert workspace.schema_version == "ConsoleWorkspace@2"
        assert workspace.runtime.status == "completed"
        assert workspace.runtime.current_stage == "completed"
        assert workspace.actions.terminal is True
        assert workspace.actions.can_approve is False
        assert workspace.actions.can_cancel is False
        assert [
            item.role_id for item in workspace.runtime.roles
        ] == [
            "research_supervisor",
            "research_worker_pool",
            "evidence_verifier",
            "synthesis_writer",
            "report_reviewer",
        ]
        assert {
            item.status for item in workspace.runtime.roles
        } == {"completed"}
        assert [
            item.step_id for item in workspace.runtime.progress
        ] == [
            "queued",
            "research",
            "verification",
            "synthesis",
            "review",
            "complete",
        ]
        assert workspace.scheduler.status == "completed"
        assert workspace.scheduler.projection_revision > 0
        assert workspace.scheduler.tasks
        research_task = next(
            item
            for item in workspace.scheduler.tasks
            if item.kind == "research"
        )
        assert research_task.task_id
        assert research_task.expected_output_schema == (
            "ResearchWorkerResult@1"
        )
        assert research_task.budget["max_tool_calls"] > 0
        assert research_task.budget_usage["tool_calls"] > 0
        assert research_task.output_artifact_ids
        assert workspace.evidence.knowledge.fact_count > 0
        assert workspace.evidence.knowledge.claim_count > 0
        assert workspace.evidence.knowledge.evidence_count > 0
        assert workspace.evidence.coverage.ready_for_reporting is True
        assert workspace.evidence.coverage.completion_ratio == 1.0
        assert workspace.evidence.packets
        assert workspace.evidence.packets[-1].verified_claims
        assert workspace.evidence.packets[-1].citations
        assert workspace.evidence.sources[0].canonical_url.startswith(
            "https://"
        )
        assert workspace.reporting.artifact_ready is True
        assert workspace.reporting.revision_count >= 1
        assert workspace.reporting.latest_revision is not None
        assert workspace.reporting.latest_revision.citation_count > 0
        assert workspace.reporting.latest_review is not None
        assert workspace.reporting.latest_review.decision == "accept"
        assert len(workspace.reporting.latest_review.scores) == 8
        assert workspace.reporting.outcome is not None
        assert workspace.navigation.studio_url == (
            f"/studio/{workspace.identity.run_id}"
        )
        assert workspace.timeline

        report = await service.get_report_view(created.research_id)
        debug = await service.get_debug_view(created.research_id)
        runs = await service.list_runs()

        assert report.schema_version == "ReportWorkspace@2"
        assert STATEMENT in report.markdown
        assert report.reporting.latest_review is not None
        assert report.reporting.latest_review.decision == "accept"
        assert debug.trace
        assert debug.state_summary["terminal_event_id"]
        assert debug.raw_state["projection_schema"] == (
            "StudioProjection@1"
        )
        assert debug.raw_state["console_schema"] == "ConsoleWorkspace@2"
        assert runs[0].schema_version == "ConsoleRunListItem@2"
        assert runs[0].current_stage == "completed"
        assert runs[0].has_report is True
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()


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
                    "decision_summary": (
                        "Explicit approval is required."
                    ),
                    "approval_reason": (
                        "The research owner must approve this "
                        "source boundary."
                    ),
                },
                usage=BudgetUsage(model_calls=1),
            )
        return await self.delegate.complete(request)

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


@pytest.mark.asyncio
async def test_console_workspace_exposes_approval_and_resume_actions(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path / "runtime",
        supervisor_model=_ApprovalThenResearchModel(),
    )
    service = ResearchConsoleService(runtime=runtime)
    try:
        created = await service.create_run(
            ResearchCreateRequest(query="Require bounded approval.")
        )
        waiting = await _wait_for_status(
            service,
            created.research_id,
            {"waiting_approval"},
        )

        assert waiting.runtime.current_stage == "waiting_approval"
        assert waiting.actions.terminal is False
        assert waiting.actions.can_approve is True
        assert waiting.actions.can_cancel is True
        assert len(waiting.actions.approvals) == 1
        approval = waiting.actions.approvals[0]
        assert approval.task_title
        assert "approve" in approval.reason
        assert approval.task_id in (
            waiting.actions.waiting_approval_task_ids
        )
        assert any(
            item.status == "blocked"
            for item in waiting.runtime.roles
        )
        assert any(
            item.status == "blocked"
            for item in waiting.runtime.progress
        )

        result = await service.approve_run(
            created.research_id,
            approved_by="user_research_owner",
            note="Approved for the bounded source scope.",
        )
        assert result["status"] == "queued"

        completed = await _wait_for_status(
            service,
            created.research_id,
            {"completed"},
        )
        assert completed.identity.resumed is True
        assert completed.actions.can_approve is False
        assert completed.reporting.artifact_ready is True
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()


@pytest.mark.asyncio
async def test_console_workspace_exposes_cancelled_and_failed_boundaries(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(
        tmp_path / "cancelled"
    )
    service = ResearchConsoleService(runtime=runtime)
    try:
        record = runtime.new_run(
            ResearchCreateRequest(query="Cancel before execution.")
        )
        queued = await service.get_console_workspace(record.research_id)
        assert queued.runtime.status == "queued"
        assert queued.actions.can_cancel is True

        await service.cancel_run(
            record.research_id,
            reason="Operator cancelled before execution.",
        )
        cancelled = await service.get_console_workspace(
            record.research_id
        )
        assert cancelled.runtime.status == "cancelled"
        assert cancelled.actions.terminal is True
        assert cancelled.actions.can_cancel is False
        assert cancelled.scheduler.cancellation_reason == (
            "Operator cancelled before execution."
        )
        assert cancelled.runtime.progress[-1].status == "cancelled"
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()

    class FailingModel:
        async def complete(self, request):
            raise RuntimeError(
                "provider failed with sk-12345678901234567890"
            )

        async def repair(self, request, invalid_response, errors):
            return await self.complete(request)

    failed_runtime, failed_tools, _ = build_deterministic_application(
        tmp_path / "failed",
        supervisor_model=FailingModel(),
    )
    failed_service = ResearchConsoleService(runtime=failed_runtime)
    try:
        failed_record = failed_runtime.new_run(
            ResearchCreateRequest(query="Fail safely.")
        )
        result = await failed_runtime.execute(
            failed_record.research_id
        )
        assert result.status == ApplicationRunStatus.FAILED

        failed = await failed_service.get_console_workspace(
            failed_record.research_id
        )
        assert failed.runtime.status == "failed"
        assert failed.runtime.error_code == "model_invocation_failed"
        assert failed.runtime.causal_errors
        assert failed.runtime.causal_errors[0].is_primary is True
        assert "sk-123" not in (failed.runtime.error_message or "")
        assert failed.actions.terminal is True
        assert failed.actions.can_cancel is False
        assert failed.runtime.progress[-1].status == "failed"
        assert any(
            item.status == "failed"
            for item in failed.runtime.roles
        )
        failed_roles = {
            item.role_id: item.status for item in failed.runtime.roles
        }
        assert failed_roles["research_supervisor"] == "failed"
        assert failed_roles["research_worker_pool"] == "waiting"
        assert failed_roles["evidence_verifier"] == "completed"
        assert failed_roles["synthesis_writer"] == "waiting"
        assert failed_roles["report_reviewer"] == "waiting"
        failed_progress = {
            item.step_id: item.status for item in failed.runtime.progress
        }
        assert failed_progress["research"] == "failed"
        assert failed_progress["verification"] == "completed"
        assert failed_progress["synthesis"] == "waiting"
        assert failed_progress["review"] == "waiting"
        assert failed.timeline[-1].event_type == "run_failed"
    finally:
        await failed_service.aclose()
        await failed_runtime.aclose()
        failed_tools.close()


@pytest.mark.asyncio
async def test_console_projects_worker_root_cause_and_failed_role_stage(
    tmp_path: Path,
):
    class MalformedWorkerModel:
        async def complete(self, request):
            return ModelResponse(
                structured={
                    "commands": [
                        {
                            "arguments": {
                                "operation": "search",
                                "query": "RAG papers",
                            }
                        }
                    ]
                },
                usage=BudgetUsage(model_calls=1),
            )

        async def repair(self, request, invalid_response, errors):
            return await self.complete(request)

    runtime, tools, _ = build_deterministic_application(
        tmp_path / "worker-root-cause",
        worker_model=MalformedWorkerModel(),
    )
    service = ResearchConsoleService(runtime=runtime)
    try:
        created = runtime.new_run(
            ResearchCreateRequest(query="Research RAG papers.")
        )
        failed = await runtime.execute(created.research_id)
        assert failed.status == ApplicationRunStatus.FAILED
        workspace = await service.get_console_workspace(
            created.research_id
        )
        assert workspace.runtime.error_code == "command_schema_invalid"
        assert workspace.runtime.causal_errors[0].code == (
            "command_schema_invalid"
        )
        assert workspace.runtime.causal_errors[0].is_primary is True
        roles = {
            item.role_id: item.status for item in workspace.runtime.roles
        }
        assert roles["research_worker_pool"] == "failed"
        assert roles["synthesis_writer"] == "waiting"
        assert roles["report_reviewer"] == "waiting"
        progress = {
            item.step_id: item.status for item in workspace.runtime.progress
        }
        assert progress["research"] == "failed"
        assert progress["synthesis"] == "waiting"
        assert workspace.reporting.revision_count == 0
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()
