from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import sqlite3

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    ArtifactKind,
    Budget,
    BudgetUsage,
    Command,
    CommandKind,
    ComponentKind,
    ComponentVersionSet,
    ErrorCategory,
    ErrorRecord,
    EventType,
    MiddlewareSpec,
    MiddlewareStage,
    ObservationStatus,
    RunEvent,
    RunStatus,
    SpanKind,
    TaskEnvelope,
    TaskKind,
    ToolGrant,
    VersionRef,
    utc_now,
)
from deep_researcher.events import (
    EventQuery,
    EventRecorder,
    SQLiteEventStore,
)
from deep_researcher.knowledge import (
    KnowledgeRepository,
    SQLiteKnowledgeStorage,
)
from deep_researcher.kernel import (
    ModelResponse,
    RawObservation,
    VerificationFeedback,
)
from deep_researcher.orchestration import SQLiteSchedulerStore
from deep_researcher.studio import (
    KernelReplayBackend,
    LiveReplayBindings,
    ReplayAttemptStatus,
    ReplayCapsule,
    ReplayMode,
    ReplayModelExchange,
    ReplayRequestStatus,
    ReplayToolExchange,
    ReplayVerificationExchange,
    SQLiteStudioAdvancedStore,
    StudioAdvancedConflict,
    StudioAdvancedService,
    StudioV2Service,
    command_fingerprint,
    source_event_fingerprint,
)
from deep_researcher.version_registry import (
    SQLiteVersionRegistryStore,
    VersionRegistry,
)


def _budget() -> Budget:
    return Budget(
        max_tokens=10_000,
        max_cost_usd=5,
        max_wall_time_seconds=30,
        max_model_calls=5,
        max_tool_calls=5,
        max_search_calls=5,
        max_retries=3,
        max_errors=3,
    )


def _version(
    kind: ComponentKind,
    name: str,
    suffix: str,
    *,
    artifact_id: str | None = None,
) -> VersionRef:
    return VersionRef(
        version_id=f"version_{kind.value}_{suffix}",
        kind=kind,
        name=name,
        version=f"1.0.{suffix[-1]}" if suffix[-1].isdigit() else "1.0.0",
        artifact_id=artifact_id,
    )


def _spec(prompt: VersionRef) -> AgentSpec:
    return AgentSpec(
        agent_spec_id="agent_spec_replay_worker",
        name="Replay worker",
        version="1.0.0",
        role=AgentRole.RESEARCH_WORKER,
        description="Deterministic replay worker for Studio tests.",
        input_schema="TaskEnvelope@1",
        output_schema="TaskResult@1",
        allowed_commands=(CommandKind.TOOL, CommandKind.STOP),
        tool_grants=(
            ToolGrant(
                tool_name="web",
                allowed_operations=("search",),
                requires_approval=True,
                argument_constraints={
                    "required": ["operation", "query"],
                    "allowed_properties": ["operation", "query"],
                },
            ),
        ),
        model=_version(ComponentKind.MODEL, "replay-model", "v1"),
        prompt=prompt,
        tool_policy=_version(
            ComponentKind.TOOL_POLICY,
            "replay-tools",
            "v1",
        ),
        stop_policy=_version(
            ComponentKind.STOP_POLICY,
            "replay-stop",
            "v1",
        ),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "replay-verifier",
            "v1",
        ),
        default_budget=_budget(),
        middleware=tuple(
            MiddlewareSpec(stage=stage, order=index)
            for index, stage in enumerate(MiddlewareStage)
        ),
        context_window_tokens=8_000,
        reserved_output_tokens=1_000,
    )


def _components(
    spec: AgentSpec,
    prompt: VersionRef,
) -> ComponentVersionSet:
    return ComponentVersionSet(
        runtime=_version(
            ComponentKind.RUNTIME,
            "studio-runtime",
            "v1",
        ),
        scheduler=_version(
            ComponentKind.SCHEDULER,
            "studio-scheduler",
            "v1",
        ),
        model=spec.model,
        agent_spec=_version(
            ComponentKind.AGENT_SPEC,
            "Replay worker",
            "v1",
        ),
        prompt=prompt,
        tool_policy=spec.tool_policy,
        stop_policy=spec.stop_policy,
        verification_policy=spec.verification_policy,
    )


def _event(
    sequence: int,
    event_type: EventType,
    *,
    components: ComponentVersionSet,
    span_id: str,
    parent_span_id: str | None,
    span_kind: SpanKind,
    status: RunStatus,
    error: ErrorRecord | None = None,
) -> RunEvent:
    return RunEvent(
        event_id=f"event_source_{sequence}",
        sequence_no=sequence,
        event_type=event_type,
        status=status,
        trace_id="trace_source",
        span_id=span_id,
        parent_span_id=parent_span_id,
        span_kind=span_kind,
        correlation_id="correlation_source",
        causation_event_id=(
            f"event_source_{sequence - 1}" if sequence > 1 else None
        ),
        run_id="run_source",
        thread_id="thread_source",
        task_id=(
            "task_source"
            if span_id == "span_source_agent"
            else None
        ),
        actor_id="agent_spec_replay_worker",
        producer_id="runtime_source",
        component_versions=components,
        error=error,
        occurred_at=utc_now(),
        recorded_at=utc_now(),
        payload={"source_sequence": sequence},
    )


class LiveModel:
    def __init__(self, query: str = "background001") -> None:
        self.query = query

    async def complete(self, request):
        return ModelResponse(
            structured={
                "commands": [
                    {
                        "kind": "tool",
                        "name": "web",
                        "arguments": {
                            "operation": "search",
                            "query": self.query,
                        },
                        "risk_level": "high",
                    }
                ]
            },
            usage=BudgetUsage(
                input_tokens=20,
                output_tokens=10,
                model_calls=1,
                cost_usd=0.02,
            ),
            response_id="response_live",
        )

    async def repair(self, request, invalid_response, errors):
        del request, invalid_response, errors
        raise AssertionError("repair is not expected")


class LiveAction:
    async def execute(self, command):
        return RawObservation(
            status=ObservationStatus.SUCCEEDED.value,
            data={
                "query": command.arguments["query"],
                "source": "live",
            },
            usage=BudgetUsage(tool_calls=1, cost_usd=0.01),
        )


class LiveVerifier:
    async def verify(self, **kwargs):
        del kwargs
        return VerificationFeedback(
            passed=True,
            success=True,
            information_gain=0.8,
            summary="Live replay result verified.",
        )


@pytest.mark.asyncio
async def test_live_replay_pauses_for_a_new_side_effect_command(
    tmp_path,
):
    runtime = _runtime(tmp_path)
    try:
        def changed_bindings(request, capsule):
            del capsule
            return LiveReplayBindings(
                model_adapter=LiveModel("new-side-effect-query"),
                action_executor=LiveAction(),
                verifier=LiveVerifier(),
                environment_label=request.environment_label,
            )

        runtime.service.replay_backend = KernelReplayBackend(
            event_store=runtime.events,
            recorder=EventRecorder(runtime.events),
            artifact_store=runtime.artifacts,
            version_store=runtime.versions,
            live_bindings_factory=changed_bindings,
        )
        record = runtime.service.prepare_replay(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            mode=ReplayMode.LIVE_ENVIRONMENT,
            selected_component_versions=runtime.components_v2,
            requested_by="principal_replay",
            reason="Exercise dynamic side-effect reapproval.",
            restart_failed_span=True,
            environment_label="staging-dynamic",
        )
        record = runtime.service.approve_replay(
            replay_request_id=record.request.replay_request_id,
            command_fingerprint=(
                record.request.required_approval_fingerprints[0]
            ),
            approved_by="principal_approver",
            reason="Approve only the original side-effect command.",
        )
        record = await runtime.service.execute_replay(
            record.request.replay_request_id
        )
        assert record.status == ReplayRequestStatus.WAITING_APPROVAL
        assert (
            record.attempts[0].status
            == ReplayAttemptStatus.WAITING_APPROVAL
        )
        pending = record.attempts[0].pending_approval_fingerprints
        assert len(pending) == 1
        assert pending[0] not in (
            record.request.required_approval_fingerprints
        )
        first_target = record.attempts[0].target_run_id
        assert runtime.events.get_run(first_target).status == RunStatus.FAILED

        record = runtime.service.approve_replay(
            replay_request_id=record.request.replay_request_id,
            command_fingerprint=pending[0],
            approved_by="principal_approver",
            reason="Approve the newly generated command for a new attempt.",
        )
        assert record.status == ReplayRequestStatus.QUEUED
        record = await runtime.service.execute_replay(
            record.request.replay_request_id
        )
        assert record.status == ReplayRequestStatus.SUCCEEDED
        assert len(record.attempts) == 2
        assert record.attempts[1].target_run_id != first_target
    finally:
        runtime.close()


@dataclass
class Runtime:
    artifacts: SQLiteArtifactStore
    events: SQLiteEventStore
    scheduler: SQLiteSchedulerStore
    knowledge: SQLiteKnowledgeStorage
    versions: SQLiteVersionRegistryStore
    advanced_store: SQLiteStudioAdvancedStore
    service: StudioAdvancedService
    capsule: ReplayCapsule
    capsule_artifact_id: str
    components_v1: ComponentVersionSet
    components_v2: ComponentVersionSet
    sample_artifact_id: str
    evaluation_artifact_id: str

    def close(self) -> None:
        self.advanced_store.close()
        self.versions.close()
        self.knowledge.close()
        self.scheduler.close()
        self.events.close()
        self.artifacts.close()


def _runtime(tmp_path) -> Runtime:
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    events = SQLiteEventStore(tmp_path / "events.sqlite3")
    scheduler = SQLiteSchedulerStore(tmp_path / "scheduler.sqlite3")
    knowledge = SQLiteKnowledgeStorage(
        tmp_path / "knowledge.sqlite3",
        artifact_store=artifacts,
    )
    versions = SQLiteVersionRegistryStore(tmp_path / "versions.sqlite3")
    advanced_store = SQLiteStudioAdvancedStore(
        tmp_path / "advanced.sqlite3"
    )
    version_registry = VersionRegistry(
        store=versions,
        artifact_store=artifacts,
    )
    prompt_v1_artifact = artifacts.put_text(
        "You are a careful researcher.\nUse governed tools.",
        kind=ArtifactKind.PROMPT,
        producer_id="producer_test",
        run_id="run_source",
        content_schema="Prompt@1",
        artifact_id="artifact_prompt_v1",
    )
    prompt_v2_artifact = artifacts.put_text(
        "You are a careful researcher.\n"
        "Use governed tools.\nPrefer primary sources.",
        kind=ArtifactKind.PROMPT,
        producer_id="producer_test",
        run_id="run_source",
        content_schema="Prompt@1",
        artifact_id="artifact_prompt_v2",
    )
    prompt_v1 = version_registry.register(
        _version(
            ComponentKind.PROMPT,
            "replay-prompt",
            "v1",
            artifact_id=prompt_v1_artifact.artifact_id,
        )
    ).manifest.version_ref
    prompt_v2 = version_registry.register(
        _version(
            ComponentKind.PROMPT,
            "replay-prompt",
            "v2",
            artifact_id=prompt_v2_artifact.artifact_id,
        ),
        parent_version_id=prompt_v1.version_id,
    ).manifest.version_ref
    spec = _spec(prompt_v1)
    components_v1 = _components(spec, prompt_v1)
    components_v2 = components_v1.model_copy(
        update={"prompt": prompt_v2}
    )
    failure = ErrorRecord(
        category=ErrorCategory.PERMANENT_PROVIDER,
        code="source_failed",
        message="The original span failed and is eligible for restart.",
        fatal=True,
        actor_id=spec.agent_spec_id,
        task_id="task_source",
    )
    source_events = (
        _event(
            1,
            EventType.RUN_STARTED,
            components=components_v1,
            span_id="span_source_root",
            parent_span_id=None,
            span_kind=SpanKind.RUN,
            status=RunStatus.RUNNING,
        ),
        _event(
            2,
            EventType.SPAN_STARTED,
            components=components_v1,
            span_id="span_source_agent",
            parent_span_id="span_source_root",
            span_kind=SpanKind.AGENT,
            status=RunStatus.RUNNING,
        ),
        _event(
            3,
            EventType.SPAN_FAILED,
            components=components_v1,
            span_id="span_source_agent",
            parent_span_id="span_source_root",
            span_kind=SpanKind.AGENT,
            status=RunStatus.FAILED,
            error=failure,
        ),
        _event(
            4,
            EventType.RUN_FAILED,
            components=components_v1,
            span_id="span_source_root",
            parent_span_id=None,
            span_kind=SpanKind.RUN,
            status=RunStatus.FAILED,
            error=failure,
        ),
    )
    for event in source_events:
        events.append(event)
    sample = artifacts.put_json(
        {
            "schema": "DatasetSample@1",
            "sample_id": "sample_studio_advanced",
            "input": "Investigate Background001.",
        },
        redact=False,
        kind=ArtifactKind.DATASET_SAMPLE,
        producer_id="producer_test",
        run_id="run_source",
        content_schema="DatasetSample@1",
        artifact_id="artifact_sample_studio",
    )
    evaluation = artifacts.put_json(
        {
            "schema": "EvaluationResult@1",
            "evaluation_id": "evaluation_source",
            "score": 0.2,
        },
        redact=False,
        kind=ArtifactKind.EVALUATION_RESULT,
        producer_id="producer_test",
        run_id="run_source",
        content_schema="EvaluationResult@1",
        artifact_id="artifact_evaluation_source",
    )
    source_tool_result = artifacts.put_json(
        {"results": [{"title": "Background001 source"}]},
        redact=False,
        kind=ArtifactKind.TOOL_RESULT,
        producer_id="producer_test",
        run_id="run_source",
        content_schema="ToolResult@1",
        artifact_id="artifact_source_tool_result",
    )
    task = TaskEnvelope(
        task_id="task_source",
        run_id="run_source",
        kind=TaskKind.RESEARCH,
        title="Research Background001",
        goal="Produce a verified evidence result.",
        input_artifact_ids=(sample.artifact_id,),
        expected_output_schema="EvidencePack@1",
        budget=_budget(),
        created_by="agent_supervisor",
    )
    semantic_command = Command(
        command_id="command_source_web",
        run_id="run_source",
        task_id=task.task_id,
        actor_id=spec.agent_spec_id,
        kind=CommandKind.TOOL,
        name="web",
        arguments={
            "operation": "search",
            "query": "background001",
        },
        idempotency_key="source-command-web",
        requires_approval=True,
        risk_level="high",
    )
    fingerprint = command_fingerprint(semantic_command)
    span_events = source_events[1:3]
    capsule = ReplayCapsule(
        capsule_id="replay_capsule_source",
        source_run_id="run_source",
        source_span_id="span_source_agent",
        source_event_ids=tuple(
            item.event_id for item in span_events
        ),
        source_event_fingerprint=source_event_fingerprint(span_events),
        task=task,
        agent_spec=spec,
        source_component_versions=components_v1,
        model_exchanges=(
            ReplayModelExchange(
                operation="complete",
                model_version=f"{spec.model.name}@{spec.model.version}",
                prompt_version=f"{spec.prompt.name}@{spec.prompt.version}",
                response={
                    "structured": {
                        "commands": [
                            {
                                "kind": "tool",
                                "name": "web",
                                "arguments": {
                                    "operation": "search",
                                    "query": "background001",
                                },
                                "risk_level": "high",
                            }
                        ]
                    },
                    "usage": BudgetUsage(
                        input_tokens=10,
                        output_tokens=5,
                        model_calls=1,
                    ).model_dump(mode="json"),
                    "response_id": "response_saved",
                },
            ),
        ),
        tool_exchanges=(
            ReplayToolExchange(
                command_fingerprint=fingerprint,
                command_kind=CommandKind.TOOL.value,
                tool_name="web",
                arguments=semantic_command.arguments,
                observation={
                    "run_id": "run_source",
                    "status": ObservationStatus.SUCCEEDED.value,
                    "normalized_data": {
                        "results": [
                            {"title": "Background001 source"}
                        ]
                    },
                    "output_artifact_ids": (
                        source_tool_result.artifact_id,
                    ),
                    "usage": BudgetUsage(
                        tool_calls=1
                    ).model_dump(mode="json"),
                },
                side_effecting=True,
                requires_approval=True,
            ),
        ),
        verification_exchanges=(
            ReplayVerificationExchange(
                command_fingerprint=fingerprint,
                feedback={
                    "passed": True,
                    "success": True,
                    "information_gain": 0.7,
                    "summary": "Saved result remains verified.",
                    "usage": BudgetUsage().model_dump(mode="json"),
                },
            ),
        ),
        dataset_sample_artifact_id=sample.artifact_id,
    )
    studio_v2 = StudioV2Service(
        event_store=events,
        scheduler_store=scheduler,
        knowledge_repository=KnowledgeRepository(knowledge),
        artifact_store=artifacts,
        version_store=versions,
    )
    recorder = EventRecorder(events)

    def live_bindings(request, loaded_capsule):
        assert loaded_capsule.capsule_id == capsule.capsule_id
        return LiveReplayBindings(
            model_adapter=LiveModel(),
            action_executor=LiveAction(),
            verifier=LiveVerifier(),
            environment_label=request.environment_label,
        )

    service = StudioAdvancedService(
        store=advanced_store,
        event_store=events,
        artifact_store=artifacts,
        version_store=versions,
        studio_v2=studio_v2,
        replay_backend=KernelReplayBackend(
            event_store=events,
            recorder=recorder,
            artifact_store=artifacts,
            version_store=versions,
            live_bindings_factory=live_bindings,
        ),
    )
    capsule_artifact_id = service.create_capsule(capsule)[
        "artifact_id"
    ]
    return Runtime(
        artifacts=artifacts,
        events=events,
        scheduler=scheduler,
        knowledge=knowledge,
        versions=versions,
        advanced_store=advanced_store,
        service=service,
        capsule=capsule,
        capsule_artifact_id=capsule_artifact_id,
        components_v1=components_v1,
        components_v2=components_v2,
        sample_artifact_id=sample.artifact_id,
        evaluation_artifact_id=evaluation.artifact_id,
    )


@pytest.mark.asyncio
async def test_saved_and_live_replay_are_real_immutable_kernel_runs(
    tmp_path,
):
    runtime = _runtime(tmp_path)
    try:
        source_before = runtime.events.list(
            EventQuery("run_source", limit=100)
        ).items
        eligibility = runtime.service.replay_eligibility(
            "run_source",
            "span_source_agent",
        )
        assert eligibility["eligible"]
        assert eligibility["terminal_failed"]
        root_eligibility = runtime.service.replay_eligibility(
            "run_source",
            "span_source_root",
        )
        assert not root_eligibility["eligible"]
        assert root_eligibility["terminal_failed"]
        assert root_eligibility["reasons"] == [
            "span has no immutable replay capsule"
        ]

        saved = runtime.service.prepare_replay(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            mode=ReplayMode.SAVED_TOOL_RESULTS,
            selected_component_versions=runtime.components_v1,
            requested_by="principal_replay",
            reason="Restart the failed span with sealed results.",
            restart_failed_span=True,
        )
        assert saved.status == ReplayRequestStatus.WAITING_APPROVAL
        fingerprint = saved.request.required_approval_fingerprints[0]
        saved = runtime.service.approve_replay(
            replay_request_id=saved.request.replay_request_id,
            command_fingerprint=fingerprint,
            approved_by="principal_approver",
            reason="Fresh approval for this replay only.",
        )
        assert saved.status == ReplayRequestStatus.QUEUED
        saved = await runtime.service.execute_replay(
            saved.request.replay_request_id
        )
        assert saved.status == ReplayRequestStatus.SUCCEEDED
        assert saved.latest_outcome is not None
        assert saved.latest_outcome.network_calls == 0
        assert (
            saved.latest_outcome.environment_label
            == "sealed-network-free"
        )
        target_run = saved.attempts[-1].target_run_id
        target = runtime.events.get_run(target_run)
        assert target is not None
        assert target.status == RunStatus.SUCCEEDED
        target_events = runtime.events.list(
            EventQuery(target_run, limit=100)
        ).items
        assert any(
            item.event_type == EventType.MODEL_COMPLETED
            for item in target_events
        )
        assert any(
            item.event_type == EventType.TOOL_COMPLETED
            for item in target_events
        )
        replay_artifact = runtime.artifacts.get(
            saved.latest_outcome.result_artifact_id
        )
        assert replay_artifact is not None
        assert replay_artifact.run_id == target_run
        assert replay_artifact.kind == ArtifactKind.STUDIO_REPLAY_RESULT

        live = runtime.service.prepare_replay(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            mode=ReplayMode.LIVE_ENVIRONMENT,
            selected_component_versions=runtime.components_v2,
            requested_by="principal_replay",
            reason="Run the candidate prompt in staging.",
            restart_failed_span=True,
            environment_label="staging-connected",
        )
        live = runtime.service.approve_replay(
            replay_request_id=live.request.replay_request_id,
            command_fingerprint=(
                live.request.required_approval_fingerprints[0]
            ),
            approved_by="principal_approver",
            reason="Fresh live side-effect approval.",
        )
        live = await runtime.service.execute_replay(
            live.request.replay_request_id
        )
        assert live.status == ReplayRequestStatus.SUCCEEDED
        assert live.latest_outcome.network_calls >= 2
        assert (
            live.latest_outcome.environment_label
            == "staging-connected"
        )
        live_events = runtime.events.list(
            EventQuery(live.attempts[-1].target_run_id, limit=100)
        ).items
        assert all(
            item.component_versions.prompt.version_id
            == runtime.components_v2.prompt.version_id
            for item in live_events
        )
        assert (
            runtime.events.list(
                EventQuery("run_source", limit=100)
            ).items
            == source_before
        )
    finally:
        runtime.close()


def test_replay_journal_recovers_crash_with_a_new_run_and_is_concurrent(
    tmp_path,
):
    runtime = _runtime(tmp_path)
    database = runtime.advanced_store.path
    try:
        record = runtime.service.prepare_replay(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            mode=ReplayMode.SAVED_TOOL_RESULTS,
            selected_component_versions=runtime.components_v1,
            requested_by="principal_replay",
            reason="Exercise restart recovery.",
            restart_failed_span=True,
        )
        record = runtime.service.approve_replay(
            replay_request_id=record.request.replay_request_id,
            command_fingerprint=(
                record.request.required_approval_fingerprints[0]
            ),
            approved_by="principal_approver",
            reason="Approve before simulated crash.",
        )
        first_attempt = runtime.advanced_store.claim(
            record.request.replay_request_id
        )
        runtime.advanced_store.close()
        runtime.advanced_store = SQLiteStudioAdvancedStore(database)
        recovered = runtime.advanced_store.recover_incomplete()
        assert len(recovered) == 1
        assert recovered[0].status == ReplayRequestStatus.QUEUED
        assert recovered[0].recovery_count == 1
        assert (
            recovered[0].attempts[0].status
            == ReplayAttemptStatus.ABANDONED
        )
        assert (
            recovered[0].attempts[0].target_run_id
            == first_attempt.target_run_id
        )
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(
                    runtime.advanced_store.claim,
                    record.request.replay_request_id,
                )
                for _ in range(2)
            ]
        results = []
        errors = []
        for future in futures:
            try:
                results.append(future.result())
            except StudioAdvancedConflict as exc:
                errors.append(exc)
        assert len(results) == 1
        assert len(errors) == 1
        assert results[0].target_run_id != first_attempt.target_run_id
        for suffix in ("page_a", "page_b"):
            runtime.advanced_store.create_request(
                record.request.model_copy(
                    update={
                        "replay_request_id": (
                            f"replay_request_{suffix}"
                        ),
                        "reason": f"Pagination request {suffix}.",
                    }
                )
            )
        first_page = runtime.advanced_store.list(limit=2)
        second_page = runtime.advanced_store.list(
            cursor=first_page.next_cursor,
            limit=2,
        )
        assert len(first_page.items) == 2
        assert len(second_page.items) == 1
        assert {
            item.request.replay_request_id
            for item in (*first_page.items, *second_page.items)
        } == {
            record.request.replay_request_id,
            "replay_request_page_a",
            "replay_request_page_b",
        }
        runtime.advanced_store.rebuild_projections()
        runtime.advanced_store.integrity_check()
        runtime.advanced_store._connection.execute(
            "UPDATE studio_replay_projection "
            "SET checksum='corrupt' "
            "WHERE replay_request_id=?",
            (record.request.replay_request_id,),
        )
        with pytest.raises(
            Exception,
            match="checksum|projection",
        ):
            runtime.advanced_store.integrity_check()
        runtime.advanced_store.rebuild_projections()
        runtime.advanced_store.integrity_check()
        with pytest.raises(sqlite3.DatabaseError):
            runtime.advanced_store._connection.execute(
                "UPDATE studio_replay_journal SET kind='failed' "
                "WHERE replay_request_id=?",
                (record.request.replay_request_id,),
            )
        backup = runtime.advanced_store.backup_to(
            tmp_path / "advanced-backup.sqlite3"
        )
        restored = SQLiteStudioAdvancedStore(backup)
        try:
            restored.integrity_check()
            assert (
                restored.get(record.request.replay_request_id)
                is not None
            )
        finally:
            restored.close()
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_ab_component_diff_and_badcase_preserve_boundaries(tmp_path):
    runtime = _runtime(tmp_path)
    try:
        replay = runtime.service.prepare_replay(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            mode=ReplayMode.LIVE_ENVIRONMENT,
            selected_component_versions=runtime.components_v2,
            requested_by="principal_replay",
            reason="Create aligned A/B candidate.",
            restart_failed_span=True,
            environment_label="evaluation-live",
        )
        replay = runtime.service.approve_replay(
            replay_request_id=replay.request.replay_request_id,
            command_fingerprint=(
                replay.request.required_approval_fingerprints[0]
            ),
            approved_by="principal_approver",
            reason="Approve aligned evaluation replay.",
        )
        replay = await runtime.service.execute_replay(
            replay.request.replay_request_id
        )
        target_run = replay.attempts[-1].target_run_id
        comparison = runtime.service.compare_runs(
            left_run_id="run_source",
            right_run_id=target_run,
            dataset_sample_artifact_id=runtime.sample_artifact_id,
        )
        assert comparison.dataset_sample_artifact_id == (
            runtime.sample_artifact_id
        )
        assert comparison.run_status == {
            "left": "failed",
            "right": "succeeded",
        }
        assert not comparison.publishes_versions
        assert "total_tokens" in comparison.metrics
        assert any(
            item.component_kind == ComponentKind.PROMPT.value
            and item.changed
            for item in comparison.components
        )
        comparison_payload = runtime.artifacts.read_json(
            comparison.result_artifact_id
        )
        assert comparison_payload["publishes_versions"] is False
        assert comparison_payload["optimizer_invoked"] is False

        unrelated_sample = runtime.artifacts.put_json(
            {"sample_id": "sample_unrelated"},
            redact=False,
            kind=ArtifactKind.DATASET_SAMPLE,
            producer_id="producer_test",
            run_id="run_source",
            content_schema="DatasetSample@1",
            artifact_id="artifact_sample_unrelated",
        )
        with pytest.raises(ValueError, match="aligned"):
            runtime.service.compare_runs(
                left_run_id="run_source",
                right_run_id=target_run,
                dataset_sample_artifact_id=unrelated_sample.artifact_id,
            )

        diff = runtime.service.component_diff(
            runtime.components_v1.prompt.version_id,
            runtime.components_v2.prompt.version_id,
        )
        assert diff.component_kind == ComponentKind.PROMPT
        assert "Prefer primary sources." in diff.unified_diff
        assert any(
            item.operation in {"insert", "replace"}
            for item in diff.changes
        )
        policy_v1_artifact = runtime.artifacts.put_text(
            "allow: [search]\nrequires_approval: true",
            kind=ArtifactKind.POLICY,
            producer_id="producer_test",
            run_id="run_source",
            content_schema="ToolPolicy@1",
            artifact_id="artifact_policy_diff_v1",
        )
        policy_v2_artifact = runtime.artifacts.put_text(
            "allow: [search, open]\nrequires_approval: true",
            kind=ArtifactKind.POLICY,
            producer_id="producer_test",
            run_id="run_source",
            content_schema="ToolPolicy@1",
            artifact_id="artifact_policy_diff_v2",
        )
        registry = VersionRegistry(
            store=runtime.versions,
            artifact_store=runtime.artifacts,
        )
        policy_v1 = registry.register(
            VersionRef(
                version_id="version_policy_diff_v1",
                kind=ComponentKind.TOOL_POLICY,
                name="diff-policy",
                version="1.0.0",
                artifact_id=policy_v1_artifact.artifact_id,
            )
        ).manifest.version_ref
        policy_v2 = registry.register(
            VersionRef(
                version_id="version_policy_diff_v2",
                kind=ComponentKind.TOOL_POLICY,
                name="diff-policy",
                version="1.1.0",
                artifact_id=policy_v2_artifact.artifact_id,
            ),
            parent_version_id=policy_v1.version_id,
        ).manifest.version_ref
        policy_diff = runtime.service.component_diff(
            policy_v1.version_id,
            policy_v2.version_id,
        )
        assert policy_diff.component_kind == ComponentKind.TOOL_POLICY
        assert "open" in policy_diff.unified_diff

        badcase = runtime.service.create_badcase(
            source_run_id="run_source",
            source_span_id="span_source_agent",
            dataset_sample_artifact_id=runtime.sample_artifact_id,
            evaluation_ids=("evaluation_source",),
            evaluation_artifact_ids=(
                runtime.evaluation_artifact_id,
            ),
            human_note=(
                "Original run failed before it could produce a supported "
                "answer; retain for offline analysis."
            ),
            created_by="principal_evaluator",
        )
        assert badcase.source_run_id == "run_source"
        assert badcase.source_span_id == "span_source_agent"
        assert badcase.component_version_ids
        assert badcase.input_artifact_ids
        assert badcase.evaluation_ids == ("evaluation_source",)
        assert not badcase.triggers_change
        payload = runtime.artifacts.read_json(badcase.artifact_id)
        assert payload["original_provenance_preserved"] is True
        assert payload["triggers_change"] is False
        assert payload["optimizer_invoked"] is False
        with pytest.raises(ValueError, match="not present"):
            runtime.service.create_badcase(
                source_run_id="run_source",
                source_span_id="span_source_agent",
                dataset_sample_artifact_id=runtime.sample_artifact_id,
                evaluation_ids=("evaluation_other",),
                evaluation_artifact_ids=(
                    runtime.evaluation_artifact_id,
                ),
                human_note="Mismatched evaluation provenance.",
                created_by="principal_evaluator",
            )
        assert (
            runtime.service.create_badcase(
                source_run_id="run_source",
                source_span_id="span_source_agent",
                dataset_sample_artifact_id=runtime.sample_artifact_id,
                evaluation_ids=("evaluation_source",),
                evaluation_artifact_ids=(
                    runtime.evaluation_artifact_id,
                ),
                human_note=badcase.human_note,
                created_by="principal_evaluator",
            )
            == badcase
        )
    finally:
        runtime.close()


def test_studio_v4_http_surface_exposes_actions_without_delete(tmp_path):
    app = create_app(runtime_dir=str(tmp_path / "console"))
    with TestClient(app) as client:
        shell = client.get("/studio/run_missing")
        assert shell.status_code == 200
        assert "Studio V4" in shell.text
        assert "Replay / A-B / Badcase" in shell.text
        missing = client.get(
            "/api/studio/advanced/runs/run_missing/spans/"
            "span_missing/replay-eligibility"
        )
        assert missing.status_code == 404
        assert (
            client.delete("/api/studio/advanced/replays/request_missing")
            .status_code
            == 405
        )
