from __future__ import annotations

import asyncio
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from deep_researcher.artifacts import ArtifactQuery, SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    Budget,
    BudgetDimension,
    BudgetUsage,
    Command,
    CommandKind,
    ConflictSeverity,
    ConflictStatus,
    Observation,
    ObservationStatus,
    SectionCoverageStatus,
    StopDecision,
    StopReason,
    TaskEnvelope,
    TaskKind,
    TaskResult,
    TaskResultStatus,
    TaskStatus,
    utc_now,
)
from deep_researcher.evidence import (
    DeterministicSemanticVerificationAdapter,
    RecordingEvidenceEventSink,
    VerificationPolicy,
    build_evidence_runtime,
)
from deep_researcher.kernel import (
    KernelRunResult,
    KernelEvent,
    ModelRequest,
    ModelResponse,
)
from deep_researcher.gateway import (
    ProtocolToolGateway,
    SQLiteToolStateStore,
    ToolAdapterResult,
    ToolDefinition,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolRegistry,
)
from deep_researcher.orchestration import (
    NativeEventSourcedScheduler,
    RunControl,
    RunControlStatus,
    SchedulerSnapshot,
    SQLiteSchedulerStore,
    TaskRecord,
    TaskLease,
)
from deep_researcher.research import (
    ConvergenceAction,
    ConvergenceEvaluator,
    ConvergencePolicy,
    DedupDecisionKind,
    DedupKind,
    InformationGainEstimator,
    InformationGainAssessment,
    ResearchCoordinationConflict,
    ResearchCoordinationCorruption,
    ResearchWorkerActionExecutor,
    ResearchWorkerResult,
    ResearchWorkerResultReconciler,
    ResearchWorkerRunner,
    SQLiteResearchCoordinationStore,
    SupervisorActionExecutor,
    SupervisorPlan,
    SupervisorPlanAction,
    SupervisorPlanningModelAdapter,
    SupervisorTaskProposal,
    build_research_runtime,
    build_research_supervisor_spec,
    build_research_worker_spec,
)
from deep_researcher.research.worker import WorkerObservationVerifier


def _budget(**updates: Any) -> Budget:
    values: dict[str, Any] = {
        "max_tokens": 20_000,
        "max_cost_usd": 20.0,
        "max_wall_time_seconds": 600.0,
        "max_model_calls": 20,
        "max_tool_calls": 30,
        "max_search_calls": 10,
        "max_retries": 5,
        "max_errors": 5,
    }
    values.update(updates)
    return Budget(**values)


def _task(
    suffix: str,
    *,
    run_id: str = "run_research",
    kind: TaskKind = TaskKind.RESEARCH,
    parent: str | None = None,
    dependencies: tuple[str, ...] = (),
    assigned_actor_id: str | None = None,
) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=f"task_{suffix}",
        run_id=run_id,
        parent_task_id=parent,
        dependency_task_ids=dependencies,
        kind=kind,
        title=f"Research task {suffix}",
        goal=f"Resolve the evidence requirement for {suffix}.",
        expected_output_schema="ResearchWorkerResult@1",
        budget=_budget(),
        created_by="agent_research_supervisor",
        assigned_actor_id=assigned_actor_id,
    )


def _command(
    suffix: str,
    *,
    kind: CommandKind,
    arguments: dict[str, Any],
    run_id: str = "run_research",
    task_id: str = "task_worker",
) -> Command:
    return Command(
        command_id=f"command_{suffix}",
        run_id=run_id,
        task_id=task_id,
        actor_id="agent_spec_research_worker_1_0_0",
        kind=kind,
        name=f"research.{kind.value}",
        arguments=arguments,
        idempotency_key=f"idempotency-{suffix}",
    )


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


class QueueModel:
    def __init__(
        self,
        responses: list[ModelResponse | Exception],
        repairs: list[ModelResponse | Exception] | None = None,
    ) -> None:
        self.responses = deque(responses)
        self.repairs = deque(repairs or [])
        self.requests: list[ModelRequest] = []
        self.repair_calls: list[
            tuple[ModelRequest, ModelResponse, tuple[str, ...]]
        ] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        self.repair_calls.append((request, invalid_response, errors))
        value = self.repairs.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class EventCollector:
    def __init__(self) -> None:
        self.events: list[KernelEvent] = []

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)


class Boundary:
    def __init__(
        self,
        *,
        delay: float = 0.0,
        fail_once: bool = False,
    ) -> None:
        self.delay = delay
        self.fail_once = fail_once
        self.calls: list[Command] = []
        self.active = 0
        self.peak = 0

    async def execute(self, command: Command) -> Observation:
        self.calls.append(command)
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            if self.fail_once:
                self.fail_once = False
                raise RuntimeError("transient boundary failure")
            query = str(command.arguments.get("query", ""))
            return Observation(
                command_id=command.command_id,
                run_id=command.run_id,
                task_id=command.task_id,
                actor_id=command.actor_id,
                status=ObservationStatus.SUCCEEDED,
                normalized_data={
                    "query": query,
                    "results": [
                        {
                            "url": (
                                "https://Example.com:443/source?utm_source=x"
                            ),
                            "title": "Primary source",
                        }
                    ],
                    "facts": [{"statement": f"verified:{query}"}],
                    "semantic_complete": True,
                },
                started_at=utc_now(),
                completed_at=utc_now(),
            )
        finally:
            self.active -= 1


class GatewayAdapter:
    def __init__(self, result: ToolAdapterResult) -> None:
        self.result = result
        self.calls = 0

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        del arguments, context
        self.calls += 1
        return self.result

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


@dataclass
class MutableSection:
    section_id: str
    coverage_status: SectionCoverageStatus
    coverage_score: float
    citation_score: float


class Collection:
    def __init__(self, values: list[Any] | None = None) -> None:
        self.values = values or []

    def list(self, run_id: str) -> tuple[Any, ...]:
        del run_id
        return tuple(self.values)


class FakeEvidence:
    def __init__(
        self,
        artifacts: SQLiteArtifactStore,
        *,
        section_id: str = "section_required",
        complete_after: int = 2,
    ) -> None:
        self.section = MutableSection(
            section_id=section_id,
            coverage_status=SectionCoverageStatus.INSUFFICIENT,
            coverage_score=0.0,
            citation_score=0.0,
        )
        self.knowledge = type(
            "Knowledge",
            (),
            {
                "artifacts": artifacts,
                "repository": type(
                    "Repository",
                    (),
                    {
                        "sections": Collection([self.section]),
                        "conflicts": Collection(),
                    },
                )(),
            },
        )()
        self.verified = type(
            "Verified",
            (),
            {
                "blocked_high_impact_claims": staticmethod(
                    lambda run_id: ()
                )
            },
        )()
        self.complete_after = complete_after
        self.verification_calls = 0
        self.engine = self

    async def verify_run(self, run_id: str, **kwargs: Any) -> None:
        del run_id, kwargs
        self.verification_calls += 1
        if self.verification_calls >= self.complete_after:
            self.section.coverage_status = SectionCoverageStatus.COMPLETE
            self.section.coverage_score = 1.0
            self.section.citation_score = 1.0

    def integrity_check(self) -> None:
        return None


def _scheduler(path: Path) -> NativeEventSourcedScheduler:
    return NativeEventSourcedScheduler(SQLiteSchedulerStore(path))


def _verification_policy() -> VerificationPolicy:
    return VerificationPolicy(
        policy_version_id="policy_research_branch_08",
    )


def _convergence_policy(
    **updates: Any,
) -> ConvergencePolicy:
    values: dict[str, Any] = {
        "run_budget": _budget(
            max_tokens=200_000,
            max_model_calls=200,
            max_tool_calls=300,
            max_search_calls=100,
        ),
        "max_cycles": 6,
        "max_low_gain_cycles": 2,
    }
    values.update(updates)
    return ConvergencePolicy(**values)


def _proposal(
    key: str,
    *,
    dependencies: tuple[str, ...] = (),
    goal: str | None = None,
) -> SupervisorTaskProposal:
    return SupervisorTaskProposal(
        proposal_key=key,
        kind=TaskKind.RESEARCH,
        title=f"Task {key}",
        goal=goal or f"Resolve {key}",
        expected_output_schema="ResearchWorkerResult@1",
        budget=_budget(),
        dependency_keys=dependencies,
        tags=("research",),
    )


def test_specs_and_plan_contract_enforce_the_full_role_boundary() -> None:
    supervisor = build_research_supervisor_spec()
    worker = build_research_worker_spec()
    assert supervisor.tool_grants == ()
    assert supervisor.metadata["provider_access"] is False
    assert supervisor.metadata["writes_final_report"] is False
    assert set(supervisor.allowed_commands) == {
        CommandKind.DELEGATE,
        CommandKind.REQUEST_APPROVAL,
        CommandKind.STOP,
    }
    assert set(worker.allowed_commands) == {
        CommandKind.SEARCH,
        CommandKind.READ,
        CommandKind.EXTRACT,
        CommandKind.DELEGATE,
        CommandKind.COMPARE,
        CommandKind.VERIFY_SOURCE,
        CommandKind.REQUEST_APPROVAL,
        CommandKind.STOP,
    }
    assert CommandKind.SYNTHESIZE not in worker.allowed_commands
    assert worker.metadata["free_form_agent_chat"] is False

    with pytest.raises(ValidationError, match="research work only"):
        SupervisorTaskProposal(
            proposal_key="write",
            kind=TaskKind.SYNTHESIS,
            title="Write",
            goal="Write final prose",
            expected_output_schema="FinalReport@1",
            budget=_budget(),
        )
    with pytest.raises(ValidationError, match="cycle"):
        SupervisorPlan(
            run_id="run_research",
            root_task_id="task_root",
            cycle=0,
            action=SupervisorPlanAction.DECOMPOSE,
            objective="Research",
            tasks=(
                _proposal("left", dependencies=("right",)),
                _proposal("right", dependencies=("left",)),
            ),
            decision_summary="Cyclic plan",
        )


def test_coordination_store_is_concurrent_durable_and_corruption_detecting(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coordination.sqlite3"
    first = SQLiteResearchCoordinationStore(path)

    def claim(task_id: str):
        other = SQLiteResearchCoordinationStore(path)
        try:
            return other.claim_key(
                run_id="run_research",
                kind=DedupKind.QUERY,
                normalized_key="  Same   Query ",
                task_id=task_id,
            )
        finally:
            other.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = tuple(
            executor.map(claim, ("task_left", "task_right"))
        )
    assert {item.decision for item in decisions} == {
        DedupDecisionKind.NEW,
        DedupDecisionKind.DUPLICATE_IN_PROGRESS,
    }
    owner = next(item for item in decisions if item.decision == DedupDecisionKind.NEW)
    first.complete_key(
        run_id="run_research",
        kind=DedupKind.QUERY,
        normalized_key="Same Query",
        task_id=owner.owner_task_id,
        artifact_ids=("artifact_query_result",),
    )
    duplicate = first.claim_key(
        run_id="run_research",
        kind=DedupKind.QUERY,
        normalized_key="same query",
        task_id="task_third",
    )
    assert duplicate.decision == DedupDecisionKind.DUPLICATE_COMPLETED
    assert duplicate.artifact_ids == ("artifact_query_result",)
    backup_path = first.backup_to(
        tmp_path / "coordination.backup.sqlite3"
    )
    first.integrity_check()
    first.close()

    backup = SQLiteResearchCoordinationStore(backup_path)
    assert backup.claim_key(
        run_id="run_research",
        kind=DedupKind.QUERY,
        normalized_key="same query",
        task_id="task_backup_reader",
    ).decision == DedupDecisionKind.DUPLICATE_COMPLETED
    backup.integrity_check()
    backup.close()

    reopened = SQLiteResearchCoordinationStore(path)
    assert reopened.claim_key(
        run_id="run_research",
        kind=DedupKind.QUERY,
        normalized_key="same query",
        task_id="task_fourth",
    ).decision == DedupDecisionKind.DUPLICATE_COMPLETED
    reopened._connection.execute(
        "UPDATE research_dedup_claims SET payload='{}'"
    )
    reopened._connection.commit()
    with pytest.raises(ResearchCoordinationCorruption):
        reopened.integrity_check()
    reopened.close()


@pytest.mark.asyncio
async def test_supervisor_model_repairs_invalid_plans_and_rejects_early_stop() -> None:
    valid = {
        "action": "decompose",
        "tasks": [
            {
                "proposal_key": "source",
                "kind": "source_discovery",
                "title": "Find sources",
                "goal": "Find primary sources",
                "expected_output_schema": "ResearchWorkerResult@1",
                "budget": _budget().model_dump(mode="json"),
            }
        ],
        "decision_summary": "Delegate source discovery.",
    }
    base = QueueModel(
        [ModelResponse(structured={"invalid": True})],
        repairs=[ModelResponse(structured=valid)],
    )
    adapter = SupervisorPlanningModelAdapter(base, max_plan_repairs=1)
    request = ModelRequest(
        run_id="run_research",
        task_id="task_root",
        actor_id="agent_spec_research_supervisor_1_0_0",
        system="generic",
        messages=(
            {
                "role": "user",
                "content": {
                    "constraints": {
                        "supervisor_context": {
                            "cycle": 2,
                            "objective": "Verify the system",
                            "allow_stop": False,
                        }
                    }
                },
            },
        ),
        command_schema={},
        model_version="model@1",
        prompt_version="prompt@1",
        max_output_tokens=2000,
    )
    result = await adapter.complete(request)
    assert len(base.repair_calls) == 1
    plan = SupervisorPlan.model_validate(
        result.structured["supervisor_plan"],
        strict=False,
    )
    assert plan.cycle == 2
    assert plan.run_id == request.run_id
    assert result.structured["commands"][0]["name"] == "supervisor.apply_plan"

    terminal = QueueModel(
        [
            ModelResponse(
                structured={
                    "action": "converged",
                    "tasks": [],
                    "decision_summary": "Stop",
                    "stop_reason": "Looks done",
                }
            )
        ]
    )
    with pytest.raises(Exception, match="cannot stop"):
        await SupervisorPlanningModelAdapter(
            terminal,
            max_plan_repairs=0,
        ).complete(request)


@pytest.mark.asyncio
async def test_supervisor_repairs_underfunded_extraction_task_budget() -> None:
    def plan(max_model_calls: int, max_tokens: int) -> dict[str, Any]:
        budget = _budget().model_dump(mode="json")
        budget["max_model_calls"] = max_model_calls
        budget["max_tokens"] = max_tokens
        return {
            "action": "decompose",
            "tasks": [
                {
                    "proposal_key": "extract_primary",
                    "kind": "research",
                    "title": "Read and extract the paper original",
                    "goal": "Read the persisted source and extract exact quotes.",
                    "expected_output_schema": "ResearchWorkerResult@1",
                    "budget": budget,
                }
            ],
            "decision_summary": "Create citation-grounded knowledge.",
        }

    base = QueueModel(
        [ModelResponse(structured=plan(2, 28_000))],
        repairs=[ModelResponse(structured=plan(4, 64_000))],
    )
    adapter = SupervisorPlanningModelAdapter(base, max_plan_repairs=1)
    request = ModelRequest(
        run_id="run_budget_admission",
        task_id="task_budget_admission_root",
        actor_id="agent_spec_research_supervisor_1_0_0",
        system="generic",
        messages=(
            {
                "role": "user",
                "content": {
                    "constraints": {
                        "supervisor_context": {
                            "cycle": 0,
                            "objective": "Ground a paper claim",
                            "allow_stop": False,
                            "max_tasks_per_plan": 6,
                            "minimum_model_calls_per_extraction_task": 4,
                            "minimum_tokens_per_extraction_task": 64_000,
                        }
                    }
                },
            },
        ),
        command_schema={},
        model_version="model@1",
        prompt_version="prompt@1",
        max_output_tokens=2000,
    )

    response = await adapter.complete(request)
    parsed = SupervisorPlan.model_validate(
        response.structured["supervisor_plan"],
        strict=False,
    )
    assert len(base.repair_calls) == 1
    assert parsed.tasks[0].budget.max_model_calls == 4
    assert parsed.tasks[0].budget.max_tokens == 64_000


@pytest.mark.asyncio
async def test_supervisor_plan_materializes_dependency_safe_dag_and_dedups(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    await scheduler.create_run(
        "run_research",
        max_concurrency=3,
        actor_id="agent_supervisor",
        mutation_id="mutation_create",
    )
    await scheduler.submit(
        _task(
            "root",
            kind=TaskKind.ROOT,
            assigned_actor_id="agent_supervisor",
        ),
        actor_id="agent_supervisor",
        mutation_id="mutation_submit",
    )
    await scheduler.claim(
        "run_research",
        worker_id="agent_supervisor",
        limit=1,
        lease_seconds=60,
        mutation_id="mutation_claim",
    )
    plan = SupervisorPlan(
        run_id="run_research",
        root_task_id="task_root",
        cycle=0,
        action=SupervisorPlanAction.DECOMPOSE,
        objective="Research dependency semantics",
        tasks=(
            _proposal("source"),
            _proposal("verify", dependencies=("source",)),
        ),
        decision_summary="Create the source and verification DAG.",
    )
    command = Command(
        command_id="command_apply_plan",
        run_id="run_research",
        task_id="task_root",
        actor_id="agent_spec_research_supervisor_1_0_0",
        kind=CommandKind.DELEGATE,
        name="supervisor.apply_plan",
        arguments={"plan": plan.model_dump(mode="json")},
        idempotency_key="idempotency-apply-plan",
    )
    executor = SupervisorActionExecutor(
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
        actor_id=command.actor_id,
    )
    first = await executor.execute(command)
    replay = await executor.execute(command)
    assert first.data["child_task_ids"] == replay.data["child_task_ids"]
    snapshot = await scheduler.snapshot("run_research")
    children = [
        item
        for item in snapshot.tasks
        if item.envelope.parent_task_id == "task_root"
    ]
    assert len(children) == 2
    by_proposal = {
        item.envelope.constraints["proposal_key"]: item for item in children
    }
    assert by_proposal["verify"].envelope.dependency_task_ids == (
        by_proposal["source"].task_id,
    )

    changed_dependency = plan.model_copy(
        update={
            "plan_id": "plan_changed_dependency",
            "cycle": 1,
            "tasks": (
                _proposal("source"),
                _proposal("verify", dependencies=()),
            ),
        }
    )
    changed_command = command.model_copy(
        update={
            "command_id": "command_changed_dependency",
            "arguments": {
                "plan": changed_dependency.model_dump(mode="json")
            },
            "idempotency_key": "idempotency-changed-dependency",
        }
    )
    changed = await executor.execute(changed_command)
    assert len(changed.data["new_task_ids"]) == 1
    assert len((await scheduler.snapshot("run_research")).tasks) == 4
    coordination.integrity_check()
    artifacts.integrity_check()
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_supervisor_replay_repairs_crash_after_plan_before_dag_split(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    await scheduler.create_run(
        "run_research",
        max_concurrency=2,
        actor_id="agent_supervisor",
        mutation_id="mutation_create_crash",
    )
    root = _task(
        "root",
        kind=TaskKind.ROOT,
        assigned_actor_id="agent_supervisor",
    )
    await scheduler.submit(
        root,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_crash",
    )
    await scheduler.claim(
        root.run_id,
        worker_id="agent_supervisor",
        limit=1,
        lease_seconds=60,
        mutation_id="mutation_claim_crash",
    )
    plan = SupervisorPlan(
        plan_id="plan_crash_replay",
        run_id=root.run_id,
        root_task_id=root.task_id,
        cycle=0,
        action=SupervisorPlanAction.DECOMPOSE,
        objective="Recover a partially applied plan",
        tasks=(_proposal("recover"),),
        decision_summary="Persist then apply the recoverable task.",
    )
    executor = SupervisorActionExecutor(
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
        actor_id="agent_spec_research_supervisor_1_0_0",
    )
    fingerprint = executor._proposal_fingerprints(plan.tasks)["recover"]
    task_id = _stable_id(
        "task",
        plan.run_id,
        plan.root_task_id,
        fingerprint,
    )
    coordination.reserve_task(
        run_id=plan.run_id,
        fingerprint=fingerprint,
        task_id=task_id,
    )
    plan_artifact_id = _stable_id("artifact", plan.plan_id)
    artifacts.put_json(
        {
            "schema": "SupervisorPlanApplication@1",
            "plan": plan.model_dump(mode="json"),
            "task_ids_by_proposal": {"recover": task_id},
            "new_task_ids": [task_id],
            "duplicate_task_ids": [],
        },
        redact=False,
        kind=ArtifactKind.SUPERVISOR_PLAN,
        producer_id="agent_spec_research_supervisor_1_0_0",
        run_id=plan.run_id,
        task_id=plan.root_task_id,
        content_schema="SupervisorPlanApplication@1",
        artifact_id=plan_artifact_id,
        idempotency_key=f"supervisor-plan:{plan.plan_id}",
    )
    assert task_id not in (await scheduler.snapshot(plan.run_id)).by_id
    result = await executor.execute(
        Command(
            command_id="command_replay_crashed_plan",
            run_id=plan.run_id,
            task_id=plan.root_task_id,
            actor_id="agent_spec_research_supervisor_1_0_0",
            kind=CommandKind.DELEGATE,
            name="supervisor.apply_plan",
            arguments={"plan": plan.model_dump(mode="json")},
            idempotency_key="idempotency-replay-crashed-plan",
        )
    )
    assert result.data["new_task_ids"] == [task_id]
    assert task_id in (await scheduler.snapshot(plan.run_id)).by_id
    artifacts.integrity_check()
    coordination.integrity_check()
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_worker_boundary_deduplicates_queries_sources_and_information_gain(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    boundary = Boundary()
    executor = ResearchWorkerActionExecutor(
        command_executor=boundary,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )
    first_command = _command(
        "search_one",
        kind=CommandKind.SEARCH,
        arguments={"query": "  Durable   Research "},
    )
    second_command = _command(
        "search_two",
        kind=CommandKind.SEARCH,
        arguments={"query": "durable research"},
    )
    first = await executor.execute(first_command)
    second = await executor.execute(second_command)
    replayed_first = await executor.execute(first_command)
    assert len(boundary.calls) == 1
    assert first.normalized_data["_research"]["duplicate"] is False
    assert second.normalized_data["_research"]["duplicate"] is True
    assert second.normalized_data["_research"]["query_keys"] == [
        "durable research"
    ]
    assert replayed_first.normalized_data["_research"]["duplicate"] is True
    assert all(
        item != _stable_id(
            "artifact",
            first_command.command_id,
            "observation",
        )
        for item in artifacts.get(
            _stable_id(
                "artifact",
                first_command.command_id,
                "observation",
            )
        ).source_artifact_ids
    )
    assert all(artifacts.get(item) is not None for item in first.output_artifact_ids)

    estimator = InformationGainEstimator(coordination)
    first_gain = estimator.assess(
        command=first_command,
        observation=first,
    )
    second_gain = estimator.assess(
        command=second_command,
        observation=second,
    )
    assert first_gain.score > 0
    assert second_gain.score == 0
    assert second_gain.duplicate is True

    read_one = _command(
        "read_one",
        kind=CommandKind.READ,
        arguments={"url": "HTTPS://Example.com:443/source?utm_source=x"},
    )
    read_two = _command(
        "read_two",
        kind=CommandKind.READ,
        arguments={"url": "https://example.com/source"},
    )
    await executor.execute(read_one)
    await executor.execute(read_two)
    assert len(boundary.calls) == 2
    coordination.integrity_check()
    artifacts.integrity_check()
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_source_discovery_completes_after_novel_governed_search(
    tmp_path: Path,
) -> None:
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    verifier = WorkerObservationVerifier(
        InformationGainEstimator(coordination)
    )
    task = _task("discovery_complete", kind=TaskKind.SOURCE_DISCOVERY)
    command = _command(
        "discovery_complete",
        kind=CommandKind.SEARCH,
        arguments={"query": "primary RAG paper"},
    )
    now = utc_now()
    observation = Observation(
        command_id=command.command_id,
        run_id=command.run_id,
        task_id=command.task_id,
        actor_id=command.actor_id,
        status=ObservationStatus.SUCCEEDED,
        output_artifact_ids=("artifact_discovery_complete",),
        normalized_data={
            "sources": [
                {
                    "source_id": "source_result_0",
                    "url": "https://arxiv.org/abs/2501.12345",
                }
            ],
            "semantic_complete": False,
            "_research": {
                "duplicate": False,
                "query_keys": ["primary rag paper"],
                "source_keys": ["https://arxiv.org/abs/2501.12345"],
                "meaningful_artifact_ids": ["artifact_discovery_complete"],
            },
        },
        started_at=now,
        completed_at=now,
    )
    try:
        result = await verifier.verify(
            spec=None,
            task=task.model_copy(
                update={
                    "run_id": command.run_id,
                    "task_id": command.task_id,
                }
            ),
            command=command,
            observation=observation,
            prior_observations=(),
        )
        assert result.passed is True
        assert result.semantic_complete is True
        assert result.success is True
    finally:
        coordination.close()


@pytest.mark.asyncio
async def test_worker_supports_all_bounded_commands_and_structured_delegation(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    boundary = Boundary()
    executor = ResearchWorkerActionExecutor(
        command_executor=boundary,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )
    for index, kind in enumerate(
        (
            CommandKind.EXTRACT,
            CommandKind.COMPARE,
            CommandKind.VERIFY_SOURCE,
        )
    ):
        arguments = (
            {"url": f"https://example.com/source-{index}"}
            if kind == CommandKind.VERIFY_SOURCE
            else {"query": f"bounded operation {index}"}
        )
        observation = await executor.execute(
            _command(
                f"bounded_{index}",
                kind=kind,
                arguments=arguments,
            )
        )
        assert observation.status == ObservationStatus.SUCCEEDED
    assert [item.kind for item in boundary.calls] == [
        CommandKind.EXTRACT,
        CommandKind.COMPARE,
        CommandKind.VERIFY_SOURCE,
    ]

    await scheduler.create_run(
        "run_research",
        max_concurrency=2,
        actor_id="agent_supervisor",
        mutation_id="mutation_create_delegation",
    )
    parent = _task(
        "worker",
        assigned_actor_id="worker_one",
    )
    await scheduler.submit(
        parent,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_parent",
    )
    await scheduler.claim(
        "run_research",
        worker_id="worker_one",
        limit=1,
        lease_seconds=60,
        mutation_id="mutation_claim_parent",
    )
    child = _task(
        "delegated_original",
        parent=parent.task_id,
    )
    first_command = _command(
        "delegate_one",
        kind=CommandKind.DELEGATE,
        arguments={"task": child.model_dump(mode="json")},
    )
    first = await executor.execute(first_command)
    semantically_same = child.model_copy(
        update={
            "task_id": "task_delegated_duplicate",
            "budget": _budget(max_tokens=9_999),
            "priority": 1.0,
            "max_attempts": 7,
        }
    )
    second = await executor.execute(
        _command(
            "delegate_two",
            kind=CommandKind.DELEGATE,
            arguments={
                "task": semantically_same.model_dump(mode="json")
            },
        )
    )
    assert first.data["new_task"] is True
    assert second.data["new_task"] is False
    assert first.data["canonical_task_id"] == second.data["canonical_task_id"]
    snapshot = await scheduler.snapshot("run_research")
    assert len(snapshot.tasks) == 2
    delegation_artifact = artifacts.get(first.output_artifact_ids[0])
    assert delegation_artifact is not None
    assert delegation_artifact.kind == ArtifactKind.WORKER_DELEGATION
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_full_runtime_executes_supervisor_worker_merge_and_convergence(
    tmp_path: Path,
) -> None:
    run_id = "run_end_to_end"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    evidence = build_evidence_runtime(
        tmp_path / "evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_verification_policy(),
    )
    supervisor_model = QueueModel(
        [
            ModelResponse(
                structured={
                    "action": "decompose",
                    "objective": "Research durable execution",
                    "tasks": [
                        {
                            "proposal_key": "primary",
                            "kind": "source_discovery",
                            "title": "Find a primary source",
                            "goal": "Find and inspect one primary source",
                            "expected_output_schema": "ResearchWorkerResult@1",
                            "budget": _budget().model_dump(mode="json"),
                            "priority": 0.9,
                        }
                    ],
                    "decision_summary": "Delegate primary-source research.",
                },
                usage=BudgetUsage(
                    input_tokens=50,
                    output_tokens=30,
                    model_calls=1,
                ),
            )
        ]
    )
    worker_model = QueueModel(
        [
            ModelResponse(
                structured={
                    "summary": "Search for the primary source.",
                    "commands": [
                        {
                            "kind": "search",
                            "name": "research.search",
                            "arguments": {
                                "query": "durable research primary source"
                            },
                            "expected_output_schema": "SearchResult@1",
                        }
                    ],
                },
                usage=BudgetUsage(
                    input_tokens=40,
                    output_tokens=20,
                    model_calls=1,
                ),
            )
        ]
    )
    boundary = Boundary()
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=supervisor_model,
        worker_model=worker_model,
        command_executor=boundary,
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(),
        worker_ids=("worker_one", "worker_two"),
    )
    root = _task(
        "root_e2e",
        run_id=run_id,
        kind=TaskKind.ROOT,
    )
    outcome = await runtime.coordinator.run(
        root,
        max_concurrency=3,
    )
    assert outcome.action == ConvergenceAction.COMPLETE
    snapshot = await scheduler.snapshot(run_id)
    assert snapshot.control.status == RunControlStatus.COMPLETED
    assert all(
        item.envelope.status == TaskStatus.COMPLETED
        for item in snapshot.tasks
    )
    root_record = snapshot.by_id[root.task_id]
    child_record = next(
        item for item in snapshot.tasks if item.task_id != root.task_id
    )
    assert root_record.budget_usage.model_calls >= 1
    assert child_record.budget_usage.model_calls >= 1
    assert child_record.budget_usage.search_calls >= 1
    assert len(snapshot.tasks) == 2
    assert len(boundary.calls) == 1
    assert runtime.coordination.worker_results(run_id)
    assert runtime.coordination.merges(run_id)
    assert runtime.coordination.convergence_decisions(run_id)[0].action == (
        ConvergenceAction.COMPLETE
    )
    replay = await runtime.coordinator.run(root, max_concurrency=3)
    assert replay.action == ConvergenceAction.COMPLETE
    assert len(boundary.calls) == 1
    runtime.integrity_check()
    runtime.close()
    evidence.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_dynamic_replanning_continues_until_semantic_section_gate_passes(
    tmp_path: Path,
) -> None:
    run_id = "run_dynamic_replan"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    evidence = FakeEvidence(artifacts, complete_after=2)
    plans = [
        ModelResponse(
            structured={
                "action": "decompose",
                "tasks": [
                    {
                        "proposal_key": "initial",
                        "kind": "research",
                        "title": "Initial research",
                        "goal": "Establish initial section evidence",
                        "expected_output_schema": "ResearchWorkerResult@1",
                        "budget": _budget(max_tokens=64_000).model_dump(
                            mode="json"
                        ),
                    }
                ],
                "decision_summary": "Run initial research.",
            }
        ),
        ModelResponse(
            structured={
                "action": "replan",
                "tasks": [
                    {
                        "proposal_key": "gap",
                        "kind": "gap",
                        "title": "Repair section gap",
                        "goal": "Acquire the missing independent section evidence",
                        "expected_output_schema": "ResearchWorkerResult@1",
                        "budget": _budget(max_tokens=64_000).model_dump(
                            mode="json"
                        ),
                    }
                ],
                "decision_summary": "Replan from the persisted coverage gap.",
            }
        ),
    ]
    worker_responses = [
        ModelResponse(
            structured={
                "summary": "Search initial evidence.",
                "commands": [
                    {
                        "kind": "search",
                        "name": "research.search",
                        "arguments": {"query": "initial evidence"},
                    }
                ],
            }
        ),
        ModelResponse(
            structured={
                "summary": "Search gap evidence.",
                "commands": [
                    {
                        "kind": "search",
                        "name": "research.search",
                        "arguments": {"query": "independent gap evidence"},
                    }
                ],
            }
        ),
    ]
    supervisor_queue = QueueModel(plans)
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=supervisor_queue,
        worker_model=QueueModel(worker_responses),
        command_executor=Boundary(),
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(
            required_section_ids=("section_required",)
        ),
        worker_ids=("worker_one", "worker_two"),
        artifact_store=artifacts,
    )
    outcome = await runtime.coordinator.run(
        _task("dynamic_root", run_id=run_id, kind=TaskKind.ROOT),
        max_concurrency=3,
    )
    assert outcome.action == ConvergenceAction.COMPLETE
    decisions = runtime.coordination.convergence_decisions(run_id)
    assert [item.action for item in decisions] == [
        ConvergenceAction.REPLAN,
        ConvergenceAction.COMPLETE,
    ]
    assert decisions[0].snapshot.coverage_gap_section_ids == (
        "section_required",
    )
    assert len(supervisor_queue.requests) == 2
    assert "section_required" in repr(supervisor_queue.requests[1].messages)
    assert "previous_action" in repr(supervisor_queue.requests[1].messages)
    assert evidence.verification_calls == 2
    snapshot = await scheduler.snapshot(run_id)
    assert len(snapshot.tasks) == 3
    assert snapshot.control.status == RunControlStatus.COMPLETED
    runtime.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_coordinator_completes_reportable_run_with_explicit_gaps(
    tmp_path: Path,
) -> None:
    run_id = "run_reportable_gaps"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    evidence = FakeEvidence(artifacts, complete_after=99)
    evidence.knowledge.repository.claims = Collection(
        [
            type(
                "SupportedClaim",
                (),
                {
                    "claim_id": "claim_reportable_gaps",
                    "status": type("Status", (), {"value": "supported"})(),
                },
            )()
        ]
    )
    evidence.knowledge.repository.citations = Collection(
        [
            type(
                "VerifiedCitation",
                (),
                {"claim_id": "claim_reportable_gaps"},
            )()
        ]
    )
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "action": "decompose",
                        "tasks": [
                            {
                                "proposal_key": "source",
                                "kind": "source_discovery",
                                "title": "Find one more source",
                                "goal": "Search for one additional source.",
                                "expected_output_schema": (
                                    "ResearchWorkerResult@1"
                                ),
                                "budget": _budget().model_dump(mode="json"),
                            }
                        ],
                        "decision_summary": "Run one bounded source search.",
                    }
                )
            ]
        ),
        worker_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "summary": "Search one source.",
                        "commands": [
                            {
                                "kind": "search",
                                "name": "research.search",
                                "arguments": {"query": "bounded source"},
                            }
                        ],
                    }
                )
            ]
        ),
        command_executor=Boundary(),
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(
            required_section_ids=("section_required",),
            max_gap_replan_cycles=0,
        ),
        worker_ids=("worker_one",),
        artifact_store=artifacts,
    )
    try:
        outcome = await runtime.coordinator.run(
            _task(
                "reportable_gaps_root",
                run_id=run_id,
                kind=TaskKind.ROOT,
            ),
            max_concurrency=2,
        )
        assert outcome.action == ConvergenceAction.COMPLETE_WITH_GAPS
        snapshot = await scheduler.snapshot(run_id)
        assert snapshot.control.status == RunControlStatus.COMPLETED
        assert snapshot.by_id[
            "task_reportable_gaps_root"
        ].envelope.status == TaskStatus.COMPLETED
    finally:
        runtime.close()
        artifacts.close()
        await scheduler.close()


def _snapshot(
    run_id: str,
    *,
    run_status: RunControlStatus = RunControlStatus.ACTIVE,
    root_status: TaskStatus = TaskStatus.PAUSED,
    root_usage: BudgetUsage | None = None,
    child_status: TaskStatus | None = None,
) -> SchedulerSnapshot:
    root = _task(
        f"{run_id}_root",
        run_id=run_id,
        kind=TaskKind.ROOT,
    ).model_copy(update={"status": root_status})
    records = [
        TaskRecord(
            envelope=root,
            budget_usage=root_usage or BudgetUsage(),
        )
    ]
    if child_status is not None:
        child = _task(
            f"{run_id}_child",
            run_id=run_id,
            parent=root.task_id,
        ).model_copy(update={"status": child_status})
        records.append(TaskRecord(envelope=child))
    return SchedulerSnapshot(
        control=RunControl(run_id=run_id, status=run_status),
        tasks=tuple(records),
    )


def test_convergence_gate_covers_low_gain_budget_cancel_and_pending_precedence(
    tmp_path: Path,
) -> None:
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    evidence = FakeEvidence(artifacts, complete_after=99)
    evaluator = ConvergenceEvaluator(
        evidence=evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=_convergence_policy(
            required_section_ids=("section_required",),
            max_low_gain_cycles=2,
        ),
    )
    first_snapshot = _snapshot("run_low_gain")
    first = evaluator.assess(
        scheduler_snapshot=first_snapshot,
        root_task_id="task_run_low_gain_root",
        cycle=0,
        latest_information_gain=0.0,
    )
    second = evaluator.assess(
        scheduler_snapshot=first_snapshot,
        root_task_id="task_run_low_gain_root",
        cycle=1,
        latest_information_gain=0.0,
    )
    assert first.action == ConvergenceAction.REPLAN
    assert second.action == ConvergenceAction.STOP_LOW_GAIN
    assert second.snapshot.low_gain_cycles == 2

    pending = evaluator.assess(
        scheduler_snapshot=_snapshot(
            "run_pending",
            child_status=TaskStatus.READY,
        ),
        root_task_id="task_run_pending_root",
        cycle=0,
        latest_information_gain=0.0,
    )
    assert pending.action == ConvergenceAction.CONTINUE

    budget_evaluator = ConvergenceEvaluator(
        evidence=evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=_convergence_policy(
            required_section_ids=("section_required",),
            run_budget=_budget(max_model_calls=1),
        ),
    )
    budget = budget_evaluator.assess(
        scheduler_snapshot=_snapshot(
            "run_budget",
            root_usage=BudgetUsage(model_calls=1),
        ),
        root_task_id="task_run_budget_root",
        cycle=0,
        latest_information_gain=1.0,
    )
    assert budget.action == ConvergenceAction.STOP_BUDGET
    assert budget.snapshot.budget_exhausted is True

    admission_evaluator = ConvergenceEvaluator(
        evidence=evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=_convergence_policy(
            required_section_ids=("section_required",),
            run_budget=_budget(
                max_tokens=200_000,
                max_model_calls=200,
            ),
            minimum_replan_token_reserve=72_000,
        ),
    )
    admission = admission_evaluator.assess(
        scheduler_snapshot=_snapshot(
            "run_budget_admission",
            root_usage=BudgetUsage(input_tokens=128_000),
        ),
        root_task_id="task_run_budget_admission_root",
        cycle=0,
        latest_information_gain=1.0,
    )
    assert admission.action == ConvergenceAction.STOP_BUDGET
    assert admission.snapshot.budget_exhausted is True

    viable_admission = ConvergenceEvaluator(
        evidence=evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=_convergence_policy(
            required_section_ids=("section_required",),
            run_budget=_budget(
                max_tokens=300_000,
                max_model_calls=200,
            ),
            minimum_replan_token_reserve=72_000,
            estimated_tokens_per_planned_task=64_000,
        ),
    ).assess(
        scheduler_snapshot=_snapshot(
            "run_budget_viable_admission",
            root_usage=BudgetUsage(input_tokens=150_000),
        ),
        root_task_id="task_run_budget_viable_admission_root",
        cycle=0,
        latest_information_gain=1.0,
    )
    assert viable_admission.action == ConvergenceAction.REPLAN
    assert viable_admission.snapshot.budget_exhausted is False

    cancelled = evaluator.assess(
        scheduler_snapshot=_snapshot(
            "run_cancelled",
            run_status=RunControlStatus.CANCELLED,
            root_status=TaskStatus.CANCELLED,
        ),
        root_task_id="task_run_cancelled_root",
        cycle=0,
        latest_information_gain=1.0,
    )
    assert cancelled.action == ConvergenceAction.CANCEL
    assert cancelled.snapshot.cancelled is True

    evidence.section.coverage_status = SectionCoverageStatus.COMPLETE
    evidence.section.coverage_score = 1.0
    evidence.section.citation_score = 1.0
    blocker = type("Blocker", (), {"claim_id": "claim_high_impact"})()
    evidence.verified = type(
        "VerifiedWithBlocker",
        (),
        {
            "blocked_high_impact_claims": staticmethod(
                lambda run_id: (blocker,)
            )
        },
    )()
    evidence.knowledge.repository.conflicts.values = [
        type(
            "SevereConflict",
            (),
            {
                "conflict_id": "conflict_severe",
                "status": ConflictStatus.OPEN,
                "severity": ConflictSeverity.CRITICAL,
            },
        )()
    ]
    blocked = evaluator.assess(
        scheduler_snapshot=_snapshot("run_blocked"),
        root_task_id="task_run_blocked_root",
        cycle=0,
        latest_information_gain=1.0,
    )
    assert blocked.action == ConvergenceAction.REPLAN
    assert blocked.snapshot.coverage_gap_section_ids == ()
    assert blocked.snapshot.blocked_high_impact_claim_ids == (
        "claim_high_impact",
    )
    assert blocked.snapshot.severe_conflict_ids == ("conflict_severe",)

    reportable_evidence = FakeEvidence(artifacts, complete_after=99)
    reportable_evidence.knowledge.repository.claims = Collection(
        [
            type(
                "SupportedClaim",
                (),
                {
                    "claim_id": "claim_reportable",
                    "status": type("Status", (), {"value": "supported"})(),
                },
            )()
        ]
    )
    reportable_evidence.knowledge.repository.citations = Collection(
        [
            type(
                "VerifiedCitation",
                (),
                {"claim_id": "claim_reportable"},
            )()
        ]
    )
    bounded_evaluator = ConvergenceEvaluator(
        evidence=reportable_evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=_convergence_policy(
            required_section_ids=("section_required",),
            max_gap_replan_cycles=3,
        ),
    )
    bounded = bounded_evaluator.assess(
        scheduler_snapshot=_snapshot("run_bounded_gaps"),
        root_task_id="task_run_bounded_gaps_root",
        cycle=3,
        latest_information_gain=1.0,
    )
    assert bounded.action == ConvergenceAction.COMPLETE_WITH_GAPS
    assert bounded.snapshot.verified_claim_count == 1
    assert bounded.snapshot.verified_citation_count == 1
    assert artifacts.get(first.decision_artifact_id) is not None
    coordination.integrity_check()
    artifacts.integrity_check()
    coordination.close()
    artifacts.close()


@pytest.mark.asyncio
async def test_worker_pool_honors_global_concurrency_and_executes_all_tasks(
    tmp_path: Path,
) -> None:
    run_id = "run_bounded_pool"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    evidence = build_evidence_runtime(
        tmp_path / "evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_verification_policy(),
    )
    supervisor_model = QueueModel(
        [
            ModelResponse(
                structured={
                    "action": "decompose",
                    "tasks": [
                        {
                            "proposal_key": f"work_{index}",
                            "kind": "research",
                            "title": f"Parallel work {index}",
                            "goal": f"Research independent source {index}",
                            "expected_output_schema": "ResearchWorkerResult@1",
                            "budget": _budget(max_tokens=64_000).model_dump(
                                mode="json"
                            ),
                        }
                        for index in range(3)
                    ],
                    "decision_summary": "Execute three independent tasks.",
                }
            )
        ]
    )
    worker_model = QueueModel(
        [
            ModelResponse(
                structured={
                    "summary": f"Search source {index}.",
                    "commands": [
                        {
                            "kind": "search",
                            "name": "research.search",
                            "arguments": {"query": f"unique source {index}"},
                        }
                    ],
                }
            )
            for index in range(3)
        ]
    )
    boundary = Boundary(delay=0.05)
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=supervisor_model,
        worker_model=worker_model,
        command_executor=boundary,
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(
            minimum_replan_token_reserve=0,
        ),
        worker_ids=("worker_one", "worker_two", "worker_three"),
    )
    outcome = await runtime.coordinator.run(
        _task("pool_root", run_id=run_id, kind=TaskKind.ROOT),
        max_concurrency=2,
    )
    assert outcome.action == ConvergenceAction.COMPLETE
    assert boundary.peak == 2
    assert len(boundary.calls) == 3
    assert len(runtime.coordination.worker_results(run_id)) == 3
    assert len((await scheduler.snapshot(run_id)).tasks) == 4
    runtime.close()
    evidence.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_worker_runtime_uses_governed_gateway_retry_fallback_path(
    tmp_path: Path,
) -> None:
    from deep_researcher.contracts import ErrorCategory, ErrorRecord

    run_id = "run_gateway_fallback"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    evidence = build_evidence_runtime(
        tmp_path / "evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_verification_policy(),
    )
    definition_values = {
        "version": "1.0.0",
        "description": "Governed search used by the branch-08 worker.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["search"]},
                "query": {"type": "string", "minLength": 1},
            },
            "required": ["operation", "query"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "facts": {"type": "array"},
                "semantic_complete": {"type": "boolean"},
            },
            "required": ["results", "facts", "semantic_complete"],
            "additionalProperties": False,
        },
        "operations": ("search",),
        "timeout_seconds": 2.0,
        "max_attempts": 1,
    }
    primary_definition = ToolDefinition(
        name="research.search",
        fallback_tools=("research.search_fallback",),
        **definition_values,
    )
    fallback_definition = ToolDefinition(
        name="research.search_fallback",
        **definition_values,
    )
    primary = GatewayAdapter(
        ToolAdapterResult(
            success=False,
            error=ErrorRecord(
                category=ErrorCategory.TRANSIENT_PROVIDER,
                code="primary_unavailable",
                message="Primary provider is unavailable.",
                retryable=True,
            ),
            retryable=True,
        )
    )
    fallback = GatewayAdapter(
        ToolAdapterResult(
            success=True,
            data={
                "results": [
                    {"url": "https://fallback.example/primary-source"}
                ],
                "facts": [{"statement": "Fallback result is governed."}],
                "semantic_complete": True,
            },
            usage=BudgetUsage(search_calls=1),
        )
    )
    registry = ToolRegistry()
    registry.register(primary_definition, primary, activate=True)
    registry.register(fallback_definition, fallback, activate=True)
    tool_state = SQLiteToolStateStore(tmp_path / "tool_state.sqlite3")
    gateway = ProtocolToolGateway(
        registry=registry,
        state_store=tool_state,
        retry_base_seconds=0,
    )
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "action": "decompose",
                        "tasks": [
                            {
                                "proposal_key": "fallback",
                                "kind": "source_discovery",
                                "title": "Use governed fallback",
                                "goal": "Find a source even if primary fails",
                                "expected_output_schema": "ResearchWorkerResult@1",
                                "budget": _budget().model_dump(mode="json"),
                            }
                        ],
                        "decision_summary": "Delegate governed source discovery.",
                    }
                )
            ]
        ),
        worker_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "summary": "Search through the governed gateway.",
                        "commands": [
                            {
                                "kind": "search",
                                "name": "research.search",
                                "arguments": {
                                    "operation": "search",
                                    "query": "fallback evidence",
                                },
                            }
                        ],
                    }
                )
            ]
        ),
        command_executor=gateway,
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(),
        worker_ids=("worker_one",),
    )
    outcome = await runtime.coordinator.run(
        _task("gateway_root", run_id=run_id, kind=TaskKind.ROOT),
        max_concurrency=2,
    )
    assert outcome.action == ConvergenceAction.COMPLETE
    assert primary.calls == 1
    assert fallback.calls == 1
    audit_rows = tool_state._connection.execute(
        "SELECT event_type FROM tool_audit_events ORDER BY sequence_no"
    ).fetchall()
    assert any("fallback" in str(row[0]) for row in audit_rows)
    runtime.close()
    tool_state.close()
    evidence.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_supervisor_approval_is_durable_and_never_bypassed(
    tmp_path: Path,
) -> None:
    run_id = "run_supervisor_approval"
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    evidence = build_evidence_runtime(
        tmp_path / "evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_verification_policy(),
    )
    runtime = build_research_runtime(
        tmp_path / "research",
        scheduler=scheduler,
        evidence=evidence,
        supervisor_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "action": "request_approval",
                        "tasks": [],
                        "decision_summary": "Approval is required.",
                        "approval_reason": (
                            "The requested source requires an approved boundary."
                        ),
                    }
                ),
                ModelResponse(
                    structured={
                        "action": "decompose",
                        "tasks": [
                            {
                                "proposal_key": "approved_source",
                                "kind": "source_discovery",
                                "title": "Use approved source boundary",
                                "goal": "Research after explicit approval",
                                "expected_output_schema": "ResearchWorkerResult@1",
                                "budget": _budget().model_dump(mode="json"),
                            }
                        ],
                        "decision_summary": (
                            "Approval is recorded; delegate bounded research."
                        ),
                    }
                ),
            ]
        ),
        worker_model=QueueModel(
            [
                ModelResponse(
                    structured={
                        "summary": "Search within the approved boundary.",
                        "commands": [
                            {
                                "kind": "search",
                                "name": "research.search",
                                "arguments": {"query": "approved source"},
                            }
                        ],
                    }
                )
            ]
        ),
        command_executor=Boundary(),
        event_sink=EventCollector(),
        convergence_policy=_convergence_policy(),
        worker_ids=("worker_one",),
    )
    root = _task("approval_root", run_id=run_id, kind=TaskKind.ROOT)
    outcome = await runtime.coordinator.run(root, max_concurrency=2)
    assert outcome.action == ConvergenceAction.AWAIT_APPROVAL
    assert outcome.waiting_approval_task_ids == (root.task_id,)
    snapshot = await scheduler.snapshot(run_id)
    record = snapshot.by_id[root.task_id]
    assert record.envelope.status == TaskStatus.WAITING_APPROVAL
    assert record.approval is not None
    assert "approved boundary" in record.approval.reason
    assert record.budget_usage.model_calls >= 1
    plan_artifacts = evidence.knowledge.artifacts.list(
        ArtifactQuery(
            run_id,
            kinds=(ArtifactKind.SUPERVISOR_PLAN,),
        )
    ).items
    assert len(plan_artifacts) == 1
    replay = await runtime.coordinator.run(root, max_concurrency=2)
    assert replay.action == ConvergenceAction.AWAIT_APPROVAL
    await scheduler.approve(
        root.task_id,
        actor_id="user_research_owner",
        note="Approved for bounded research.",
        mutation_id="mutation_approve_supervisor",
    )
    completed = await runtime.coordinator.run(root, max_concurrency=2)
    assert completed.action == ConvergenceAction.COMPLETE
    assert (
        await scheduler.snapshot(run_id)
    ).control.status == RunControlStatus.COMPLETED
    runtime.close()
    evidence.close()
    await scheduler.close()


class FakeKernel:
    def __init__(self, values: list[KernelRunResult]) -> None:
        self.values = deque(values)

    async def run(self, **kwargs: Any) -> KernelRunResult:
        del kwargs
        return self.values.popleft()


class EmptyVerifier:
    def __init__(self) -> None:
        self.assessments: dict[str, Any] = {}


def _kernel_result(
    task: TaskEnvelope,
    *,
    status: TaskResultStatus,
    retryable: bool = False,
) -> KernelRunResult:
    from deep_researcher.contracts import ErrorCategory, ErrorRecord

    error = (
        ErrorRecord(
            category=ErrorCategory.TRANSIENT_PROVIDER,
            code="retryable_worker_failure",
            message="The provider failed transiently.",
            retryable=retryable,
            fatal=not retryable,
        )
        if status
        in {
            TaskResultStatus.FAILED,
            TaskResultStatus.CANCELLED,
            TaskResultStatus.REJECTED,
        }
        else None
    )
    reason = (
        StopReason.APPROVAL_REQUIRED
        if status == TaskResultStatus.DEFERRED
        else (
            StopReason.NO_ACTION_AVAILABLE
            if status == TaskResultStatus.PARTIAL
            else StopReason.REPEATED_ERROR
        )
    )
    return KernelRunResult(
        task_result=TaskResult(
            result_id=f"result_{task.task_id}_{status.value}",
            task_id=task.task_id,
            run_id=task.run_id,
            actor_id="agent_spec_research_worker_1_0_0",
            status=status,
            summary=f"Worker result: {status.value}",
            error=error,
            started_at=utc_now(),
            completed_at=utc_now(),
        ),
        stop_decision=StopDecision(
            should_stop=True,
            reason=reason,
            summary=f"Worker stopped: {status.value}",
            approval_required=status == TaskResultStatus.DEFERRED,
        ),
        effective_budget=task.budget,
    )


@pytest.mark.asyncio
async def test_worker_runner_schedules_retry_and_preserves_approval_state(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    await scheduler.create_run(
        "run_research",
        max_concurrency=2,
        actor_id="agent_supervisor",
        mutation_id="mutation_create",
    )
    retry_task = _task(
        "retry",
        assigned_actor_id="worker_retry",
    )
    approval_task = _task(
        "approval",
        assigned_actor_id="worker_approval",
    )
    partial_task = _task(
        "partial",
        assigned_actor_id="worker_partial",
    )
    await scheduler.submit(
        retry_task,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_retry",
    )
    await scheduler.submit(
        approval_task,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_approval",
    )
    await scheduler.submit(
        partial_task,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_partial",
    )
    retry_lease = (
        await scheduler.claim(
            "run_research",
            worker_id="worker_retry",
            limit=1,
            lease_seconds=60,
            mutation_id="mutation_claim_retry",
        )
    )[0]
    approval_lease = (
        await scheduler.claim(
            "run_research",
            worker_id="worker_approval",
            limit=1,
            lease_seconds=60,
            mutation_id="mutation_claim_approval",
        )
    )[0]
    verifier = EmptyVerifier()
    retry_runner = ResearchWorkerRunner(
        worker_id="worker_retry",
        agent_spec_id="agent_spec_research_worker_1_0_0",
        kernel=FakeKernel(
            [
                _kernel_result(
                    retry_lease.task,
                    status=TaskResultStatus.FAILED,
                    retryable=True,
                )
            ]
        ),
        verifier=verifier,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )
    retry_result = await retry_runner.run_lease(retry_lease)
    assert retry_result.retry_scheduled is True
    assert (
        await scheduler.snapshot("run_research")
    ).by_id[retry_task.task_id].envelope.status == TaskStatus.READY

    approval_runner = ResearchWorkerRunner(
        worker_id="worker_approval",
        agent_spec_id="agent_spec_research_worker_1_0_0",
        kernel=FakeKernel(
            [
                _kernel_result(
                    approval_lease.task,
                    status=TaskResultStatus.DEFERRED,
                )
            ]
        ),
        verifier=verifier,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )
    await approval_runner.run_lease(approval_lease)
    approval_record = (
        await scheduler.snapshot("run_research")
    ).by_id[approval_task.task_id]
    assert approval_record.envelope.status == TaskStatus.WAITING_APPROVAL
    assert approval_record.approval is not None
    partial_lease = (
        await scheduler.claim(
            "run_research",
            worker_id="worker_partial",
            limit=1,
            lease_seconds=60,
            mutation_id="mutation_claim_partial",
        )
    )[0]
    partial_runner = ResearchWorkerRunner(
        worker_id="worker_partial",
        agent_spec_id="agent_spec_research_worker_1_0_0",
        kernel=FakeKernel(
            [
                _kernel_result(
                    partial_lease.task,
                    status=TaskResultStatus.PARTIAL,
                )
            ]
        ),
        verifier=verifier,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )
    partial_result = await partial_runner.run_lease(partial_lease)
    assert partial_result.retry_scheduled is True
    assert (
        await scheduler.snapshot("run_research")
    ).by_id[partial_task.task_id].envelope.status == TaskStatus.READY
    assert len(coordination.worker_results("run_research")) == 3
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_meaningful_budget_partial_completes_discovery_dependency(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination = SQLiteResearchCoordinationStore(
        tmp_path / "coordination.sqlite3"
    )
    await scheduler.create_run(
        "run_meaningful_partial",
        max_concurrency=1,
        actor_id="agent_supervisor",
        mutation_id="mutation_create_meaningful_partial",
    )
    task = _task(
        "meaningful_partial",
        run_id="run_meaningful_partial",
        kind=TaskKind.SOURCE_DISCOVERY,
        assigned_actor_id="worker_meaningful",
    )
    await scheduler.submit(
        task,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_meaningful_partial",
    )
    lease = (
        await scheduler.claim(
            task.run_id,
            worker_id="worker_meaningful",
            limit=1,
            lease_seconds=60,
            mutation_id="mutation_claim_meaningful_partial",
        )
    )[0]
    source_artifact = artifacts.put_json(
        {"sources": [{"url": "https://example.org/source"}]},
        kind=ArtifactKind.SEARCH_RESPONSE,
        producer_id="agent_spec_research_worker_1_0_0",
        run_id=task.run_id,
        task_id=task.task_id,
        content_schema="ResearchObservation@1",
    )
    command = Command(
        command_id="command_meaningful_partial",
        run_id=task.run_id,
        task_id=task.task_id,
        actor_id="agent_spec_research_worker_1_0_0",
        kind=CommandKind.SEARCH,
        name="research.search",
        arguments={"operation": "search", "query": "bounded evidence"},
        expected_output_schema="ResearchSearchResult@1",
        idempotency_key="meaningful-partial-search",
    )
    verifier = EmptyVerifier()
    verifier.assessments[command.command_id] = InformationGainAssessment(
        task_id=task.task_id,
        command_id=command.command_id,
        score=0.6,
        new_artifact_ids=(source_artifact.artifact_id,),
        new_source_keys=("https://example.org/source",),
        new_query_keys=("bounded evidence",),
        decision_summary="A new governed source was persisted.",
    )
    kernel_result = KernelRunResult(
        task_result=TaskResult(
            result_id="result_meaningful_partial",
            task_id=task.task_id,
            run_id=task.run_id,
            actor_id="agent_spec_research_worker_1_0_0",
            status=TaskResultStatus.PARTIAL,
            output_artifact_ids=(source_artifact.artifact_id,),
            summary="Budget ended after useful source discovery.",
            started_at=utc_now(),
            completed_at=utc_now(),
        ),
        stop_decision=StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            summary="The bounded token budget was exhausted.",
            exhausted_dimensions=(BudgetDimension.TOKENS,),
        ),
        effective_budget=task.budget,
        commands=(command,),
    )
    runner = ResearchWorkerRunner(
        worker_id="worker_meaningful",
        agent_spec_id="agent_spec_research_worker_1_0_0",
        kernel=FakeKernel([kernel_result]),
        verifier=verifier,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
    )

    result = await runner.run_lease(lease)

    assert result.status == TaskResultStatus.PARTIAL
    assert result.information_gain == pytest.approx(0.6)
    assert result.retry_scheduled is False
    record = (await scheduler.snapshot(task.run_id)).by_id[task.task_id]
    assert record.envelope.status == TaskStatus.COMPLETED
    assert source_artifact.artifact_id in record.output_artifact_ids
    coordination.close()
    artifacts.close()
    await scheduler.close()


@pytest.mark.asyncio
async def test_worker_result_reconciles_restart_crash_before_scheduler_commit(
    tmp_path: Path,
) -> None:
    scheduler = _scheduler(tmp_path / "scheduler.sqlite3")
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    coordination_path = tmp_path / "coordination.sqlite3"
    coordination = SQLiteResearchCoordinationStore(coordination_path)
    await scheduler.create_run(
        "run_research",
        max_concurrency=1,
        actor_id="agent_supervisor",
        mutation_id="mutation_create_reconcile",
    )
    task = _task(
        "crash_reconcile",
        assigned_actor_id="worker_reconcile",
    )
    await scheduler.submit(
        task,
        actor_id="agent_supervisor",
        mutation_id="mutation_submit_reconcile",
    )
    lease = (
        await scheduler.claim(
            task.run_id,
            worker_id="worker_reconcile",
            limit=1,
            lease_seconds=600,
            mutation_id="mutation_claim_reconcile",
        )
    )[0]
    artifact = artifacts.put_json(
        {"schema": "ResearchWorkerResult@1", "recovered": True},
        kind=ArtifactKind.WORKER_RESULT,
        producer_id="agent_spec_research_worker_1_0_0",
        run_id=task.run_id,
        task_id=task.task_id,
    )
    intent = ResearchWorkerResult(
        worker_result_id="worker_result_crash_reconcile",
        run_id=task.run_id,
        task_id=task.task_id,
        worker_id="worker_reconcile",
        task_result_id="result_crash_reconcile",
        task_attempt=lease.task.attempt,
        status=TaskResultStatus.SUCCEEDED,
        output_artifact_ids=(artifact.artifact_id,),
        information_gain=0.8,
        command_count=1,
        usage=BudgetUsage(
            model_calls=1,
            tool_calls=1,
            search_calls=1,
        ),
        summary="The Worker finished before the process crashed.",
    )
    coordination.save_worker_result(intent)
    coordination.close()
    assert (
        await scheduler.snapshot(task.run_id)
    ).by_id[task.task_id].envelope.status == TaskStatus.RUNNING

    reopened = SQLiteResearchCoordinationStore(coordination_path)
    reconciler = ResearchWorkerResultReconciler(
        scheduler=scheduler,
        coordination=reopened,
    )
    assert await reconciler.reconcile(task.run_id) == (
        intent.worker_result_id,
    )
    record = (
        await scheduler.snapshot(task.run_id)
    ).by_id[task.task_id]
    assert record.envelope.status == TaskStatus.COMPLETED
    assert record.result_id == intent.task_result_id
    assert record.output_artifact_ids == intent.output_artifact_ids
    assert await reconciler.reconcile(task.run_id) == ()
    reopened.integrity_check()
    reopened.close()
    artifacts.close()
    await scheduler.close()


def test_coordination_identity_conflicts_are_rejected(tmp_path: Path) -> None:
    store = SQLiteResearchCoordinationStore(tmp_path / "coord.sqlite3")
    first = store.reserve_task(
        run_id="run_research",
        fingerprint="a" * 64,
        task_id="task_first",
    )
    assert first.inserted is True
    replay = store.reserve_task(
        run_id="run_research",
        fingerprint="a" * 64,
        task_id="task_second",
    )
    assert replay.canonical_task_id == "task_first"
    with pytest.raises(ResearchCoordinationConflict):
        store.reserve_task(
            run_id="run_research",
            fingerprint="b" * 64,
            task_id="task_first",
        )
    store.close()
