from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest

from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    Budget,
    BudgetDimension,
    BudgetUsage,
    Command,
    CommandKind,
    ComponentKind,
    ErrorCategory,
    ErrorRecord,
    EventType,
    MiddlewareSpec,
    MiddlewareStage,
    Observation,
    ObservationStatus,
    StopReason,
    TaskEnvelope,
    TaskKind,
    TaskResultStatus,
    TaskStatus,
    ToolGrant,
    VersionRef,
    utc_now,
)
from deep_researcher.kernel import (
    ActionExecutionError,
    AgentKernel,
    AgentSpecRegistry,
    CancellationToken,
    CommandPolicyChecker,
    ContextBuilder,
    EventRecorderKernelSink,
    KernelConfig,
    KernelEvent,
    ModelInvocationError,
    ModelRequest,
    ModelResponse,
    RawObservation,
    StopPolicyEngine,
    StopPolicyState,
    VerificationFeedback,
    component_versions_for_agent,
    effective_budget,
    estimate_tokens,
    normalize_commands,
    validate_middleware_pipeline,
)
from deep_researcher.events import EventQuery, EventRecorder, SQLiteEventStore
from deep_researcher.research import build_research_worker_spec


def _version(kind: ComponentKind, name: str) -> VersionRef:
    return VersionRef(kind=kind, name=name, version="1.0.0")


def _budget(**updates: Any) -> Budget:
    values: dict[str, Any] = {
        "max_tokens": 20_000,
        "max_cost_usd": 10.0,
        "max_wall_time_seconds": 10.0,
        "max_model_calls": 10,
        "max_tool_calls": 10,
        "max_search_calls": 10,
        "max_retries": 5,
        "max_errors": 5,
    }
    values.update(updates)
    return Budget(**values)


def _middleware() -> tuple[MiddlewareSpec, ...]:
    return tuple(MiddlewareSpec(stage=stage, order=index) for index, stage in enumerate(MiddlewareStage))


def _spec(
    name: str = "Worker",
    *,
    role: AgentRole = AgentRole.RESEARCH_WORKER,
    allowed: tuple[CommandKind, ...] = (CommandKind.TOOL, CommandKind.SEARCH, CommandKind.STOP),
    budget: Budget | None = None,
    max_parallel_commands: int = 2,
    supports_delegation: bool = False,
    enabled: bool = True,
    middleware: tuple[MiddlewareSpec, ...] | None = None,
) -> AgentSpec:
    return AgentSpec(
        name=name,
        version="1.0.0",
        role=role,
        description=f"Deterministic {name} test AgentSpec.",
        input_schema="TaskEnvelope@1",
        output_schema="TaskResult@1",
        allowed_commands=allowed,
        tool_grants=(
            ToolGrant(
                tool_name="web",
                allowed_operations=("search", "open"),
                max_calls_per_task=3,
                argument_constraints={
                    "required": ["operation"],
                    "allowed_properties": ["operation", "query"],
                    "properties": {
                        "operation": {"type": "string", "enum": ["search", "open"]},
                        "query": {"type": "string", "max_length": 200},
                    },
                },
            ),
        ),
        model=_version(ComponentKind.MODEL, f"{name.casefold()}-model"),
        prompt=_version(ComponentKind.PROMPT, f"{name.casefold()}-prompt"),
        tool_policy=_version(ComponentKind.TOOL_POLICY, f"{name.casefold()}-tools"),
        stop_policy=_version(ComponentKind.STOP_POLICY, f"{name.casefold()}-stop"),
        verification_policy=_version(ComponentKind.VERIFICATION_POLICY, f"{name.casefold()}-verify"),
        default_budget=budget or _budget(),
        middleware=middleware or _middleware(),
        context_window_tokens=8_000,
        reserved_output_tokens=1_000,
        max_parallel_commands=max_parallel_commands,
        supports_delegation=supports_delegation,
        enabled=enabled,
    )


def _task(
    suffix: str = "one",
    *,
    budget: Budget | None = None,
    constraints: dict[str, Any] | None = None,
) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=f"task_{suffix}",
        run_id=f"run_{suffix}",
        kind=TaskKind.RESEARCH,
        title="Research the specified question",
        goal="Produce a verified result without exposing private reasoning.",
        constraints=constraints or {},
        expected_output_schema="EvidencePack@1",
        budget=budget or _budget(),
        created_by="agent_supervisor",
    )


def _tool_response(*, operation: str = "search", query: str = "topic", summary: str = "Use the web tool") -> ModelResponse:
    return ModelResponse(
        structured={
            "summary": summary,
            "commands": [
                {
                    "kind": "tool",
                    "name": "web",
                    "arguments": {"operation": operation, "query": query},
                }
            ],
        },
        usage=BudgetUsage(input_tokens=20, output_tokens=10, cost_usd=0.01),
        response_id="response_tool",
    )


def _search_response(*, count: int = 1) -> ModelResponse:
    return ModelResponse(
        structured={
            "commands": [
                {"kind": "search", "name": f"search-{index}", "arguments": {"query": f"q{index}"}}
                for index in range(count)
            ]
        },
        usage=BudgetUsage(input_tokens=5, output_tokens=5),
    )


class QueueModel:
    def __init__(self, responses: list[ModelResponse | Exception], repairs: list[ModelResponse | Exception] | None = None) -> None:
        self.responses = deque(responses)
        self.repairs = deque(repairs or [])
        self.requests: list[ModelRequest] = []
        self.repair_requests: list[tuple[ModelRequest, ModelResponse, tuple[str, ...]]] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        value = self.responses.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    async def repair(self, request: ModelRequest, invalid_response: ModelResponse, errors: tuple[str, ...]) -> ModelResponse:
        self.repair_requests.append((request, invalid_response, errors))
        value = self.repairs.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class QueueExecutor:
    def __init__(self, values: list[RawObservation | Observation | Exception], *, delay: float = 0.0) -> None:
        self.values = deque(values)
        self.delay = delay
        self.commands: list[Command] = []

    async def execute(self, command: Command) -> RawObservation | Observation:
        self.commands.append(command)
        if self.delay:
            await asyncio.sleep(self.delay)
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class QueueVerifier:
    def __init__(self, values: list[VerificationFeedback | Exception]) -> None:
        self.values = deque(values)
        self.calls: list[tuple[Command, Observation, tuple[Observation, ...]]] = []

    async def verify(
        self,
        *,
        spec: AgentSpec,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        self.calls.append((command, observation, prior_observations))
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return value


class EventCollector:
    def __init__(self) -> None:
        self.events: list[KernelEvent] = []

    def emit(self, event: KernelEvent) -> None:
        self.events.append(event)


def _kernel(
    spec: AgentSpec,
    model: QueueModel,
    executor: QueueExecutor,
    verifier: QueueVerifier,
    *,
    config: KernelConfig | None = None,
    collector: EventCollector | None = None,
) -> tuple[AgentKernel, EventCollector]:
    registry = AgentSpecRegistry()
    registry.register(spec)
    sink = collector or EventCollector()
    return (
        AgentKernel(
            registry=registry,
            model_adapter=model,
            action_executor=executor,
            verifier=verifier,
            event_sink=sink,
            config=config,
        ),
        sink,
    )


@pytest.mark.asyncio
async def test_multiple_agent_specs_share_one_kernel_and_emit_complete_safe_trace():
    worker = _spec("Worker")
    reviewer = _spec("Reviewer", role=AgentRole.REPORT_REVIEWER)
    registry = AgentSpecRegistry()
    registry.register(worker)
    registry.register(reviewer)
    model = QueueModel([_tool_response(summary="Search primary sources"), _tool_response(operation="open", summary="Review source")])
    executor = QueueExecutor([
        RawObservation(status="success", data={"api_key": "secret", "result": "worker"}, output_artifact_ids=("artifact_worker",)),
        RawObservation(status="succeeded", data={"result": "reviewer"}, output_artifact_ids=("artifact_reviewer",)),
    ])
    verifier = QueueVerifier([
        VerificationFeedback(passed=True, success=True, information_gain=0.9, summary="Worker result verified."),
        VerificationFeedback(passed=True, success=True, information_gain=0.8, summary="Reviewer result verified."),
    ])
    events = EventCollector()
    kernel = AgentKernel(registry=registry, model_adapter=model, action_executor=executor, verifier=verifier, event_sink=events)

    worker_result = await kernel.run(agent_spec_id=worker.agent_spec_id, task=_task("worker", constraints={"api_key": "top-secret"}))
    reviewer_result = await kernel.run(agent_spec_id=reviewer.agent_spec_id, task=_task("reviewer"))

    assert worker_result.task_result.status == TaskResultStatus.SUCCEEDED
    assert reviewer_result.task_result.status == TaskResultStatus.SUCCEEDED
    assert worker_result.stop_decision.reason == StopReason.SUCCESS
    assert worker_result.task_result.output_artifact_ids == ("artifact_worker",)
    assert worker_result.task_result.usage.model_calls == 1
    assert worker_result.task_result.usage.tool_calls == 1
    assert executor.commands[0].actor_id == worker.agent_spec_id
    assert executor.commands[1].actor_id == reviewer.agent_spec_id
    assert model.requests[0].model_version == "worker-model@1.0.0"
    assert model.requests[1].model_version == "reviewer-model@1.0.0"
    assert "top-secret" not in repr(model.requests[0])
    assert "secret" not in repr(events.events)
    event_types = [event.event_type for event in events.events]
    for required in (
        "kernel.started",
        "model.started",
        "model.completed",
        "decision.recorded",
        "command.proposed",
        "policy.decided",
        "action.started",
        "action.completed",
        "verification.completed",
        "kernel.stopped",
    ):
        assert required in event_types


@pytest.mark.asyncio
async def test_schema_repair_is_bounded_budgeted_and_never_persists_raw_reasoning():
    spec = _spec()
    invalid = ModelResponse(content="I reasoned privately and chose something else")
    repaired = _tool_response(summary="Repaired structured decision")
    model = QueueModel([invalid], repairs=[repaired])
    executor = QueueExecutor([RawObservation(status="ok", data={"answer": 1})])
    verifier = QueueVerifier([VerificationFeedback(passed=True, success=True, information_gain=1.0, summary="Verified")])
    kernel, events = _kernel(spec, model, executor, verifier)

    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task())

    assert result.task_result.status == TaskResultStatus.SUCCEEDED
    assert result.task_result.usage.model_calls == 2
    assert result.task_result.usage.retries == 1
    assert result.task_result.usage.errors == 1
    assert len(model.repair_requests) == 1
    assert "privately" not in repr(result.decisions)
    assert "privately" not in repr(events.events)
    assert [event.event_type for event in events.events].count("model.started") == 2


@pytest.mark.asyncio
async def test_live_worker_repairs_deepseek_shape_then_executes_canonical_tool():
    spec = build_research_worker_spec()
    model = QueueModel(
        [
            ModelResponse(
                structured={
                    "commands": [
                        {
                            "arguments": {
                                "operation": "search",
                                "query": "RAG papers",
                            }
                        }
                    ]
                }
            )
        ],
        repairs=[
            ModelResponse(
                structured={
                    "commands": [
                        {
                            "kind": "search",
                            "arguments": {
                                "operation": "search",
                                "query": "RAG papers",
                            },
                        }
                    ]
                }
            )
        ],
    )
    executor = QueueExecutor(
        [RawObservation(status="succeeded", data={"items": []})]
    )
    verifier = QueueVerifier(
        [
            VerificationFeedback(
                passed=True,
                success=True,
                information_gain=0.1,
                summary="Canonical search executed.",
            )
        ]
    )
    kernel, _ = _kernel(spec, model, executor, verifier)
    result = await kernel.run(
        agent_spec_id=spec.agent_spec_id,
        task=_live_worker_task(),
    )
    assert result.task_result.status == TaskResultStatus.SUCCEEDED
    assert len(model.repair_requests) == 1
    assert executor.commands[0].name == "research.search"
    assert executor.commands[0].kind == CommandKind.SEARCH


@pytest.mark.asyncio
async def test_schema_repair_exhaustion_returns_structured_failure():
    spec = _spec()
    invalid = ModelResponse(content="not json")
    model = QueueModel([invalid], repairs=[invalid, invalid])
    kernel, _ = _kernel(
        spec,
        model,
        QueueExecutor([]),
        QueueVerifier([]),
        config=KernelConfig(max_schema_repairs=2),
    )

    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task())

    assert result.stop_decision.reason == StopReason.REPEATED_ERROR
    assert result.task_result.status == TaskResultStatus.FAILED
    assert result.task_result.error is not None
    assert result.task_result.error.category == ErrorCategory.SCHEMA_VALIDATION
    assert result.task_result.usage.model_calls == 3
    assert result.task_result.usage.retries == 2
    assert result.task_result.usage.errors == 3


def test_command_normalization_forces_identity_is_deterministic_and_redacts():
    spec = _spec()
    task = _task()
    response = ModelResponse(
        structured={
            "commands": [{
                "command_id": "command_injected",
                "run_id": "run_injected",
                "actor_id": "actor_injected",
                "kind": "tool",
                "name": "web",
                "risk_level": "high",
                "arguments": {"operation": "search", "api_key": "secret", "query": "q"},
                "metadata": {"chain_of_thought": "hidden", "round": 999},
            }]
        }
    )
    first = normalize_commands(response, spec=spec, task=task, round_no=2)[0]
    second = normalize_commands(response, spec=spec, task=task, round_no=2)[0]

    assert first == second
    assert first.command_id.startswith("command_")
    assert first.run_id == task.run_id
    assert first.task_id == task.task_id
    assert first.actor_id == spec.agent_spec_id
    assert first.requires_approval is True
    assert first.arguments["api_key"] == "[REDACTED]"
    assert "chain_of_thought" not in first.metadata
    assert first.metadata["round"] == 2


def _live_worker_task() -> TaskEnvelope:
    return _task(
        "live_worker_contract",
        constraints={
            "available_worker_tools": [
                "research.search",
                "research.read",
            ],
            "worker_tool_contracts": {
                "research.search": {
                    "kind": "search",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "search"},
                            "query": {"type": "string", "minLength": 1},
                        },
                        "required": ["operation", "query"],
                        "additionalProperties": False,
                    },
                },
                "research.read": {
                    "kind": "read",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "operation": {"const": "read"},
                            "url": {"type": "string", "minLength": 1},
                        },
                        "required": ["operation", "url"],
                        "additionalProperties": False,
                    },
                },
            },
        },
    )


def test_live_worker_schema_is_strict_and_uses_canonical_tool_contracts():
    spec = build_research_worker_spec(max_parallel_commands=5)
    task = _live_worker_task()
    request = ContextBuilder().build(
        spec=spec,
        task=task,
        observations=(),
        feedback=(),
        budget=spec.default_budget,
        usage=BudgetUsage(),
    )
    command_schema = request.command_schema["properties"]["commands"]
    assert command_schema["minItems"] == 1
    assert command_schema["maxItems"] == 5
    variants = command_schema["items"]["oneOf"]
    search = next(
        item
        for item in variants
        if item["properties"]["name"].get("const") == "research.search"
    )
    assert search["required"] == ["kind", "name", "arguments"]
    assert search["properties"]["kind"] == {"const": "search"}
    assert search["properties"]["arguments"]["required"] == [
        "operation",
        "query",
    ]


@pytest.mark.parametrize(
    ("raw", "expected_note"),
    [
        (
            {
                "name": "research.search",
                "arguments": {"operation": "search", "query": "RAG"},
            },
            "kind inferred from canonical name",
        ),
        (
            {
                "kind": "search",
                "arguments": {"operation": "search", "query": "RAG"},
            },
            "name inferred from command kind",
        ),
        (
            {
                "name": "search",
                "arguments": {"operation": "search", "query": "RAG"},
            },
            "short command alias normalized to canonical name",
        ),
    ],
)
def test_live_worker_normalizes_deepseek_command_identity(
    raw: dict[str, Any],
    expected_note: str,
):
    command = normalize_commands(
        ModelResponse(structured={"commands": [raw]}),
        spec=build_research_worker_spec(),
        task=_live_worker_task(),
        round_no=1,
    )[0]
    assert command.kind == CommandKind.SEARCH
    assert command.name == "research.search"
    assert expected_note in command.metadata["identity_normalization"]


@pytest.mark.parametrize(
    "raw",
    [
        {
            "kind": "read",
            "name": "research.search",
            "arguments": {"operation": "search", "query": "RAG"},
        },
        {
            "kind": "search",
            "name": "unknown.search",
            "arguments": {"operation": "search", "query": "RAG"},
        },
        {
            "kind": "search",
            "name": "research.search",
            "arguments": {"query": "RAG"},
        },
    ],
)
def test_live_worker_rejects_conflicting_unknown_or_invalid_tools(
    raw: dict[str, Any],
):
    with pytest.raises(Exception, match=r"commands\[0\]"):
        normalize_commands(
            ModelResponse(structured={"commands": [raw]}),
            spec=build_research_worker_spec(),
            task=_live_worker_task(),
            round_no=1,
        )


@pytest.mark.parametrize("kind", list(CommandKind))
def test_every_command_kind_has_a_normalized_policy_path(kind: CommandKind):
    spec = _spec(
        "AllCommands",
        allowed=tuple(CommandKind),
        supports_delegation=True,
    )
    task = _task(f"command_{kind.value}")
    arguments: dict[str, Any] = {}
    name = kind.value
    if kind == CommandKind.TOOL:
        name = "web"
        arguments = {"operation": "search", "query": "topic"}
    response = ModelResponse(structured={"commands": [{"kind": kind.value, "name": name, "arguments": arguments}]})

    command = normalize_commands(response, spec=spec, task=task, round_no=1)[0]
    policy = CommandPolicyChecker().check(command, spec)

    assert command.kind == kind
    assert command.run_id == task.run_id
    assert command.task_id == task.task_id
    assert policy.allowed
    assert policy.approval_required is (kind == CommandKind.REQUEST_APPROVAL)


def test_registry_is_version_immutable_supports_resolution_and_filters_disabled_specs():
    registry = AgentSpecRegistry()
    enabled = _spec("Shared")
    disabled = _spec("Disabled", enabled=False)
    registry.register(enabled)
    registry.register(enabled)
    registry.register(disabled)

    assert registry.require(enabled.agent_spec_id) == enabled
    assert registry.resolve("SHARED", "1.0.0") == enabled
    assert registry.list() == (enabled,)
    assert {item.agent_spec_id for item in registry.list(enabled_only=False)} == {enabled.agent_spec_id, disabled.agent_spec_id}
    with pytest.raises(ValueError, match="disabled"):
        registry.require(disabled.agent_spec_id)
    with pytest.raises(ValueError, match="immutable"):
        registry.register(enabled.model_copy(update={"description": "Changed immutable version"}))
    with pytest.raises(ValueError, match="immutable"):
        registry.register(_spec("Shared"))


def test_middleware_pipeline_and_effective_budget_are_strict():
    spec = _spec()
    assert validate_middleware_pipeline(spec) == tuple(MiddlewareStage)
    wrong = tuple(MiddlewareSpec(stage=stage, order=index) for index, stage in enumerate(reversed(tuple(MiddlewareStage))))
    with pytest.raises(ValueError, match="required kernel order"):
        validate_middleware_pipeline(_spec("Wrong", middleware=wrong))
    missing = tuple(item for item in _middleware() if item.stage != MiddlewareStage.POLICY_CHECK)
    with pytest.raises(ValueError, match="missing required"):
        validate_middleware_pipeline(_spec("Missing", middleware=missing))

    merged = effective_budget(
        Budget(max_tokens=1000, max_cost_usd=2.0),
        _budget(max_tokens=2000, max_cost_usd=1.0),
    )
    assert merged.max_tokens == 1000
    assert merged.max_cost_usd == 1.0
    assert merged.max_tool_calls == 10
    with pytest.raises(ValueError, match="all budget dimensions"):
        effective_budget(Budget(max_tokens=100), Budget(max_cost_usd=1.0))


def test_context_builder_trims_old_observations_redacts_and_injects_versions():
    spec = _spec().model_copy(update={"context_window_tokens": 1_200, "reserved_output_tokens": 300})
    task = _task(constraints={"authorization": "Bearer abc123", "scope": "public"})
    command = normalize_commands(_tool_response(), spec=spec, task=task, round_no=1)[0]
    observations = tuple(
        Observation(
            command_id=command.command_id,
            run_id=task.run_id,
            task_id=task.task_id,
            actor_id=spec.agent_spec_id,
            status=ObservationStatus.SUCCEEDED,
            normalized_data={"index": index, "body": "x" * 1400, "token": "private"},
            started_at=utc_now(),
        )
        for index in range(8)
    )
    request = ContextBuilder().build(
        spec=spec,
        task=task,
        observations=observations,
        feedback=("focus on sources",),
        budget=_budget(max_tokens=2_000),
        usage=BudgetUsage(),
    )

    assert estimate_tokens(request.system) + sum(estimate_tokens(item) for item in request.messages) <= 900
    assert len(request.messages) < len(observations) + 1
    assert all(item["role"] != "tool" for item in request.messages)
    assert any(
        isinstance(item["content"], dict)
        and item["content"].get("message_type")
        == "governed_tool_observation"
        for item in request.messages
    )
    assert request.model_version == "worker-model@1.0.0"
    assert request.prompt_version == "worker-prompt@1.0.0"
    assert "abc123" not in repr(request)
    assert "'token': 'private'" not in repr(request)
    assert "'token': '[REDACTED]'" in repr(request)


def test_research_phase_requires_read_before_exact_quote_extraction():
    now = utc_now()
    search = Observation(
        command_id="command_phase_search",
        run_id="run_phase",
        task_id="task_phase",
        actor_id="agent_phase",
        status=ObservationStatus.SUCCEEDED,
        normalized_data={
            "sources": [{"url": "https://arxiv.org/abs/2501.12345"}],
            "passages": [{"text": "Search result snippet only."}],
            "_tool": {"name": "research.search"},
        },
        started_at=now,
        completed_at=now,
    )
    after_search = ContextBuilder._research_phase((search,))
    assert after_search["phase"] == "source_read_required"
    assert "research.read" in after_search["required_next_action"]

    read = search.model_copy(
        update={
            "observation_id": "observation_phase_read",
            "command_id": "command_phase_read",
            "normalized_data": {
                "sources": [{"url": "https://arxiv.org/abs/2501.12345"}],
                "passages": [{"text": "Verbatim paper text."}],
                "_tool": {"name": "research.read"},
            },
        }
    )
    after_read = ContextBuilder._research_phase((search, read))
    assert after_read["phase"] == "extraction_required"
    assert "research.extract" in after_read["required_next_action"]


def test_policy_checks_kind_delegation_grants_arguments_limits_expiry_and_approval():
    spec = _spec()
    task = _task()
    checker = CommandPolicyChecker()
    allowed = normalize_commands(_tool_response(), spec=spec, task=task, round_no=1)[0]
    assert checker.check(allowed, spec).allowed

    not_granted = allowed.model_copy(update={"name": "shell"})
    assert not checker.check(not_granted, spec).allowed
    bad_operation = allowed.model_copy(update={"arguments": {"operation": "delete"}})
    assert not checker.check(bad_operation, spec).allowed
    missing = allowed.model_copy(update={"arguments": {}})
    assert not checker.check(missing, spec).allowed
    long_query = allowed.model_copy(update={"arguments": {"operation": "search", "query": "x" * 201}})
    assert not checker.check(long_query, spec).allowed
    assert not checker.check(allowed, spec, tool_call_counts={"web": 3}).allowed
    expired = allowed.model_copy(update={"expires_at": utc_now() - timedelta(seconds=1)})
    assert not checker.check(expired, spec).allowed
    high_risk = allowed.model_copy(update={"risk_level": "high", "requires_approval": True})
    assert checker.check(high_risk, spec).approval_required


@pytest.mark.asyncio
async def test_policy_denial_and_approval_stop_before_action_execution():
    spec = _spec()
    denied_model = QueueModel([ModelResponse(structured={"commands": [{"kind": "tool", "name": "shell", "arguments": {"operation": "exec"}}]})])
    denied_executor = QueueExecutor([])
    kernel, _ = _kernel(spec, denied_model, denied_executor, QueueVerifier([]))
    denied = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("denied"))
    assert denied.stop_decision.reason == StopReason.POLICY_DENIED
    assert denied.task_result.status == TaskResultStatus.REJECTED
    assert denied.task_result.error.category == ErrorCategory.POLICY_DENIED
    assert denied_executor.commands == []

    approval_model = QueueModel([ModelResponse(structured={"commands": [{
        "kind": "tool", "name": "web", "arguments": {"operation": "search"}, "risk_level": "high"
    }]})])
    approval_executor = QueueExecutor([])
    kernel, events = _kernel(spec, approval_model, approval_executor, QueueVerifier([]))
    approval = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("approval"))
    assert approval.stop_decision.reason == StopReason.APPROVAL_REQUIRED
    assert approval.stop_decision.approval_required
    assert approval.task_result.status == TaskResultStatus.DEFERRED
    assert approval_executor.commands == []
    assert "approval.requested" in [event.event_type for event in events.events]


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["success", "semantic_complete"])
async def test_explicit_stop_commands(reason: str):
    spec = _spec()
    model = QueueModel([ModelResponse(structured={"commands": [{"kind": "stop", "name": "done", "arguments": {"reason": reason, "summary": "Complete"}}]})])
    kernel, _ = _kernel(spec, model, QueueExecutor([]), QueueVerifier([]))
    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task(reason))
    assert result.stop_decision.reason.value == reason
    assert result.task_result.status == TaskResultStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_verifier_semantic_completion_and_no_information_gain_stop():
    spec = _spec()
    semantic_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([RawObservation(status="succeeded", data={})]),
        QueueVerifier([VerificationFeedback(passed=True, semantic_complete=True, summary="Enough verified evidence.")]),
    )
    semantic = await semantic_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("semantic"))
    assert semantic.stop_decision.reason == StopReason.SEMANTIC_COMPLETE

    low_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response(), _tool_response(), _tool_response()]),
        QueueExecutor([RawObservation(status="succeeded", data={}) for _ in range(3)]),
        QueueVerifier([VerificationFeedback(passed=False, information_gain=0.0, summary="No gain") for _ in range(3)]),
    )
    low = await low_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("low_gain"))
    assert low.stop_decision.reason == StopReason.LOW_INFORMATION_GAIN
    assert low.task_result.status == TaskResultStatus.PARTIAL
    assert len(low.observations) == 3


@pytest.mark.asyncio
async def test_repeated_action_error_stops_and_records_each_normalized_observation():
    spec = _spec(budget=_budget(max_errors=10, max_retries=5, max_tool_calls=10))
    error = ActionExecutionError("same transient failure", retryable=True)
    model = QueueModel([_tool_response()])
    executor = QueueExecutor([error, error, error])
    verifier = QueueVerifier([VerificationFeedback(passed=False, information_gain=0.2, summary="failed") for _ in range(3)])
    kernel, events = _kernel(spec, model, executor, verifier, config=KernelConfig(max_action_retries=5))
    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("repeat"))

    assert result.stop_decision.reason == StopReason.REPEATED_ERROR
    assert result.task_result.status == TaskResultStatus.FAILED
    assert len(result.observations) == 3
    assert all(item.status == ObservationStatus.FAILED for item in result.observations)
    assert [item.attempt for item in result.observations] == [1, 2, 3]
    assert result.task_result.usage.tool_calls == 3
    assert result.task_result.usage.errors == 3
    assert result.task_result.usage.retries == 2
    assert [event.event_type for event in events.events].count("action.failed") == 3


@pytest.mark.asyncio
async def test_model_and_verifier_retry_paths_are_bounded_and_accounted():
    spec = _spec()
    model = QueueModel([ModelInvocationError("temporary", retryable=True), _tool_response()])
    executor = QueueExecutor([RawObservation(status="succeeded", data={})])
    verifier = QueueVerifier([
        ModelInvocationError("verifier temporary", retryable=True),
        VerificationFeedback(passed=True, success=True, information_gain=0.5, summary="Recovered"),
    ])
    kernel, _ = _kernel(spec, model, executor, verifier)
    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("retry_paths"))

    assert result.task_result.status == TaskResultStatus.SUCCEEDED
    assert result.task_result.usage.model_calls == 2
    assert result.task_result.usage.retries == 2
    assert result.task_result.usage.errors == 2
    assert len(verifier.calls) == 2


@pytest.mark.asyncio
async def test_nonretryable_action_and_verifier_failures_are_structured():
    spec = _spec()
    action_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([ActionExecutionError("permanent action failure")]),
        QueueVerifier([VerificationFeedback(passed=False, information_gain=0.0, summary="Action failed")]),
    )
    action_result = await action_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("action_permanent"))
    assert action_result.stop_decision.reason == StopReason.VERIFICATION_FAILED
    assert action_result.task_result.status == TaskResultStatus.FAILED
    assert action_result.task_result.error.category == ErrorCategory.PERMANENT_PROVIDER

    verifier_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([RawObservation(status="succeeded", data={})]),
        QueueVerifier([RuntimeError("verifier unavailable")]),
    )
    verifier_result = await verifier_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("verifier_permanent"))
    assert verifier_result.stop_decision.reason == StopReason.VERIFICATION_FAILED
    assert verifier_result.task_result.status == TaskResultStatus.FAILED
    assert verifier_result.task_result.error.category == ErrorCategory.VERIFICATION


@pytest.mark.asyncio
async def test_cancellation_before_and_during_operation_is_terminal_and_interrupts_work():
    spec = _spec()
    before = CancellationToken()
    before.cancel()
    kernel, _ = _kernel(spec, QueueModel([_tool_response()]), QueueExecutor([]), QueueVerifier([]))
    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("cancel_before"), cancellation=before)
    assert result.stop_decision.reason == StopReason.USER_CANCELLED
    assert result.task_result.status == TaskResultStatus.CANCELLED
    assert result.task_result.error.category == ErrorCategory.CANCELLED

    class SlowModel(QueueModel):
        async def complete(self, request: ModelRequest) -> ModelResponse:
            self.requests.append(request)
            await asyncio.sleep(10)
            return _tool_response()

    during = CancellationToken()
    slow = SlowModel([])
    kernel, _ = _kernel(spec, slow, QueueExecutor([]), QueueVerifier([]))
    running = asyncio.create_task(kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("cancel_during"), cancellation=during))
    await asyncio.sleep(0.01)
    during.cancel()
    result = await asyncio.wait_for(running, timeout=1)
    assert result.stop_decision.reason == StopReason.USER_CANCELLED
    assert result.task_result.status == TaskResultStatus.CANCELLED


@pytest.mark.asyncio
async def test_no_action_and_permanent_model_failure_are_structured_terminal_results():
    spec = _spec()
    no_action_kernel, _ = _kernel(
        spec,
        QueueModel([ModelResponse(structured={"commands": []})]),
        QueueExecutor([]),
        QueueVerifier([]),
    )
    no_action = await no_action_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("no_action"))
    assert no_action.stop_decision.reason == StopReason.NO_ACTION_AVAILABLE
    assert no_action.task_result.status == TaskResultStatus.PARTIAL

    failed_kernel, _ = _kernel(
        spec,
        QueueModel([ModelInvocationError("bad request")]),
        QueueExecutor([]),
        QueueVerifier([]),
    )
    failed = await failed_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("model_failed"))
    assert failed.stop_decision.reason == StopReason.NO_ACTION_AVAILABLE
    assert failed.task_result.status == TaskResultStatus.FAILED
    assert failed.task_result.error.category == ErrorCategory.PERMANENT_PROVIDER


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("dimension", "task_budget", "model_response", "command_count"),
    [
        (BudgetDimension.TOKENS, _budget(max_tokens=2_000), ModelResponse(structured=_tool_response().structured, usage=BudgetUsage(input_tokens=2_000)), 0),
        (BudgetDimension.COST, _budget(max_cost_usd=1.0), ModelResponse(structured=_tool_response().structured, usage=BudgetUsage(cost_usd=1.0)), 0),
        (BudgetDimension.MODEL_CALLS, _budget(max_model_calls=1), _tool_response(), 1),
        (BudgetDimension.TOOL_CALLS, _budget(max_tool_calls=1), _search_response(count=2), 1),
        (BudgetDimension.SEARCH_CALLS, _budget(max_search_calls=1), _search_response(count=2), 1),
    ],
)
async def test_kernel_enforces_token_cost_model_tool_and_search_budgets(
    dimension: BudgetDimension,
    task_budget: Budget,
    model_response: ModelResponse,
    command_count: int,
):
    spec = _spec(max_parallel_commands=2)
    responses = [model_response]
    if dimension == BudgetDimension.MODEL_CALLS:
        responses = [model_response]
    model = QueueModel(responses)
    executor = QueueExecutor([RawObservation(status="succeeded", data={}) for _ in range(max(1, command_count))])
    verifier = QueueVerifier([VerificationFeedback(passed=False, information_gain=0.5, summary="continue") for _ in range(max(1, command_count))])
    kernel, _ = _kernel(spec, model, executor, verifier)

    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=_task(f"budget_{dimension.value}", budget=task_budget))

    assert result.stop_decision.reason == StopReason.BUDGET_EXHAUSTED
    assert dimension in result.stop_decision.exhausted_dimensions
    assert len(executor.commands) == command_count


@pytest.mark.asyncio
async def test_kernel_enforces_retry_error_and_wall_time_budgets():
    spec = _spec()
    retry_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([ActionExecutionError("retry", retryable=True)]),
        QueueVerifier([VerificationFeedback(passed=False, information_gain=0.5, summary="retry")]),
    )
    retry = await retry_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("budget_retry", budget=_budget(max_retries=0)))
    assert retry.stop_decision.reason == StopReason.BUDGET_EXHAUSTED
    assert BudgetDimension.RETRIES in retry.stop_decision.exhausted_dimensions

    error_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([ActionExecutionError("retry", retryable=True)]),
        QueueVerifier([VerificationFeedback(passed=False, information_gain=0.5, summary="retry")]),
    )
    error = await error_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("budget_error", budget=_budget(max_errors=1)))
    assert error.stop_decision.reason == StopReason.BUDGET_EXHAUSTED
    assert BudgetDimension.ERRORS in error.stop_decision.exhausted_dimensions

    wall_verifier = QueueVerifier([])
    wall_kernel, _ = _kernel(
        spec,
        QueueModel([_tool_response()]),
        QueueExecutor([RawObservation(status="succeeded", data={})], delay=0.2),
        wall_verifier,
    )
    wall = await wall_kernel.run(agent_spec_id=spec.agent_spec_id, task=_task("budget_wall", budget=_budget(max_wall_time_seconds=0.02)))
    assert wall.stop_decision.reason == StopReason.BUDGET_EXHAUSTED
    assert BudgetDimension.WALL_TIME in wall.stop_decision.exhausted_dimensions
    assert wall_verifier.calls == []
    assert wall.observations[-1].status == ObservationStatus.TIMEOUT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("raw_status", "expected"),
    [
        ("success", ObservationStatus.SUCCEEDED),
        ("failed", ObservationStatus.FAILED),
        ("timed_out", ObservationStatus.TIMEOUT),
        ("cancelled", ObservationStatus.CANCELLED),
        ("rejected", ObservationStatus.REJECTED),
        ("unknown", ObservationStatus.FAILED),
    ],
)
async def test_observation_normalization_covers_every_status_and_identity(raw_status: str, expected: ObservationStatus):
    spec = _spec()
    kernel, _ = _kernel(spec, QueueModel([_tool_response()]), QueueExecutor([]), QueueVerifier([]))
    task = _task(f"normalize_{raw_status}")
    command = normalize_commands(_tool_response(), spec=spec, task=task, round_no=1)[0]
    error = None
    if raw_status not in {"success", "unknown"}:
        error = ErrorRecord(category=ErrorCategory.TRANSIENT_PROVIDER, code="raw_error", message="raw failure", retryable=False)
    raw = RawObservation(
        status=raw_status,
        data={"password": "secret", "value": 1},
        usage=BudgetUsage(),
        error=error,
    )

    observation = kernel._normalize_observation(raw, command=command, attempt=2)

    assert observation.status == expected
    assert observation.command_id == command.command_id
    assert observation.run_id == task.run_id
    assert observation.task_id == task.task_id
    assert observation.actor_id == spec.agent_spec_id
    assert observation.attempt == 2
    assert observation.usage.tool_calls == 1
    assert observation.normalized_data["password"] == "[REDACTED]"
    if expected == ObservationStatus.SUCCEEDED:
        assert observation.error is None
    else:
        assert observation.error is not None
        assert observation.usage.errors == 1


def test_stop_policy_tracks_deadline_low_gain_repeated_error_and_cancellation():
    engine = StopPolicyEngine(max_low_gain_rounds=2, max_repeated_errors=2)
    state = StopPolicyState()
    token = CancellationToken()
    budget = _budget()
    engine.record_gain(state, 0.0)
    assert engine.evaluate(budget=budget, usage=BudgetUsage(), state=state, cancellation=token) is None
    engine.record_gain(state, 0.0)
    assert engine.evaluate(budget=budget, usage=BudgetUsage(), state=state, cancellation=token).reason == StopReason.LOW_INFORMATION_GAIN
    state = StopPolicyState()
    engine.record_error(state, "same")
    engine.record_error(state, "same")
    assert engine.evaluate(budget=budget, usage=BudgetUsage(), state=state, cancellation=token).reason == StopReason.REPEATED_ERROR
    token.cancel()
    assert engine.evaluate(budget=budget, usage=BudgetUsage(), state=state, cancellation=token).reason == StopReason.USER_CANCELLED
    deadline = _budget(deadline=utc_now() - timedelta(milliseconds=1))
    assert engine.evaluate(budget=deadline, usage=BudgetUsage(), state=StopPolicyState(), cancellation=CancellationToken()).reason == StopReason.BUDGET_EXHAUSTED


def test_kernel_rejects_terminal_or_wrongly_assigned_tasks_and_requires_event_sink():
    spec = _spec()
    registry = AgentSpecRegistry()
    registry.register(spec)
    with pytest.raises(ValueError, match="event sink"):
        AgentKernel(
            registry=registry,
            model_adapter=QueueModel([]),
            action_executor=QueueExecutor([]),
            verifier=QueueVerifier([]),
            event_sink=None,  # type: ignore[arg-type]
        )
    kernel = AgentKernel(
        registry=registry,
        model_adapter=QueueModel([]),
        action_executor=QueueExecutor([]),
        verifier=QueueVerifier([]),
        event_sink=EventCollector(),
    )
    terminal = _task("terminal").model_copy(update={"status": TaskStatus.COMPLETED})
    with pytest.raises(ValueError, match="terminal"):
        asyncio.run(kernel.run(agent_spec_id=spec.agent_spec_id, task=terminal))
    assigned = _task("assigned").model_copy(update={"assigned_actor_id": "agent_spec_other"})
    with pytest.raises(ValueError, match="different"):
        asyncio.run(kernel.run(agent_spec_id=spec.agent_spec_id, task=assigned))


@pytest.mark.asyncio
async def test_event_recorder_sink_persists_balanced_agent_model_tool_and_fact_events(tmp_path):
    spec = _spec()
    task = _task("persistent_trace")
    versions = component_versions_for_agent(
        spec,
        runtime=_version(ComponentKind.RUNTIME, "kernel-runtime"),
        scheduler=_version(ComponentKind.SCHEDULER, "test-scheduler"),
    )
    with SQLiteEventStore(tmp_path / "kernel-events.sqlite3") as store:
        sink = EventRecorderKernelSink(
            EventRecorder(store),
            run_id=task.run_id,
            thread_id="thread_kernel",
            trace_id="trace_kernel",
            correlation_id="correlation_kernel",
            component_versions=versions,
        )
        registry = AgentSpecRegistry()
        registry.register(spec)
        kernel = AgentKernel(
            registry=registry,
            model_adapter=QueueModel([_tool_response()]),
            action_executor=QueueExecutor([RawObservation(status="succeeded", data={"answer": 1})]),
            verifier=QueueVerifier([VerificationFeedback(passed=True, success=True, information_gain=1.0, summary="Persisted")]),
            event_sink=sink,
        )
        result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=task)

        assert result.task_result.status == TaskResultStatus.SUCCEEDED
        page = store.list(EventQuery(task.run_id, limit=1000))
        types = [event.event_type for event in page.items]
        assert types[0] == EventType.RUN_STARTED
        assert EventType.SPAN_STARTED in types
        assert EventType.MODEL_STARTED in types
        assert EventType.MODEL_COMPLETED in types
        assert EventType.COMMAND_PROPOSED in types
        assert EventType.POLICY_DECIDED in types
        assert EventType.TOOL_STARTED in types
        assert EventType.TOOL_COMPLETED in types
        assert EventType.VERIFICATION_COMPLETED in types
        assert types[-1] == EventType.SPAN_COMPLETED
        assert [event.sequence_no for event in page.items] == list(range(1, len(page.items) + 1))
        assert all(event.component_versions == versions for event in page.items)
        store.integrity_check()

        with pytest.raises(ValueError, match="run mismatch"):
            sink.emit(KernelEvent(event_type="kernel.started", run_id="run_other", task_id=task.task_id, actor_id=spec.agent_spec_id))


@pytest.mark.asyncio
async def test_persistent_sink_closes_schema_repair_failure_spans(tmp_path):
    spec = _spec()
    task = _task("persistent_repair_failure")
    versions = component_versions_for_agent(
        spec,
        runtime=_version(ComponentKind.RUNTIME, "kernel-runtime"),
        scheduler=_version(ComponentKind.SCHEDULER, "test-scheduler"),
    )
    with SQLiteEventStore(tmp_path / "kernel-repair-failure.sqlite3") as store:
        sink = EventRecorderKernelSink(
            EventRecorder(store),
            run_id=task.run_id,
            thread_id="thread_kernel",
            trace_id="trace_kernel_repair",
            correlation_id="correlation_kernel",
            component_versions=versions,
        )
        registry = AgentSpecRegistry()
        registry.register(spec)
        kernel = AgentKernel(
            registry=registry,
            model_adapter=QueueModel(
                [ModelResponse(content="not-json")],
                repairs=[ModelInvocationError("repair provider failed")],
            ),
            action_executor=QueueExecutor([]),
            verifier=QueueVerifier([]),
            event_sink=sink,
        )
        result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=task)

        assert result.task_result.status == TaskResultStatus.FAILED
        page = store.list(EventQuery(task.run_id, limit=1000))
        types = [event.event_type for event in page.items]
        assert types.count(EventType.MODEL_STARTED) == 2
        assert types.count(EventType.MODEL_COMPLETED) == 1
        assert types.count(EventType.MODEL_FAILED) == 1
        assert types.count(EventType.SPAN_STARTED) == 1
        assert types.count(EventType.SPAN_FAILED) == 1
        store.integrity_check()


@pytest.mark.asyncio
async def test_model_retry_cannot_cross_model_call_budget_and_task_deadline_preempts_calls():
    spec = _spec()
    model = QueueModel([ModelInvocationError("temporary", retryable=True), _tool_response()])
    kernel, _ = _kernel(spec, model, QueueExecutor([]), QueueVerifier([]))
    limited = await kernel.run(
        agent_spec_id=spec.agent_spec_id,
        task=_task("model_retry_budget", budget=_budget(max_model_calls=1)),
    )
    assert limited.stop_decision.reason == StopReason.BUDGET_EXHAUSTED
    assert BudgetDimension.MODEL_CALLS in limited.stop_decision.exhausted_dimensions
    assert limited.task_result.status == TaskResultStatus.PARTIAL
    assert len(model.requests) == 1

    deadline_model = QueueModel([_tool_response()])
    deadline_kernel, _ = _kernel(spec, deadline_model, QueueExecutor([]), QueueVerifier([]))
    expired = _task("task_deadline").model_copy(update={"deadline": utc_now() - timedelta(milliseconds=1)})
    deadline_result = await deadline_kernel.run(agent_spec_id=spec.agent_spec_id, task=expired)
    assert deadline_result.stop_decision.reason == StopReason.DEADLINE_REACHED
    assert deadline_result.task_result.status == TaskResultStatus.PARTIAL
    assert deadline_model.requests == []
