from __future__ import annotations

import asyncio
from collections import deque
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    Budget,
    BudgetUsage,
    Command,
    CommandKind,
    ComponentKind,
    ErrorCategory,
    ErrorRecord,
    MiddlewareSpec,
    MiddlewareStage,
    TaskEnvelope,
    TaskKind,
    TaskResultStatus,
    ToolGrant,
    VersionRef,
)
from deep_researcher.gateway import (
    CircuitBreakerPolicy,
    FunctionCallNormalizationError,
    FunctionCallNormalizer,
    PatternSafetyScanner,
    ProtocolToolGateway,
    RateLimit,
    SQLiteToolStateStore,
    SafetyDecision,
    ToolAdapterError,
    ToolAdapterResult,
    ToolDefinition,
    ToolErrorKind,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
    ToolRegistry,
    ToolRiskLevel,
)
from deep_researcher.kernel import (
    AgentKernel,
    AgentSpecRegistry,
    CancellationToken,
    ModelResponse,
    VerificationFeedback,
)


def _definition(
    name: str = "search.primary",
    version: str = "1.0.0",
    **updates: Any,
) -> ToolDefinition:
    values: dict[str, Any] = {
        "name": name,
        "version": version,
        "description": f"Test adapter for {name}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["search"]},
                "query": {"type": "string", "minLength": 1},
                "url": {"type": "string"},
            },
            "required": ["operation", "query"],
            "additionalProperties": False,
        },
        "output_schema": {
            "type": "object",
            "properties": {"items": {"type": "array"}},
            "required": ["items"],
            "additionalProperties": False,
        },
        "operations": ("search",),
        "permission_scopes": ("web:search",),
        "timeout_seconds": 1.0,
        "max_attempts": 3,
        "rate_limit": RateLimit(calls=20, window_seconds=60.0),
        "estimated_cost_usd": 0.02,
        "protocol": ToolProtocol.NATIVE,
        "provider": "test",
    }
    values.update(updates)
    return ToolDefinition(**values)


def _command(
    name: str = "search.primary",
    suffix: str = "one",
    *,
    arguments: dict[str, Any] | None = None,
    approval: bool = False,
) -> Command:
    return Command(
        command_id=f"command_{suffix}",
        run_id=f"run_{suffix}",
        task_id=f"task_{suffix}",
        actor_id="agent_worker",
        kind=CommandKind.TOOL,
        name=name,
        arguments=arguments or {"operation": "search", "query": "topic"},
        expected_output_schema="ToolExecutionResult@1",
        idempotency_key=f"idempotency-{suffix}",
        requires_approval=approval,
    )


class QueueAdapter:
    def __init__(self, values: list[ToolAdapterResult | Exception], *, delay: float = 0.0) -> None:
        self.values = deque(values)
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        self.calls.append(arguments)
        if self.delay:
            await asyncio.sleep(self.delay)
        value = self.values.popleft()
        if isinstance(value, Exception):
            raise value
        return value

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


def _ok(value: str = "result") -> ToolAdapterResult:
    return ToolAdapterResult(success=True, data={"items": [value]})


def _failed(message: str = "temporary", *, retryable: bool = True) -> ToolAdapterResult:
    return ToolAdapterResult(
        success=False,
        error=ErrorRecord(
            category=ErrorCategory.TRANSIENT_PROVIDER if retryable else ErrorCategory.PERMANENT_PROVIDER,
            code="adapter_failure",
            message=message,
            retryable=retryable,
            fatal=not retryable,
        ),
        retryable=retryable,
    )


def _context(*, approved: bool = False, permissions: frozenset[str] = frozenset({"web:search"}), cancellation=None, on_retry=None) -> ToolInvocationContext:
    return ToolInvocationContext(
        principal_id="principal_test",
        permissions=permissions,
        approved=approved,
        correlation_id="correlation_test",
        trace_id="trace_test",
        cancellation=cancellation,
        on_retry=on_retry,
    )


def _gateway(tmp_path: Path, definition: ToolDefinition, adapter: QueueAdapter, *, retry_base_seconds: float = 0.0):
    registry = ToolRegistry()
    registry.register(definition, adapter, activate=True)
    store = SQLiteToolStateStore(tmp_path / "tool-state.sqlite3")
    return ProtocolToolGateway(registry=registry, state_store=store, retry_base_seconds=retry_base_seconds), registry, store


def test_tool_definition_and_registry_enforce_versions_schemas_and_activation():
    adapter = QueueAdapter([_ok()])
    registry = ToolRegistry()
    v1 = _definition(version="1.0.0")
    v2 = _definition(version="2.0.0")
    registry.register(v1, adapter)
    registry.register(v2, adapter)
    assert registry.resolve(v1.name)[0] == v1
    registry.activate(v2.name, v2.version)
    assert registry.resolve(v2.name)[0] == v2
    assert registry.definitions(active_only=False) == (v1, v2)
    with pytest.raises(ValueError, match="immutable"):
        registry.register(v1.model_copy(update={"description": "changed"}), adapter)
    with pytest.raises(Exception):
        _definition(input_schema={"type": "not-a-json-schema-type"})
    with pytest.raises(Exception, match="high-risk"):
        _definition(risk_level=ToolRiskLevel.HIGH)
    with pytest.raises(Exception, match="non-idempotent"):
        _definition(side_effecting=True, idempotent=False)


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "search.primary", "arguments": {"operation": "search", "query": "q"}},
        {"id": "call_openai", "type": "function", "function": {"name": "search.primary", "arguments": '{"operation":"search","query":"q"}'}},
        {"type": "tool_use", "id": "call_anthropic", "name": "search.primary", "input": {"operation": "search", "query": "q"}},
        {"choices": [{"message": {"tool_calls": [{"id": "call_nested", "function": {"name": "search.primary", "arguments": {"operation": "search", "query": "q"}}}]}}]},
    ],
)
def test_function_calling_normalizes_common_wire_shapes(payload: dict[str, Any]):
    registry = ToolRegistry()
    definition = _definition(protocol=ToolProtocol.FUNCTION_CALLING)
    registry.register(definition, QueueAdapter([_ok()]))
    normalizer = FunctionCallNormalizer(registry)
    command = normalizer.normalize(payload, run_id="run_function", task_id="task_function", actor_id="agent_worker", provider="test")[0]
    duplicate = normalizer.normalize(payload, run_id="run_function", task_id="task_function", actor_id="agent_worker", provider="test")[0]
    assert command.command_id == duplicate.command_id
    assert command.idempotency_key == duplicate.idempotency_key
    assert command.arguments == duplicate.arguments
    assert command.kind == CommandKind.TOOL
    assert command.name == definition.name
    assert command.metadata["tool_version"] == definition.version
    assert command.metadata["protocol"] == "function_calling"
    assert normalizer.schemas()[0]["function"]["parameters"] == definition.input_schema


def test_function_calling_rejects_missing_calls_unknown_tools_and_bad_arguments():
    registry = ToolRegistry()
    registry.register(_definition(), QueueAdapter([_ok()]))
    normalizer = FunctionCallNormalizer(registry)
    with pytest.raises(FunctionCallNormalizationError, match="no tool calls"):
        normalizer.normalize({}, run_id="run_f", task_id="task_f", actor_id="agent_f")
    with pytest.raises(FunctionCallNormalizationError, match="unknown tool"):
        normalizer.normalize({"name": "missing", "arguments": {}}, run_id="run_f", task_id="task_f", actor_id="agent_f")
    with pytest.raises(FunctionCallNormalizationError, match="invalid JSON"):
        normalizer.normalize({"name": "search.primary", "arguments": "{"}, run_id="run_f", task_id="task_f", actor_id="agent_f")


@pytest.mark.asyncio
async def test_gateway_success_normalizes_usage_cost_observation_and_audit(tmp_path):
    adapter = QueueAdapter([
        ToolAdapterResult(success=True, data={"items": ["a"]}, usage=BudgetUsage(search_calls=1)),
        ToolAdapterResult(success=True, data={"items": ["b"]}, usage=BudgetUsage(search_calls=1)),
    ])
    gateway, _, store = _gateway(tmp_path, _definition(), adapter)
    command = _command()
    result = await gateway.execute_tool(command, _context())
    observation = await gateway.execute(
        command.model_copy(update={"command_id": "command_observation", "idempotency_key": "idempotency-observation"}),
        _context(),
    )
    assert result.success
    assert result.usage.tool_calls == 1
    assert result.usage.search_calls == 1
    assert result.usage.cost_usd == pytest.approx(0.02)
    assert observation.status.value == "succeeded"
    assert observation.normalized_data["_tool"]["version"] == "1.0.0"
    events = store.list_audit(command.command_id)
    assert [event["event_type"] for event in events] == ["tool.started", "tool.completed"]
    store.integrity_check()


@pytest.mark.asyncio
async def test_permissions_approval_schema_and_safety_fail_closed_before_adapter(tmp_path):
    definition = _definition(requires_approval=True, risk_level=ToolRiskLevel.HIGH)
    adapter = QueueAdapter([_ok()])
    gateway, _, _ = _gateway(tmp_path, definition, adapter)
    denied = await gateway.execute_tool(_command("search.primary", "permission"), _context(permissions=frozenset()))
    assert denied.error.category == ErrorCategory.POLICY_DENIED
    approval = await gateway.execute_tool(_command("search.primary", "approval", approval=True), _context())
    assert approval.error.category == ErrorCategory.APPROVAL_REQUIRED
    invalid = await gateway.execute_tool(_command("search.primary", "schema", arguments={"operation": "search"}), _context(approved=True))
    assert invalid.error.category == ErrorCategory.SCHEMA_VALIDATION
    unsafe = await gateway.execute_tool(
        _command("search.primary", "ssrf", arguments={"operation": "search", "query": "q", "url": "http://127.0.0.1/admin"}),
        _context(approved=True),
    )
    assert unsafe.error.code == ToolErrorKind.SAFETY.value
    assert adapter.calls == []


def test_safety_scanner_blocks_size_depth_control_and_non_http_urls():
    scanner = PatternSafetyScanner(max_payload_bytes=100, max_depth=2)
    definition = _definition()
    assert not scanner.scan({"value": "x" * 200}, definition, phase="input").allowed
    assert not scanner.scan({"a": {"b": {"c": 1}}}, definition, phase="input").allowed
    assert not scanner.scan({"value": "bad\x00value"}, definition, phase="input").allowed
    assert not scanner.scan({"url": "file:///etc/passwd"}, definition, phase="input").allowed
    assert scanner.scan({"url": "https://example.com"}, definition, phase="input").allowed


@pytest.mark.asyncio
async def test_retry_is_bounded_audited_and_accounted(tmp_path):
    retries: list[dict[str, Any]] = []
    adapter = QueueAdapter([_failed(), ToolAdapterError("network", kind=ToolErrorKind.TRANSIENT, retryable=True), _ok()])
    gateway, _, store = _gateway(tmp_path, _definition(max_attempts=3), adapter)
    result = await gateway.execute_tool(_command(suffix="retry"), _context(on_retry=retries.append))
    assert result.success
    assert result.attempts == 3
    assert result.usage.tool_calls == 3
    assert result.usage.retries == 2
    assert result.usage.errors == 2
    assert len(retries) == 2
    assert [item["event_type"] for item in store.list_audit("command_retry")].count("tool.retry_scheduled") == 2


@pytest.mark.asyncio
async def test_rate_limit_counts_each_retry_and_is_restart_safe(tmp_path):
    definition = _definition(max_attempts=3, rate_limit=RateLimit(calls=1, window_seconds=60))
    adapter = QueueAdapter([_failed(), _ok()])
    gateway, _, store = _gateway(tmp_path, definition, adapter)
    result = await gateway.execute_tool(_command(suffix="rate-retry"), _context())
    assert not result.success
    assert result.error.code == ToolErrorKind.RATE_LIMIT.value
    assert len(adapter.calls) == 1
    store.close()

    reopened = SQLiteToolStateStore(tmp_path / "tool-state.sqlite3")
    registry = ToolRegistry()
    registry.register(definition, adapter)
    gateway = ProtocolToolGateway(registry=registry, state_store=reopened, retry_base_seconds=0)
    blocked = await gateway.execute_tool(_command(suffix="rate-restart"), _context())
    assert blocked.error.code == ToolErrorKind.RATE_LIMIT.value
    reopened.close()


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_fallback_recovers(tmp_path):
    primary_definition = _definition(
        fallback_tools=("search.fallback",),
        max_attempts=1,
        circuit_breaker=CircuitBreakerPolicy(failure_threshold=1, recovery_seconds=60),
    )
    fallback_definition = _definition(name="search.fallback", max_attempts=1)
    primary = QueueAdapter([_failed()])
    fallback = QueueAdapter([_ok("fallback-one"), _ok("fallback-two")])
    gateway, registry, _ = _gateway(tmp_path, primary_definition, primary)
    registry.register(fallback_definition, fallback)
    first = await gateway.execute_tool(_command(suffix="fallback-one"), _context())
    second = await gateway.execute_tool(_command(suffix="fallback-two"), _context())
    assert first.success and second.success
    assert first.tool_name == "search.fallback"
    assert second.tool_name == "search.fallback"
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 2
    assert second.fallback_chain == ("search.fallback@1.0.0",)


@pytest.mark.asyncio
async def test_idempotency_replays_across_restart_and_rejects_key_reuse(tmp_path):
    definition = _definition()
    adapter = QueueAdapter([_ok("once")])
    gateway, registry, store = _gateway(tmp_path, definition, adapter)
    command = _command(suffix="durable")
    first = await gateway.execute_tool(command, _context())
    store.close()

    reopened = SQLiteToolStateStore(tmp_path / "tool-state.sqlite3")
    gateway = ProtocolToolGateway(registry=registry, state_store=reopened, retry_base_seconds=0)
    replay = await gateway.execute_tool(command, _context())
    assert first.success and replay.success
    assert replay.idempotent_replay
    assert len(adapter.calls) == 1
    changed = command.model_copy(update={"arguments": {"operation": "search", "query": "different"}})
    conflict = await gateway.execute_tool(changed, _context())
    assert conflict.error.code == ToolErrorKind.IDEMPOTENCY.value
    reopened.close()


@pytest.mark.asyncio
async def test_concurrent_duplicate_is_busy_and_does_not_double_execute(tmp_path):
    adapter = QueueAdapter([_ok()], delay=0.05)
    gateway, _, _ = _gateway(tmp_path, _definition(), adapter)
    command = _command(suffix="concurrent")
    first, second = await asyncio.gather(
        gateway.execute_tool(command, _context()),
        gateway.execute_tool(command, _context()),
    )
    assert sorted([first.success, second.success]) == [False, True]
    failed = first if not first.success else second
    assert failed.error.code == ToolErrorKind.IDEMPOTENCY.value
    assert failed.error.retryable
    assert len(adapter.calls) == 1


@pytest.mark.asyncio
async def test_cache_reuses_result_for_distinct_commands_and_survives_restart(tmp_path):
    definition = _definition(cache_ttl_seconds=60)
    adapter = QueueAdapter([_ok("cached")])
    gateway, registry, store = _gateway(tmp_path, definition, adapter)
    first = await gateway.execute_tool(_command(suffix="cache-one"), _context())
    second = await gateway.execute_tool(_command(suffix="cache-two"), _context())
    assert first.success and second.cached
    assert len(adapter.calls) == 1
    store.close()
    reopened = SQLiteToolStateStore(tmp_path / "tool-state.sqlite3")
    gateway = ProtocolToolGateway(registry=registry, state_store=reopened, retry_base_seconds=0)
    third = await gateway.execute_tool(_command(suffix="cache-three"), _context())
    assert third.cached
    assert len(adapter.calls) == 1
    reopened.close()


@pytest.mark.asyncio
async def test_timeout_and_cancellation_interrupt_adapter(tmp_path):
    slow = QueueAdapter([_ok()], delay=1)
    gateway, _, _ = _gateway(tmp_path, _definition(timeout_seconds=0.01, max_attempts=1), slow)
    timed_out = await gateway.execute_tool(_command(suffix="timeout"), _context())
    assert timed_out.error.code == ToolErrorKind.TIMEOUT.value
    token = CancellationToken()
    token.cancel()
    cancelled_adapter = QueueAdapter([_ok()])
    cancelled_gateway, _, _ = _gateway(tmp_path / "cancelled", _definition(), cancelled_adapter)
    cancelled = await cancelled_gateway.execute_tool(_command(suffix="cancelled"), _context(cancellation=token))
    assert cancelled.error.category == ErrorCategory.CANCELLED
    assert cancelled_adapter.calls == []


@pytest.mark.asyncio
async def test_output_schema_and_output_safety_failures_are_not_cached(tmp_path):
    invalid = ToolAdapterResult(success=True, data={"wrong": []})
    unsafe = ToolAdapterResult(success=True, data={"items": ["http://10.0.0.1/private"]})
    adapter = QueueAdapter([invalid, unsafe])
    gateway, _, _ = _gateway(tmp_path, _definition(cache_ttl_seconds=60, max_attempts=1), adapter)
    schema = await gateway.execute_tool(_command(suffix="output-schema"), _context())
    safety = await gateway.execute_tool(_command(suffix="output-safety"), _context())
    assert schema.error.category == ErrorCategory.PROTOCOL
    assert safety.error.code == ToolErrorKind.SAFETY.value
    assert not schema.cached and not safety.cached


def test_state_store_detects_audit_corruption(tmp_path):
    path = tmp_path / "corrupt.sqlite3"
    store = SQLiteToolStateStore(path)
    store.append_audit(run_id="run_a", task_id="task_a", command_id="command_a", event_type="test", payload={"ok": True})
    store.close()
    connection = sqlite3.connect(path)
    connection.execute("UPDATE tool_audit_events SET event_json='{}'")
    connection.commit()
    connection.close()
    reopened = SQLiteToolStateStore(path)
    with pytest.raises(RuntimeError, match="checksum"):
        reopened.integrity_check()
    reopened.close()


@pytest.mark.asyncio
async def test_agent_kernel_executes_a_governed_gateway_command_end_to_end(tmp_path):
    definition = _definition(max_attempts=1)
    adapter = QueueAdapter([_ok("kernel-result")])
    registry = ToolRegistry()
    registry.register(definition, adapter)
    store = SQLiteToolStateStore(tmp_path / "kernel-gateway.sqlite3")
    gateway = ProtocolToolGateway(
        registry=registry,
        state_store=store,
        retry_base_seconds=0,
        context_factory=lambda command: _context(),
    )

    version = lambda kind, name: VersionRef(kind=kind, name=name, version="1.0.0")
    budget = Budget(
        max_tokens=5_000,
        max_cost_usd=5,
        max_wall_time_seconds=10,
        max_model_calls=2,
        max_tool_calls=2,
        max_search_calls=2,
        max_retries=1,
        max_errors=1,
    )
    spec = AgentSpec(
        name="Gateway Worker",
        version="1.0.0",
        role=AgentRole.RESEARCH_WORKER,
        description="Executes a governed search command through AgentKernel.",
        input_schema="TaskEnvelope@1",
        output_schema="TaskResult@1",
        allowed_commands=(CommandKind.TOOL, CommandKind.STOP),
        tool_grants=(
            ToolGrant(
                tool_name=definition.name,
                allowed_operations=("search",),
                max_calls_per_task=2,
                argument_constraints={
                    "required": ["operation", "query"],
                    "allowed_properties": ["operation", "query"],
                    "properties": {
                        "operation": {"type": "string", "enum": ["search"]},
                        "query": {"type": "string", "max_length": 200},
                    },
                },
            ),
        ),
        model=version(ComponentKind.MODEL, "gateway-model"),
        prompt=version(ComponentKind.PROMPT, "gateway-prompt"),
        tool_policy=version(ComponentKind.TOOL_POLICY, "gateway-tools"),
        stop_policy=version(ComponentKind.STOP_POLICY, "gateway-stop"),
        verification_policy=version(ComponentKind.VERIFICATION_POLICY, "gateway-verify"),
        default_budget=budget,
        middleware=tuple(MiddlewareSpec(stage=stage, order=index) for index, stage in enumerate(MiddlewareStage)),
        context_window_tokens=4_000,
        reserved_output_tokens=500,
        max_parallel_commands=1,
    )

    class Model:
        async def complete(self, request):
            return ModelResponse(
                structured={
                    "summary": "Search through the governed gateway.",
                    "commands": [
                        {
                            "kind": "tool",
                            "name": definition.name,
                            "arguments": {"operation": "search", "query": "protocol gateway"},
                        }
                    ],
                },
                usage=BudgetUsage(input_tokens=10, output_tokens=5),
            )

        async def repair(self, request, invalid_response, errors):
            raise AssertionError("repair should not be needed")

    class Verifier:
        async def verify(self, **kwargs):
            observation = kwargs["observation"]
            assert observation.normalized_data["items"] == ["kernel-result"]
            assert observation.normalized_data["_tool"]["version"] == "1.0.0"
            return VerificationFeedback(
                passed=True,
                success=True,
                semantic_complete=True,
                information_gain=1.0,
                summary="Gateway result verified.",
            )

    class Events:
        def __init__(self):
            self.values = []

        def emit(self, event):
            self.values.append(event)

    spec_registry = AgentSpecRegistry()
    spec_registry.register(spec)
    events = Events()
    kernel = AgentKernel(
        registry=spec_registry,
        model_adapter=Model(),
        action_executor=gateway,
        verifier=Verifier(),
        event_sink=events,
    )
    task = TaskEnvelope(
        task_id="task_gateway_kernel",
        run_id="run_gateway_kernel",
        kind=TaskKind.RESEARCH,
        title="Gateway integration",
        goal="Verify AgentKernel to ProtocolToolGateway execution.",
        expected_output_schema="ToolExecutionResult@1",
        budget=budget,
        created_by="agent_supervisor",
    )

    result = await kernel.run(agent_spec_id=spec.agent_spec_id, task=task)

    assert result.task_result.status == TaskResultStatus.SUCCEEDED
    assert adapter.calls == [{"operation": "search", "query": "protocol gateway"}]
    assert [event.event_type for event in events.values].count("action.completed") == 1
    store.integrity_check()
    store.close()
