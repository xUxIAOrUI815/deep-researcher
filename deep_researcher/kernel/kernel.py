from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass
import hashlib
import inspect
import time
from typing import Any, Awaitable, TypeVar

from deep_researcher.contracts import (
    AgentDecisionSummary,
    AgentSpec,
    Budget,
    BudgetDimension,
    BudgetUsage,
    Command,
    CommandKind,
    ErrorCategory,
    ErrorRecord,
    Observation,
    ObservationStatus,
    StopDecision,
    StopReason,
    TaskEnvelope,
    TaskResult,
    TaskResultStatus,
    TaskStatus,
    utc_now,
)

from .middleware import (
    CommandPolicyChecker,
    ContextBuilder,
    effective_budget,
    normalize_commands,
    redact,
    validate_middleware_pipeline,
)
from .registry import AgentSpecRegistry
from .stop import StopPolicyEngine, StopPolicyState
from .types import (
    ActionExecutionError,
    ActionExecutor,
    CancellationToken,
    CommandSchemaError,
    KernelCancelled,
    KernelEvent,
    KernelEventSink,
    KernelTimedOut,
    KernelVerifier,
    ModelAdapter,
    ModelInvocationError,
    ModelRequest,
    ModelResponse,
    RawObservation,
    VerificationFeedback,
)


@dataclass(frozen=True)
class KernelConfig:
    max_schema_repairs: int = 2
    max_model_retries: int = 2
    max_action_retries: int = 2
    max_verification_retries: int = 1

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True)
class KernelRunResult:
    task_result: TaskResult
    stop_decision: StopDecision
    effective_budget: Budget
    commands: tuple[Command, ...] = ()
    observations: tuple[Observation, ...] = ()
    decisions: tuple[AgentDecisionSummary, ...] = ()
    verification_feedback: tuple[VerificationFeedback, ...] = ()


_T = TypeVar("_T")


class AgentKernel:
    """Framework-independent Observe -> Decide -> Act -> Verify runtime.

    The kernel owns a single task loop. Scheduling, provider transports,
    evidence persistence, and role-specific behavior are injected through the
    typed boundaries and remain outside this package.
    """

    def __init__(
        self,
        *,
        registry: AgentSpecRegistry,
        model_adapter: ModelAdapter,
        action_executor: ActionExecutor,
        verifier: KernelVerifier,
        event_sink: KernelEventSink,
        context_builder: ContextBuilder | None = None,
        policy_checker: CommandPolicyChecker | None = None,
        stop_policy: StopPolicyEngine | None = None,
        config: KernelConfig | None = None,
    ) -> None:
        if event_sink is None:
            raise ValueError("AgentKernel requires a durable event sink")
        self.registry = registry
        self.model_adapter = model_adapter
        self.action_executor = action_executor
        self.verifier = verifier
        self.event_sink = event_sink
        self.context_builder = context_builder or ContextBuilder()
        self.policy_checker = policy_checker or CommandPolicyChecker()
        self.stop_policy = stop_policy or StopPolicyEngine()
        self.config = config or KernelConfig()

    async def run(
        self,
        *,
        agent_spec_id: str,
        task: TaskEnvelope,
        cancellation: CancellationToken | None = None,
    ) -> KernelRunResult:
        spec = self.registry.require(agent_spec_id)
        validate_middleware_pipeline(spec)
        budget = effective_budget(task.budget, spec.default_budget, task_deadline=task.deadline)
        self._validate_task(task, spec)
        token = cancellation or CancellationToken()
        started_at = utc_now()
        started_monotonic = time.monotonic()
        usage = BudgetUsage()
        state = StopPolicyState()
        commands: list[Command] = []
        observations: list[Observation] = []
        decisions: list[AgentDecisionSummary] = []
        feedback_items: list[VerificationFeedback] = []
        repair_feedback: list[str] = []
        output_artifact_ids: list[str] = []
        tool_call_counts: dict[str, int] = {}
        round_no = 0
        total_information_gain = 0.0

        self._emit(
            "kernel.started",
            spec,
            task,
            {
                "agent_spec": f"{spec.name}@{spec.version}",
                "role": spec.role.value,
                "effective_budget": budget.model_dump(mode="json"),
            },
        )

        async def finish(stop: StopDecision, *, error: ErrorRecord | None = None) -> KernelRunResult:
            nonlocal usage
            usage = self._with_elapsed(usage, started_monotonic)
            status = self._result_status(stop.reason, error)
            if status in {TaskResultStatus.FAILED, TaskResultStatus.CANCELLED, TaskResultStatus.REJECTED} and error is None:
                error = self._stop_error(stop, spec, task)
            result = TaskResult(
                task_id=task.task_id,
                run_id=task.run_id,
                actor_id=spec.agent_spec_id,
                status=status,
                output_artifact_ids=tuple(output_artifact_ids),
                summary=stop.summary[:4000],
                usage=usage,
                metrics={
                    "kernel_rounds": float(round_no),
                    "commands": float(len(commands)),
                    "observations": float(len(observations)),
                    "information_gain": float(total_information_gain),
                },
                error=error,
                started_at=started_at,
                completed_at=utc_now(),
            )
            self._emit(
                "kernel.stopped",
                spec,
                task,
                {
                    "reason": stop.reason.value,
                    "summary": stop.summary,
                    "status": status.value,
                    "usage": usage.model_dump(mode="json"),
                    "exhausted_dimensions": [item.value for item in stop.exhausted_dimensions],
                    "approval_required": stop.approval_required,
                    "error": error.model_dump(mode="json") if error else None,
                },
            )
            return KernelRunResult(
                task_result=result,
                stop_decision=stop,
                effective_budget=budget,
                commands=tuple(commands),
                observations=tuple(observations),
                decisions=tuple(decisions),
                verification_feedback=tuple(feedback_items),
            )

        while True:
            usage = self._with_elapsed(usage, started_monotonic)
            stop = self._check_stop(
                budget=budget,
                usage=usage,
                state=state,
                cancellation=token,
                consumes=(BudgetDimension.MODEL_CALLS,),
            )
            if stop is not None:
                return await finish(stop)

            round_no += 1
            try:
                request = self.context_builder.build(
                    spec=spec,
                    task=task,
                    observations=tuple(observations),
                    feedback=tuple(repair_feedback),
                    budget=budget,
                    usage=usage,
                )
            except Exception as exc:
                error = self._error(
                    ErrorCategory.BUDGET_EXHAUSTED,
                    "context_budget_exhausted",
                    str(exc),
                    spec,
                    task,
                    fatal=True,
                )
                usage = usage.plus(errors=1)
                stop = StopDecision(
                    should_stop=True,
                    reason=StopReason.BUDGET_EXHAUSTED,
                    summary="The task context cannot fit within the effective token budget.",
                    exhausted_dimensions=(BudgetDimension.TOKENS,),
                )
                return await finish(stop, error=error)

            response: ModelResponse | None = None
            model_failure: ErrorRecord | None = None
            for model_attempt in range(1, self.config.max_model_retries + 2):
                self._emit(
                    "model.started",
                    spec,
                    task,
                    {
                        "round": round_no,
                        "attempt": model_attempt,
                        "model_version": request.model_version,
                        "prompt_version": request.prompt_version,
                        "estimated_input_tokens": request.metadata.get("estimated_input_tokens", 0),
                    },
                )
                call_started = time.monotonic()
                try:
                    response = await self._await_controlled(
                        self.model_adapter.complete(request),
                        cancellation=token,
                        timeout_seconds=self._remaining_seconds(budget, started_monotonic),
                    )
                    if not isinstance(response, ModelResponse):
                        raise ModelInvocationError("ModelAdapter returned a non-ModelResponse value")
                except KernelCancelled:
                    usage = self._with_elapsed(usage.plus(model_calls=1), started_monotonic)
                    error = self._error(
                        ErrorCategory.CANCELLED,
                        "model_cancelled",
                        "Model invocation was cancelled.",
                        spec,
                        task,
                        attempt=model_attempt,
                        fatal=True,
                    )
                    self._emit("model.failed", spec, task, {"round": round_no, "attempt": model_attempt, "error": error.model_dump(mode="json")})
                    return await finish(self._cancelled_stop(), error=error)
                except KernelTimedOut:
                    usage = self._with_elapsed(usage.plus(model_calls=1, errors=1), started_monotonic)
                    error = self._error(
                        ErrorCategory.BUDGET_EXHAUSTED,
                        "model_wall_time_exhausted",
                        "Model invocation exceeded the remaining wall-time budget.",
                        spec,
                        task,
                        attempt=model_attempt,
                        fatal=True,
                    )
                    self._emit("model.failed", spec, task, {"round": round_no, "attempt": model_attempt, "error": error.model_dump(mode="json")})
                    return await finish(self._wall_time_stop(budget), error=error)
                except Exception as exc:
                    retryable = isinstance(exc, ModelInvocationError) and exc.retryable
                    usage = self._with_elapsed(usage.plus(model_calls=1, errors=1), started_monotonic)
                    fingerprint = self._fingerprint("model", type(exc).__name__, str(exc))
                    self.stop_policy.record_error(state, fingerprint)
                    model_failure = self._error(
                        ErrorCategory.TRANSIENT_PROVIDER if retryable else ErrorCategory.PERMANENT_PROVIDER,
                        "model_invocation_failed",
                        str(exc) or type(exc).__name__,
                        spec,
                        task,
                        attempt=model_attempt,
                        retryable=retryable,
                        fatal=not retryable,
                    )
                    self._emit(
                        "model.failed",
                        spec,
                        task,
                        {
                            "round": round_no,
                            "attempt": model_attempt,
                            "latency_ms": (time.monotonic() - call_started) * 1000,
                            "error": model_failure.model_dump(mode="json"),
                        },
                    )
                    if not retryable or model_attempt > self.config.max_model_retries:
                        stop = StopDecision(
                            should_stop=True,
                            reason=StopReason.REPEATED_ERROR if model_attempt > 1 else StopReason.NO_ACTION_AVAILABLE,
                            summary="The model could not produce a decision.",
                        )
                        return await finish(stop, error=model_failure)
                    retry_stop = self._before_model_retry(budget, usage, state, token)
                    if retry_stop is not None:
                        return await finish(retry_stop, error=model_failure if retry_stop.reason == StopReason.REPEATED_ERROR else None)
                    usage = usage.plus(retries=1)
                    self._emit("retry.scheduled", spec, task, {"operation": "model", "attempt": model_attempt + 1})
                    continue
                else:
                    response_usage = response.usage.model_copy(
                        update={"model_calls": max(1, response.usage.model_calls)}
                    )
                    usage = self._merge_usage(usage, response_usage, started_monotonic)
                    self._emit(
                        "model.completed",
                        spec,
                        task,
                        {
                            "round": round_no,
                            "attempt": model_attempt,
                            "response_id": response.response_id,
                            "finish_reason": response.finish_reason,
                            "latency_ms": max(response.latency_ms, (time.monotonic() - call_started) * 1000),
                            "usage": response_usage.model_dump(mode="json"),
                        },
                    )
                    break

            assert response is not None
            parsed_commands: tuple[Command, ...] | None = None
            invalid_response = response
            schema_error: ErrorRecord | None = None
            for repair_attempt in range(0, self.config.max_schema_repairs + 1):
                try:
                    parsed_commands = normalize_commands(invalid_response, spec=spec, task=task, round_no=round_no)
                    break
                except CommandSchemaError as exc:
                    usage = usage.plus(errors=1)
                    fingerprint = self._fingerprint("schema", *exc.errors)
                    self.stop_policy.record_error(state, fingerprint)
                    schema_error = self._error(
                        ErrorCategory.SCHEMA_VALIDATION,
                        "command_schema_invalid",
                        "; ".join(exc.errors),
                        spec,
                        task,
                        attempt=repair_attempt + 1,
                        retryable=repair_attempt < self.config.max_schema_repairs,
                        fatal=repair_attempt >= self.config.max_schema_repairs,
                    )
                    self._emit(
                        "command.schema_invalid",
                        spec,
                        task,
                        {"round": round_no, "repair_attempt": repair_attempt, "errors": list(exc.errors)},
                    )
                    if repair_attempt >= self.config.max_schema_repairs:
                        stop = StopDecision(
                            should_stop=True,
                            reason=StopReason.REPEATED_ERROR,
                            summary="Structured command generation failed after bounded schema repair.",
                        )
                        return await finish(stop, error=schema_error)
                    retry_stop = self._before_model_repair(budget, usage, state, token)
                    if retry_stop is not None:
                        return await finish(retry_stop, error=schema_error if retry_stop.reason == StopReason.REPEATED_ERROR else None)
                    usage = usage.plus(retries=1)
                    self._emit(
                        "retry.scheduled",
                        spec,
                        task,
                        {"operation": "schema_repair", "attempt": repair_attempt + 1, "errors": list(exc.errors)},
                    )
                    try:
                        repair_request = self.context_builder.build(
                            spec=spec,
                            task=task,
                            observations=tuple(observations),
                            feedback=tuple((*repair_feedback, *exc.errors)),
                            budget=budget,
                            usage=usage,
                        )
                    except Exception as context_exc:
                        error = self._error(
                            ErrorCategory.BUDGET_EXHAUSTED,
                            "repair_context_budget_exhausted",
                            str(context_exc),
                            spec,
                            task,
                            fatal=True,
                        )
                        usage = usage.plus(errors=1)
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=StopReason.BUDGET_EXHAUSTED,
                                summary="Schema repair context cannot fit within the remaining token budget.",
                                exhausted_dimensions=(BudgetDimension.TOKENS,),
                            ),
                            error=error,
                        )
                    self._emit(
                        "model.started",
                        spec,
                        task,
                        {
                            "round": round_no,
                            "repair_attempt": repair_attempt + 1,
                            "operation": "schema_repair",
                            "model_version": repair_request.model_version,
                            "prompt_version": repair_request.prompt_version,
                            "estimated_input_tokens": repair_request.metadata.get("estimated_input_tokens", 0),
                        },
                    )
                    try:
                        invalid_response = await self._await_controlled(
                            self.model_adapter.repair(repair_request, invalid_response, exc.errors),
                            cancellation=token,
                            timeout_seconds=self._remaining_seconds(budget, started_monotonic),
                        )
                        if not isinstance(invalid_response, ModelResponse):
                            raise ModelInvocationError("ModelAdapter.repair returned a non-ModelResponse value")
                    except KernelCancelled:
                        usage = self._with_elapsed(usage.plus(model_calls=1), started_monotonic)
                        error = self._error(
                            ErrorCategory.CANCELLED,
                            "schema_repair_cancelled",
                            "Schema repair was cancelled.",
                            spec,
                            task,
                            attempt=repair_attempt + 1,
                            fatal=True,
                        )
                        self._emit(
                            "model.failed",
                            spec,
                            task,
                            {"round": round_no, "repair_attempt": repair_attempt + 1, "error": error.model_dump(mode="json")},
                        )
                        return await finish(self._cancelled_stop(), error=error)
                    except KernelTimedOut:
                        usage = self._with_elapsed(usage.plus(model_calls=1, errors=1), started_monotonic)
                        error = self._error(
                            ErrorCategory.BUDGET_EXHAUSTED,
                            "schema_repair_wall_time_exhausted",
                            "Schema repair exceeded the remaining wall-time budget.",
                            spec,
                            task,
                            attempt=repair_attempt + 1,
                            fatal=True,
                        )
                        self._emit(
                            "model.failed",
                            spec,
                            task,
                            {"round": round_no, "repair_attempt": repair_attempt + 1, "error": error.model_dump(mode="json")},
                        )
                        return await finish(self._wall_time_stop(budget), error=error)
                    except Exception as repair_exc:
                        usage = self._with_elapsed(usage.plus(model_calls=1, errors=1), started_monotonic)
                        schema_error = self._error(
                            ErrorCategory.TRANSIENT_PROVIDER if getattr(repair_exc, "retryable", False) else ErrorCategory.PERMANENT_PROVIDER,
                            "schema_repair_failed",
                            str(repair_exc) or type(repair_exc).__name__,
                            spec,
                            task,
                            attempt=repair_attempt + 1,
                            retryable=bool(getattr(repair_exc, "retryable", False)),
                            fatal=True,
                        )
                        self._emit(
                            "model.failed",
                            spec,
                            task,
                            {
                                "round": round_no,
                                "repair_attempt": repair_attempt + 1,
                                "error": schema_error.model_dump(mode="json"),
                            },
                        )
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=StopReason.REPEATED_ERROR,
                                summary="The model failed while repairing its structured command response.",
                            ),
                            error=schema_error,
                        )
                    repair_usage = invalid_response.usage.model_copy(
                        update={"model_calls": max(1, invalid_response.usage.model_calls)}
                    )
                    usage = self._merge_usage(usage, repair_usage, started_monotonic)
                    self._emit(
                        "model.completed",
                        spec,
                        task,
                        {
                            "round": round_no,
                            "repair_attempt": repair_attempt + 1,
                            "response_id": invalid_response.response_id,
                            "finish_reason": invalid_response.finish_reason,
                            "usage": repair_usage.model_dump(mode="json"),
                        },
                    )

            assert parsed_commands is not None
            if not parsed_commands:
                return await finish(
                    StopDecision(
                        should_stop=True,
                        reason=StopReason.NO_ACTION_AVAILABLE,
                        summary="The model returned no executable command.",
                    )
                )

            commands.extend(parsed_commands)
            decision = AgentDecisionSummary(
                run_id=task.run_id,
                task_id=task.task_id,
                actor_id=spec.agent_spec_id,
                observation_summary=self._decision_summary(invalid_response, len(observations), len(parsed_commands)),
                selected_command_ids=tuple(item.command_id for item in parsed_commands),
                alternatives_considered=(),
                policy_checks=("schema_validated", "commands_normalized"),
            )
            decisions.append(decision)
            self._emit("decision.recorded", spec, task, decision.model_dump(mode="json"))

            for command in parsed_commands:
                self._emit("command.proposed", spec, task, command.model_dump(mode="json"))
                policy = self.policy_checker.check(
                    command,
                    spec,
                    tool_call_counts=tool_call_counts,
                )
                self._emit(
                    "policy.decided",
                    spec,
                    task,
                    {
                        "command_id": command.command_id,
                        "allowed": policy.allowed,
                        "approval_required": policy.approval_required,
                        "reason": policy.reason,
                        "checks": list(policy.checks),
                    },
                )
                if not policy.allowed:
                    error = self._error(
                        ErrorCategory.POLICY_DENIED,
                        "command_policy_denied",
                        policy.reason,
                        spec,
                        task,
                        command=command,
                        fatal=True,
                    )
                    return await finish(
                        StopDecision(
                            should_stop=True,
                            reason=StopReason.POLICY_DENIED,
                            summary=f"Command policy denied {command.name}: {policy.reason}"[:1000],
                        ),
                        error=error,
                    )
                if policy.approval_required:
                    self._emit(
                        "approval.requested",
                        spec,
                        task,
                        {"command_id": command.command_id, "name": command.name, "risk_level": command.risk_level},
                    )
                    return await finish(
                        StopDecision(
                            should_stop=True,
                            reason=StopReason.APPROVAL_REQUIRED,
                            summary=f"Command {command.name} requires approval before execution."[:1000],
                            approval_required=True,
                        )
                    )
                if command.kind == CommandKind.STOP:
                    reason_value = str(command.arguments.get("reason", StopReason.SUCCESS.value))
                    reason = StopReason.SEMANTIC_COMPLETE if reason_value == StopReason.SEMANTIC_COMPLETE.value else StopReason.SUCCESS
                    summary = str(command.arguments.get("summary") or "The agent explicitly completed the task.")[:1000]
                    return await finish(StopDecision(should_stop=True, reason=reason, summary=summary))

                action_stop = self._check_stop(
                    budget=budget,
                    usage=self._with_elapsed(usage, started_monotonic),
                    state=state,
                    cancellation=token,
                    consumes=self._action_dimensions(command),
                )
                if action_stop is not None:
                    return await finish(action_stop)

                max_attempts = self.config.max_action_retries + 1
                for action_attempt in range(1, max_attempts + 1):
                    self._emit(
                        "action.started",
                        spec,
                        task,
                        {"command_id": command.command_id, "kind": command.kind.value, "name": command.name, "attempt": action_attempt},
                    )
                    action_started = utc_now()
                    raw: RawObservation | Observation
                    retryable = False
                    try:
                        raw = await self._await_controlled(
                            self.action_executor.execute(command),
                            cancellation=token,
                            timeout_seconds=self._remaining_seconds(budget, started_monotonic),
                        )
                    except KernelCancelled:
                        raw = RawObservation(
                            status=ObservationStatus.CANCELLED.value,
                            error=self._error(
                                ErrorCategory.CANCELLED,
                                "action_cancelled",
                                "Action execution was cancelled.",
                                spec,
                                task,
                                command=command,
                                attempt=action_attempt,
                                fatal=True,
                            ),
                            started_at=action_started,
                            completed_at=utc_now(),
                        )
                    except KernelTimedOut:
                        raw = RawObservation(
                            status=ObservationStatus.TIMEOUT.value,
                            error=self._error(
                                ErrorCategory.BUDGET_EXHAUSTED,
                                "action_wall_time_exhausted",
                                "Action execution exceeded the remaining wall-time budget.",
                                spec,
                                task,
                                command=command,
                                attempt=action_attempt,
                                fatal=True,
                            ),
                            started_at=action_started,
                            completed_at=utc_now(),
                        )
                    except Exception as exc:
                        retryable = isinstance(exc, ActionExecutionError) and exc.retryable
                        raw = RawObservation(
                            status=ObservationStatus.FAILED.value,
                            error=self._error(
                                ErrorCategory.TRANSIENT_PROVIDER if retryable else ErrorCategory.PERMANENT_PROVIDER,
                                "action_execution_failed",
                                str(exc) or type(exc).__name__,
                                spec,
                                task,
                                command=command,
                                attempt=action_attempt,
                                retryable=retryable,
                                fatal=not retryable,
                            ),
                            started_at=action_started,
                            completed_at=utc_now(),
                        )

                    try:
                        observation = self._normalize_observation(raw, command=command, attempt=action_attempt)
                    except Exception as exc:
                        observation = self._normalize_observation(
                            RawObservation(
                                status=ObservationStatus.FAILED.value,
                                error=self._error(
                                    ErrorCategory.PROTOCOL,
                                    "observation_normalization_failed",
                                    str(exc) or type(exc).__name__,
                                    spec,
                                    task,
                                    command=command,
                                    attempt=action_attempt,
                                    fatal=True,
                                ),
                                started_at=action_started,
                                completed_at=utc_now(),
                            ),
                            command=command,
                            attempt=action_attempt,
                        )
                    observations.append(observation)
                    for artifact_id in observation.output_artifact_ids:
                        if artifact_id not in output_artifact_ids:
                            output_artifact_ids.append(artifact_id)
                    usage = self._merge_usage(usage, observation.usage, started_monotonic)
                    tool_call_counts[command.name] = tool_call_counts.get(command.name, 0) + 1
                    if observation.error is not None:
                        self.stop_policy.record_error(
                            state,
                            self._fingerprint(
                                observation.error.category.value,
                                observation.error.code,
                                observation.error.message,
                            ),
                        )
                    self._emit(
                        "action.completed" if observation.status == ObservationStatus.SUCCEEDED else "action.failed",
                        spec,
                        task,
                        observation.model_dump(mode="json"),
                    )
                    if observation.status == ObservationStatus.CANCELLED:
                        return await finish(self._cancelled_stop(), error=observation.error)
                    if observation.status == ObservationStatus.TIMEOUT:
                        return await finish(self._wall_time_stop(budget), error=observation.error)

                    verification, verification_error, usage = await self._verify(
                        spec=spec,
                        task=task,
                        command=command,
                        observation=observation,
                        observations=tuple(observations),
                        cancellation=token,
                        budget=budget,
                        usage=usage,
                        state=state,
                        started_monotonic=started_monotonic,
                    )
                    if verification_error is not None:
                        if verification_error.category == ErrorCategory.CANCELLED:
                            return await finish(self._cancelled_stop(), error=verification_error)
                        if verification_error.category == ErrorCategory.BUDGET_EXHAUSTED:
                            return await finish(self._wall_time_stop(budget), error=verification_error)
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=StopReason.VERIFICATION_FAILED,
                                summary="The independent verifier could not evaluate the action result.",
                            ),
                            error=verification_error,
                        )
                    assert verification is not None
                    feedback_items.append(verification)
                    total_information_gain += max(0.0, verification.information_gain)
                    self.stop_policy.record_gain(state, verification.information_gain)
                    repair_feedback.extend(verification.repair_feedback)
                    self._emit(
                        "verification.completed",
                        spec,
                        task,
                        {
                            "command_id": command.command_id,
                            "observation_id": observation.observation_id,
                            "passed": verification.passed,
                            "success": verification.success,
                            "semantic_complete": verification.semantic_complete,
                            "information_gain": verification.information_gain,
                            "summary": verification.summary,
                            "repair_feedback": list(verification.repair_feedback),
                            "usage": verification.usage.model_dump(mode="json"),
                        },
                    )
                    if verification.success and observation.status == ObservationStatus.SUCCEEDED:
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=StopReason.SUCCESS,
                                summary=verification.summary[:1000],
                            )
                        )
                    if verification.semantic_complete:
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=StopReason.SEMANTIC_COMPLETE,
                                summary=verification.summary[:1000],
                            )
                        )
                    retryable = retryable or bool(observation.error and observation.error.retryable)
                    if observation.status != ObservationStatus.SUCCEEDED and retryable and action_attempt < max_attempts:
                        retry_stop = self._before_action_retry(budget, usage, state, token, command)
                        if retry_stop is not None:
                            return await finish(retry_stop)
                        usage = usage.plus(retries=1)
                        self._emit(
                            "retry.scheduled",
                            spec,
                            task,
                            {"operation": "action", "command_id": command.command_id, "attempt": action_attempt + 1},
                        )
                        continue
                    if observation.status != ObservationStatus.SUCCEEDED:
                        reason = StopReason.REPEATED_ERROR if action_attempt > 1 else StopReason.VERIFICATION_FAILED
                        return await finish(
                            StopDecision(
                                should_stop=True,
                                reason=reason,
                                summary=f"Action {command.name} failed without a permitted recovery path."[:1000],
                            ),
                            error=observation.error,
                        )
                    break

    async def _verify(
        self,
        *,
        spec: AgentSpec,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        observations: tuple[Observation, ...],
        cancellation: CancellationToken,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        started_monotonic: float,
    ) -> tuple[VerificationFeedback | None, ErrorRecord | None, BudgetUsage]:
        for attempt in range(1, self.config.max_verification_retries + 2):
            try:
                feedback = await self._await_controlled(
                    self.verifier.verify(
                        spec=spec,
                        task=task,
                        command=command,
                        observation=observation,
                        prior_observations=observations[:-1],
                    ),
                    cancellation=cancellation,
                    timeout_seconds=self._remaining_seconds(budget, started_monotonic),
                )
                if not isinstance(feedback, VerificationFeedback):
                    raise TypeError("KernelVerifier returned a non-VerificationFeedback value")
            except KernelCancelled:
                error = self._error(ErrorCategory.CANCELLED, "verification_cancelled", "Verification was cancelled.", spec, task, command=command, fatal=True)
                return None, error, self._with_elapsed(usage, started_monotonic)
            except KernelTimedOut:
                error = self._error(ErrorCategory.BUDGET_EXHAUSTED, "verification_wall_time_exhausted", "Verification exceeded the wall-time budget.", spec, task, command=command, fatal=True)
                return None, error, self._with_elapsed(usage.plus(errors=1), started_monotonic)
            except Exception as exc:
                retryable = bool(getattr(exc, "retryable", False))
                usage = self._with_elapsed(usage.plus(errors=1), started_monotonic)
                self.stop_policy.record_error(state, self._fingerprint("verification", type(exc).__name__, str(exc)))
                error = self._error(
                    ErrorCategory.VERIFICATION,
                    "verification_execution_failed",
                    str(exc) or type(exc).__name__,
                    spec,
                    task,
                    command=command,
                    attempt=attempt,
                    retryable=retryable,
                    fatal=not retryable,
                )
                if not retryable or attempt > self.config.max_verification_retries:
                    return None, error, usage
                retry_stop = self._before_retry(budget, usage, state, cancellation)
                if retry_stop is not None:
                    return None, self._stop_error(retry_stop, spec, task), usage
                usage = usage.plus(retries=1)
                self._emit("retry.scheduled", spec, task, {"operation": "verification", "attempt": attempt + 1})
                continue
            return feedback, None, self._merge_usage(usage, feedback.usage, started_monotonic)
        raise AssertionError("unreachable verifier retry state")

    @staticmethod
    def _validate_task(task: TaskEnvelope, spec: AgentSpec) -> None:
        terminal = {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}
        if task.status in terminal:
            raise ValueError(f"AgentKernel cannot execute terminal task: {task.status.value}")
        if task.assigned_actor_id is not None and task.assigned_actor_id != spec.agent_spec_id:
            raise ValueError("task is assigned to a different AgentSpec")

    def _normalize_observation(self, raw: RawObservation | Observation, *, command: Command, attempt: int) -> Observation:
        if isinstance(raw, Observation):
            status_value: Any = raw.status.value
            data: Any = raw.normalized_data
            artifact_ids = raw.output_artifact_ids
            item_usage = raw.usage
            error = raw.error
            started_at = raw.started_at
            completed_at = raw.completed_at
        elif isinstance(raw, RawObservation):
            status_value = raw.status
            data = raw.data
            artifact_ids = raw.output_artifact_ids
            item_usage = raw.usage
            error = raw.error
            started_at = raw.started_at
            completed_at = raw.completed_at
        else:
            raise TypeError("ActionExecutor must return RawObservation or Observation")

        aliases = {
            "success": ObservationStatus.SUCCEEDED,
            "ok": ObservationStatus.SUCCEEDED,
            "error": ObservationStatus.FAILED,
            "timed_out": ObservationStatus.TIMEOUT,
            "canceled": ObservationStatus.CANCELLED,
        }
        normalized_value = str(status_value).casefold()
        try:
            status = aliases[normalized_value] if normalized_value in aliases else ObservationStatus(normalized_value)
        except ValueError:
            status = ObservationStatus.FAILED
            error = error or ErrorRecord(
                category=ErrorCategory.PROTOCOL,
                code="invalid_observation_status",
                message=f"Action returned an unknown status: {status_value}",
                fatal=True,
                actor_id=command.actor_id,
                task_id=command.task_id,
                command_id=command.command_id,
            )
        if status == ObservationStatus.SUCCEEDED and error is not None:
            status = ObservationStatus.FAILED
        if status != ObservationStatus.SUCCEEDED and error is None:
            error = ErrorRecord(
                category=ErrorCategory.PROTOCOL,
                code="action_failed_without_error",
                message="Action returned a non-success status without a structured error.",
                retryable=False,
                fatal=status in {ObservationStatus.CANCELLED, ObservationStatus.REJECTED},
                actor_id=command.actor_id,
                task_id=command.task_id,
                command_id=command.command_id,
            )
        if error is not None:
            error = ErrorRecord(
                **{
                    **error.model_dump(),
                    "actor_id": command.actor_id,
                    "task_id": command.task_id,
                    "command_id": command.command_id,
                    "attempt": attempt,
                    "metadata": redact(error.metadata),
                }
            )
        normalized_data = redact(data if isinstance(data, dict) else {"value": data})
        updates = {
            "tool_calls": max(1, item_usage.tool_calls),
            "search_calls": max(1, item_usage.search_calls) if command.kind == CommandKind.SEARCH else item_usage.search_calls,
            "errors": max(1, item_usage.errors) if status != ObservationStatus.SUCCEEDED else item_usage.errors,
        }
        item_usage = item_usage.model_copy(update=updates)
        if completed_at < started_at:
            completed_at = started_at
        return Observation(
            command_id=command.command_id,
            run_id=command.run_id,
            task_id=command.task_id,
            actor_id=command.actor_id,
            status=status,
            output_artifact_ids=tuple(artifact_ids),
            normalized_data=normalized_data,
            usage=item_usage,
            error=error,
            attempt=attempt,
            started_at=started_at,
            completed_at=completed_at,
        )

    async def _await_controlled(
        self,
        awaitable: Awaitable[_T],
        *,
        cancellation: CancellationToken,
        timeout_seconds: float,
    ) -> _T:
        if cancellation.cancelled:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise KernelCancelled("operation cancelled")
        if timeout_seconds <= 0:
            if inspect.iscoroutine(awaitable):
                awaitable.close()
            raise KernelTimedOut("wall-time budget exhausted")
        operation = asyncio.ensure_future(awaitable)
        cancellation_wait = asyncio.create_task(cancellation.wait())
        try:
            done, _ = await asyncio.wait(
                {operation, cancellation_wait},
                timeout=timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_wait in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise KernelCancelled("operation cancelled")
            if operation not in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise KernelTimedOut("wall-time budget exhausted")
            return await operation
        finally:
            cancellation_wait.cancel()
            with suppress(asyncio.CancelledError):
                await cancellation_wait

    def _check_stop(
        self,
        *,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
        consumes: tuple[BudgetDimension, ...] = (),
    ) -> StopDecision | None:
        if budget.deadline is not None and utc_now() >= budget.deadline:
            return StopDecision(
                should_stop=True,
                reason=StopReason.DEADLINE_REACHED,
                summary="The task deadline was reached.",
            )
        stop = self.stop_policy.evaluate(
            budget=budget,
            usage=usage,
            state=state,
            cancellation=cancellation,
            budget_dimensions=consumes,
        )
        if stop is not None:
            return stop
        current = {
            BudgetDimension.MODEL_CALLS: (usage.model_calls, budget.max_model_calls),
            BudgetDimension.TOOL_CALLS: (usage.tool_calls, budget.max_tool_calls),
            BudgetDimension.SEARCH_CALLS: (usage.search_calls, budget.max_search_calls),
            BudgetDimension.RETRIES: (usage.retries, budget.max_retries),
        }
        exhausted = tuple(
            dimension
            for dimension in consumes
            if dimension in current and current[dimension][1] is not None and current[dimension][0] >= current[dimension][1]
        )
        if exhausted:
            return StopDecision(
                should_stop=True,
                reason=StopReason.BUDGET_EXHAUSTED,
                summary="Execution budget was exhausted before the next operation.",
                exhausted_dimensions=exhausted,
            )
        return None

    def _before_retry(
        self,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
    ) -> StopDecision | None:
        return self._check_stop(
            budget=budget,
            usage=usage,
            state=state,
            cancellation=cancellation,
            consumes=(BudgetDimension.RETRIES,),
        )

    def _before_model_retry(
        self,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
    ) -> StopDecision | None:
        return self._check_stop(
            budget=budget,
            usage=usage,
            state=state,
            cancellation=cancellation,
            consumes=(BudgetDimension.RETRIES, BudgetDimension.MODEL_CALLS),
        )

    def _before_model_repair(
        self,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
    ) -> StopDecision | None:
        return self._check_stop(
            budget=budget,
            usage=usage,
            state=state,
            cancellation=cancellation,
            consumes=(BudgetDimension.RETRIES, BudgetDimension.MODEL_CALLS),
        )

    def _before_action_retry(
        self,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
        command: Command,
    ) -> StopDecision | None:
        return self._check_stop(
            budget=budget,
            usage=usage,
            state=state,
            cancellation=cancellation,
            consumes=(BudgetDimension.RETRIES, *self._action_dimensions(command)),
        )

    @staticmethod
    def _action_dimensions(command: Command) -> tuple[BudgetDimension, ...]:
        if command.kind == CommandKind.SEARCH:
            return (BudgetDimension.TOOL_CALLS, BudgetDimension.SEARCH_CALLS)
        return (BudgetDimension.TOOL_CALLS,)

    @staticmethod
    def _remaining_seconds(budget: Budget, started_monotonic: float) -> float:
        remaining: list[float] = []
        if budget.max_wall_time_seconds is not None:
            remaining.append(budget.max_wall_time_seconds - (time.monotonic() - started_monotonic))
        if budget.deadline is not None:
            remaining.append((budget.deadline - utc_now()).total_seconds())
        return max(0.0, min(remaining)) if remaining else 365 * 24 * 60 * 60.0

    @staticmethod
    def _merge_usage(base: BudgetUsage, item: BudgetUsage, started_monotonic: float) -> BudgetUsage:
        return BudgetUsage(
            input_tokens=base.input_tokens + item.input_tokens,
            output_tokens=base.output_tokens + item.output_tokens,
            cost_usd=base.cost_usd + item.cost_usd,
            wall_time_seconds=max(base.wall_time_seconds, item.wall_time_seconds, time.monotonic() - started_monotonic),
            model_calls=base.model_calls + item.model_calls,
            tool_calls=base.tool_calls + item.tool_calls,
            search_calls=base.search_calls + item.search_calls,
            retries=base.retries + item.retries,
            errors=base.errors + item.errors,
        )

    @staticmethod
    def _with_elapsed(usage: BudgetUsage, started_monotonic: float) -> BudgetUsage:
        return usage.model_copy(update={"wall_time_seconds": max(usage.wall_time_seconds, time.monotonic() - started_monotonic)})

    def _emit(self, event_type: str, spec: AgentSpec, task: TaskEnvelope, payload: dict[str, Any]) -> None:
        self.event_sink.emit(
            KernelEvent(
                event_type=event_type,
                run_id=task.run_id,
                task_id=task.task_id,
                actor_id=spec.agent_spec_id,
                payload=redact(payload),
            )
        )

    @staticmethod
    def _decision_summary(response: ModelResponse, observation_count: int, command_count: int) -> str:
        summary = ""
        if isinstance(response.structured, dict):
            candidate = redact(response.structured).get("summary")
            if isinstance(candidate, str):
                summary = candidate.strip()
        return (summary or f"Selected {command_count} structured command(s) after {observation_count} observation(s).")[:4000]

    @staticmethod
    def _result_status(reason: StopReason, error: ErrorRecord | None) -> TaskResultStatus:
        if reason in {StopReason.SUCCESS, StopReason.SEMANTIC_COMPLETE}:
            return TaskResultStatus.SUCCEEDED
        if reason in {
            StopReason.BUDGET_EXHAUSTED,
            StopReason.DEADLINE_REACHED,
            StopReason.LOW_INFORMATION_GAIN,
        }:
            return TaskResultStatus.PARTIAL
        if reason == StopReason.USER_CANCELLED:
            return TaskResultStatus.CANCELLED
        if reason == StopReason.APPROVAL_REQUIRED:
            return TaskResultStatus.DEFERRED
        if reason == StopReason.POLICY_DENIED:
            return TaskResultStatus.REJECTED
        if error is not None or reason in {StopReason.REPEATED_ERROR, StopReason.VERIFICATION_FAILED}:
            return TaskResultStatus.FAILED
        return TaskResultStatus.PARTIAL

    @staticmethod
    def _cancelled_stop() -> StopDecision:
        return StopDecision(should_stop=True, reason=StopReason.USER_CANCELLED, summary="Execution was cancelled.")

    @staticmethod
    def _wall_time_stop(budget: Budget) -> StopDecision:
        if budget.deadline is not None and utc_now() >= budget.deadline:
            return StopDecision(
                should_stop=True,
                reason=StopReason.DEADLINE_REACHED,
                summary="The task deadline was reached.",
            )
        return StopDecision(
            should_stop=True,
            reason=StopReason.BUDGET_EXHAUSTED,
            summary="The execution wall-time budget was exhausted.",
            exhausted_dimensions=(BudgetDimension.WALL_TIME,),
        )

    def _stop_error(self, stop: StopDecision, spec: AgentSpec, task: TaskEnvelope) -> ErrorRecord:
        category = {
            StopReason.USER_CANCELLED: ErrorCategory.CANCELLED,
            StopReason.POLICY_DENIED: ErrorCategory.POLICY_DENIED,
            StopReason.APPROVAL_REQUIRED: ErrorCategory.APPROVAL_REQUIRED,
            StopReason.BUDGET_EXHAUSTED: ErrorCategory.BUDGET_EXHAUSTED,
            StopReason.VERIFICATION_FAILED: ErrorCategory.VERIFICATION,
        }.get(stop.reason, ErrorCategory.INTERNAL)
        return self._error(category, f"kernel_{stop.reason.value}", stop.summary, spec, task, fatal=True)

    @staticmethod
    def _error(
        category: ErrorCategory,
        code: str,
        message: str,
        spec: AgentSpec,
        task: TaskEnvelope,
        *,
        command: Command | None = None,
        attempt: int = 1,
        retryable: bool = False,
        fatal: bool = False,
    ) -> ErrorRecord:
        return ErrorRecord(
            category=category,
            code=code[:120],
            message=str(redact(message))[:2000],
            retryable=retryable,
            fatal=fatal,
            attempt=attempt,
            actor_id=spec.agent_spec_id,
            task_id=task.task_id,
            command_id=command.command_id if command else None,
        )

    @staticmethod
    def _fingerprint(*parts: str) -> str:
        return hashlib.sha256("\0".join(parts).encode("utf-8", errors="replace")).hexdigest()
