from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Callable, Protocol

from deep_researcher.artifacts.store import (
    ArtifactQuery,
    ArtifactStore,
)
from deep_researcher.contracts import (
    AgentSpec,
    ArtifactKind,
    BudgetUsage,
    Command,
    ComponentKind,
    ErrorCategory,
    ErrorRecord,
    EventLevel,
    EventType,
    Observation,
    ObservationStatus,
    RunEvent,
    RunStatus,
    SpanKind,
    StopReason,
    TaskResultStatus,
    TaskStatus,
    VersionRef,
    utc_now,
)
from deep_researcher.events import (
    EventQuery,
    EventRecorder,
    EventStore,
)
from deep_researcher.kernel import (
    AgentKernel,
    AgentSpecRegistry,
    EventRecorderKernelSink,
)
from deep_researcher.kernel.middleware import CommandPolicyChecker
from deep_researcher.kernel.types import (
    ActionExecutor,
    KernelEvent,
    KernelEventSink,
    KernelVerifier,
    ModelAdapter,
    ModelRequest,
    ModelResponse,
    PolicyDecision,
    RawObservation,
    VerificationFeedback,
)
from deep_researcher.version_registry.store import (
    SQLiteVersionRegistryStore,
)

from .advanced_models import (
    ReplayAttempt,
    ReplayAttemptStatus,
    ReplayCapsule,
    ReplayEligibility,
    ReplayExecutionOutcome,
    ReplayMode,
    ReplayRequest,
)


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()
    return f"{prefix}_{digest[:32]}"


def command_fingerprint(command: Command) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "kind": command.kind.value,
                "name": command.name,
                "arguments": command.arguments,
                "input_artifact_ids": command.input_artifact_ids,
                "expected_output_schema": command.expected_output_schema,
                "risk_level": command.risk_level,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def source_event_fingerprint(events: tuple[RunEvent, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            [item.model_dump(mode="json") for item in events],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def all_run_events(
    event_store: EventStore,
    run_id: str,
    *,
    span_id: str | None = None,
) -> tuple[RunEvent, ...]:
    after = 0
    events: list[RunEvent] = []
    while True:
        page = event_store.list(
            EventQuery(
                run_id,
                span_id=span_id,
                after_sequence=after,
                limit=1000,
            )
        )
        events.extend(page.items)
        if page.next_after_sequence is None:
            break
        after = page.next_after_sequence
    return tuple(events)


def span_tree_events(
    event_store: EventStore,
    run_id: str,
    span_id: str,
) -> tuple[RunEvent, ...]:
    events = all_run_events(event_store, run_id)
    parents = {
        item.span_id: item.parent_span_id for item in events
    }
    if span_id not in parents:
        return ()

    def belongs(candidate: str) -> bool:
        seen: set[str] = set()
        while candidate not in seen:
            if candidate == span_id:
                return True
            seen.add(candidate)
            parent = parents.get(candidate)
            if parent is None:
                return False
            candidate = parent
        return False

    return tuple(item for item in events if belongs(item.span_id))


class ReplayCapsuleRepository:
    """Creates and discovers immutable capsules without changing source events."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        producer_id: str = "runtime_studio_replay_capsule",
    ) -> None:
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.producer_id = producer_id

    def create(self, capsule: ReplayCapsule):
        events = span_tree_events(
            self.event_store,
            capsule.source_run_id,
            capsule.source_span_id,
        )
        if tuple(item.event_id for item in events) != (
            capsule.source_event_ids
        ):
            raise ValueError(
                "replay capsule source events do not match the source span"
            )
        if source_event_fingerprint(events) != (
            capsule.source_event_fingerprint
        ):
            raise ValueError(
                "replay capsule source event fingerprint does not match"
            )
        for exchange in capsule.tool_exchanges:
            for artifact_id in (
                *exchange.input_artifact_ids,
                *tuple(
                    exchange.observation.get(
                        "output_artifact_ids",
                        (),
                    )
                ),
            ):
                if self.artifact_store.get(artifact_id) is None:
                    raise ValueError(
                        "replay tool exchange references a missing artifact: "
                        f"{artifact_id}"
                    )
        artifact_id = stable_id("artifact", capsule.capsule_id)
        return self.artifact_store.put_json(
            {
                "schema": "ReplayCapsule@1",
                "capsule": capsule.model_dump(mode="json"),
                "immutable": True,
                "network_free_material": True,
            },
            redact=False,
            kind=ArtifactKind.REPLAY_CAPSULE,
            producer_id=self.producer_id,
            run_id=capsule.source_run_id,
            task_id=capsule.task.task_id,
            content_schema="ReplayCapsule@1",
            # Cross-run component artifacts remain explicit IDs in the
            # capsule. ArtifactStore provenance edges intentionally stay
            # run-local.
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"replay-capsule:{capsule.capsule_id}",
            metadata={
                "source_span_id": capsule.source_span_id,
                "dataset_sample_artifact_id": (
                    capsule.dataset_sample_artifact_id
                ),
            },
        )

    def load(self, artifact_id: str) -> ReplayCapsule:
        envelope = self.artifact_store.get(artifact_id)
        if envelope is None:
            raise KeyError(f"unknown replay capsule: {artifact_id}")
        if (
            envelope.kind != ArtifactKind.REPLAY_CAPSULE
            or envelope.content_schema != "ReplayCapsule@1"
        ):
            raise ValueError("artifact is not a ReplayCapsule@1")
        payload = json.loads(
            self.artifact_store.read_bytes(artifact_id).decode()
        )
        return ReplayCapsule.model_validate(
            payload["capsule"],
            strict=False,
        )

    def find(
        self,
        source_run_id: str,
        source_span_id: str,
    ) -> tuple[str, ReplayCapsule] | None:
        current_events = span_tree_events(
            self.event_store,
            source_run_id,
            source_span_id,
        )
        current_ids = tuple(item.event_id for item in current_events)
        current_fingerprint = (
            source_event_fingerprint(current_events)
            if current_events
            else None
        )
        fallback: tuple[str, ReplayCapsule] | None = None
        exact: tuple[str, ReplayCapsule] | None = None
        cursor = None
        while True:
            page = self.artifact_store.list(
                ArtifactQuery(
                    run_id=source_run_id,
                    kinds=(ArtifactKind.REPLAY_CAPSULE,),
                    after_created_at=cursor[0] if cursor else None,
                    after_artifact_id=cursor[1] if cursor else None,
                    limit=1000,
                )
            )
            for envelope in page.items:
                if (
                    envelope.metadata.get("source_span_id")
                    != source_span_id
                ):
                    continue
                capsule = self.load(envelope.artifact_id)
                if capsule.source_span_id == source_span_id:
                    fallback = envelope.artifact_id, capsule
                    if (
                        capsule.source_event_ids == current_ids
                        and capsule.source_event_fingerprint
                        == current_fingerprint
                    ):
                        exact = envelope.artifact_id, capsule
            if page.next_cursor is None:
                return exact or fallback
            cursor = page.next_cursor

    def eligibility(
        self,
        source_run_id: str,
        source_span_id: str,
    ) -> ReplayEligibility:
        if self.event_store.get_run(source_run_id) is None:
            raise KeyError(f"unknown source run: {source_run_id}")
        events = span_tree_events(
            self.event_store,
            source_run_id,
            source_span_id,
        )
        reasons: list[str] = []
        terminal = next(
            (
                item
                for item in reversed(events)
                if item.span_id == source_span_id
                and item.event_type
                in {
                    EventType.RUN_COMPLETED,
                    EventType.RUN_FAILED,
                    EventType.RUN_CANCELLED,
                    EventType.SPAN_COMPLETED,
                    EventType.SPAN_FAILED,
                    EventType.MODEL_COMPLETED,
                    EventType.MODEL_FAILED,
                    EventType.TOOL_COMPLETED,
                    EventType.TOOL_FAILED,
                }
            ),
            None,
        )
        if not events:
            reasons.append("span has no durable events")
        if terminal is None:
            reasons.append("span is not terminal")
        found = self.find(source_run_id, source_span_id)
        capsule_artifact_id = found[0] if found else None
        capsule = found[1] if found else None
        if capsule is None:
            reasons.append("span has no immutable replay capsule")
        elif (
            tuple(item.event_id for item in events)
            != capsule.source_event_ids
            or source_event_fingerprint(events)
            != capsule.source_event_fingerprint
        ):
            reasons.append("replay capsule no longer matches source events")
        modes = (
            (
                ReplayMode.SAVED_TOOL_RESULTS,
                ReplayMode.LIVE_ENVIRONMENT,
            )
            if not reasons
            else ()
        )
        return ReplayEligibility(
            source_run_id=source_run_id,
            source_span_id=source_span_id,
            eligible=not reasons,
            terminal_event_id=terminal.event_id if terminal else None,
            terminal_failed=bool(
                terminal
                and terminal.event_type
                in {
                    EventType.RUN_FAILED,
                    EventType.SPAN_FAILED,
                    EventType.MODEL_FAILED,
                    EventType.TOOL_FAILED,
                }
            ),
            capsule_artifact_id=capsule_artifact_id,
            reasons=tuple(reasons),
            supported_modes=modes,
        )


class SavedReplayModelAdapter:
    def __init__(self, capsule: ReplayCapsule) -> None:
        self._items = list(capsule.model_exchanges)
        self._index = 0

    def _next(
        self,
        operation: str,
        request: ModelRequest,
    ) -> ModelResponse:
        if self._index >= len(self._items):
            raise RuntimeError(
                "sealed replay model cassette is exhausted"
            )
        exchange = self._items[self._index]
        self._index += 1
        if exchange.operation != operation:
            raise RuntimeError(
                "sealed replay model operation order changed"
            )
        if (
            exchange.model_version != request.model_version
            or exchange.prompt_version != request.prompt_version
        ):
            raise RuntimeError(
                "sealed model response cannot be used with another "
                "model or prompt version"
            )
        response = dict(exchange.response)
        if isinstance(response.get("usage"), dict):
            response["usage"] = BudgetUsage.model_validate(
                response["usage"],
                strict=False,
            )
        return ModelResponse(**response)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        return self._next("complete", request)

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        del invalid_response, errors
        return self._next("repair", request)


class SavedReplayActionExecutor:
    def __init__(
        self,
        *,
        capsule: ReplayCapsule,
        artifact_store: ArtifactStore,
        target_run_id: str,
        target_task_id: str,
        approved_fingerprints: frozenset[str],
    ) -> None:
        self._items = list(capsule.tool_exchanges)
        self._consumed: set[int] = set()
        self.artifact_store = artifact_store
        self.target_run_id = target_run_id
        self.target_task_id = target_task_id
        self.approved_fingerprints = approved_fingerprints

    async def execute(self, command: Command) -> RawObservation:
        fingerprint = command_fingerprint(command)
        match = next(
            (
                (index, item)
                for index, item in enumerate(self._items)
                if index not in self._consumed
                and item.command_fingerprint == fingerprint
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                "sealed replay has no tool result for command "
                f"{command.name}/{fingerprint}"
            )
        index, exchange = match
        if (
            exchange.side_effecting or exchange.requires_approval
        ) and fingerprint not in self.approved_fingerprints:
            raise RuntimeError(
                "saved side-effect result requires a fresh replay approval"
            )
        self._consumed.add(index)
        observation = dict(exchange.observation)
        source_output_ids = tuple(
            observation.get("output_artifact_ids", ())
        )
        alias_id = stable_id(
            "artifact",
            self.target_run_id,
            "saved_tool_result",
            str(index),
            fingerprint,
        )
        self.artifact_store.put_json(
            {
                "schema": "SavedToolReplayObservation@1",
                "source_run_id": self._items[index].observation.get(
                    "run_id"
                ),
                "command_fingerprint": fingerprint,
                "observation": observation,
                "source_output_artifact_ids": source_output_ids,
                "side_effect_executed": False,
                "network_calls": 0,
            },
            redact=False,
            kind=ArtifactKind.TOOL_RESULT,
            producer_id="runtime_studio_saved_tool_replay",
            run_id=self.target_run_id,
            task_id=self.target_task_id,
            content_schema="SavedToolReplayObservation@1",
            source_artifact_ids=(),
            artifact_id=alias_id,
            idempotency_key=(
                f"saved-tool-replay:{self.target_run_id}:{index}"
            ),
        )
        status = str(
            observation.get(
                "status",
                ObservationStatus.SUCCEEDED.value,
            )
        )
        error_data = observation.get("error")
        error = (
            ErrorRecord.model_validate(error_data, strict=False)
            if error_data
            else None
        )
        usage_data = observation.get("usage") or {}
        return RawObservation(
            status=status,
            data=observation.get(
                "normalized_data",
                observation.get("data"),
            ),
            output_artifact_ids=(alias_id,),
            usage=BudgetUsage.model_validate(
                usage_data,
                strict=False,
            ),
            error=error,
            started_at=utc_now(),
            completed_at=utc_now(),
        )


class SavedReplayVerifier:
    def __init__(self, capsule: ReplayCapsule) -> None:
        self._items = list(capsule.verification_exchanges)
        self._consumed: set[int] = set()

    async def verify(
        self,
        *,
        spec: AgentSpec,
        task,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        del spec, task, observation, prior_observations
        fingerprint = command_fingerprint(command)
        match = next(
            (
                (index, item)
                for index, item in enumerate(self._items)
                if index not in self._consumed
                and item.command_fingerprint == fingerprint
            ),
            None,
        )
        if match is None:
            raise RuntimeError(
                "sealed replay has no verification result for command "
                f"{command.name}/{fingerprint}"
            )
        index, exchange = match
        self._consumed.add(index)
        feedback = dict(exchange.feedback)
        if isinstance(feedback.get("usage"), dict):
            feedback["usage"] = BudgetUsage.model_validate(
                feedback["usage"],
                strict=False,
            )
        return VerificationFeedback(**feedback)


class ReplayCommandPolicyChecker:
    """Preserves normal policy checks while consuming only fresh approvals."""

    def __init__(
        self,
        approved_fingerprints: frozenset[str],
    ) -> None:
        self.base = CommandPolicyChecker()
        self.approved_fingerprints = approved_fingerprints
        self.pending_fingerprints: set[str] = set()

    def check(
        self,
        command: Command,
        spec: AgentSpec,
        *,
        tool_call_counts: dict[str, int] | None = None,
        now=None,
    ) -> PolicyDecision:
        decision = self.base.check(
            command,
            spec,
            tool_call_counts=tool_call_counts,
            now=now,
        )
        if not decision.allowed or not decision.approval_required:
            return decision
        fingerprint = command_fingerprint(command)
        if fingerprint in self.approved_fingerprints:
            return PolicyDecision(
                allowed=True,
                reason="policy allowed with fresh replay approval",
                approval_required=False,
                checks=(*decision.checks, "fresh_replay_approval"),
            )
        self.pending_fingerprints.add(fingerprint)
        return decision


class CountingKernelEventSink:
    def __init__(
        self,
        delegate: KernelEventSink,
        *,
        count_live_calls: bool,
    ) -> None:
        self.delegate = delegate
        self.count_live_calls = count_live_calls
        self.counts: Counter[str] = Counter()

    def emit(self, event: KernelEvent) -> None:
        if event.event_type in {"model.started", "action.started"}:
            self.counts[event.event_type] += 1
        self.delegate.emit(event)

    @property
    def network_calls(self) -> int:
        if not self.count_live_calls:
            return 0
        return sum(self.counts.values())


@dataclass(frozen=True)
class LiveReplayBindings:
    model_adapter: ModelAdapter
    action_executor: ActionExecutor
    verifier: KernelVerifier
    environment_label: str
    network_call_counter: Callable[[], int] | None = None

    def __post_init__(self) -> None:
        if (
            not self.environment_label.strip()
            or self.environment_label == "sealed-network-free"
        ):
            raise ValueError(
                "live bindings require an explicit live environment label"
            )


class LiveReplayBindingFactory(Protocol):
    def __call__(
        self,
        request: ReplayRequest,
        capsule: ReplayCapsule,
    ) -> LiveReplayBindings: ...


class KernelReplayBackend:
    """Runs a replay through the real AgentKernel and durable event sink."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        recorder: EventRecorder,
        artifact_store: ArtifactStore,
        version_store: SQLiteVersionRegistryStore,
        live_bindings_factory: LiveReplayBindingFactory | None = None,
    ) -> None:
        self.event_store = event_store
        self.recorder = recorder
        self.artifact_store = artifact_store
        self.version_store = version_store
        self.live_bindings_factory = live_bindings_factory

    async def execute(
        self,
        *,
        request: ReplayRequest,
        attempt: ReplayAttempt,
        capsule: ReplayCapsule,
        approved_fingerprints: frozenset[str],
    ) -> ReplayExecutionOutcome:
        spec = self._selected_agent_spec(request, capsule)
        target_task_id = stable_id(
            "task",
            attempt.target_run_id,
            capsule.task.task_id,
        )
        target_task = capsule.task.model_copy(
            update={
                "task_id": target_task_id,
                "run_id": attempt.target_run_id,
                "parent_task_id": None,
                "dependency_task_ids": (),
                "status": TaskStatus.READY,
                "attempt": 0,
                "input_artifact_ids": capsule.task.input_artifact_ids,
                "constraints": {
                    **capsule.task.constraints,
                    "replay_source_run_id": request.source_run_id,
                    "replay_source_span_id": request.source_span_id,
                    "replay_request_id": request.replay_request_id,
                    "replay_mode": request.mode.value,
                },
                "created_by": request.requested_by,
                "assigned_actor_id": spec.agent_spec_id,
                "created_at": attempt.started_at,
                "updated_at": attempt.started_at,
            }
        )
        if request.mode == ReplayMode.SAVED_TOOL_RESULTS:
            self._validate_saved_component_selection(request, capsule)
            model_adapter: ModelAdapter = SavedReplayModelAdapter(capsule)
            action_executor: ActionExecutor = SavedReplayActionExecutor(
                capsule=capsule,
                artifact_store=self.artifact_store,
                target_run_id=attempt.target_run_id,
                target_task_id=target_task_id,
                approved_fingerprints=approved_fingerprints,
            )
            verifier: KernelVerifier = SavedReplayVerifier(capsule)
            environment_label = "sealed-network-free"
            count_live_calls = False
            network_counter = None
            network_before = 0
        else:
            if self.live_bindings_factory is None:
                raise RuntimeError(
                    "live replay requires configured live runtime bindings"
                )
            bindings = self.live_bindings_factory(request, capsule)
            if bindings.environment_label != request.environment_label:
                raise ValueError(
                    "live environment label differs from replay request"
                )
            model_adapter = bindings.model_adapter
            action_executor = bindings.action_executor
            verifier = bindings.verifier
            environment_label = bindings.environment_label
            count_live_calls = True
            network_counter = bindings.network_call_counter
            network_before = (
                network_counter() if network_counter is not None else 0
            )

        registry = AgentSpecRegistry()
        registry.register(spec)
        durable_sink = EventRecorderKernelSink(
            self.recorder,
            run_id=attempt.target_run_id,
            thread_id=attempt.target_thread_id,
            trace_id=attempt.target_trace_id,
            correlation_id=request.replay_request_id,
            component_versions=request.selected_component_versions,
            producer_id="agent_kernel_studio_replay",
        )
        sink = CountingKernelEventSink(
            durable_sink,
            count_live_calls=count_live_calls,
        )
        policy = ReplayCommandPolicyChecker(approved_fingerprints)
        kernel = AgentKernel(
            registry=registry,
            model_adapter=model_adapter,
            action_executor=action_executor,
            verifier=verifier,
            event_sink=sink,
            policy_checker=policy,
        )
        kernel_result = await kernel.run(
            agent_spec_id=spec.agent_spec_id,
            task=target_task,
        )
        if network_counter is not None:
            network_calls = max(0, network_counter() - network_before)
            network_accounting_complete = True
        else:
            network_calls = sink.network_calls
            network_accounting_complete = not count_live_calls
        pending = tuple(sorted(policy.pending_fingerprints))
        if (
            kernel_result.stop_decision.reason
            == StopReason.APPROVAL_REQUIRED
        ):
            attempt_status = ReplayAttemptStatus.WAITING_APPROVAL
        elif kernel_result.task_result.status in {
            TaskResultStatus.SUCCEEDED,
            TaskResultStatus.PARTIAL,
        }:
            attempt_status = ReplayAttemptStatus.SUCCEEDED
        else:
            attempt_status = ReplayAttemptStatus.FAILED
        error = kernel_result.task_result.error
        if attempt_status == ReplayAttemptStatus.WAITING_APPROVAL:
            error = None
        elif attempt_status == ReplayAttemptStatus.FAILED and error is None:
            error = ErrorRecord(
                category=ErrorCategory.INTERNAL,
                code="replay_kernel_failed",
                message=kernel_result.stop_decision.summary,
                fatal=True,
                task_id=target_task_id,
                actor_id=spec.agent_spec_id,
            )

        result_artifact_id = stable_id(
            "artifact",
            attempt.attempt_id,
            "result",
        )
        self.artifact_store.put_json(
            {
                "schema": "StudioReplayResult@1",
                "replay_request": request.model_dump(mode="json"),
                "attempt": attempt.model_dump(mode="json"),
                "source_provenance": {
                    "run_id": request.source_run_id,
                    "span_id": request.source_span_id,
                    "terminal_event_id": request.source_terminal_event_id,
                    "capsule_artifact_id": request.capsule_artifact_id,
                    "source_event_fingerprint": (
                        request.source_event_fingerprint
                    ),
                },
                "kernel_result": {
                    "task_result": kernel_result.task_result.model_dump(
                        mode="json"
                    ),
                    "stop_decision": kernel_result.stop_decision.model_dump(
                        mode="json"
                    ),
                    "commands": [
                        item.model_dump(mode="json")
                        for item in kernel_result.commands
                    ],
                    "observations": [
                        item.model_dump(mode="json")
                        for item in kernel_result.observations
                    ],
                },
                "selected_component_versions": (
                    request.selected_component_versions.model_dump(
                        mode="json"
                    )
                ),
                "mode": request.mode.value,
                "environment_label": environment_label,
                "network_calls": network_calls,
                "network_accounting_complete": (
                    network_accounting_complete
                ),
                "pending_approval_fingerprints": pending,
                "historical_events_rewritten": False,
            },
            redact=False,
            kind=ArtifactKind.STUDIO_REPLAY_RESULT,
            producer_id="runtime_studio_replay",
            run_id=attempt.target_run_id,
            task_id=target_task_id,
            content_schema="StudioReplayResult@1",
            source_artifact_ids=(),
            artifact_id=result_artifact_id,
            idempotency_key=f"studio-replay-result:{attempt.attempt_id}",
        )
        self._close_target_run(
            request=request,
            attempt=attempt,
            task_id=target_task_id,
            actor_id=spec.agent_spec_id,
            attempt_status=attempt_status,
            result_artifact_id=result_artifact_id,
            output_artifact_ids=kernel_result.task_result.output_artifact_ids,
            error=error,
            pending=pending,
        )
        return ReplayExecutionOutcome(
            replay_request_id=request.replay_request_id,
            attempt_id=attempt.attempt_id,
            target_run_id=attempt.target_run_id,
            status=attempt_status,
            output_artifact_ids=(
                *kernel_result.task_result.output_artifact_ids,
                result_artifact_id,
            ),
            result_artifact_id=result_artifact_id,
            network_calls=network_calls,
            network_accounting_complete=network_accounting_complete,
            environment_label=environment_label,
            usage=kernel_result.task_result.usage,
            pending_approval_fingerprints=pending,
            error=error,
            completed_at=utc_now(),
        )

    def _close_target_run(
        self,
        *,
        request: ReplayRequest,
        attempt: ReplayAttempt,
        task_id: str,
        actor_id: str,
        attempt_status: ReplayAttemptStatus,
        result_artifact_id: str,
        output_artifact_ids: tuple[str, ...],
        error: ErrorRecord | None,
        pending: tuple[str, ...],
    ) -> None:
        run = self.event_store.get_run(attempt.target_run_id)
        if run is None:
            raise RuntimeError("AgentKernel did not create a target run")
        events = all_run_events(self.event_store, attempt.target_run_id)
        causation = events[-1].event_id if events else None
        if attempt_status == ReplayAttemptStatus.SUCCEEDED:
            event_type = EventType.RUN_COMPLETED
            status = RunStatus.SUCCEEDED
            terminal_error = None
        else:
            event_type = EventType.RUN_FAILED
            status = RunStatus.FAILED
            terminal_error = error or ErrorRecord(
                category=ErrorCategory.APPROVAL_REQUIRED,
                code="fresh_replay_approval_required",
                message=(
                    "Replay generated a side-effecting or governed command "
                    "that needs a fresh approval."
                ),
                fatal=False,
                task_id=task_id,
                actor_id=actor_id,
            )
        now = utc_now()
        self.recorder.record(
            RunEvent(
                event_id=stable_id(
                    "event",
                    attempt.attempt_id,
                    "terminal",
                ),
                sequence_no=self.event_store.next_sequence(
                    attempt.target_run_id
                ),
                event_type=event_type,
                level=(
                    EventLevel.INFO
                    if terminal_error is None
                    else EventLevel.ERROR
                ),
                status=status,
                trace_id=attempt.target_trace_id,
                span_id=run.root_span_id,
                parent_span_id=None,
                span_kind=SpanKind.RUN,
                correlation_id=request.replay_request_id,
                causation_event_id=causation,
                run_id=attempt.target_run_id,
                thread_id=attempt.target_thread_id,
                task_id=task_id,
                actor_id=actor_id,
                producer_id="runtime_studio_replay",
                output_artifact_ids=(
                    *output_artifact_ids,
                    result_artifact_id,
                ),
                error=terminal_error,
                component_versions=request.selected_component_versions,
                occurred_at=now,
                recorded_at=now,
                payload={
                    "replay_request_id": request.replay_request_id,
                    "source_run_id": request.source_run_id,
                    "source_span_id": request.source_span_id,
                    "replay_mode": request.mode.value,
                    "environment_label": request.environment_label,
                    "pending_approval_fingerprints": pending,
                },
            )
        )

    def _selected_agent_spec(
        self,
        request: ReplayRequest,
        capsule: ReplayCapsule,
    ) -> AgentSpec:
        versions = request.selected_component_versions
        selected_agent_ref = versions.agent_spec
        source_agent_ref = capsule.source_component_versions.agent_spec
        if (
            selected_agent_ref is not None
            and (
                source_agent_ref is None
                or selected_agent_ref.version_id
                != source_agent_ref.version_id
            )
        ):
            base = self._load_registered_agent_spec(selected_agent_ref)
        else:
            base = capsule.agent_spec
        updates: dict[str, Any] = {
            "model": versions.model or base.model,
            "prompt": versions.prompt or base.prompt,
            "skill": versions.skill,
            "tool_policy": versions.tool_policy or base.tool_policy,
            "stop_policy": versions.stop_policy or base.stop_policy,
            "verification_policy": (
                versions.verification_policy
                if versions.verification_policy is not None
                else base.verification_policy
            ),
            "metadata": {
                **base.metadata,
                "studio_replay_request_id": request.replay_request_id,
                "selected_agent_version_id": (
                    selected_agent_ref.version_id
                    if selected_agent_ref
                    else None
                ),
            },
        }
        selected = base.model_copy(update=updates)
        self._validate_selected_versions(request, capsule)
        return AgentSpec.model_validate(
            selected.model_dump(mode="json"),
            strict=False,
        )

    def _validate_selected_versions(
        self,
        request: ReplayRequest,
        capsule: ReplayCapsule,
    ) -> None:
        source_by_id = {
            item.version_id: item
            for item in self._component_refs(
                capsule.source_component_versions
            )
        }
        for ref in self._component_refs(
            request.selected_component_versions
        ):
            if ref.version_id in source_by_id:
                if ref != source_by_id[ref.version_id]:
                    raise ValueError(
                        "selected component version ID changed identity"
                    )
                continue
            if ref.kind in {
                ComponentKind.AGENT_SPEC,
                ComponentKind.PROMPT,
                ComponentKind.SKILL,
                ComponentKind.TOOL_POLICY,
                ComponentKind.STOP_POLICY,
                ComponentKind.VERIFICATION_POLICY,
                ComponentKind.RUBRIC,
            }:
                manifest = self.version_store.manifest(ref.version_id)
                if manifest is None:
                    raise ValueError(
                        "selected component is not registered: "
                        f"{ref.version_id}"
                    )
                registered = manifest.version_ref
                if registered != ref:
                    raise ValueError(
                        "selected component differs from its immutable "
                        "version manifest"
                    )
            elif ref.artifact_id is not None:
                envelope = self.artifact_store.get(ref.artifact_id)
                if envelope is None:
                    raise ValueError(
                        "selected component artifact is missing: "
                        f"{ref.artifact_id}"
                    )
                if (
                    ref.content_hash is not None
                    and ref.content_hash != envelope.content_hash
                ):
                    raise ValueError(
                        "selected component content hash does not match"
                    )

    @staticmethod
    def _component_refs(versions) -> tuple[VersionRef, ...]:
        return tuple(
            item
            for item in (
                versions.runtime,
                versions.scheduler,
                versions.model,
                versions.agent_spec,
                versions.prompt,
                versions.skill,
                versions.tool_policy,
                versions.stop_policy,
                versions.verification_policy,
                versions.rubric,
                *versions.tools,
            )
            if item is not None
        )

    def _validate_saved_component_selection(
        self,
        request: ReplayRequest,
        capsule: ReplayCapsule,
    ) -> None:
        source = capsule.source_component_versions
        selected = request.selected_component_versions
        for field_name in ("model", "prompt", "skill"):
            if getattr(source, field_name) != getattr(
                selected,
                field_name,
            ):
                raise ValueError(
                    "sealed replay can change policies and runtime controls, "
                    "but model/prompt/skill changes require live-environment "
                    f"execution ({field_name} changed)"
                )

    def _load_registered_agent_spec(
        self,
        ref: VersionRef,
    ) -> AgentSpec:
        manifest = self.version_store.manifest(ref.version_id)
        if manifest is None:
            raise ValueError(
                f"AgentSpec version is not registered: {ref.version_id}"
            )
        content = json.loads(
            self.artifact_store.read_bytes(
                manifest.content_artifact_id
            ).decode()
        )
        candidates = [
            content,
            content.get("agent_spec") if isinstance(content, dict) else None,
            content.get("spec") if isinstance(content, dict) else None,
            content.get("content") if isinstance(content, dict) else None,
        ]
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            try:
                return AgentSpec.model_validate(candidate, strict=False)
            except Exception:
                continue
        raise ValueError(
            "registered AgentSpec artifact has no AgentSpec payload"
        )
