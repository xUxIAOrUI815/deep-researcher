from __future__ import annotations

import hashlib
import threading
from typing import Any

from deep_researcher.contracts import (
    AgentSpec,
    BudgetUsage,
    ComponentKind,
    ComponentVersionSet,
    ErrorCategory,
    ErrorRecord,
    EventLevel,
    EventType,
    RunEvent,
    RunStatus,
    SpanKind,
    VersionRef,
    new_id,
    utc_now,
)
from deep_researcher.events import EventQuery, EventRecorder, RunTerminalError, SequenceConflict

from .middleware import redact
from .types import KernelEvent


def component_versions_for_agent(
    spec: AgentSpec,
    *,
    runtime: VersionRef,
    scheduler: VersionRef,
) -> ComponentVersionSet:
    if runtime.kind != ComponentKind.RUNTIME:
        raise ValueError("runtime must be a runtime VersionRef")
    if scheduler.kind != ComponentKind.SCHEDULER:
        raise ValueError("scheduler must be a scheduler VersionRef")
    return ComponentVersionSet(
        runtime=runtime,
        scheduler=scheduler,
        model=spec.model,
        agent_spec=VersionRef(kind=ComponentKind.AGENT_SPEC, name=spec.name, version=spec.version),
        prompt=spec.prompt,
        skill=spec.skill,
        tool_policy=spec.tool_policy,
        stop_policy=spec.stop_policy,
        verification_policy=spec.verification_policy,
    )


class EventRecorderKernelSink:
    """Persist KernelEvents as balanced RunEvent spans and structured facts.

    The sink may join an existing open run or create its root span. It closes
    only the task/agent span that it owns; run termination remains the
    orchestration runtime's responsibility.
    """

    def __init__(
        self,
        recorder: EventRecorder,
        *,
        run_id: str,
        thread_id: str,
        trace_id: str,
        correlation_id: str,
        component_versions: ComponentVersionSet,
        producer_id: str = "agent_kernel",
    ) -> None:
        self.recorder = recorder
        self.run_id = run_id
        self.thread_id = thread_id
        self.trace_id = trace_id
        self.correlation_id = correlation_id
        self.component_versions = component_versions
        self.producer_id = producer_id
        self._lock = threading.RLock()
        self._root_span_id: str | None = None
        self._agent_spans: dict[str, str] = {}
        self._model_spans: dict[tuple[str, str], str] = {}
        self._action_spans: dict[tuple[str, str], str] = {}
        self._last_event_id: str | None = None

    def emit(self, event: KernelEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError(f"kernel event run mismatch: {event.run_id}")
        with self._lock:
            self._ensure_run(event)
            if event.event_type == "kernel.started":
                self._start_agent_span(event)
            elif event.event_type == "kernel.stopped":
                self._finish_agent_span(event)
            elif event.event_type == "model.started":
                self._start_operation_span(event, SpanKind.MODEL)
            elif event.event_type in {"model.completed", "model.failed"}:
                self._finish_operation_span(event, SpanKind.MODEL)
            elif event.event_type == "action.started":
                self._start_operation_span(event, SpanKind.TOOL)
            elif event.event_type in {"action.completed", "action.failed"}:
                self._finish_operation_span(event, SpanKind.TOOL)
            else:
                self._record_fact(event)

    def _ensure_run(self, event: KernelEvent) -> None:
        record = self.recorder.store.get_run(self.run_id)
        if record is not None:
            if record.thread_id != self.thread_id or record.trace_id != self.trace_id:
                raise ValueError("kernel sink run identity does not match the existing event run")
            if record.terminal_event_id is not None:
                raise ValueError("kernel sink cannot append to a terminal run")
            self._root_span_id = record.root_span_id
            if self._last_event_id is None:
                previous = self.recorder.store.list(
                    EventQuery(self.run_id, after_sequence=max(0, record.next_sequence - 2), limit=1)
                ).items
                self._last_event_id = previous[-1].event_id if previous else None
            return
        root_span = self._stable_id("span", self.run_id, "root")
        root_event = self._build(
            source=event,
            event_type=EventType.RUN_STARTED,
            status=RunStatus.RUNNING,
            span_id=root_span,
            parent_span_id=None,
            span_kind=SpanKind.RUN,
            actor_id="runtime_agent_kernel",
            payload={"message": "Run created by AgentKernel event sink."},
        )
        try:
            outcome = self._record(root_event)
        except RunTerminalError:
            # A sibling sink may have created the same run after our read.
            record = self.recorder.store.get_run(self.run_id)
            if record is None or record.terminal_event_id is not None:
                raise
            if record.thread_id != self.thread_id or record.trace_id != self.trace_id:
                raise ValueError("concurrently created run has a different identity")
            self._root_span_id = record.root_span_id
            previous = self.recorder.store.list(
                EventQuery(self.run_id, after_sequence=max(0, record.next_sequence - 2), limit=1)
            ).items
            self._last_event_id = previous[-1].event_id if previous else None
        else:
            self._root_span_id = root_span
            self._last_event_id = outcome.event_id

    def _start_agent_span(self, event: KernelEvent) -> None:
        if event.task_id in self._agent_spans:
            raise ValueError(f"kernel task span is already active: {event.task_id}")
        span_id = new_id("span")
        self._agent_spans[event.task_id] = span_id
        self._record_and_track(
            self._build(
                source=event,
                event_type=EventType.SPAN_STARTED,
                status=RunStatus.RUNNING,
                span_id=span_id,
                parent_span_id=self._root_span_id,
                span_kind=SpanKind.AGENT,
                payload=event.payload,
            )
        )

    def _finish_agent_span(self, event: KernelEvent) -> None:
        span_id = self._agent_spans.get(event.task_id)
        if span_id is None:
            raise ValueError(f"kernel task span was not started: {event.task_id}")
        if any(task_id == event.task_id for task_id, _ in self._model_spans):
            raise ValueError("cannot stop kernel task with an open model span")
        if any(task_id == event.task_id for task_id, _ in self._action_spans):
            raise ValueError("cannot stop kernel task with an open action span")
        status_value = str(event.payload.get("status", "partial"))
        failed = status_value in {"failed", "cancelled", "rejected"}
        error = self._payload_error(event.payload) if failed else None
        self._record_and_track(
            self._build(
                source=event,
                event_type=EventType.SPAN_FAILED if failed else EventType.SPAN_COMPLETED,
                status=RunStatus.FAILED if failed else RunStatus.RUNNING,
                span_id=span_id,
                parent_span_id=self._root_span_id,
                span_kind=SpanKind.AGENT,
                payload=event.payload,
                error=error,
            )
        )
        del self._agent_spans[event.task_id]

    def _start_operation_span(self, event: KernelEvent, kind: SpanKind) -> None:
        parent = self._agent_spans.get(event.task_id)
        if parent is None:
            raise ValueError(f"operation has no active kernel task span: {event.task_id}")
        key = self._operation_key(event, kind)
        target = self._model_spans if kind == SpanKind.MODEL else self._action_spans
        if key in target:
            raise ValueError(f"kernel operation span is already active: {key[1]}")
        span_id = new_id("span")
        target[key] = span_id
        event_type = EventType.MODEL_STARTED if kind == SpanKind.MODEL else EventType.TOOL_STARTED
        self._record_and_track(
            self._build(
                source=event,
                event_type=event_type,
                status=RunStatus.RUNNING,
                span_id=span_id,
                parent_span_id=parent,
                span_kind=kind,
                payload=event.payload,
            )
        )

    def _finish_operation_span(self, event: KernelEvent, kind: SpanKind) -> None:
        key = self._operation_key(event, kind)
        target = self._model_spans if kind == SpanKind.MODEL else self._action_spans
        span_id = target.get(key)
        if span_id is None:
            raise ValueError(f"kernel operation completion has no start: {key[1]}")
        parent = self._agent_spans.get(event.task_id)
        if parent is None:
            raise ValueError(f"operation has no active kernel task span: {event.task_id}")
        failed = event.event_type.endswith(".failed")
        if kind == SpanKind.MODEL:
            event_type = EventType.MODEL_FAILED if failed else EventType.MODEL_COMPLETED
        else:
            event_type = EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED
        error = self._payload_error(event.payload) if failed else None
        self._record_and_track(
            self._build(
                source=event,
                event_type=event_type,
                status=RunStatus.FAILED if failed else RunStatus.RUNNING,
                span_id=span_id,
                parent_span_id=parent,
                span_kind=kind,
                payload=event.payload,
                error=error,
            )
        )
        del target[key]

    def _record_fact(self, event: KernelEvent) -> None:
        span_id = self._agent_spans.get(event.task_id) or self._root_span_id
        parent = self._root_span_id if span_id != self._root_span_id else None
        mapping = {
            "command.proposed": EventType.COMMAND_PROPOSED,
            "decision.recorded": EventType.DECISION_RECORDED,
            "policy.decided": EventType.POLICY_DECIDED,
            "approval.requested": EventType.APPROVAL_REQUESTED,
            "retry.scheduled": EventType.RETRY_SCHEDULED,
            "verification.completed": EventType.VERIFICATION_COMPLETED,
        }
        self._record_and_track(
            self._build(
                source=event,
                event_type=mapping.get(event.event_type, EventType.CUSTOM),
                status=RunStatus.RUNNING,
                span_id=span_id,
                parent_span_id=parent,
                span_kind=SpanKind.AGENT if span_id != self._root_span_id else SpanKind.RUN,
                payload={"kernel_event_type": event.event_type, **event.payload},
            )
        )

    def _build(
        self,
        *,
        source: KernelEvent,
        event_type: EventType,
        status: RunStatus,
        span_id: str,
        parent_span_id: str | None,
        span_kind: SpanKind,
        payload: dict[str, Any],
        actor_id: str | None = None,
        error: ErrorRecord | None = None,
    ) -> RunEvent:
        safe_payload = redact(payload)
        usage_data = safe_payload.get("usage", {}) if isinstance(safe_payload.get("usage"), dict) else {}
        usage = BudgetUsage(**{
            name: usage_data.get(name, 0)
            for name in BudgetUsage.model_fields
            if name != "schema_version"
        })
        latency_ms = max(0.0, float(safe_payload.get("latency_ms", 0.0) or 0.0))
        attempt = max(1, int(safe_payload.get("attempt", safe_payload.get("repair_attempt", 1)) or 1))
        return RunEvent(
            sequence_no=self.recorder.store.next_sequence(self.run_id),
            event_type=event_type,
            level=EventLevel.ERROR if error is not None else EventLevel.INFO,
            status=status,
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            correlation_id=self.correlation_id,
            causation_event_id=self._last_event_id,
            run_id=self.run_id,
            thread_id=self.thread_id,
            task_id=source.task_id,
            actor_id=actor_id or source.actor_id,
            producer_id=self.producer_id,
            usage=usage,
            latency_ms=latency_ms,
            attempt=attempt,
            error=error,
            component_versions=self.component_versions,
            occurred_at=source.occurred_at,
            recorded_at=max(source.occurred_at, utc_now()),
            payload={"kernel_event_type": source.event_type, **safe_payload},
        )

    def _record(self, event: RunEvent) -> RunEvent:
        for _ in range(8):
            try:
                return self.recorder.record(event).append.event
            except SequenceConflict:
                event = event.model_copy(update={"sequence_no": self.recorder.store.next_sequence(event.run_id)})
        raise SequenceConflict(f"could not allocate sequence for kernel run {event.run_id}")

    def _record_and_track(self, event: RunEvent) -> RunEvent:
        recorded = self._record(event)
        self._last_event_id = recorded.event_id
        return recorded

    @staticmethod
    def _operation_key(event: KernelEvent, kind: SpanKind) -> tuple[str, str]:
        if kind == SpanKind.MODEL:
            repair = event.payload.get("repair_attempt")
            suffix = f"repair:{repair}" if repair is not None else f"decision:{event.payload.get('attempt', 1)}"
            return event.task_id, f"round:{event.payload.get('round', 0)}:{suffix}"
        return event.task_id, f"command:{event.payload.get('command_id', 'unknown')}:attempt:{event.payload.get('attempt', 1)}"

    @staticmethod
    def _payload_error(payload: dict[str, Any]) -> ErrorRecord:
        candidate = payload.get("error")
        if isinstance(candidate, dict):
            try:
                return ErrorRecord.model_validate(candidate)
            except Exception:
                candidate = None
        return ErrorRecord(
            category=ErrorCategory.INTERNAL,
            code="kernel_operation_failed",
            message=str(payload.get("summary") or payload.get("reason") or "Kernel operation failed.")[:2000],
            fatal=True,
        )

    @staticmethod
    def _stable_id(prefix: str, *parts: str) -> str:
        digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]
        return f"{prefix}_{digest}"
