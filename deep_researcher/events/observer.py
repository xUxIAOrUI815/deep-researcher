from __future__ import annotations

from datetime import datetime, timezone
import re
import threading
from typing import Any

from core.observability.tracing import EventLevel as LegacyLevel
from core.observability.tracing import EventType as LegacyType
from core.observability.tracing import NoopObserver, ObservabilityEvent

from deep_researcher.contracts import (
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

from .recorder import EventRecorder, RecordOutcome
from .redaction import RedactionPolicy
from .store import EventQuery, SequenceConflict


_LEGACY_EVENT_MAP: dict[LegacyType, EventType] = {
    LegacyType.TASK_CREATED: EventType.TASK_CREATED,
    LegacyType.TASK_UPDATED: EventType.TASK_STATE_CHANGED,
    LegacyType.TASK_STATUS_CHANGED: EventType.TASK_STATE_CHANGED,
    LegacyType.TASK_MERGED: EventType.TASK_STATE_CHANGED,
    LegacyType.TASK_PRUNED: EventType.TASK_STATE_CHANGED,
    LegacyType.TASK_DEFERRED: EventType.TASK_STATE_CHANGED,
    LegacyType.EVIDENCE_CREATED: EventType.EVIDENCE_CHANGED,
    LegacyType.EVIDENCE_LINKED: EventType.EVIDENCE_CHANGED,
    LegacyType.CONFLICT_DETECTED: EventType.EVIDENCE_CHANGED,
    LegacyType.REPORT_FINALIZED: EventType.REPORT_CHANGED,
    LegacyType.QUERY_GENERATED: EventType.DECISION_RECORDED,
    LegacyType.QUERY_REJECTED: EventType.DECISION_RECORDED,
    LegacyType.QUERY_DROPPED: EventType.DECISION_RECORDED,
    LegacyType.SOURCE_ACCEPTED: EventType.CUSTOM,
    LegacyType.SOURCE_REJECTED: EventType.CUSTOM,
    LegacyType.EXPLORATION_STOPPED: EventType.BUDGET_CHANGED,
    LegacyType.DISTILL_STARTED: EventType.EVIDENCE_CHANGED,
    LegacyType.PASSAGE_CLEANED: EventType.EVIDENCE_CHANGED,
    LegacyType.CLAIM_EXTRACTED: EventType.EVIDENCE_CHANGED,
    LegacyType.FACT_EXTRACTED: EventType.EVIDENCE_CHANGED,
    LegacyType.EVIDENCE_PACK_CREATED: EventType.EVIDENCE_CHANGED,
    LegacyType.COMPRESSION_COMPLETED: EventType.EVIDENCE_CHANGED,
    LegacyType.WRITER_STARTED: EventType.REPORT_CHANGED,
    LegacyType.SECTION_GENERATED: EventType.REPORT_CHANGED,
    LegacyType.WRITER_COMPLETED: EventType.REPORT_CHANGED,
    LegacyType.RETRY_SCHEDULED: EventType.RETRY_SCHEDULED,
    LegacyType.BUDGET_SNAPSHOT: EventType.BUDGET_CHANGED,
}


def _legacy_id(prefix: str, value: str | None) -> str | None:
    if not value:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value)).strip("-.") or "unknown"
    if re.fullmatch(r"[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*", cleaned):
        return cleaned
    return f"{prefix}_{cleaned}"


def _version(kind: ComponentKind, name: str) -> VersionRef:
    return VersionRef(kind=kind, name=name, version="1.0.0")


def legacy_component_versions() -> ComponentVersionSet:
    return ComponentVersionSet(
        runtime=_version(ComponentKind.RUNTIME, "draft-runtime"),
        scheduler=_version(ComponentKind.SCHEDULER, "langgraph-adapter"),
        prompt=_version(ComponentKind.PROMPT, "draft-prompts"),
    )


class PersistentEventObserver(NoopObserver):
    """Compatibility observer that persists the draft pipeline as RunEvents."""

    def __init__(
        self,
        recorder: EventRecorder,
        *,
        component_versions: ComponentVersionSet | None = None,
        redaction_policy: RedactionPolicy | None = None,
    ) -> None:
        self.recorder = recorder
        self.component_versions = component_versions or legacy_component_versions()
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self._lock = threading.RLock()
        self._root_spans: dict[str, str] = {}
        self._node_spans: dict[tuple[str, str], str] = {}
        self._operation_spans: dict[tuple[str, str], tuple[str, SpanKind]] = {}
        self._active_spans: dict[str, list[str]] = {}
        self._last_event_ids: dict[str, str] = {}
        self.failures: list[str] = []

    def emit(self, event: ObservabilityEvent) -> None:
        with self._lock:
            try:
                if event.event_type == LegacyType.RUN_STARTED:
                    self._ensure_root(event)
                    return
                self._ensure_root(event)
                if event.event_type == LegacyType.NODE_STARTED:
                    self._start_node_span(event)
                elif event.event_type in {LegacyType.NODE_COMPLETED, LegacyType.NODE_FAILED}:
                    self._finish_node_span(event)
                elif event.event_type in {LegacyType.MODEL_STARTED, LegacyType.TOOL_STARTED}:
                    self._start_operation_span(event)
                elif event.event_type in {
                    LegacyType.MODEL_COMPLETED,
                    LegacyType.MODEL_FAILED,
                    LegacyType.TOOL_COMPLETED,
                    LegacyType.TOOL_FAILED,
                }:
                    self._finish_operation_span(event)
                elif event.event_type in {LegacyType.RUN_COMPLETED, LegacyType.RUN_FAILED}:
                    self._finish_run(event)
                else:
                    self._record_legacy_event(event)
            except Exception as exc:
                self.failures.append(f"{type(exc).__name__}: {exc}")
                raise

    def _run_id(self, event: ObservabilityEvent) -> str:
        return _legacy_id("run", event.run_id) or "run_unknown"

    def _thread_id(self, event: ObservabilityEvent) -> str:
        return _legacy_id("thread", event.thread_id) or "thread_unknown"

    def _trace_id(self, event: ObservabilityEvent) -> str:
        return _legacy_id("trace", event.trace_id) or f"trace_{self._run_id(event)}"

    def _timestamp(self, event: ObservabilityEvent) -> datetime:
        value = event.timestamp
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    def _ensure_root(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        if run_id in self._root_spans:
            return
        existing_run = self.recorder.store.get_run(run_id)
        if existing_run is not None:
            self._root_spans[run_id] = existing_run.root_span_id
            self._active_spans[run_id] = []
            previous = self.recorder.store.list(
                EventQuery(
                    run_id,
                    after_sequence=max(0, existing_run.next_sequence - 2),
                    limit=1,
                )
            ).items
            if previous:
                self._last_event_ids[run_id] = previous[-1].event_id
            return
        root_span = _legacy_id("span", event.span_id) or new_id("span")
        self._root_spans[run_id] = root_span
        self._active_spans[run_id] = []
        root_event = self._build(
            event,
            event_type=EventType.RUN_STARTED,
            status=RunStatus.RUNNING,
            span_id=root_span,
            parent_span_id=None,
            span_kind=SpanKind.RUN,
            actor_id="runtime_legacy",
            payload={"message": event.message or "Research run started", "legacy_event_type": event.event_type.value},
        )
        if event.event_type != LegacyType.RUN_STARTED:
            root_event = root_event.model_copy(update={"event_id": new_id("event")})
        outcome = self._record_with_sequence_retry(root_event)
        self._last_event_ids[run_id] = outcome.append.event.event_id

    def _start_node_span(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        node_name = event.node_name or "unknown"
        span_id = new_id("span")
        parent_span = self._active_spans[run_id][-1] if self._active_spans[run_id] else self._root_spans[run_id]
        self._node_spans[(run_id, node_name)] = span_id
        self._active_spans[run_id].append(span_id)
        domain = self._build(
            event,
            event_type=EventType.SPAN_STARTED,
            status=RunStatus.RUNNING,
            span_id=span_id,
            parent_span_id=parent_span,
            span_kind=SpanKind.AGENT,
            actor_id=_legacy_id("agent", node_name) or "agent_unknown",
            payload={"message": event.message, "node_name": node_name, **event.payload},
        )
        self._record_and_track(domain)

    def _finish_node_span(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        node_name = event.node_name or "unknown"
        span_id = self._node_spans.pop((run_id, node_name), None)
        if span_id is None:
            self._record_legacy_event(event)
            return
        active = self._active_spans[run_id]
        if span_id in active:
            active.remove(span_id)
        failed = event.event_type == LegacyType.NODE_FAILED
        error = self._error(event) if failed else None
        domain = self._build(
            event,
            event_type=EventType.SPAN_FAILED if failed else EventType.SPAN_COMPLETED,
            status=RunStatus.FAILED if failed else RunStatus.RUNNING,
            span_id=span_id,
            parent_span_id=self._root_spans[run_id],
            span_kind=SpanKind.AGENT,
            actor_id=_legacy_id("agent", node_name) or "agent_unknown",
            payload={"message": event.message, "node_name": node_name, **event.payload},
            error=error,
        )
        self._record_and_track(domain)

    def _operation_key(self, event: ObservabilityEvent) -> str:
        return str(event.payload.get("operation_id") or event.payload.get("call_id") or "")

    def _start_operation_span(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        operation_id = self._operation_key(event)
        if not operation_id:
            raise ValueError("model and tool events require an operation_id")
        span_id = new_id("span")
        span_kind = SpanKind.MODEL if event.event_type == LegacyType.MODEL_STARTED else SpanKind.TOOL
        parent_span = self._active_spans[run_id][-1] if self._active_spans[run_id] else self._root_spans[run_id]
        self._operation_spans[(run_id, operation_id)] = (span_id, span_kind)
        self._active_spans[run_id].append(span_id)
        domain = self._build(
            event,
            event_type=EventType.MODEL_STARTED if span_kind == SpanKind.MODEL else EventType.TOOL_STARTED,
            status=RunStatus.RUNNING,
            span_id=span_id,
            parent_span_id=parent_span,
            span_kind=span_kind,
            actor_id=_legacy_id("agent", event.agent_name or self._actor_for(event.event_type)) or "agent_runtime",
            payload={"message": event.message, **event.payload},
        )
        self._record_and_track(domain)

    def _finish_operation_span(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        operation_id = self._operation_key(event)
        operation = self._operation_spans.pop((run_id, operation_id), None)
        if operation is None:
            raise ValueError(f"operation completion has no start event: {operation_id or '<missing>'}")
        span_id, span_kind = operation
        active = self._active_spans[run_id]
        if span_id in active:
            active.remove(span_id)
        failed = event.event_type in {LegacyType.MODEL_FAILED, LegacyType.TOOL_FAILED}
        if span_kind == SpanKind.MODEL:
            event_type = EventType.MODEL_FAILED if failed else EventType.MODEL_COMPLETED
        else:
            event_type = EventType.TOOL_FAILED if failed else EventType.TOOL_COMPLETED
        parent_span = active[-1] if active else self._root_spans[run_id]
        domain = self._build(
            event,
            event_type=event_type,
            status=RunStatus.FAILED if failed else RunStatus.RUNNING,
            span_id=span_id,
            parent_span_id=parent_span,
            span_kind=span_kind,
            actor_id=_legacy_id("agent", event.agent_name or self._actor_for(event.event_type)) or "agent_runtime",
            payload={"message": event.message, **event.payload},
            error=self._error(event) if failed else None,
        )
        self._record_and_track(domain)

    def _finish_run(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        if self._active_spans.get(run_id):
            raise RuntimeError("cannot terminate a run with active node spans")
        failed = event.event_type == LegacyType.RUN_FAILED
        domain = self._build(
            event,
            event_type=EventType.RUN_FAILED if failed else EventType.RUN_COMPLETED,
            status=RunStatus.FAILED if failed else RunStatus.SUCCEEDED,
            span_id=self._root_spans[run_id],
            parent_span_id=None,
            span_kind=SpanKind.RUN,
            actor_id="runtime_legacy",
            payload={"message": event.message, **event.payload},
            error=self._error(event) if failed else None,
        )
        self._record_and_track(domain)

    def _record_legacy_event(self, event: ObservabilityEvent) -> None:
        run_id = self._run_id(event)
        event_type = _LEGACY_EVENT_MAP.get(event.event_type, EventType.CUSTOM)
        span_id = self._active_spans[run_id][-1] if self._active_spans[run_id] else self._root_spans[run_id]
        parent_span_id = self._root_spans[run_id] if span_id != self._root_spans[run_id] else None
        actor_name = event.agent_name or event.node_name or self._actor_for(event.event_type)
        payload = {
            "message": event.message,
            "legacy_event_type": event.event_type.value,
            **event.payload,
        }
        for name in ("source_id", "fact_id", "claim_id", "evidence_id", "conflict_id", "section_id"):
            value = getattr(event, name)
            if value:
                payload[name] = value
        if event_type == EventType.DECISION_RECORDED:
            payload = {
                "observation_summary": event.message or event.event_type.value,
                "selected_command_ids": [],
                "alternatives_considered": [],
                "policy_checks": [],
                "metadata": self.redaction_policy.redact(payload),
            }
        failed = event.event_type in {LegacyType.AGENT_FAILED}
        domain = self._build(
            event,
            event_type=event_type,
            status=RunStatus.FAILED if failed else RunStatus.RUNNING,
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=SpanKind.AGENT,
            actor_id=_legacy_id("agent", actor_name) or "agent_runtime",
            payload=payload,
            error=self._error(event) if failed else None,
        )
        self._record_and_track(domain)

    @staticmethod
    def _actor_for(event_type: LegacyType) -> str:
        value = event_type.value
        if value.startswith(("query", "source", "exploration")):
            return "researcher"
        if value.startswith("tool"):
            return "researcher"
        if value.startswith("model"):
            return "model_adapter"
        if value.startswith(("distill", "passage", "claim", "fact", "evidence", "conflict", "compression")):
            return "distiller"
        if value.startswith(("writer", "section", "report")):
            return "writer"
        return "runtime"

    def _error(self, event: ObservabilityEvent) -> ErrorRecord:
        payload = self.redaction_policy.redact(event.payload)
        return ErrorRecord(
            category=ErrorCategory.INTERNAL,
            code=str(payload.get("error_code") or event.event_type.value).replace(".", "_")[:120],
            message=str(payload.get("error") or event.message or "legacy pipeline failure")[:2000],
            retryable=bool(payload.get("retryable", False)),
            fatal=bool(
                payload.get(
                    "fatal",
                    event.event_type in {LegacyType.RUN_FAILED, LegacyType.NODE_FAILED, LegacyType.AGENT_FAILED},
                )
            ),
            actor_id=_legacy_id("agent", event.agent_name or event.node_name or "runtime"),
            task_id=_legacy_id("task", event.task_id),
        )

    def _build(
        self,
        legacy: ObservabilityEvent,
        *,
        event_type: EventType,
        status: RunStatus,
        span_id: str,
        parent_span_id: str | None,
        span_kind: SpanKind,
        actor_id: str,
        payload: dict[str, Any],
        error: ErrorRecord | None = None,
    ) -> RunEvent:
        usage_payload = payload.get("usage", {}) if isinstance(payload.get("usage"), dict) else {}
        usage = BudgetUsage(
            input_tokens=max(0, int(usage_payload.get("input_tokens", 0) or 0)),
            output_tokens=max(0, int(usage_payload.get("output_tokens", 0) or 0)),
            cost_usd=max(0.0, float(usage_payload.get("cost_usd", 0.0) or 0.0)),
            wall_time_seconds=max(0.0, float(usage_payload.get("wall_time_seconds", 0.0) or 0.0)),
            model_calls=max(0, int(usage_payload.get("model_calls", 0) or 0)),
            tool_calls=max(0, int(usage_payload.get("tool_calls", 0) or 0)),
            search_calls=max(0, int(usage_payload.get("search_calls", 0) or 0)),
            retries=max(0, int(usage_payload.get("retries", 0) or 0)),
            errors=max(0, int(usage_payload.get("errors", 0) or 0)),
        )
        occurred = self._timestamp(legacy)
        return RunEvent(
            event_id=_legacy_id("event", legacy.event_id) or new_id("event"),
            sequence_no=self.recorder.store.next_sequence(self._run_id(legacy)),
            event_type=event_type,
            level={
                LegacyLevel.DEBUG: EventLevel.DEBUG,
                LegacyLevel.INFO: EventLevel.INFO,
                LegacyLevel.WARNING: EventLevel.WARNING,
                LegacyLevel.ERROR: EventLevel.ERROR,
            }[legacy.level],
            status=status,
            trace_id=self._trace_id(legacy),
            span_id=span_id,
            parent_span_id=parent_span_id,
            span_kind=span_kind,
            correlation_id=_legacy_id("correlation", legacy.research_id) or "correlation_unknown",
            causation_event_id=self._last_event_ids.get(self._run_id(legacy)),
            run_id=self._run_id(legacy),
            thread_id=self._thread_id(legacy),
            task_id=_legacy_id("task", legacy.task_id),
            actor_id=actor_id,
            producer_id="observer_legacy_adapter",
            usage=usage,
            latency_ms=max(0.0, float(payload.get("latency_ms", 0.0) or 0.0)),
            attempt=max(1, int(payload.get("attempt", 1) or 1)),
            error=error,
            component_versions=self.component_versions,
            occurred_at=occurred,
            recorded_at=max(occurred, utc_now()),
            payload=self.redaction_policy.redact(payload),
        )

    def _record_with_sequence_retry(self, event: RunEvent) -> RecordOutcome:
        for _ in range(8):
            try:
                return self.recorder.record(event)
            except SequenceConflict:
                event = event.model_copy(update={"sequence_no": self.recorder.store.next_sequence(event.run_id)})
        raise SequenceConflict(f"could not allocate sequence for run {event.run_id}")

    def _record_and_track(self, event: RunEvent) -> RecordOutcome:
        outcome = self._record_with_sequence_retry(event)
        self._last_event_ids[event.run_id] = outcome.append.event.event_id
        return outcome
