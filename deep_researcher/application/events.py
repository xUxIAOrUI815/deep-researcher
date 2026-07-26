from __future__ import annotations

import hashlib
import threading
from typing import Any

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
    utc_now,
)
from deep_researcher.evidence import (
    EvidenceDomainEvent,
    EvidenceEventSink,
    EvidenceTraceContext,
    EventRecorderEvidenceSink,
)
from deep_researcher.events import EventQuery, EventRecorder, SequenceConflict
from deep_researcher.kernel import EventRecorderKernelSink


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


def _version(kind: ComponentKind, name: str, version: str = "1.0.0") -> VersionRef:
    return VersionRef(
        version_id=f"version_{name.replace('-', '_')}_{version.replace('.', '_')}",
        kind=kind,
        name=name,
        version=version,
    )


def application_component_versions(
    *,
    model_name: str = "configured-model",
    model_version: str = "1.0.0",
) -> ComponentVersionSet:
    return ComponentVersionSet(
        runtime=_version(
            ComponentKind.RUNTIME,
            "background001-application-runtime",
        ),
        scheduler=_version(
            ComponentKind.SCHEDULER,
            "native-event-sourced-scheduler",
        ),
        model=_version(ComponentKind.MODEL, model_name, model_version),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "background001-verification-policy",
        ),
        tools=tuple(
            _version(ComponentKind.TOOL, name)
            for name in (
                "research-search",
                "research-read",
                "research-extract",
                "research-compare",
                "research-verify-source",
            )
        ),
    )


class ManagedEvidenceEventSink(EvidenceEventSink):
    def __init__(
        self,
        recorder: EventRecorder,
        *,
        run_id: str,
        thread_id: str,
        trace_id: str,
        root_span_id: str,
        correlation_id: str,
        component_versions: ComponentVersionSet,
        attempt: int = 1,
    ) -> None:
        self.recorder = recorder
        self.run_id = run_id
        self.thread_id = thread_id
        self.trace_id = trace_id
        self.root_span_id = root_span_id
        self.correlation_id = correlation_id
        self.component_versions = component_versions
        if attempt < 1:
            raise ValueError("evidence span attempt must be positive")
        self.span_id = _stable_id(
            "span",
            run_id,
            "evidence",
            str(attempt),
        )
        self._started = False
        self._closed = False
        self._lock = threading.RLock()
        self._delegate = EventRecorderEvidenceSink(
            recorder,
            EvidenceTraceContext(
                thread_id=thread_id,
                trace_id=trace_id,
                span_id=self.span_id,
                parent_span_id=root_span_id,
                correlation_id=correlation_id,
                component_versions=component_versions,
            ),
        )

    def emit(self, event: EvidenceDomainEvent) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("evidence event sink is closed")
            if event.run_id != self.run_id:
                raise ValueError("evidence event run does not match sink")
            if not self._started:
                self._append(
                    EventType.SPAN_STARTED,
                    RunStatus.RUNNING,
                    payload={"stage": "evidence_verification"},
                )
                self._started = True
            self._delegate.emit(event)

    def close(self, *, failed: bool = False) -> None:
        with self._lock:
            if self._closed:
                return
            if self._started:
                self._append(
                    EventType.SPAN_FAILED if failed else EventType.SPAN_COMPLETED,
                    RunStatus.FAILED if failed else RunStatus.RUNNING,
                    payload={"stage": "evidence_verification"},
                    error=(
                        ErrorRecord(
                            category=ErrorCategory.INTERNAL,
                            code="evidence_stage_failed",
                            message="Evidence verification stage failed.",
                            fatal=True,
                        )
                        if failed
                        else None
                    ),
                )
            self._closed = True

    def _append(
        self,
        event_type: EventType,
        status: RunStatus,
        *,
        payload: dict[str, Any],
        error: ErrorRecord | None = None,
    ) -> None:
        event = RunEvent(
            sequence_no=self.recorder.store.next_sequence(self.run_id),
            event_type=event_type,
            level=EventLevel.ERROR if error else EventLevel.INFO,
            status=status,
            trace_id=self.trace_id,
            span_id=self.span_id,
            parent_span_id=self.root_span_id,
            span_kind=SpanKind.VERIFICATION,
            correlation_id=self.correlation_id,
            run_id=self.run_id,
            thread_id=self.thread_id,
            actor_id="agent_spec_evidence_verifier_1_0_0",
            producer_id="runtime_application_evidence",
            error=error,
            component_versions=self.component_versions,
            payload=payload,
        )
        for _ in range(20):
            try:
                self.recorder.record(event)
                return
            except SequenceConflict:
                event = event.model_copy(
                    update={
                        "sequence_no": self.recorder.store.next_sequence(
                            self.run_id
                        )
                    }
                )
        raise SequenceConflict("could not append managed evidence span event")


class ApplicationRunEventController:
    """Owns the root run span and terminal event for an application run."""

    def __init__(
        self,
        recorder: EventRecorder,
        *,
        run_id: str,
        thread_id: str,
        trace_id: str,
        correlation_id: str,
        component_versions: ComponentVersionSet,
        attempt: int = 1,
    ) -> None:
        self.recorder = recorder
        self.run_id = run_id
        self.thread_id = thread_id
        self.trace_id = trace_id
        self.correlation_id = correlation_id
        self.component_versions = component_versions
        self.root_span_id = _stable_id("span", run_id, "root")
        self.last_event_id: str | None = None
        self.evidence_sink = ManagedEvidenceEventSink(
            recorder,
            run_id=run_id,
            thread_id=thread_id,
            trace_id=trace_id,
            root_span_id=self.root_span_id,
            correlation_id=correlation_id,
            component_versions=component_versions,
            attempt=attempt,
        )

    def start(self, *, query: str, research_id: str) -> None:
        existing = self.recorder.store.get_run(self.run_id)
        if existing is not None:
            if existing.terminal_event_id is not None:
                raise RuntimeError("cannot resume a terminal event run")
            if (
                existing.thread_id != self.thread_id
                or existing.trace_id != self.trace_id
            ):
                raise ValueError("event run identity mismatch")
            self.root_span_id = existing.root_span_id
            page = self.recorder.store.list(
                EventQuery(
                    self.run_id,
                    after_sequence=max(0, existing.next_sequence - 2),
                    limit=1,
                )
            )
            self.last_event_id = page.items[-1].event_id if page.items else None
            return
        self._record(
            EventType.RUN_STARTED,
            RunStatus.RUNNING,
            actor_id="runtime_application",
            payload={
                "research_id": research_id,
                "query": query,
                "scheduler": "native_event_sourced",
            },
        )

    def kernel_sink(self) -> EventRecorderKernelSink:
        return EventRecorderKernelSink(
            self.recorder,
            run_id=self.run_id,
            thread_id=self.thread_id,
            trace_id=self.trace_id,
            correlation_id=self.correlation_id,
            component_versions=self.component_versions,
        )

    def stage(
        self,
        name: str,
        *,
        payload: dict[str, Any] | None = None,
        output_artifact_ids: tuple[str, ...] = (),
    ) -> None:
        self._record(
            EventType.REPORT_CHANGED if name == "reporting" else EventType.CUSTOM,
            RunStatus.RUNNING,
            actor_id="runtime_application",
            payload={"stage": name, **(payload or {})},
            output_artifact_ids=output_artifact_ids,
        )

    def complete(
        self,
        *,
        output_artifact_ids: tuple[str, ...],
        usage: BudgetUsage,
    ) -> None:
        if self._terminal():
            return
        self.evidence_sink.close()
        self._record(
            EventType.RUN_COMPLETED,
            RunStatus.SUCCEEDED,
            actor_id="runtime_application",
            payload={"stage": "completed"},
            output_artifact_ids=output_artifact_ids,
            usage=usage,
        )

    def cancel(self, *, reason: str, usage: BudgetUsage) -> None:
        if self._terminal():
            return
        self.evidence_sink.close()
        self._record(
            EventType.RUN_CANCELLED,
            RunStatus.CANCELLED,
            actor_id="runtime_application",
            payload={"stage": "cancelled", "reason": reason},
            usage=usage,
        )

    def fail(self, *, error: ErrorRecord, usage: BudgetUsage) -> None:
        if self._terminal():
            return
        self.evidence_sink.close(failed=True)
        self._record(
            EventType.RUN_FAILED,
            RunStatus.FAILED,
            actor_id="runtime_application",
            payload={"stage": "failed", "error_code": error.code},
            error=error,
            usage=usage,
        )

    def _record(
        self,
        event_type: EventType,
        status: RunStatus,
        *,
        actor_id: str,
        payload: dict[str, Any],
        output_artifact_ids: tuple[str, ...] = (),
        error: ErrorRecord | None = None,
        usage: BudgetUsage | None = None,
    ) -> RunEvent:
        event = RunEvent(
            sequence_no=self.recorder.store.next_sequence(self.run_id),
            event_type=event_type,
            level=EventLevel.ERROR if error else EventLevel.INFO,
            status=status,
            trace_id=self.trace_id,
            span_id=self.root_span_id,
            span_kind=SpanKind.RUN,
            correlation_id=self.correlation_id,
            causation_event_id=self.last_event_id,
            run_id=self.run_id,
            thread_id=self.thread_id,
            actor_id=actor_id,
            producer_id="runtime_background001_application",
            output_artifact_ids=output_artifact_ids,
            usage=usage or BudgetUsage(),
            error=error,
            component_versions=self.component_versions,
            occurred_at=utc_now(),
            payload=payload,
        )
        for _ in range(20):
            try:
                recorded = self.recorder.record(event).append.event
                self.last_event_id = recorded.event_id
                return recorded
            except SequenceConflict:
                event = event.model_copy(
                    update={
                        "sequence_no": self.recorder.store.next_sequence(
                            self.run_id
                        ),
                        "causation_event_id": self.last_event_id,
                    }
                )
        raise SequenceConflict("could not append application run event")

    def _terminal(self) -> bool:
        record = self.recorder.store.get_run(self.run_id)
        return record is not None and record.terminal_event_id is not None
