from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pydantic import Field, field_validator

from deep_researcher.contracts import (
    BudgetUsage,
    ComponentVersionSet,
    ContractModel,
    EventType,
    RunEvent,
    RunStatus,
    SpanKind,
    utc_now,
)
from deep_researcher.contracts._base import validate_identifier
from deep_researcher.events import EventRecorder, SequenceConflict


class EvidenceDomainEvent(ContractModel):
    event_id: str
    event_type: EventType
    run_id: str
    task_id: str | None = None
    subject_id: str
    actor_id: str
    input_artifact_ids: tuple[str, ...] = ()
    output_artifact_ids: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    payload: dict[str, Any] = Field(default_factory=dict)
    occurred_at: datetime = Field(default_factory=utc_now)

    @field_validator("event_id", "run_id", "task_id", "subject_id", "actor_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("input_artifact_ids", "output_artifact_ids")
    @classmethod
    def _artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return tuple(dict.fromkeys(value))

    @field_validator("event_type")
    @classmethod
    def _event_type(cls, value: EventType) -> EventType:
        allowed = {EventType.EVIDENCE_CHANGED, EventType.VERIFICATION_COMPLETED}
        if value not in allowed:
            raise ValueError(f"unsupported evidence-domain event type: {value.value}")
        return value


class EvidenceEventSink(Protocol):
    def emit(self, event: EvidenceDomainEvent) -> None: ...


class RecordingEvidenceEventSink:
    """Test/offline sink that retains every event rather than discarding it."""

    def __init__(self) -> None:
        self.events: list[EvidenceDomainEvent] = []

    def emit(self, event: EvidenceDomainEvent) -> None:
        if any(item.event_id == event.event_id for item in self.events):
            return
        self.events.append(event)


@dataclass(frozen=True)
class EvidenceTraceContext:
    thread_id: str
    trace_id: str
    span_id: str
    correlation_id: str
    component_versions: ComponentVersionSet
    parent_span_id: str | None = None
    producer_id: str = "runtime_evidence_engine"
    causation_event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for value in (
            self.thread_id,
            self.trace_id,
            self.span_id,
            self.correlation_id,
            self.producer_id,
            self.parent_span_id,
            self.causation_event_id,
        ):
            if value is not None:
                validate_identifier(value)


class EventRecorderEvidenceSink:
    """Publishes durable evidence feedback into the Background001 RunEvent store."""

    def __init__(
        self,
        recorder: EventRecorder,
        context: EvidenceTraceContext,
        *,
        max_sequence_retries: int = 20,
    ) -> None:
        if max_sequence_retries < 1:
            raise ValueError("max_sequence_retries must be positive")
        self.recorder = recorder
        self.context = context
        self.max_sequence_retries = max_sequence_retries

    def emit(self, event: EvidenceDomainEvent) -> None:
        if self.recorder.store.get(event.event_id) is not None:
            return
        for _ in range(self.max_sequence_retries):
            run_event = RunEvent(
                event_id=event.event_id,
                sequence_no=self.recorder.store.next_sequence(event.run_id),
                event_type=event.event_type,
                status=RunStatus.RUNNING,
                trace_id=self.context.trace_id,
                span_id=self.context.span_id,
                parent_span_id=self.context.parent_span_id,
                span_kind=SpanKind.VERIFICATION,
                correlation_id=self.context.correlation_id,
                causation_event_id=self.context.causation_event_id,
                run_id=event.run_id,
                thread_id=self.context.thread_id,
                task_id=event.task_id,
                actor_id=event.actor_id,
                producer_id=self.context.producer_id,
                input_artifact_ids=event.input_artifact_ids,
                output_artifact_ids=event.output_artifact_ids,
                usage=event.usage,
                component_versions=self.context.component_versions,
                occurred_at=event.occurred_at,
                payload={
                    "subject_id": event.subject_id,
                    **self.context.metadata,
                    **event.payload,
                },
            )
            try:
                self.recorder.record(run_event)
            except SequenceConflict:
                continue
            return
        raise SequenceConflict(
            "could not allocate evidence event sequence after bounded retries"
        )
