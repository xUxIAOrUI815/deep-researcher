from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import uuid
from typing import Any, Optional, Protocol

from core.run_context import RunContext


class EventLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class EventType(str, Enum):
    RUN_STARTED = "run.started"
    RUN_COMPLETED = "run.completed"
    RUN_FAILED = "run.failed"
    NODE_STARTED = "node.started"
    NODE_COMPLETED = "node.completed"
    NODE_FAILED = "node.failed"
    AGENT_STARTED = "agent.started"
    AGENT_COMPLETED = "agent.completed"
    AGENT_FAILED = "agent.failed"
    TASK_CREATED = "task.created"
    TASK_UPDATED = "task.updated"
    TASK_STATUS_CHANGED = "task.status_changed"
    TASK_MERGED = "task.merged"
    TASK_PRUNED = "task.pruned"
    TASK_DEFERRED = "task.deferred"
    EVIDENCE_CREATED = "evidence.created"
    EVIDENCE_LINKED = "evidence.linked"
    CONFLICT_DETECTED = "conflict.detected"
    REPORT_FINALIZED = "report.finalized"
    QUERY_GENERATED = "query.generated"
    QUERY_REJECTED = "query.rejected"
    QUERY_DROPPED = "query.dropped"
    SOURCE_ACCEPTED = "source.accepted"
    SOURCE_REJECTED = "source.rejected"
    EXPLORATION_STOPPED = "exploration.stopped"
    DISTILL_STARTED = "distill.started"
    PASSAGE_CLEANED = "passage.cleaned"
    CLAIM_EXTRACTED = "claim.extracted"
    FACT_EXTRACTED = "fact.extracted"
    EVIDENCE_PACK_CREATED = "evidence_pack.created"
    COMPRESSION_COMPLETED = "compression.completed"
    WRITER_STARTED = "writer.started"
    SECTION_GENERATED = "section.generated"
    WRITER_COMPLETED = "writer.completed"
    CITATION_RESOLVED = "writer.citation_resolved"
    CITATION_UNRESOLVED = "writer.citation_unresolved"


@dataclass
class ObservabilityEvent:
    event_type: EventType
    level: EventLevel = EventLevel.INFO
    event_id: str = field(default_factory=lambda: f"evt_{uuid.uuid4().hex}")
    timestamp: datetime = field(default_factory=datetime.now)
    research_id: str = ""
    thread_id: str = ""
    run_id: str = ""
    trace_id: str = ""
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    node_name: Optional[str] = None
    agent_name: Optional[str] = None
    phase: str = ""
    task_id: Optional[str] = None
    source_id: Optional[str] = None
    fact_id: Optional[str] = None
    claim_id: Optional[str] = None
    evidence_id: Optional[str] = None
    conflict_id: Optional[str] = None
    section_id: Optional[str] = None
    message: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    error: Optional[dict[str, Any]] = None
    state_version: Optional[int] = None

    @classmethod
    def from_context(
        cls,
        context: RunContext,
        event_type: EventType,
        *,
        level: EventLevel = EventLevel.INFO,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
        **ids: Any,
    ) -> "ObservabilityEvent":
        return cls(
            event_type=event_type,
            level=level,
            research_id=context.research_id,
            thread_id=context.thread_id,
            run_id=context.run_id,
            trace_id=context.trace_id,
            message=message,
            payload=payload or {},
            **ids,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["event_type"] = self.event_type.value
        data["level"] = self.level.value
        data["timestamp"] = self.timestamp.isoformat()
        data["node"] = self.node_name
        data["agent"] = self.agent_name
        if not data.get("phase"):
            data["phase"] = _phase_for_event(self.event_type)
        return data


class Observer(Protocol):
    def emit(self, event: ObservabilityEvent) -> None:
        ...

    def record_run_event(
        self,
        context: RunContext,
        event_type: EventType,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    def record_node_event(
        self,
        context: RunContext,
        event_type: EventType,
        node_name: str,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    def record_task_event(
        self,
        context: RunContext,
        event_type: EventType,
        task_id: str,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        ...

    def record_evidence_event(
        self,
        context: RunContext,
        event_type: EventType,
        *,
        evidence_id: Optional[str] = None,
        fact_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        source_id: Optional[str] = None,
        section_id: Optional[str] = None,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        ...


class NoopObserver:
    """Default observer that keeps call sites stable without external dependencies."""

    def emit(self, event: ObservabilityEvent) -> None:
        return None

    def record_run_event(
        self,
        context: RunContext,
        event_type: EventType,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.emit(
            ObservabilityEvent.from_context(
                context, event_type, message=message, payload=payload
            )
        )

    def record_node_event(
        self,
        context: RunContext,
        event_type: EventType,
        node_name: str,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.emit(
            ObservabilityEvent.from_context(
                context,
                event_type,
                node_name=node_name,
                message=message,
                payload=payload,
            )
        )

    def record_task_event(
        self,
        context: RunContext,
        event_type: EventType,
        task_id: str,
        *,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.emit(
            ObservabilityEvent.from_context(
                context,
                event_type,
                task_id=task_id,
                message=message,
                payload=payload,
            )
        )

    def record_evidence_event(
        self,
        context: RunContext,
        event_type: EventType,
        *,
        evidence_id: Optional[str] = None,
        fact_id: Optional[str] = None,
        claim_id: Optional[str] = None,
        source_id: Optional[str] = None,
        section_id: Optional[str] = None,
        message: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        self.emit(
            ObservabilityEvent.from_context(
                context,
                event_type,
                evidence_id=evidence_id,
                fact_id=fact_id,
                claim_id=claim_id,
                source_id=source_id,
                section_id=section_id,
                message=message,
                payload=payload,
            )
        )


def _phase_for_event(event_type: EventType) -> str:
    value = event_type.value
    if value.startswith("run."):
        return "run"
    if value.startswith("node."):
        return "node"
    if value.startswith("task."):
        return "planning"
    if value.startswith("query.") or value.startswith("source.") or value.startswith("exploration."):
        return "research"
    if value.startswith("distill.") or value.startswith("passage.") or value.startswith("claim.") or value.startswith("fact.") or value.startswith("evidence"):
        return "distillation"
    if value.startswith("writer.") or value.startswith("section.") or value.startswith("report."):
        return "writing"
    if value.startswith("conflict.") or value.startswith("compression."):
        return "distillation"
    return "unknown"


class JsonlObserver(NoopObserver):
    """Observer that appends one JSON event per line under event_dir."""

    def __init__(self, event_dir: str | Path = ".console_runtime/events") -> None:
        self.event_dir = Path(event_dir)
        self.event_dir.mkdir(parents=True, exist_ok=True)

    def emit(self, event: ObservabilityEvent) -> None:
        research_id = event.research_id or "unknown"
        self.event_dir.mkdir(parents=True, exist_ok=True)
        path = self.event_dir / f"{research_id}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_dict(), ensure_ascii=False, default=str) + "\n")


class CompositeObserver(NoopObserver):
    """Fan out events to multiple observers without coupling call sites."""

    def __init__(self, observers: list[Observer]) -> None:
        self.observers = list(observers)

    def emit(self, event: ObservabilityEvent) -> None:
        for observer in self.observers:
            observer.emit(event)


_observer: Observer = NoopObserver()


def set_observer(observer: Observer) -> None:
    global _observer
    _observer = observer


def get_observer() -> Observer:
    return _observer
