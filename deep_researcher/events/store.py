from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from deep_researcher.contracts import EventType, RunEvent, RunStatus


class EventStoreError(RuntimeError):
    """Base class for persistent event-store failures."""


class SequenceConflict(EventStoreError):
    """Raised when an append does not use the next run-local sequence."""


class DuplicateEventConflict(EventStoreError):
    """Raised when an event ID is reused for different canonical content."""


class SpanLifecycleError(EventStoreError):
    """Raised when a span start, use, or terminal transition is invalid."""


class RunTerminalError(EventStoreError):
    """Raised for invalid run start or terminal-event behavior."""


class EventStoreCorruption(EventStoreError):
    """Raised when durable event data fails an integrity invariant."""


@dataclass(frozen=True)
class EventQuery:
    run_id: str
    after_sequence: int = 0
    limit: int = 100
    event_types: tuple[EventType, ...] = ()
    trace_id: str | None = None
    span_id: str | None = None
    actor_id: str | None = None
    task_id: str | None = None
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None

    def __post_init__(self) -> None:
        if self.after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        for value in (self.occurred_from, self.occurred_to):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("query timestamps must be timezone-aware")
        if self.occurred_from and self.occurred_to and self.occurred_to < self.occurred_from:
            raise ValueError("occurred_to cannot precede occurred_from")


@dataclass(frozen=True)
class EventPage:
    items: tuple[RunEvent, ...]
    next_after_sequence: int | None


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    thread_id: str
    trace_id: str
    root_span_id: str
    status: RunStatus
    next_sequence: int
    terminal_event_id: str | None
    created_at: datetime


@dataclass(frozen=True)
class RunQuery:
    thread_id: str | None = None
    statuses: tuple[RunStatus, ...] = ()
    after_created_at: datetime | None = None
    after_run_id: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if self.after_created_at is not None and (
            self.after_created_at.tzinfo is None or self.after_created_at.utcoffset() is None
        ):
            raise ValueError("run cursor timestamp must be timezone-aware")
        if (self.after_created_at is None) != (self.after_run_id is None):
            raise ValueError("run cursor timestamp and run ID must be supplied together")


@dataclass(frozen=True)
class RunPage:
    items: tuple[RunRecord, ...]
    next_cursor: tuple[datetime, str] | None


@dataclass(frozen=True)
class AppendResult:
    event: RunEvent
    inserted: bool


@dataclass(frozen=True)
class PendingExport:
    event: RunEvent
    exporter_name: str
    attempts: int
    last_error: str | None


class EventStore(Protocol):
    def append(self, event: RunEvent, *, export_targets: tuple[str, ...] = ()) -> AppendResult:
        ...

    def get(self, event_id: str) -> RunEvent | None:
        ...

    def list(self, query: EventQuery) -> EventPage:
        ...

    def next_sequence(self, run_id: str) -> int:
        ...

    def get_run(self, run_id: str) -> RunRecord | None:
        ...

    def list_runs(self, query: RunQuery) -> RunPage:
        ...

    def mark_exported(self, event_id: str, exporter_name: str) -> None:
        ...

    def mark_export_failed(self, event_id: str, exporter_name: str, error: str) -> None:
        ...

    def pending_exports(self, *, exporter_name: str | None = None, limit: int = 100) -> tuple[PendingExport, ...]:
        ...

    def integrity_check(self) -> None:
        ...

    def close(self) -> None:
        ...
