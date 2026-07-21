from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re


class StudioProjectionError(RuntimeError):
    pass


class ProjectionGap(StudioProjectionError):
    pass


class ProjectionConflict(StudioProjectionError):
    pass


class ProjectionCorruption(StudioProjectionError):
    pass


def normalize_projection_id(prefix: str, value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value)).strip("-.") or "unknown"
    if re.fullmatch(r"[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*", cleaned):
        return cleaned
    return f"{prefix}_{cleaned}"


@dataclass(frozen=True)
class TimelineQuery:
    run_id: str
    after_sequence: int = 0
    limit: int = 100
    event_types: tuple[str, ...] = ()
    span_kinds: tuple[str, ...] = ()
    statuses: tuple[str, ...] = ()
    actor_id: str | None = None
    task_id: str | None = None
    text: str | None = None
    error_only: bool = False
    has_artifacts: bool | None = None
    occurred_from: datetime | None = None
    occurred_to: datetime | None = None

    def __post_init__(self) -> None:
        if self.after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        for value in (self.occurred_from, self.occurred_to):
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError("timeline timestamps must be timezone-aware")
        if self.occurred_from and self.occurred_to and self.occurred_to < self.occurred_from:
            raise ValueError("occurred_to cannot precede occurred_from")


@dataclass(frozen=True)
class TimelinePage:
    items: tuple[dict, ...]
    next_after_sequence: int | None


@dataclass(frozen=True)
class ProjectionPage:
    items: tuple[dict, ...]
    next_cursor: tuple[str, str] | None
