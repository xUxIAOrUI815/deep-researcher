from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator

from deep_researcher.contracts import ContractModel
from deep_researcher.contracts._base import validate_identifier


class StudioResourceKind(str, Enum):
    RUN_EVENT = "run_event"
    SCHEDULER_EVENT = "scheduler_event"
    ARTIFACT = "artifact"
    SOURCE_SNAPSHOT = "source_snapshot"


class StudioResourceLink(ContractModel):
    kind: StudioResourceKind
    resource_id: str
    href: str
    label: str

    @field_validator("resource_id")
    @classmethod
    def _resource_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioGraphNode(ContractModel):
    node_id: str
    node_type: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=1000)
    status: str | None = Field(default=None, max_length=120)
    data: dict[str, Any] = Field(default_factory=dict)
    links: tuple[StudioResourceLink, ...]

    @field_validator("node_id")
    @classmethod
    def _node_id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("links")
    @classmethod
    def _resolvable(cls, value: tuple[StudioResourceLink, ...]) -> tuple[StudioResourceLink, ...]:
        if not value:
            raise ValueError("Studio graph nodes require an event or artifact link")
        identities = {(item.kind, item.resource_id) for item in value}
        if len(identities) != len(value):
            raise ValueError("Studio graph node links must be unique")
        return value


class StudioGraphEdge(ContractModel):
    edge_id: str
    edge_type: str = Field(min_length=1, max_length=120)
    source_node_id: str
    target_node_id: str
    directed: bool = True
    data: dict[str, Any] = Field(default_factory=dict)
    links: tuple[StudioResourceLink, ...]

    @field_validator("edge_id", "source_node_id", "target_node_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("links")
    @classmethod
    def _edge_resolvable(cls, value: tuple[StudioResourceLink, ...]) -> tuple[StudioResourceLink, ...]:
        if not value:
            raise ValueError("Studio graph edges require an event or artifact link")
        return value


class StudioGraphPage(ContractModel):
    graph_kind: str
    run_id: str
    nodes: tuple[StudioGraphNode, ...]
    edges: tuple[StudioGraphEdge, ...]
    frontier_node_ids: tuple[str, ...] = ()
    next_cursor: str | None = None

    @field_validator("run_id")
    @classmethod
    def _run_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioStateFieldChange(ContractModel):
    entity_kind: str = Field(min_length=1, max_length=120)
    entity_id: str
    field: str = Field(min_length=1, max_length=120)
    before: Any = None
    after: Any = None

    @field_validator("entity_id")
    @classmethod
    def _entity_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioStateDiff(ContractModel):
    domain: str
    sequence_no: int = Field(ge=1)
    event_id: str
    event_type: str
    occurred_at: str
    task_id: str | None = None
    changes: tuple[StudioStateFieldChange, ...]
    links: tuple[StudioResourceLink, ...]

    @field_validator("event_id", "task_id")
    @classmethod
    def _event_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class StudioStateDiffPage(ContractModel):
    run_id: str
    domain: str
    items: tuple[StudioStateDiff, ...]
    next_after_sequence: int | None = Field(default=None, ge=1)
    rebuilt_through_sequence: int = Field(ge=0)

    @field_validator("run_id")
    @classmethod
    def _diff_run_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioErrorRetryNode(ContractModel):
    node_id: str
    event_id: str
    sequence_no: int = Field(ge=1)
    event_type: str
    status: str
    task_id: str | None = None
    span_id: str
    attempt: int = Field(ge=1)
    retry_of_event_id: str | None = None
    error: dict[str, Any] | None = None
    links: tuple[StudioResourceLink, ...]

    @field_validator("node_id", "event_id", "task_id", "span_id", "retry_of_event_id")
    @classmethod
    def _error_ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None


class StudioErrorRetryPage(ContractModel):
    run_id: str
    items: tuple[StudioErrorRetryNode, ...]
    next_after_sequence: int | None = Field(default=None, ge=1)

    @field_validator("run_id")
    @classmethod
    def _error_run_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioComponentVersionView(ContractModel):
    scope: str
    kind: str
    name: str
    version_id: str
    semantic_version: str
    state: str
    content_hash: str | None = None
    links: tuple[StudioResourceLink, ...]

    @field_validator("version_id")
    @classmethod
    def _version_id(cls, value: str) -> str:
        return validate_identifier(value)


class StudioMetricsView(ContractModel):
    run_id: str
    totals: dict[str, int | float]
    by_actor: dict[str, dict[str, int | float]]
    by_task: dict[str, dict[str, Any]]
    budget_health: tuple[dict[str, Any], ...]
    component_versions: tuple[StudioComponentVersionView, ...]
    source_event_links: tuple[StudioResourceLink, ...]

    @field_validator("run_id")
    @classmethod
    def _metrics_run_id(cls, value: str) -> str:
        return validate_identifier(value)
