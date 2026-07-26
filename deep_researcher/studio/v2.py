from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
from typing import Any, Iterable

from deep_researcher.artifacts import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ArtifactStatus,
    AtomicFact,
    Citation,
    Claim,
    Conflict,
    EventType,
    Evidence,
    Passage,
    Report,
    RunEvent,
    Section,
    Source,
    SourceSnapshot,
    TaskStatus,
)
from deep_researcher.events import EventQuery, EventStore
from deep_researcher.knowledge import (
    KnowledgeQuery,
    KnowledgeRelation,
    KnowledgeRepository,
)
from deep_researcher.knowledge.sqlite_storage import entity_id
from deep_researcher.orchestration import (
    SQLiteSchedulerStore,
    SchedulerEvent,
    SchedulerEventQuery,
    SchedulerEventType,
    SchedulerTaskQuery,
    TaskRecord,
)
from deep_researcher.version_registry import (
    SQLiteVersionRegistryStore,
    VersionLifecycleState,
)

from .v2_models import (
    StudioComponentVersionView,
    StudioErrorRetryNode,
    StudioErrorRetryPage,
    StudioGraphEdge,
    StudioGraphNode,
    StudioGraphPage,
    StudioMetricsView,
    StudioResourceKind,
    StudioResourceLink,
    StudioStateDiff,
    StudioStateDiffPage,
    StudioStateFieldChange,
)


_KNOWLEDGE_ENTITY_TYPES = (
    "Source",
    "SourceSnapshot",
    "Passage",
    "Evidence",
    "AtomicFact",
    "Claim",
    "Citation",
    "Conflict",
    "Section",
    "Report",
)
_SNAPSHOT_CONTENT_KINDS = frozenset(
    {
        ArtifactKind.SOURCE_SNAPSHOT,
        ArtifactKind.CLEANED_CONTENT,
        ArtifactKind.PASSAGE,
    }
)
_MODEL_TERMINALS = {
    EventType.MODEL_COMPLETED,
    EventType.MODEL_FAILED,
}
_TOOL_TERMINALS = {
    EventType.TOOL_COMPLETED,
    EventType.TOOL_FAILED,
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _encode_cursor(kind: str, values: dict[str, Any]) -> str:
    payload = json.dumps(
        {"schema": "StudioCursor@1", "kind": kind, **values},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None, kind: str) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        padded = value + "=" * (-len(value) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded).decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid Studio cursor") from exc
    if payload.get("schema") != "StudioCursor@1" or payload.get("kind") != kind:
        raise ValueError("Studio cursor belongs to another view")
    return payload


def _dedupe_links(
    links: Iterable[StudioResourceLink],
) -> tuple[StudioResourceLink, ...]:
    output: list[StudioResourceLink] = []
    seen: set[tuple[StudioResourceKind, str]] = set()
    for link in links:
        key = (link.kind, link.resource_id)
        if key not in seen:
            output.append(link)
            seen.add(key)
    return tuple(output)


class StudioV2Service:
    """Read-only Studio V2 views over public event, graph, and artifact APIs."""

    def __init__(
        self,
        *,
        event_store: EventStore,
        scheduler_store: SQLiteSchedulerStore,
        knowledge_repository: KnowledgeRepository,
        artifact_store: ArtifactStore,
        version_store: SQLiteVersionRegistryStore,
    ) -> None:
        self.event_store = event_store
        self.scheduler_store = scheduler_store
        self.knowledge_repository = knowledge_repository
        self.artifact_store = artifact_store
        self.version_store = version_store

    @staticmethod
    def _run_event_link(event: RunEvent) -> StudioResourceLink:
        return StudioResourceLink(
            kind=StudioResourceKind.RUN_EVENT,
            resource_id=event.event_id,
            href=f"/api/studio/v2/events/{event.event_id}",
            label=f"{event.event_type.value} #{event.sequence_no}",
        )

    @staticmethod
    def _scheduler_event_link(event: SchedulerEvent) -> StudioResourceLink:
        return StudioResourceLink(
            kind=StudioResourceKind.SCHEDULER_EVENT,
            resource_id=event.event_id,
            href=(
                f"/api/studio/v2/runs/{event.run_id}/scheduler-events/"
                f"{event.sequence_no}"
            ),
            label=f"{event.event_type.value} #{event.sequence_no}",
        )

    def _artifact_link(self, artifact_id: str) -> StudioResourceLink | None:
        envelope = self.artifact_store.get(artifact_id)
        if envelope is None:
            return None
        return StudioResourceLink(
            kind=StudioResourceKind.ARTIFACT,
            resource_id=artifact_id,
            href=f"/api/studio/v2/artifacts/{artifact_id}",
            label=envelope.kind.value,
        )

    @staticmethod
    def _snapshot_link(snapshot_id: str) -> StudioResourceLink:
        return StudioResourceLink(
            kind=StudioResourceKind.SOURCE_SNAPSHOT,
            resource_id=snapshot_id,
            href=f"/api/studio/v2/snapshots/{snapshot_id}",
            label="source snapshot",
        )

    def _all_scheduler_events(self, run_id: str) -> tuple[SchedulerEvent, ...]:
        output: list[SchedulerEvent] = []
        cursor = 0
        while True:
            page = self.scheduler_store.list_event_page(
                SchedulerEventQuery(
                    run_id=run_id,
                    after_sequence=cursor,
                    limit=1000,
                )
            )
            output.extend(page.items)
            if page.next_after_sequence is None:
                break
            cursor = page.next_after_sequence
        return tuple(output)

    def _all_run_events(
        self,
        run_id: str,
        *,
        event_types: tuple[EventType, ...] = (),
    ) -> tuple[RunEvent, ...]:
        output: list[RunEvent] = []
        cursor = 0
        while True:
            page = self.event_store.list(
                EventQuery(
                    run_id=run_id,
                    after_sequence=cursor,
                    event_types=event_types,
                    limit=1000,
                )
            )
            output.extend(page.items)
            if page.next_after_sequence is None:
                break
            cursor = page.next_after_sequence
        return tuple(output)

    def task_graph(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
        statuses: tuple[str, ...] = (),
    ) -> StudioGraphPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        decoded = _decode_cursor(cursor, "task_graph")
        status_values = tuple(
            TaskStatus(value) for value in statuses
        )
        page = self.scheduler_store.list_task_page(
            SchedulerTaskQuery(
                run_id=run_id,
                after_task_id=decoded.get("after_task_id"),
                statuses=status_values,
                limit=limit,
            )
        )
        events = self._all_scheduler_events(run_id)
        events_by_task: dict[str, list[SchedulerEvent]] = {}
        split_events_by_child: dict[str, SchedulerEvent] = {}
        for event in events:
            referenced = set()
            if event.task_id:
                referenced.add(event.task_id)
            for raw in event.payload.get("records", ()):
                try:
                    referenced.add(TaskRecord.model_validate(raw, strict=False).task_id)
                except (TypeError, ValueError):
                    continue
            for task_id in referenced:
                events_by_task.setdefault(task_id, []).append(event)
            if event.event_type == SchedulerEventType.TASK_SPLIT:
                for child_id in event.payload.get("child_task_ids", ()):
                    split_events_by_child[str(child_id)] = event

        nodes: list[StudioGraphNode] = []
        edges: list[StudioGraphEdge] = []
        page_ids = {record.task_id for record in page.items}
        frontier: set[str] = set()
        for record in page.items:
            envelope = record.envelope
            task_events = events_by_task.get(record.task_id, [])
            artifacts = (
                *envelope.input_artifact_ids,
                *record.output_artifact_ids,
            )
            links = _dedupe_links(
                [
                    *(self._scheduler_event_link(item) for item in task_events),
                    *(
                        link
                        for artifact_id in artifacts
                        if (link := self._artifact_link(artifact_id)) is not None
                    ),
                ]
            )
            if not links:
                raise RuntimeError(
                    f"task node has no scheduler event or artifact: {record.task_id}"
                )
            operation_history = [
                {
                    "event_type": item.event_type.value,
                    "sequence_no": item.sequence_no,
                    "event_id": item.event_id,
                }
                for item in task_events
                if item.event_type
                in {
                    SchedulerEventType.TASK_SPLIT,
                    SchedulerEventType.TASK_MERGED,
                    SchedulerEventType.TASK_PRUNED,
                    SchedulerEventType.TASK_FAILED,
                    SchedulerEventType.TASK_RETRIED,
                }
            ]
            nodes.append(
                StudioGraphNode(
                    node_id=record.task_id,
                    node_type=envelope.kind.value,
                    label=envelope.title,
                    status=envelope.status.value,
                    data={
                        "goal": envelope.goal,
                        "priority": envelope.priority,
                        "attempt": envelope.attempt,
                        "max_attempts": envelope.max_attempts,
                        "revision": record.revision,
                        "assigned_actor_id": envelope.assigned_actor_id,
                        "error_ref": record.error_ref,
                        "merged_into_task_id": record.merged_into_task_id,
                        "defer_reason": record.defer_reason,
                        "budget": envelope.budget.model_dump(mode="json"),
                        "budget_usage": record.budget_usage.model_dump(mode="json"),
                        "operations": operation_history,
                    },
                    links=links,
                )
            )
            parent_id = envelope.parent_task_id
            if parent_id:
                if parent_id not in page_ids:
                    frontier.add(parent_id)
                event = split_events_by_child.get(record.task_id)
                edge_links = (
                    (self._scheduler_event_link(event),)
                    if event is not None
                    else (links[0],)
                )
                edge_type = "split" if event is not None else "parent"
                edges.append(
                    StudioGraphEdge(
                        edge_id=_stable_id(
                            "edge",
                            run_id,
                            edge_type,
                            parent_id,
                            record.task_id,
                        ),
                        edge_type=edge_type,
                        source_node_id=parent_id,
                        target_node_id=record.task_id,
                        links=edge_links,
                    )
                )
            for dependency_id in envelope.dependency_task_ids:
                if dependency_id not in page_ids:
                    frontier.add(dependency_id)
                edges.append(
                    StudioGraphEdge(
                        edge_id=_stable_id(
                            "edge",
                            run_id,
                            "dependency",
                            dependency_id,
                            record.task_id,
                        ),
                        edge_type="dependency",
                        source_node_id=dependency_id,
                        target_node_id=record.task_id,
                        links=(links[0],),
                    )
                )
            if record.merged_into_task_id:
                target = record.merged_into_task_id
                if target not in page_ids:
                    frontier.add(target)
                merge_event = next(
                    (
                        item
                        for item in reversed(task_events)
                        if item.event_type == SchedulerEventType.TASK_MERGED
                    ),
                    None,
                )
                edges.append(
                    StudioGraphEdge(
                        edge_id=_stable_id(
                            "edge",
                            run_id,
                            "merge",
                            record.task_id,
                            target,
                        ),
                        edge_type="merge",
                        source_node_id=record.task_id,
                        target_node_id=target,
                        links=(
                            (self._scheduler_event_link(merge_event),)
                            if merge_event is not None
                            else (links[0],)
                        ),
                    )
                )
        next_cursor = (
            _encode_cursor(
                "task_graph",
                {"after_task_id": page.next_after_task_id},
            )
            if page.next_after_task_id is not None
            else None
        )
        return StudioGraphPage(
            graph_kind="task_dag",
            run_id=run_id,
            nodes=tuple(nodes),
            edges=tuple(edges),
            frontier_node_ids=tuple(sorted(frontier - page_ids)),
            next_cursor=next_cursor,
        )

    def _fallback_event(self, run_id: str, task_id: str | None) -> RunEvent | None:
        page = self.event_store.list(
            EventQuery(
                run_id=run_id,
                task_id=task_id,
                limit=1,
            )
        )
        if page.items:
            return page.items[0]
        if task_id is not None:
            root = self.event_store.list(EventQuery(run_id=run_id, limit=1))
            return root.items[0] if root.items else None
        return None

    def _entity_links(self, entity: Any) -> tuple[StudioResourceLink, ...]:
        run_id = str(entity.provenance.run_id)
        provenance = entity.provenance
        links: list[StudioResourceLink] = []
        if provenance.causation_event_id:
            event = self.event_store.get(provenance.causation_event_id)
            if event is not None:
                links.append(self._run_event_link(event))
        artifact_ids = list(provenance.source_artifact_ids)
        for name in ("artifact_id", "text_artifact_id", "content_artifact_id"):
            artifact_id = getattr(entity, name, None)
            if artifact_id:
                artifact_ids.append(artifact_id)
        for artifact_id in artifact_ids:
            link = self._artifact_link(str(artifact_id))
            if link is not None:
                links.append(link)
        if isinstance(entity, SourceSnapshot):
            links.append(self._snapshot_link(entity.snapshot_id))
        if not links:
            fallback = self._fallback_event(run_id, provenance.task_id)
            if fallback is not None:
                links.append(self._run_event_link(fallback))
        if not links:
            raise RuntimeError(
                f"knowledge node has no event or artifact provenance: {entity_id(entity)}"
            )
        return _dedupe_links(links)

    @staticmethod
    def _entity_label(entity: Any) -> str:
        for name in ("title", "statement", "summary", "heading", "canonical_url"):
            value = getattr(entity, name, None)
            if value:
                return str(value)[:1000]
        return entity_id(entity)

    @staticmethod
    def _entity_status(entity: Any) -> str | None:
        status = getattr(entity, "status", None)
        return status.value if hasattr(status, "value") else str(status) if status else None

    @staticmethod
    def _entity_data(entity: Any, revision: int) -> dict[str, Any]:
        data: dict[str, Any] = {"revision": revision}
        fields_by_type = {
            Source: (
                "canonical_url",
                "source_type",
                "source_level",
                "publisher",
                "authority_score",
                "published_at",
            ),
            SourceSnapshot: (
                "source_id",
                "source_level",
                "source_version",
                "final_url",
                "media_type",
                "http_status",
                "fetched_at",
                "capture_method",
                "artifact_id",
            ),
            Passage: (
                "snapshot_id",
                "ordinal",
                "locator",
                "char_start",
                "char_end",
                "language",
                "text_artifact_id",
            ),
            Evidence: (
                "passage_ids",
                "relation",
                "summary",
                "confidence",
                "relevance",
                "source_quality",
                "verification_id",
            ),
            AtomicFact: (
                "statement",
                "evidence_ids",
                "confidence",
                "verification_id",
            ),
            Claim: (
                "statement",
                "fact_ids",
                "evidence_ids",
                "confidence",
                "importance",
                "high_impact",
                "support_score",
                "verification_id",
            ),
            Citation: (
                "claim_id",
                "evidence_id",
                "passage_id",
                "snapshot_id",
                "source_id",
            ),
            Conflict: (
                "claim_ids",
                "fact_ids",
                "severity",
                "high_impact",
                "resolution",
                "resolution_kind",
                "resolution_evidence_ids",
            ),
            Section: (
                "report_id",
                "parent_section_id",
                "title",
                "order",
                "claim_ids",
                "citation_ids",
                "coverage_status",
                "coverage_score",
                "citation_score",
                "unsupported_claim_ids",
                "conflicted_claim_ids",
            ),
            Report: (
                "title",
                "version",
                "section_ids",
                "content_artifact_id",
                "quality_scores",
            ),
        }
        for model, names in fields_by_type.items():
            if isinstance(entity, model):
                for name in names:
                    value = getattr(entity, name, None)
                    if hasattr(value, "value"):
                        value = value.value
                    elif isinstance(value, tuple):
                        value = list(value)
                    elif isinstance(value, datetime):
                        value = value.isoformat()
                    data[name] = value
                break
        return data

    def _entity_node(self, saved: Any) -> StudioGraphNode:
        entity = saved.entity
        return StudioGraphNode(
            node_id=entity_id(entity),
            node_type=type(entity).__name__,
            label=self._entity_label(entity),
            status=self._entity_status(entity),
            data=self._entity_data(entity, saved.revision),
            links=self._entity_links(entity),
        )

    @staticmethod
    def _entity_relationships(entity: Any) -> tuple[tuple[str, str], ...]:
        pairs: list[tuple[str, str]] = []

        def add(edge_type: str, values: str | tuple[str, ...] | None) -> None:
            if isinstance(values, str):
                pairs.append((edge_type, values))
            elif values:
                pairs.extend((edge_type, value) for value in values)

        if isinstance(entity, SourceSnapshot):
            add("snapshot_source", entity.source_id)
        elif isinstance(entity, Passage):
            add("passage_snapshot", entity.snapshot_id)
        elif isinstance(entity, Evidence):
            add("evidence_passage", entity.passage_ids)
        elif isinstance(entity, AtomicFact):
            add("fact_evidence", entity.evidence_ids)
        elif isinstance(entity, Claim):
            add("claim_fact", entity.fact_ids)
            add("claim_evidence", entity.evidence_ids)
        elif isinstance(entity, Citation):
            add("citation_claim", entity.claim_id)
            add("citation_evidence", entity.evidence_id)
            add("citation_passage", entity.passage_id)
            add("citation_snapshot", entity.snapshot_id)
            add("citation_source", entity.source_id)
        elif isinstance(entity, Conflict):
            add("conflict_claim", entity.claim_ids)
            add("conflict_fact", entity.fact_ids)
            add("conflict_resolution_evidence", entity.resolution_evidence_ids)
        elif isinstance(entity, Section):
            add("section_report", entity.report_id)
            add("section_parent", entity.parent_section_id)
            add("section_claim", entity.claim_ids)
            add("section_citation", entity.citation_ids)
        elif isinstance(entity, Report):
            add("report_section", entity.section_ids)
        return tuple(pairs)

    def evidence_graph(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 100,
        entity_types: tuple[str, ...] = _KNOWLEDGE_ENTITY_TYPES,
        statuses: tuple[str, ...] = (),
    ) -> StudioGraphPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        unknown = set(entity_types) - set(_KNOWLEDGE_ENTITY_TYPES)
        if unknown:
            raise ValueError(f"unsupported evidence graph entity types: {sorted(unknown)}")
        decoded = _decode_cursor(cursor, "evidence_graph")
        created = (
            datetime.fromisoformat(decoded["after_created_at"])
            if decoded.get("after_created_at")
            else None
        )
        page = self.knowledge_repository.storage.list_latest(
            KnowledgeQuery(
                run_id=run_id,
                entity_types=entity_types,
                statuses=statuses,
                after_created_at=created,
                after_entity_id=decoded.get("after_entity_id"),
                limit=limit,
            )
        )
        nodes = tuple(self._entity_node(saved) for saved in page.items)
        page_ids = {item.node_id for item in nodes}
        edges: list[StudioGraphEdge] = []
        frontier: set[str] = set()
        for saved, node in zip(page.items, nodes, strict=True):
            for edge_type, target in self._entity_relationships(saved.entity):
                if target not in page_ids:
                    frontier.add(target)
                edges.append(
                    StudioGraphEdge(
                        edge_id=_stable_id(
                            "edge",
                            run_id,
                            edge_type,
                            node.node_id,
                            target,
                        ),
                        edge_type=edge_type,
                        source_node_id=node.node_id,
                        target_node_id=target,
                        links=(node.links[0],),
                    )
                )
        next_cursor = (
            _encode_cursor(
                "evidence_graph",
                {
                    "after_created_at": page.next_cursor[0].isoformat(),
                    "after_entity_id": page.next_cursor[1],
                },
            )
            if page.next_cursor is not None
            else None
        )
        return StudioGraphPage(
            graph_kind="evidence_graph",
            run_id=run_id,
            nodes=nodes,
            edges=tuple(edges),
            frontier_node_ids=tuple(sorted(frontier - page_ids)),
            next_cursor=next_cursor,
        )

    @staticmethod
    def _task_state(record: TaskRecord) -> dict[str, Any]:
        envelope = record.envelope
        return {
            "status": envelope.status.value,
            "attempt": envelope.attempt,
            "revision": record.revision,
            "priority": envelope.priority,
            "assigned_actor_id": envelope.assigned_actor_id,
            "result_id": record.result_id,
            "error_ref": record.error_ref,
            "merged_into_task_id": record.merged_into_task_id,
            "output_artifact_ids": list(record.output_artifact_ids),
        }

    @staticmethod
    def _budget_state(record: TaskRecord) -> dict[str, Any]:
        return record.budget_usage.model_dump(mode="json")

    def scheduler_state_diffs(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> StudioStateDiffPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        events = self._all_scheduler_events(run_id)
        task_state: dict[str, dict[str, Any]] = {}
        budget_state: dict[str, dict[str, Any]] = {}
        run_status: str | None = None
        output: list[StudioStateDiff] = []
        has_more = False
        for event in events:
            changes: list[StudioStateFieldChange] = []
            control = event.payload.get("control")
            if isinstance(control, dict):
                after_status = str(control.get("status"))
                if after_status != run_status:
                    changes.append(
                        StudioStateFieldChange(
                            entity_kind="run",
                            entity_id=run_id,
                            field="status",
                            before=run_status,
                            after=after_status,
                        )
                    )
                    run_status = after_status
            for raw in event.payload.get("records", ()):
                record = TaskRecord.model_validate(raw, strict=False)
                task_id = record.task_id
                after_task = self._task_state(record)
                before_task = task_state.get(task_id, {})
                for field, after in after_task.items():
                    before = before_task.get(field)
                    if before != after:
                        changes.append(
                            StudioStateFieldChange(
                                entity_kind="task",
                                entity_id=task_id,
                                field=field,
                                before=before,
                                after=after,
                            )
                        )
                task_state[task_id] = after_task
                after_budget = self._budget_state(record)
                before_budget = budget_state.get(task_id, {})
                for field, after in after_budget.items():
                    before = before_budget.get(field, 0)
                    if before != after:
                        changes.append(
                            StudioStateFieldChange(
                                entity_kind="budget",
                                entity_id=task_id,
                                field=field,
                                before=before,
                                after=after,
                            )
                        )
                budget_state[task_id] = after_budget
            if event.sequence_no <= after_sequence or not changes:
                continue
            if len(output) >= limit:
                has_more = True
                break
            artifact_links = [
                link
                for raw in event.payload.get("records", ())
                for artifact_id in (
                    *TaskRecord.model_validate(raw, strict=False).envelope.input_artifact_ids,
                    *TaskRecord.model_validate(raw, strict=False).output_artifact_ids,
                )
                if (link := self._artifact_link(artifact_id)) is not None
            ]
            output.append(
                StudioStateDiff(
                    domain="scheduler",
                    sequence_no=event.sequence_no,
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    occurred_at=event.occurred_at.isoformat(),
                    task_id=event.task_id,
                    changes=tuple(changes),
                    links=_dedupe_links(
                        [self._scheduler_event_link(event), *artifact_links]
                    ),
                )
            )
        next_sequence = output[-1].sequence_no if has_more and output else None
        rebuilt = output[-1].sequence_no if output else min(after_sequence, len(events))
        return StudioStateDiffPage(
            run_id=run_id,
            domain="scheduler",
            items=tuple(output),
            next_after_sequence=next_sequence,
            rebuilt_through_sequence=rebuilt,
        )

    @staticmethod
    def _evidence_state_changes(event: RunEvent) -> tuple[dict[str, Any], ...]:
        payload = event.payload
        explicit = payload.get("state_changes")
        if isinstance(explicit, list):
            return tuple(item for item in explicit if isinstance(item, dict))
        subject_id = payload.get("subject_id")
        change = payload.get("change")
        inferred: list[dict[str, Any]] = []
        if change == "claim_evidence_verified" and subject_id:
            inferred.append(
                {
                    "entity_type": "Claim",
                    "entity_id": subject_id,
                    "field": "status",
                    "after": payload.get("claim_status"),
                }
            )
            inferred.extend(
                {
                    "entity_type": "Evidence",
                    "entity_id": item,
                    "field": "status",
                    "after": "verified",
                }
                for item in payload.get("verified_evidence_ids", ())
            )
            inferred.extend(
                {
                    "entity_type": "Citation",
                    "entity_id": item,
                    "field": "status",
                    "after": "verified",
                }
                for item in payload.get("verified_citation_ids", ())
            )
        elif change == "conflict_resolved" and subject_id:
            inferred.append(
                {
                    "entity_type": "Conflict",
                    "entity_id": subject_id,
                    "field": "status",
                    "after": "resolved",
                }
            )
            inferred.extend(
                {
                    "entity_type": "Claim",
                    "entity_id": item,
                    "field": "status",
                    "after": "contested",
                }
                for item in payload.get("invalidated_claim_ids", ())
            )
        elif change == "conflict_accepted_unresolved" and subject_id:
            inferred.append(
                {
                    "entity_type": "Conflict",
                    "entity_id": subject_id,
                    "field": "status",
                    "after": "accepted_unresolved",
                }
            )
        elif change == "section_coverage_assessed" and subject_id:
            inferred.append(
                {
                    "entity_type": "Section",
                    "entity_id": subject_id,
                    "field": "coverage_status",
                    "after": payload.get("coverage_status"),
                }
            )
        return tuple(inferred)

    def evidence_state_diffs(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> StudioStateDiffPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        events = self._all_run_events(
            run_id,
            event_types=(EventType.EVIDENCE_CHANGED,),
        )
        state: dict[tuple[str, str, str], Any] = {}
        output: list[StudioStateDiff] = []
        has_more = False
        rebuilt = 0
        for event in events:
            changes: list[StudioStateFieldChange] = []
            for raw in self._evidence_state_changes(event):
                entity_kind = str(raw.get("entity_type") or "Evidence")
                entity_id_value = str(
                    raw.get("entity_id")
                    or event.payload.get("subject_id")
                    or event.task_id
                    or run_id
                )
                field = str(raw.get("field") or "status")
                key = (entity_kind, entity_id_value, field)
                before = raw.get("before", state.get(key))
                after = raw.get("after")
                if before != after:
                    changes.append(
                        StudioStateFieldChange(
                            entity_kind=entity_kind,
                            entity_id=entity_id_value,
                            field=field,
                            before=before,
                            after=after,
                        )
                    )
                state[key] = after
            rebuilt = event.sequence_no
            if event.sequence_no <= after_sequence or not changes:
                continue
            if len(output) >= limit:
                has_more = True
                break
            links = [
                self._run_event_link(event),
                *(
                    link
                    for artifact_id in (
                        *event.input_artifact_ids,
                        *event.output_artifact_ids,
                    )
                    if (link := self._artifact_link(artifact_id)) is not None
                ),
            ]
            output.append(
                StudioStateDiff(
                    domain="evidence",
                    sequence_no=event.sequence_no,
                    event_id=event.event_id,
                    event_type=event.event_type.value,
                    occurred_at=event.occurred_at.isoformat(),
                    task_id=event.task_id,
                    changes=tuple(changes),
                    links=_dedupe_links(links),
                )
            )
        return StudioStateDiffPage(
            run_id=run_id,
            domain="evidence",
            items=tuple(output),
            next_after_sequence=(
                output[-1].sequence_no if has_more and output else None
            ),
            rebuilt_through_sequence=rebuilt,
        )

    def error_retry_chain(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> StudioErrorRetryPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        events = self._all_run_events(run_id)
        last_failure: dict[tuple[str | None, str], str] = {}
        output: list[StudioErrorRetryNode] = []
        has_more = False
        for event in events:
            relevant = event.error is not None or event.event_type == EventType.RETRY_SCHEDULED
            key = (event.task_id, event.span_id)
            retry_of = None
            if event.event_type == EventType.RETRY_SCHEDULED:
                retry_of = event.causation_event_id or last_failure.get(key)
            if event.error is not None:
                retry_of = (
                    event.causation_event_id
                    if event.attempt > 1
                    else None
                )
                last_failure[key] = event.event_id
            if not relevant or event.sequence_no <= after_sequence:
                continue
            if len(output) >= limit:
                has_more = True
                break
            links: list[StudioResourceLink] = [self._run_event_link(event)]
            for artifact_id in (
                *event.input_artifact_ids,
                *event.output_artifact_ids,
            ):
                link = self._artifact_link(artifact_id)
                if link is not None:
                    links.append(link)
            if event.error and event.error.detail_artifact_id:
                link = self._artifact_link(event.error.detail_artifact_id)
                if link is not None:
                    links.append(link)
            output.append(
                StudioErrorRetryNode(
                    node_id=_stable_id(
                        "error_chain",
                        run_id,
                        event.event_id,
                    ),
                    event_id=event.event_id,
                    sequence_no=event.sequence_no,
                    event_type=event.event_type.value,
                    status=event.status.value,
                    task_id=event.task_id,
                    span_id=event.span_id,
                    attempt=event.attempt,
                    retry_of_event_id=retry_of,
                    error=(
                        event.error.model_dump(mode="json")
                        if event.error is not None
                        else None
                    ),
                    links=_dedupe_links(links),
                )
            )
        return StudioErrorRetryPage(
            run_id=run_id,
            items=tuple(output),
            next_after_sequence=(
                output[-1].sequence_no if has_more and output else None
            ),
        )

    def scheduler_error_retry_chain(
        self,
        run_id: str,
        *,
        after_sequence: int = 0,
        limit: int = 100,
    ) -> StudioErrorRetryPage:
        if after_sequence < 0:
            raise ValueError("after_sequence must be non-negative")
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        events = self._all_scheduler_events(run_id)
        last_failure: dict[str, str] = {}
        output: list[StudioErrorRetryNode] = []
        has_more = False
        relevant_types = {
            SchedulerEventType.TASK_FAILED,
            SchedulerEventType.TASK_RETRIED,
            SchedulerEventType.LEASE_RECOVERED,
        }
        for event in events:
            if event.event_type not in relevant_types:
                continue
            records = tuple(
                TaskRecord.model_validate(item, strict=False)
                for item in event.payload.get("records", ())
            )
            record = next(
                (
                    item
                    for item in records
                    if event.task_id is not None
                    and item.task_id == event.task_id
                ),
                records[0] if records else None,
            )
            task_id = event.task_id or (
                record.task_id if record is not None else None
            )
            if task_id is None:
                continue
            retry_of = (
                last_failure.get(task_id)
                if event.event_type
                in {
                    SchedulerEventType.TASK_RETRIED,
                    SchedulerEventType.LEASE_RECOVERED,
                }
                else None
            )
            if event.event_type == SchedulerEventType.TASK_FAILED:
                last_failure[task_id] = event.event_id
            if event.sequence_no <= after_sequence:
                continue
            if len(output) >= limit:
                has_more = True
                break
            links: list[StudioResourceLink] = [
                self._scheduler_event_link(event)
            ]
            if record is not None:
                for artifact_id in (
                    *record.envelope.input_artifact_ids,
                    *record.output_artifact_ids,
                ):
                    link = self._artifact_link(artifact_id)
                    if link is not None:
                        links.append(link)
            output.append(
                StudioErrorRetryNode(
                    node_id=_stable_id(
                        "error_chain",
                        run_id,
                        event.event_id,
                    ),
                    event_id=event.event_id,
                    sequence_no=event.sequence_no,
                    event_type=event.event_type.value,
                    status=(
                        record.envelope.status.value
                        if record is not None
                        else "recovered"
                    ),
                    task_id=task_id,
                    span_id=_stable_id(
                        "span_scheduler",
                        run_id,
                        task_id,
                    ),
                    attempt=(
                        max(1, record.envelope.attempt)
                        if record is not None
                        else 1
                    ),
                    retry_of_event_id=retry_of,
                    error=(
                        {
                            "error_ref": record.error_ref,
                            "retryable": (
                                event.event_type
                                != SchedulerEventType.TASK_FAILED
                                or record.envelope.attempt
                                < record.envelope.max_attempts
                            ),
                        }
                        if record is not None and record.error_ref is not None
                        else None
                    ),
                    links=_dedupe_links(links),
                )
            )
        return StudioErrorRetryPage(
            run_id=run_id,
            items=tuple(output),
            next_after_sequence=(
                output[-1].sequence_no if has_more and output else None
            ),
        )

    def conflict_navigation(
        self,
        run_id: str,
        *,
        cursor: str | None = None,
        limit: int = 50,
        statuses: tuple[str, ...] = (),
    ) -> StudioGraphPage:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        decoded = _decode_cursor(cursor, "conflict_navigation")
        created = (
            datetime.fromisoformat(decoded["after_created_at"])
            if decoded.get("after_created_at")
            else None
        )
        page = self.knowledge_repository.storage.list_latest(
            KnowledgeQuery(
                run_id=run_id,
                entity_types=("Conflict",),
                statuses=statuses,
                after_created_at=created,
                after_entity_id=decoded.get("after_entity_id"),
                limit=limit,
            )
        )
        saved_by_id: dict[str, Any] = {}
        conflict_ids: set[str] = set()
        for saved in page.items:
            conflict = saved.entity
            conflict_id = entity_id(conflict)
            saved_by_id[conflict_id] = saved
            conflict_ids.add(conflict_id)
            for _, target_id in self._entity_relationships(conflict):
                target = self.knowledge_repository.storage.get_latest(target_id)
                if target is not None:
                    saved_by_id[target_id] = target
        nodes = tuple(
            self._entity_node(saved)
            for _, saved in sorted(saved_by_id.items())
        )
        links_by_id = {node.node_id: node.links for node in nodes}
        edges: list[StudioGraphEdge] = []
        for saved in page.items:
            source = entity_id(saved.entity)
            for edge_type, target in self._entity_relationships(saved.entity):
                edges.append(
                    StudioGraphEdge(
                        edge_id=_stable_id(
                            "edge",
                            run_id,
                            edge_type,
                            source,
                            target,
                        ),
                        edge_type=edge_type,
                        source_node_id=source,
                        target_node_id=target,
                        links=(links_by_id[source][0],),
                    )
                )
        next_cursor = (
            _encode_cursor(
                "conflict_navigation",
                {
                    "after_created_at": page.next_cursor[0].isoformat(),
                    "after_entity_id": page.next_cursor[1],
                },
            )
            if page.next_cursor is not None
            else None
        )
        return StudioGraphPage(
            graph_kind="conflict_navigation",
            run_id=run_id,
            nodes=nodes,
            edges=tuple(edges),
            next_cursor=next_cursor,
        )

    def component_versions(
        self,
        run_id: str,
    ) -> tuple[StudioComponentVersionView, ...]:
        run = self.event_store.get_run(run_id)
        latest: RunEvent | None = None
        if run is not None and run.next_sequence > 1:
            page = self.event_store.list(
                EventQuery(
                    run_id=run_id,
                    after_sequence=max(0, run.next_sequence - 2),
                    limit=1,
                )
            )
            latest = page.items[0] if page.items else None
        output: list[StudioComponentVersionView] = []
        if latest is not None:
            versions = latest.component_versions
            refs = [
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
            ]
            for ref in refs:
                if ref is None:
                    continue
                links: list[StudioResourceLink] = [
                    self._run_event_link(latest)
                ]
                if ref.artifact_id:
                    artifact = self._artifact_link(ref.artifact_id)
                    if artifact is not None:
                        links.append(artifact)
                output.append(
                    StudioComponentVersionView(
                        scope="run_pinned",
                        kind=ref.kind.value,
                        name=ref.name,
                        version_id=ref.version_id,
                        semantic_version=ref.version,
                        state="pinned",
                        content_hash=ref.content_hash,
                        links=_dedupe_links(links),
                    )
                )
        for record in self.version_store.records():
            if record.state != VersionLifecycleState.PROMOTED:
                continue
            ref = record.manifest.version_ref
            links = [
                link
                for artifact_id in (
                    record.manifest.manifest_artifact_id,
                    (
                        record.transitions[-1].transition_artifact_id
                        if record.transitions
                        else None
                    ),
                )
                if artifact_id is not None
                and (link := self._artifact_link(artifact_id)) is not None
            ]
            if not links:
                continue
            output.append(
                StudioComponentVersionView(
                    scope="registry_active",
                    kind=ref.kind.value,
                    name=ref.name,
                    version_id=ref.version_id,
                    semantic_version=ref.version,
                    state=record.state.value,
                    content_hash=record.manifest.content_hash,
                    links=_dedupe_links(links),
                )
            )
        return tuple(output)

    @staticmethod
    def _empty_usage() -> dict[str, int | float]:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "model_calls": 0,
            "tool_calls": 0,
            "retries": 0,
            "errors": 0,
        }

    @staticmethod
    def _add_metric(
        target: dict[str, int | float],
        event: RunEvent,
    ) -> None:
        is_model = event.event_type in _MODEL_TERMINALS
        is_tool = event.event_type in _TOOL_TERMINALS
        if is_model:
            target["input_tokens"] += event.usage.input_tokens
            target["output_tokens"] += event.usage.output_tokens
            target["total_tokens"] += event.usage.total_tokens
            target["cost_usd"] += event.usage.cost_usd
            target["model_calls"] += 1
        if is_tool:
            target["tool_calls"] += 1
        if is_model or is_tool:
            target["latency_ms"] += event.latency_ms
        if event.event_type == EventType.RETRY_SCHEDULED:
            target["retries"] += 1
        if event.error is not None:
            target["errors"] += 1

    def metrics(self, run_id: str) -> StudioMetricsView:
        events = self._all_run_events(run_id)
        if not events and self.scheduler_store.get_control(run_id) is None:
            raise KeyError(run_id)
        totals = self._empty_usage()
        by_actor: dict[str, dict[str, int | float]] = {}
        by_task: dict[str, dict[str, Any]] = {}
        for event in events:
            actor = by_actor.setdefault(event.actor_id, self._empty_usage())
            self._add_metric(totals, event)
            self._add_metric(actor, event)
            if event.task_id:
                task = by_task.setdefault(
                    event.task_id,
                    {"event_usage": self._empty_usage()},
                )
                self._add_metric(task["event_usage"], event)

        budget_health: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            page = self.scheduler_store.list_task_page(
                SchedulerTaskQuery(
                    run_id=run_id,
                    after_task_id=cursor,
                    limit=1000,
                )
            )
            for record in page.items:
                budget = record.envelope.budget
                usage = record.budget_usage
                exceeded = [
                    item.value
                    for item in budget.exceeded_dimensions(usage)
                ]
                ratios: dict[str, float] = {}
                limits = {
                    "tokens": budget.max_tokens,
                    "cost": budget.max_cost_usd,
                    "wall_time": budget.max_wall_time_seconds,
                    "model_calls": budget.max_model_calls,
                    "tool_calls": budget.max_tool_calls,
                    "search_calls": budget.max_search_calls,
                    "retries": budget.max_retries,
                    "errors": budget.max_errors,
                }
                actuals = {
                    "tokens": usage.total_tokens,
                    "cost": usage.cost_usd,
                    "wall_time": usage.wall_time_seconds,
                    "model_calls": usage.model_calls,
                    "tool_calls": usage.tool_calls,
                    "search_calls": usage.search_calls,
                    "retries": usage.retries,
                    "errors": usage.errors,
                }
                for name, maximum in limits.items():
                    if maximum is not None:
                        ratios[name] = actuals[name] / maximum
                item = {
                    "task_id": record.task_id,
                    "status": record.envelope.status.value,
                    "limits": budget.model_dump(mode="json"),
                    "usage": usage.model_dump(mode="json"),
                    "ratios": ratios,
                    "exceeded_dimensions": exceeded,
                }
                budget_health.append(item)
                by_task.setdefault(
                    record.task_id,
                    {"event_usage": self._empty_usage()},
                )["scheduler_budget"] = item
            if page.next_after_task_id is None:
                break
            cursor = page.next_after_task_id
        source_links = (
            _dedupe_links(
                (
                    self._run_event_link(events[0]),
                    self._run_event_link(events[-1]),
                )
            )
            if events
            else ()
        )
        if not source_links:
            scheduler_events = self._all_scheduler_events(run_id)
            if scheduler_events:
                source_links = (
                    self._scheduler_event_link(scheduler_events[-1]),
                )
        return StudioMetricsView(
            run_id=run_id,
            totals=totals,
            by_actor=by_actor,
            by_task=by_task,
            budget_health=tuple(budget_health),
            component_versions=self.component_versions(run_id),
            source_event_links=source_links,
        )

    def snapshot_navigation(self, snapshot_id: str) -> dict[str, Any]:
        snapshot = self.knowledge_repository.snapshots.require(snapshot_id)
        source = self.knowledge_repository.sources.require(snapshot.source_id)
        passages = self.knowledge_repository.related(
            snapshot_id,
            KnowledgeRelation.PASSAGE_SNAPSHOT,
            incoming=True,
        )
        artifact = (
            self.artifact_store.get(snapshot.artifact_id)
            if snapshot.artifact_id
            else None
        )
        return {
            "snapshot": snapshot.model_dump(mode="json"),
            "source": source.model_dump(mode="json"),
            "passages": [
                item.model_dump(mode="json")
                for item in passages
                if isinstance(item, Passage)
            ],
            "artifact": (
                artifact.model_dump(mode="json")
                if artifact is not None
                else None
            ),
            "content_url": (
                f"/api/studio/v2/artifacts/{artifact.artifact_id}/content"
                if artifact is not None
                and artifact.kind in _SNAPSHOT_CONTENT_KINDS
                else None
            ),
            "links": [
                item.model_dump(mode="json")
                for item in self._entity_links(snapshot)
            ],
        }

    def artifact_metadata(self, artifact_id: str) -> dict[str, Any]:
        envelope = self.artifact_store.get(artifact_id)
        if envelope is None:
            raise KeyError(artifact_id)
        return envelope.model_dump(mode="json")

    def artifact_content(
        self,
        artifact_id: str,
    ) -> tuple[bytes, str, str]:
        envelope = self.artifact_store.get(artifact_id)
        if envelope is None:
            raise KeyError(artifact_id)
        if (
            envelope.kind not in _SNAPSHOT_CONTENT_KINDS
            or envelope.status != ArtifactStatus.AVAILABLE
        ):
            raise PermissionError(
                "Studio V2 exposes content only for available source snapshots "
                "and passage artifacts"
            )
        return (
            self.artifact_store.read_bytes(artifact_id),
            envelope.media_type,
            envelope.content_hash,
        )

    def run_event(self, event_id: str) -> dict[str, Any]:
        event = self.event_store.get(event_id)
        if event is None:
            raise KeyError(event_id)
        return event.model_dump(mode="json")

    def scheduler_event(
        self,
        run_id: str,
        sequence_no: int,
    ) -> dict[str, Any]:
        if sequence_no < 1:
            raise ValueError("sequence_no must be positive")
        page = self.scheduler_store.list_event_page(
            SchedulerEventQuery(
                run_id=run_id,
                after_sequence=sequence_no - 1,
                limit=1,
            )
        )
        if not page.items or page.items[0].sequence_no != sequence_no:
            raise KeyError(f"{run_id}/{sequence_no}")
        return page.items[0].model_dump(mode="json")
