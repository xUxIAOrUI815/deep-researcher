from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
import hashlib
from typing import Any, Protocol

from deep_researcher.contracts import BudgetDimension, BudgetUsage, TaskEnvelope, TaskStatus, utc_now

from .models import (
    ApprovalRecord,
    ApprovalStatus,
    RecoveryReport,
    RunControl,
    RunControlStatus,
    SchedulerEvent,
    SchedulerEventType,
    SchedulerSnapshot,
    TaskCompletion,
    TaskEdit,
    TaskLease,
    TaskRecord,
)
from .store import SQLiteSchedulerStore, SchedulerMutationConflict


class SchedulerError(RuntimeError):
    pass


class SchedulerStateError(SchedulerError):
    pass


class SchedulerLeaseError(SchedulerError):
    pass


class Scheduler(Protocol):
    async def create_run(self, run_id: str, *, max_concurrency: int, actor_id: str, mutation_id: str) -> RunControl: ...
    async def submit(self, task: TaskEnvelope, *, actor_id: str, mutation_id: str, available_at: datetime | None = None) -> TaskRecord: ...
    async def split(self, parent_task_id: str, children: tuple[TaskEnvelope, ...], *, actor_id: str, mutation_id: str) -> tuple[TaskRecord, ...]: ...
    async def merge(self, source_task_id: str, target_task_id: str, *, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def defer(self, task_id: str, *, until: datetime, reason: str, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def prune(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def cancel_task(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def claim(self, run_id: str, *, worker_id: str, limit: int, lease_seconds: float, mutation_id: str) -> tuple[TaskLease, ...]: ...
    async def heartbeat(self, task_id: str, *, worker_id: str, lease_seconds: float, mutation_id: str) -> TaskLease: ...
    async def update_usage(self, task_id: str, usage: BudgetUsage, *, worker_id: str, mutation_id: str) -> TaskRecord: ...
    async def complete(self, task_id: str, completion: TaskCompletion, *, worker_id: str, mutation_id: str) -> TaskRecord: ...
    async def fail(self, task_id: str, *, error_ref: str, worker_id: str, mutation_id: str, usage: BudgetUsage | None = None) -> TaskRecord: ...
    async def retry(self, task_id: str, *, actor_id: str, mutation_id: str, available_at: datetime | None = None) -> TaskRecord: ...
    async def pause_task(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord: ...
    async def resume_task(self, task_id: str, *, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def request_approval(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord: ...
    async def approve(self, task_id: str, *, actor_id: str, note: str, mutation_id: str) -> TaskRecord: ...
    async def reject(self, task_id: str, *, actor_id: str, note: str, error_ref: str, mutation_id: str) -> TaskRecord: ...
    async def edit(self, task_id: str, edit: TaskEdit, *, actor_id: str, mutation_id: str) -> TaskRecord: ...
    async def pause_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl: ...
    async def resume_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl: ...
    async def cancel_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl: ...
    async def complete_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl: ...
    async def snapshot(self, run_id: str) -> SchedulerSnapshot: ...
    async def recover(self, run_id: str, *, actor_id: str, mutation_id: str) -> RecoveryReport: ...
    async def rebuild_projection(self, run_id: str) -> SchedulerSnapshot: ...
    async def close(self) -> None: ...


class NativeEventSourcedScheduler:
    """Durable priority/DAG scheduler whose projection is rebuilt from events."""

    def __init__(self, store: SQLiteSchedulerStore, *, clock=utc_now) -> None:
        self.store = store
        self.clock = clock
        self._lock = asyncio.Lock()

    async def create_run(self, run_id: str, *, max_concurrency: int = 4, actor_id: str, mutation_id: str) -> RunControl:
        payload = {"run_id": run_id, "max_concurrency": max_concurrency}
        fingerprint = self.store.fingerprint("create_run", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                replay = self.store.mutation_event(mutation_id, fingerprint, connection=connection)
                if replay:
                    return self._control(run_id, connection)
                if self.store.get_control(run_id, connection=connection) is not None:
                    raise SchedulerStateError(f"scheduler run already exists: {run_id}")
                now = self.clock()
                control = RunControl(
                    run_id=run_id,
                    max_concurrency=max_concurrency,
                    projection_revision=1,
                    created_at=now,
                    updated_at=now,
                )
                self.store.write_control(control, connection=connection)
                self.store.append_event(
                    self._event(
                        control,
                        SchedulerEventType.RUN_CREATED,
                        actor_id,
                        mutation_id,
                        fingerprint,
                        records=(),
                    ),
                    connection=connection,
                )
                return control

    async def submit(
        self,
        task: TaskEnvelope,
        *,
        actor_id: str,
        mutation_id: str,
        available_at: datetime | None = None,
    ) -> TaskRecord:
        payload = {"task": task.model_dump(mode="json"), "available_at": (available_at or task.created_at).isoformat()}
        fingerprint = self.store.fingerprint("submit", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task.task_id, connection)
                control = self._mutable_control(task.run_id, connection)
                if task.status != TaskStatus.PENDING:
                    raise SchedulerStateError("submitted task must be pending")
                if self.store.get_task(task.task_id, connection=connection) is not None:
                    raise SchedulerStateError(f"task already exists: {task.task_id}")
                self._validate_references(task, connection)
                now = self.clock()
                record = TaskRecord(envelope=task, available_at=available_at or now, updated_at=now)
                records = [record]
                all_records = list(self.store.list_tasks(task.run_id, connection=connection)) + records
                self._validate_dag(all_records)
                records = self._promote_ready(records, all_records, now)
                control = self._advance(control, now)
                self._persist_event(connection, control, SchedulerEventType.TASK_CREATED, actor_id, mutation_id, fingerprint, records, task.task_id)
                return next(item for item in records if item.task_id == task.task_id)

    async def split(
        self,
        parent_task_id: str,
        children: tuple[TaskEnvelope, ...],
        *,
        actor_id: str,
        mutation_id: str,
    ) -> tuple[TaskRecord, ...]:
        payload = {"parent_task_id": parent_task_id, "children": [item.model_dump(mode="json") for item in children]}
        fingerprint = self.store.fingerprint("split", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                replay = self.store.mutation_event(mutation_id, fingerprint, connection=connection)
                if replay:
                    return tuple(self._task(task.task_id, connection) for task in children)
                parent = self._task(parent_task_id, connection)
                control = self._mutable_control(parent.run_id, connection)
                if not children:
                    raise SchedulerStateError("split requires at least one child")
                if parent.envelope.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}:
                    raise SchedulerStateError("terminal task cannot be split")
                child_ids = {item.task_id for item in children}
                if len(child_ids) != len(children):
                    raise SchedulerStateError("split child IDs must be unique")
                existing = list(self.store.list_tasks(parent.run_id, connection=connection))
                existing_ids = {item.task_id for item in existing}
                if child_ids & existing_ids:
                    raise SchedulerStateError("split child already exists")
                for child in children:
                    if child.run_id != parent.run_id or child.parent_task_id != parent_task_id:
                        raise SchedulerStateError("split children must share the run and parent")
                    unknown = set(child.dependency_task_ids) - existing_ids - child_ids
                    if unknown:
                        raise SchedulerStateError(f"unknown child dependencies: {sorted(unknown)}")
                self._validate_dag([*existing, *(TaskRecord(envelope=item) for item in children)])
                now = self.clock()
                ordered = self._toposort_children(children)
                records: list[TaskRecord] = []
                projected = existing[:]
                for child in ordered:
                    record = TaskRecord(envelope=child, available_at=now, updated_at=now)
                    projected.append(record)
                    promoted = self._promote_ready([record], projected, now)[0]
                    records.append(promoted)
                    projected[-1] = promoted
                control = self._advance(control, now)
                self._persist_event(
                    connection,
                    control,
                    SchedulerEventType.TASK_SPLIT,
                    actor_id,
                    mutation_id,
                    fingerprint,
                    records,
                    parent_task_id,
                    extra={"child_task_ids": [item.task_id for item in records]},
                )
                by_id = {item.task_id: item for item in records}
                return tuple(by_id[item.task_id] for item in children)

    async def merge(
        self,
        source_task_id: str,
        target_task_id: str,
        *,
        actor_id: str,
        mutation_id: str,
    ) -> TaskRecord:
        payload = {"source_task_id": source_task_id, "target_task_id": target_task_id}
        fingerprint = self.store.fingerprint("merge", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(source_task_id, connection)
                source = self._task(source_task_id, connection)
                target = self._task(target_task_id, connection)
                if source.run_id != target.run_id or source_task_id == target_task_id:
                    raise SchedulerStateError("merge requires distinct tasks in one run")
                control = self._mutable_control(source.run_id, connection)
                if source.envelope.status not in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.PAUSED, TaskStatus.DEFERRED}:
                    raise SchedulerStateError("running or terminal task cannot be merged")
                now = self.clock()
                merged = self._transition(source, TaskStatus.MERGED, now, merged_into_task_id=target_task_id)
                records = [merged]
                all_records = list(self.store.list_tasks(source.run_id, connection=connection))
                updated_all: list[TaskRecord] = []
                for record in all_records:
                    if record.task_id == source_task_id:
                        updated_all.append(merged)
                        continue
                    if source_task_id not in record.envelope.dependency_task_ids:
                        updated_all.append(record)
                        continue
                    dependencies = tuple(dict.fromkeys(target_task_id if item == source_task_id else item for item in record.envelope.dependency_task_ids))
                    updated = self._edit_envelope(record, {"dependency_task_ids": dependencies}, now)
                    records.append(updated)
                    updated_all.append(updated)
                self._validate_dag(updated_all)
                control = self._advance(control, now)
                self._persist_event(connection, control, SchedulerEventType.TASK_MERGED, actor_id, mutation_id, fingerprint, records, source_task_id, extra={"target_task_id": target_task_id})
                return merged

    async def defer(self, task_id: str, *, until: datetime, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        if until.tzinfo is None or until.utcoffset() is None:
            raise ValueError("defer timestamp must be timezone-aware")
        return await self._simple_transition(
            "defer",
            SchedulerEventType.TASK_DEFERRED,
            task_id,
            TaskStatus.DEFERRED,
            actor_id,
            mutation_id,
            payload={"until": until.isoformat(), "reason": reason},
            updates={"available_at": until, "defer_reason": reason},
        )

    async def prune(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        return await self._simple_transition(
            "prune",
            SchedulerEventType.TASK_PRUNED,
            task_id,
            TaskStatus.PRUNED,
            actor_id,
            mutation_id,
            payload={"reason": reason},
            updates={"error_ref": self._error_ref("pruned", mutation_id)},
        )

    async def cancel_task(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        return await self._simple_transition(
            "cancel_task",
            SchedulerEventType.TASK_CANCELLED,
            task_id,
            TaskStatus.CANCELLED,
            actor_id,
            mutation_id,
            payload={"reason": reason},
            updates={"error_ref": self._error_ref("cancelled", mutation_id)},
        )

    async def claim(
        self,
        run_id: str,
        *,
        worker_id: str,
        limit: int = 1,
        lease_seconds: float = 30.0,
        mutation_id: str,
    ) -> tuple[TaskLease, ...]:
        if limit < 1 or lease_seconds <= 0:
            raise ValueError("claim limit and lease duration must be positive")
        payload = {"run_id": run_id, "worker_id": worker_id, "limit": limit, "lease_seconds": lease_seconds}
        fingerprint = self.store.fingerprint("claim", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                replay = self.store.mutation_event(mutation_id, fingerprint, connection=connection)
                if replay:
                    return tuple(TaskLease.model_validate(item, strict=False) for item in replay.payload.get("leases", []))
                control = self._control(run_id, connection)
                now = self.clock()
                records, recovered, failed = self._refresh_records(run_id, connection, now)
                projected = {item.task_id: item for item in self.store.list_tasks(run_id, connection=connection)}
                projected.update({item.task_id: item for item in records})
                leases: list[TaskLease] = []
                if control.status == RunControlStatus.ACTIVE:
                    active_count = sum(
                        1
                        for record in projected.values()
                        if record.envelope.status == TaskStatus.RUNNING and record.lease_expires_at and record.lease_expires_at > now
                    )
                    slots = max(0, control.max_concurrency - active_count)
                    candidates = [
                        item
                        for item in projected.values()
                        if item.envelope.status == TaskStatus.READY
                        and item.available_at <= now
                        and (item.envelope.assigned_actor_id is None or item.envelope.assigned_actor_id == worker_id)
                    ]
                    candidates.sort(
                        key=lambda item: (
                            -item.envelope.priority,
                            item.envelope.deadline or datetime.max.replace(tzinfo=now.tzinfo),
                            item.envelope.created_at,
                            item.task_id,
                        )
                    )
                    for candidate in candidates[: min(limit, slots)]:
                        running = self._transition(
                            candidate,
                            TaskStatus.RUNNING,
                            now,
                            lease_owner=worker_id,
                            lease_expires_at=now + timedelta(seconds=lease_seconds),
                            last_heartbeat_at=now,
                            pause_reason=None,
                            paused_by_run=False,
                        )
                        records.append(running)
                        projected[running.task_id] = running
                        leases.append(
                            TaskLease(
                                task=running.envelope,
                                worker_id=worker_id,
                                lease_expires_at=running.lease_expires_at,
                                projection_revision=control.projection_revision + 1,
                            )
                        )
                control = self._advance(control, now)
                self._persist_event(
                    connection,
                    control,
                    SchedulerEventType.TASK_CLAIMED,
                    worker_id,
                    mutation_id,
                    fingerprint,
                    self._dedupe_records(records),
                    extra={
                        "leases": [item.model_dump(mode="json") for item in leases],
                        "recovered_task_ids": recovered,
                        "failed_task_ids": failed,
                    },
                )
                return tuple(leases)

    async def heartbeat(self, task_id: str, *, worker_id: str, lease_seconds: float, mutation_id: str) -> TaskLease:
        if lease_seconds <= 0:
            raise ValueError("lease duration must be positive")
        payload = {"task_id": task_id, "worker_id": worker_id, "lease_seconds": lease_seconds}
        fingerprint = self.store.fingerprint("heartbeat", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                replay = self.store.mutation_event(mutation_id, fingerprint, connection=connection)
                if replay:
                    return TaskLease.model_validate(replay.payload["lease"], strict=False)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                updated = record.model_copy(
                    update={
                        "lease_expires_at": now + timedelta(seconds=lease_seconds),
                        "last_heartbeat_at": now,
                        "revision": record.revision + 1,
                        "updated_at": now,
                    }
                )
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                lease = TaskLease(task=updated.envelope, worker_id=worker_id, lease_expires_at=updated.lease_expires_at, projection_revision=control.projection_revision)
                self._persist_event(connection, control, SchedulerEventType.TASK_HEARTBEAT, worker_id, mutation_id, fingerprint, [updated], task_id, extra={"lease": lease.model_dump(mode="json")})
                return lease

    async def complete(
        self,
        task_id: str,
        completion: TaskCompletion,
        *,
        worker_id: str,
        mutation_id: str,
    ) -> TaskRecord:
        payload = {"task_id": task_id, "worker_id": worker_id, "completion": completion.model_dump(mode="json")}
        fingerprint = self.store.fingerprint("complete", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                usage = self._combine_usage(record.budget_usage, completion.usage)
                budget_candidate = record.model_copy(update={"budget_usage": usage})
                if self._scheduling_budget_exhausted(budget_candidate, now=now):
                    completed = self._transition(
                        record,
                        TaskStatus.FAILED,
                        now,
                        error_ref="error_budget_exhausted",
                        budget_usage=usage,
                    )
                    changed = [completed]
                    event_type = SchedulerEventType.TASK_FAILED
                else:
                    completed = self._transition(
                        record,
                        TaskStatus.COMPLETED,
                        now,
                        result_id=completion.result_id,
                        output_artifact_ids=completion.output_artifact_ids,
                        budget_usage=usage,
                    )
                    all_records = [
                        item if item.task_id != task_id else completed
                        for item in self.store.list_tasks(record.run_id, connection=connection)
                    ]
                    promoted = self._promote_ready(
                        [item for item in all_records if item.envelope.status == TaskStatus.PENDING],
                        all_records,
                        now,
                    )
                    changed = [completed, *(item for item in promoted if item.envelope.status == TaskStatus.READY)]
                    event_type = SchedulerEventType.TASK_COMPLETED
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                self._persist_event(
                    connection,
                    control,
                    event_type,
                    worker_id,
                    mutation_id,
                    fingerprint,
                    self._dedupe_records(changed),
                    task_id,
                )
                return completed

    async def update_usage(
        self,
        task_id: str,
        usage: BudgetUsage,
        *,
        worker_id: str,
        mutation_id: str,
    ) -> TaskRecord:
        payload = {"task_id": task_id, "worker_id": worker_id, "usage": usage.model_dump(mode="json")}
        fingerprint = self.store.fingerprint("update_usage", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                combined = self._combine_usage(record.budget_usage, usage)
                updated = record.model_copy(
                    update={"budget_usage": combined, "revision": record.revision + 1, "updated_at": now}
                )
                if self._scheduling_budget_exhausted(updated, now=now):
                    updated = self._transition(
                        updated,
                        TaskStatus.FAILED,
                        now,
                        error_ref="error_budget_exhausted",
                        budget_usage=combined,
                    )
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                self._persist_event(connection, control, SchedulerEventType.BUDGET_UPDATED, worker_id, mutation_id, fingerprint, [updated], task_id)
                return updated

    async def fail(
        self,
        task_id: str,
        *,
        error_ref: str,
        worker_id: str,
        mutation_id: str,
        usage: BudgetUsage | None = None,
    ) -> TaskRecord:
        payload = {"task_id": task_id, "worker_id": worker_id, "error_ref": error_ref, "usage": (usage or BudgetUsage()).model_dump(mode="json")}
        fingerprint = self.store.fingerprint("fail", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                failed = self._transition(
                    record,
                    TaskStatus.FAILED,
                    now,
                    error_ref=error_ref,
                    budget_usage=self._combine_usage(record.budget_usage, usage or BudgetUsage()).plus(errors=1),
                )
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                self._persist_event(connection, control, SchedulerEventType.TASK_FAILED, worker_id, mutation_id, fingerprint, [failed], task_id)
                return failed

    async def retry(
        self,
        task_id: str,
        *,
        actor_id: str,
        mutation_id: str,
        available_at: datetime | None = None,
    ) -> TaskRecord:
        payload = {"task_id": task_id, "available_at": available_at.isoformat() if available_at else None}
        fingerprint = self.store.fingerprint("retry", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                control = self._mutable_control(record.run_id, connection)
                if record.envelope.status != TaskStatus.FAILED:
                    raise SchedulerStateError("only failed tasks may be retried")
                if record.envelope.attempt >= record.envelope.max_attempts:
                    raise SchedulerStateError("task attempt limit is exhausted")
                now = self.clock()
                pending = self._transition(
                    record,
                    TaskStatus.PENDING,
                    now,
                    available_at=available_at or now,
                    error_ref=None,
                    budget_usage=record.budget_usage.plus(retries=1),
                )
                all_records = [item if item.task_id != task_id else pending for item in self.store.list_tasks(record.run_id, connection=connection)]
                pending = self._promote_ready([pending], all_records, now)[0]
                control = self._advance(control, now)
                self._persist_event(connection, control, SchedulerEventType.TASK_RETRIED, actor_id, mutation_id, fingerprint, [pending], task_id)
                return pending

    async def pause_task(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord:
        payload = {"task_id": task_id, "worker_id": worker_id, "reason": reason}
        fingerprint = self.store.fingerprint("pause_task", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                paused = self._transition(record, TaskStatus.PAUSED, now, pause_reason=reason, paused_by_run=False)
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                self._persist_event(connection, control, SchedulerEventType.TASK_PAUSED, worker_id, mutation_id, fingerprint, [paused], task_id)
                return paused

    async def resume_task(self, task_id: str, *, actor_id: str, mutation_id: str) -> TaskRecord:
        return await self._simple_transition(
            "resume_task",
            SchedulerEventType.TASK_RESUMED,
            task_id,
            TaskStatus.READY,
            actor_id,
            mutation_id,
            payload={},
            updates={"pause_reason": None, "paused_by_run": False},
            allowed_run_statuses={RunControlStatus.ACTIVE},
        )

    async def request_approval(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord:
        payload = {"task_id": task_id, "worker_id": worker_id, "reason": reason}
        fingerprint = self.store.fingerprint("request_approval", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                now = self.clock()
                self._require_lease(record, worker_id, now)
                approval = ApprovalRecord(
                    approval_id=f"approval_{hashlib.sha256(mutation_id.encode()).hexdigest()[:24]}",
                    task_id=task_id,
                    requested_by=worker_id,
                    reason=reason,
                    requested_at=now,
                )
                waiting = self._transition(record, TaskStatus.WAITING_APPROVAL, now, approval=approval)
                control = self._advance(self._mutable_control(record.run_id, connection), now)
                self._persist_event(connection, control, SchedulerEventType.APPROVAL_REQUESTED, worker_id, mutation_id, fingerprint, [waiting], task_id)
                return waiting

    async def approve(self, task_id: str, *, actor_id: str, note: str, mutation_id: str) -> TaskRecord:
        return await self._resolve_approval(task_id, approved=True, actor_id=actor_id, note=note, error_ref=None, mutation_id=mutation_id)

    async def reject(self, task_id: str, *, actor_id: str, note: str, error_ref: str, mutation_id: str) -> TaskRecord:
        return await self._resolve_approval(task_id, approved=False, actor_id=actor_id, note=note, error_ref=error_ref, mutation_id=mutation_id)

    async def edit(self, task_id: str, edit: TaskEdit, *, actor_id: str, mutation_id: str) -> TaskRecord:
        payload = {"task_id": task_id, "edit": edit.model_dump(mode="json")}
        fingerprint = self.store.fingerprint("edit", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                control = self._mutable_control(record.run_id, connection)
                if record.envelope.status in {TaskStatus.RUNNING, TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}:
                    raise SchedulerStateError("running or terminal task cannot be edited")
                now = self.clock()
                updates = {
                    key: value
                    for key, value in edit.model_dump(exclude={"schema_version", "clear_deadline", "clear_assigned_actor"}).items()
                    if key in edit.model_fields_set
                }
                if edit.clear_deadline:
                    updates["deadline"] = None
                if edit.clear_assigned_actor:
                    updates["assigned_actor_id"] = None
                updated = self._edit_envelope(record, updates, now)
                self._validate_references(updated.envelope, connection, allow_self=True)
                all_records = [item if item.task_id != task_id else updated for item in self.store.list_tasks(record.run_id, connection=connection)]
                self._validate_dag(all_records)
                if updated.envelope.status == TaskStatus.PENDING:
                    updated = self._promote_ready([updated], all_records, now)[0]
                control = self._advance(control, now)
                self._persist_event(connection, control, SchedulerEventType.TASK_EDITED, actor_id, mutation_id, fingerprint, [updated], task_id)
                return updated

    async def pause_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl:
        payload = {"run_id": run_id, "reason": reason}
        fingerprint = self.store.fingerprint("pause_run", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._control(run_id, connection)
                control = self._control(run_id, connection)
                if control.status != RunControlStatus.ACTIVE:
                    raise SchedulerStateError("only active runs may be paused")
                now = self.clock()
                records = [
                    self._transition(item, TaskStatus.PAUSED, now, pause_reason=reason, paused_by_run=True)
                    for item in self.store.list_tasks(run_id, connection=connection)
                    if item.envelope.status == TaskStatus.RUNNING
                ]
                control = self._advance(control.model_copy(update={"status": RunControlStatus.PAUSED}), now)
                self._persist_event(connection, control, SchedulerEventType.RUN_PAUSED, actor_id, mutation_id, fingerprint, records, extra={"reason": reason})
                return control

    async def resume_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl:
        payload = {"run_id": run_id}
        fingerprint = self.store.fingerprint("resume_run", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._control(run_id, connection)
                control = self._control(run_id, connection)
                if control.status != RunControlStatus.PAUSED:
                    raise SchedulerStateError("only paused runs may be resumed")
                now = self.clock()
                records = [
                    self._transition(item, TaskStatus.READY, now, pause_reason=None, paused_by_run=False)
                    for item in self.store.list_tasks(run_id, connection=connection)
                    if item.envelope.status == TaskStatus.PAUSED and item.paused_by_run
                ]
                control = self._advance(control.model_copy(update={"status": RunControlStatus.ACTIVE}), now)
                self._persist_event(connection, control, SchedulerEventType.RUN_RESUMED, actor_id, mutation_id, fingerprint, records)
                return control

    async def cancel_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl:
        payload = {"run_id": run_id, "reason": reason}
        fingerprint = self.store.fingerprint("cancel_run", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._control(run_id, connection)
                control = self._control(run_id, connection)
                if control.status in {RunControlStatus.CANCELLED, RunControlStatus.COMPLETED}:
                    raise SchedulerStateError("terminal run cannot be cancelled")
                now = self.clock()
                records = []
                for item in self.store.list_tasks(run_id, connection=connection):
                    if item.envelope.status in {TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}:
                        continue
                    records.append(self._transition(item, TaskStatus.CANCELLED, now, error_ref=self._error_ref("run_cancelled", mutation_id, item.task_id)))
                control = self._advance(
                    control.model_copy(update={"status": RunControlStatus.CANCELLED, "cancellation_reason": reason}), now
                )
                self._persist_event(connection, control, SchedulerEventType.RUN_CANCELLED, actor_id, mutation_id, fingerprint, records, extra={"reason": reason})
                return control

    async def complete_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl:
        payload = {"run_id": run_id}
        fingerprint = self.store.fingerprint("complete_run", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._control(run_id, connection)
                control = self._control(run_id, connection)
                if control.status != RunControlStatus.ACTIVE:
                    raise SchedulerStateError("only active runs may complete")
                nonterminal = [
                    item.task_id
                    for item in self.store.list_tasks(run_id, connection=connection)
                    if item.envelope.status not in {
                        TaskStatus.COMPLETED,
                        TaskStatus.FAILED,
                        TaskStatus.CANCELLED,
                        TaskStatus.PRUNED,
                        TaskStatus.MERGED,
                    }
                ]
                if nonterminal:
                    raise SchedulerStateError(f"run still has nonterminal tasks: {nonterminal}")
                now = self.clock()
                control = self._advance(control.model_copy(update={"status": RunControlStatus.COMPLETED}), now)
                self._persist_event(connection, control, SchedulerEventType.RUN_COMPLETED, actor_id, mutation_id, fingerprint, [])
                return control

    async def snapshot(self, run_id: str) -> SchedulerSnapshot:
        async with self._lock:
            control = self.store.get_control(run_id)
            if control is None:
                raise KeyError(f"unknown scheduler run: {run_id}")
            return SchedulerSnapshot(control=control, tasks=self.store.list_tasks(run_id))

    async def recover(self, run_id: str, *, actor_id: str, mutation_id: str) -> RecoveryReport:
        payload = {"run_id": run_id}
        fingerprint = self.store.fingerprint("recover", payload)
        async with self._lock:
            with self.store.transaction() as connection:
                replay = self.store.mutation_event(mutation_id, fingerprint, connection=connection)
                if replay:
                    return RecoveryReport.model_validate(replay.payload["report"], strict=False)
                control = self._control(run_id, connection)
                now = self.clock()
                records, recovered, failed = self._refresh_records(run_id, connection, now)
                promoted = [item.task_id for item in records if item.envelope.status == TaskStatus.READY]
                control = self._advance(control, now)
                report = RecoveryReport(
                    run_id=run_id,
                    recovered_task_ids=tuple(recovered),
                    failed_task_ids=tuple(failed),
                    promoted_task_ids=tuple(promoted),
                    projection_revision=control.projection_revision,
                )
                self._persist_event(connection, control, SchedulerEventType.LEASE_RECOVERED, actor_id, mutation_id, fingerprint, records, extra={"report": report.model_dump(mode="json")})
                return report

    async def rebuild_projection(self, run_id: str) -> SchedulerSnapshot:
        async with self._lock:
            self.store.rebuild_projection(run_id)
            return SchedulerSnapshot(control=self._control(run_id), tasks=self.store.list_tasks(run_id))

    async def close(self) -> None:
        self.store.close()

    async def _simple_transition(
        self,
        operation: str,
        event_type: SchedulerEventType,
        task_id: str,
        target: TaskStatus,
        actor_id: str,
        mutation_id: str,
        *,
        payload: dict[str, Any],
        updates: dict[str, Any],
        allowed_run_statuses: set[RunControlStatus] | None = None,
    ) -> TaskRecord:
        fingerprint = self.store.fingerprint(operation, {"task_id": task_id, **payload})
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                control = self._control(record.run_id, connection)
                allowed = allowed_run_statuses or {RunControlStatus.ACTIVE, RunControlStatus.PAUSED}
                if control.status not in allowed:
                    raise SchedulerStateError(f"run does not allow task mutation: {control.status.value}")
                now = self.clock()
                updated = self._transition(record, target, now, **updates)
                control = self._advance(control, now)
                self._persist_event(connection, control, event_type, actor_id, mutation_id, fingerprint, [updated], task_id, extra=payload)
                return updated

    async def _resolve_approval(
        self,
        task_id: str,
        *,
        approved: bool,
        actor_id: str,
        note: str,
        error_ref: str | None,
        mutation_id: str,
    ) -> TaskRecord:
        operation = "approve" if approved else "reject"
        payload = {"task_id": task_id, "note": note, "error_ref": error_ref}
        fingerprint = self.store.fingerprint(operation, payload)
        async with self._lock:
            with self.store.transaction() as connection:
                if self.store.mutation_event(mutation_id, fingerprint, connection=connection):
                    return self._task(task_id, connection)
                record = self._task(task_id, connection)
                control = self._mutable_control(record.run_id, connection)
                if record.envelope.status != TaskStatus.WAITING_APPROVAL or record.approval is None:
                    raise SchedulerStateError("task is not waiting for approval")
                now = self.clock()
                approval = record.approval.model_copy(
                    update={
                        "status": ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED,
                        "resolved_by": actor_id,
                        "resolution_note": note,
                        "resolved_at": now,
                    }
                )
                target = TaskStatus.READY if approved else TaskStatus.FAILED
                updated = self._transition(record, target, now, approval=approval, error_ref=error_ref)
                control = self._advance(control, now)
                self._persist_event(
                    connection,
                    control,
                    SchedulerEventType.APPROVAL_APPROVED if approved else SchedulerEventType.APPROVAL_REJECTED,
                    actor_id,
                    mutation_id,
                    fingerprint,
                    [updated],
                    task_id,
                )
                return updated

    def _refresh_records(
        self,
        run_id: str,
        connection,
        now: datetime,
    ) -> tuple[list[TaskRecord], list[str], list[str]]:
        current = list(self.store.list_tasks(run_id, connection=connection))
        projected: dict[str, TaskRecord] = {item.task_id: item for item in current}
        changed: list[TaskRecord] = []
        recovered: list[str] = []
        failed_ids: list[str] = []
        for item in current:
            updated = item
            if item.envelope.status == TaskStatus.RUNNING and item.lease_expires_at and item.lease_expires_at <= now:
                if item.envelope.attempt >= item.envelope.max_attempts:
                    updated = self._transition(item, TaskStatus.FAILED, now, error_ref="error_lease_attempts_exhausted")
                    failed_ids.append(item.task_id)
                else:
                    updated = self._transition(
                        item,
                        TaskStatus.FAILED,
                        now,
                        error_ref="error_lease_expired",
                        budget_usage=item.budget_usage.plus(errors=1),
                    )
                    updated = self._transition(
                        updated,
                        TaskStatus.PENDING,
                        now,
                        error_ref=None,
                        budget_usage=updated.budget_usage.plus(retries=1),
                    )
                    recovered.append(item.task_id)
            if updated.envelope.status in {TaskStatus.PENDING, TaskStatus.READY, TaskStatus.DEFERRED}:
                deadlines = [
                    deadline
                    for deadline in (updated.envelope.deadline, updated.envelope.budget.deadline)
                    if deadline is not None
                ]
                deadline = min(deadlines) if deadlines else None
                exhausted = self._scheduling_budget_exhausted(updated, now=now)
                if deadline is not None and deadline <= now:
                    updated = self._transition(updated, TaskStatus.FAILED, now, error_ref="error_deadline_reached")
                    failed_ids.append(item.task_id)
                elif exhausted:
                    updated = self._transition(updated, TaskStatus.FAILED, now, error_ref="error_budget_exhausted")
                    failed_ids.append(item.task_id)
                elif updated.envelope.status == TaskStatus.DEFERRED and updated.available_at <= now:
                    updated = self._transition(updated, TaskStatus.PENDING, now, defer_reason=None)
            if updated != item:
                projected[item.task_id] = updated
                changed.append(updated)
        pending = [item for item in projected.values() if item.envelope.status == TaskStatus.PENDING]
        promoted = self._promote_ready(pending, list(projected.values()), now)
        for item in promoted:
            if item != projected[item.task_id]:
                projected[item.task_id] = item
                changed.append(item)
        return self._dedupe_records(changed), recovered, failed_ids

    def _promote_ready(self, candidates: list[TaskRecord], all_records: list[TaskRecord], now: datetime) -> list[TaskRecord]:
        by_id = {item.task_id: item for item in all_records}
        output: list[TaskRecord] = []
        for item in candidates:
            if item.envelope.status != TaskStatus.PENDING or item.available_at > now:
                output.append(item)
                continue
            dependencies = [by_id.get(task_id) for task_id in item.envelope.dependency_task_ids]
            if all(dependency is not None and dependency.envelope.status == TaskStatus.COMPLETED for dependency in dependencies):
                output.append(self._transition(item, TaskStatus.READY, now))
            else:
                output.append(item)
        return output

    def _persist_event(
        self,
        connection,
        control: RunControl,
        event_type: SchedulerEventType,
        actor_id: str,
        mutation_id: str,
        fingerprint: str,
        records: list[TaskRecord] | tuple[TaskRecord, ...],
        task_id: str | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        records = self._dedupe_records(list(records))
        self.store.write_control(control, connection=connection)
        for record in records:
            self.store.write_task(record, connection=connection)
        self.store.append_event(
            self._event(control, event_type, actor_id, mutation_id, fingerprint, records, task_id=task_id, extra=extra),
            connection=connection,
        )

    def _event(
        self,
        control: RunControl,
        event_type: SchedulerEventType,
        actor_id: str,
        mutation_id: str,
        fingerprint: str,
        records: list[TaskRecord] | tuple[TaskRecord, ...],
        task_id: str | None = None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> SchedulerEvent:
        return SchedulerEvent(
            event_id=self.store.event_id(mutation_id),
            mutation_id=mutation_id,
            fingerprint=fingerprint,
            run_id=control.run_id,
            sequence_no=control.projection_revision,
            event_type=event_type,
            task_id=task_id,
            actor_id=actor_id,
            payload={
                "control": control.model_dump(mode="json"),
                "records": [item.model_dump(mode="json") for item in records],
                **(extra or {}),
            },
            occurred_at=control.updated_at,
        )

    @staticmethod
    def _advance(control: RunControl, now: datetime) -> RunControl:
        return control.model_copy(update={"projection_revision": control.projection_revision + 1, "updated_at": now})

    def _control(self, run_id: str, connection=None) -> RunControl:
        control = self.store.get_control(run_id, connection=connection)
        if control is None:
            raise KeyError(f"unknown scheduler run: {run_id}")
        return control

    def _mutable_control(self, run_id: str, connection) -> RunControl:
        control = self._control(run_id, connection)
        if control.status in {RunControlStatus.CANCELLED, RunControlStatus.COMPLETED}:
            raise SchedulerStateError(f"scheduler run is terminal: {control.status.value}")
        return control

    def _task(self, task_id: str, connection=None) -> TaskRecord:
        record = self.store.get_task(task_id, connection=connection)
        if record is None:
            raise KeyError(f"unknown scheduler task: {task_id}")
        return record

    @staticmethod
    def _transition(record: TaskRecord, target: TaskStatus, now: datetime, **updates: Any) -> TaskRecord:
        envelope = record.envelope.transition(target, at=now)
        cleared = {"lease_owner": None, "lease_expires_at": None, "last_heartbeat_at": None}
        return record.model_copy(
            update={
                "envelope": envelope,
                "revision": record.revision + 1,
                "updated_at": now,
                **cleared,
                **updates,
            }
        )

    @staticmethod
    def _edit_envelope(record: TaskRecord, updates: dict[str, Any], now: datetime) -> TaskRecord:
        values = record.envelope.model_dump(mode="python")
        values.update(updates)
        values["updated_at"] = now
        envelope = TaskEnvelope.model_validate(values, strict=False)
        return record.model_copy(update={"envelope": envelope, "revision": record.revision + 1, "updated_at": now})

    def _validate_references(self, task: TaskEnvelope, connection, *, allow_self: bool = False) -> None:
        if task.parent_task_id:
            parent = self.store.get_task(task.parent_task_id, connection=connection)
            if parent is None or parent.run_id != task.run_id:
                raise SchedulerStateError("parent task must exist in the same run")
        for dependency_id in task.dependency_task_ids:
            dependency = self.store.get_task(dependency_id, connection=connection)
            if dependency is None or dependency.run_id != task.run_id:
                if allow_self and dependency_id == task.task_id:
                    continue
                raise SchedulerStateError(f"dependency must exist in the same run: {dependency_id}")

    @staticmethod
    def _validate_dag(records: list[TaskRecord]) -> None:
        graph = {item.task_id: item.envelope.dependency_task_ids for item in records}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(task_id: str) -> None:
            if task_id in visiting:
                raise SchedulerStateError("task dependency cycle detected")
            if task_id in visited:
                return
            visiting.add(task_id)
            for dependency in graph.get(task_id, ()):
                if dependency not in graph:
                    raise SchedulerStateError(f"unknown dependency in DAG: {dependency}")
                visit(dependency)
            visiting.remove(task_id)
            visited.add(task_id)

        for task_id in graph:
            visit(task_id)

    @staticmethod
    def _toposort_children(children: tuple[TaskEnvelope, ...]) -> tuple[TaskEnvelope, ...]:
        by_id = {item.task_id: item for item in children}
        output: list[TaskEnvelope] = []
        visited: set[str] = set()

        def visit(item: TaskEnvelope) -> None:
            if item.task_id in visited:
                return
            for dependency in item.dependency_task_ids:
                if dependency in by_id:
                    visit(by_id[dependency])
            visited.add(item.task_id)
            output.append(item)

        for child in children:
            visit(child)
        return tuple(output)

    @staticmethod
    def _require_lease(record: TaskRecord, worker_id: str, now: datetime) -> None:
        if record.envelope.status != TaskStatus.RUNNING or record.lease_owner != worker_id:
            raise SchedulerLeaseError("task is not leased by this worker")
        if record.lease_expires_at is None or record.lease_expires_at <= now:
            raise SchedulerLeaseError("task lease has expired")

    @staticmethod
    def _combine_usage(left: BudgetUsage, right: BudgetUsage) -> BudgetUsage:
        return left.plus(
            input_tokens=right.input_tokens,
            output_tokens=right.output_tokens,
            cost_usd=right.cost_usd,
            wall_time_seconds=right.wall_time_seconds,
            model_calls=right.model_calls,
            tool_calls=right.tool_calls,
            search_calls=right.search_calls,
            retries=right.retries,
            errors=right.errors,
        )

    @staticmethod
    def _scheduling_budget_exhausted(record: TaskRecord, *, now: datetime) -> bool:
        exhausted = set(record.envelope.budget.exceeded_dimensions(record.budget_usage, now=now))
        if record.budget_usage.retries == 0:
            exhausted.discard(BudgetDimension.RETRIES)
        if record.budget_usage.errors == 0:
            exhausted.discard(BudgetDimension.ERRORS)
        return bool(exhausted)

    @staticmethod
    def _dedupe_records(records: list[TaskRecord]) -> list[TaskRecord]:
        output: dict[str, TaskRecord] = {}
        for record in records:
            output[record.task_id] = record
        return list(output.values())

    @staticmethod
    def _error_ref(kind: str, mutation_id: str, task_id: str = "") -> str:
        digest = hashlib.sha256(f"{kind}\0{mutation_id}\0{task_id}".encode("utf-8")).hexdigest()
        return f"error_{digest[:24]}"
