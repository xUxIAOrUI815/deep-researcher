from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
from typing import Any

import pytest

from deep_researcher.contracts import Budget, BudgetUsage, TaskEnvelope, TaskKind, TaskStatus
from deep_researcher.orchestration import (
    NativeEventSourcedScheduler,
    RunControlStatus,
    SchedulerCorruption,
    SchedulerLeaseError,
    SchedulerMutationConflict,
    SchedulerStateError,
    SQLiteSchedulerStore,
    TaskCompletion,
    TaskEdit,
)


class MutableClock:
    def __init__(self) -> None:
        self.value = datetime(2026, 7, 23, tzinfo=timezone.utc)

    def __call__(self) -> datetime:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += timedelta(seconds=seconds)


def _budget(**updates: Any) -> Budget:
    values: dict[str, Any] = {
        "max_tokens": 10_000,
        "max_cost_usd": 10,
        "max_wall_time_seconds": 300,
        "max_model_calls": 10,
        "max_tool_calls": 10,
        "max_search_calls": 10,
        "max_retries": 3,
        "max_errors": 3,
    }
    values.update(updates)
    return Budget(**values)


def _task(
    suffix: str,
    *,
    run_id: str = "run_scheduler",
    dependencies: tuple[str, ...] = (),
    parent: str | None = None,
    priority: float = 0.5,
    deadline: datetime | None = None,
    max_attempts: int = 3,
    budget: Budget | None = None,
    assigned_actor_id: str | None = None,
) -> TaskEnvelope:
    return TaskEnvelope(
        task_id=f"task_{suffix}",
        run_id=run_id,
        parent_task_id=parent,
        dependency_task_ids=dependencies,
        kind=TaskKind.RESEARCH,
        title=f"Task {suffix}",
        goal=f"Execute bounded task {suffix}.",
        expected_output_schema="TaskResult@1",
        budget=budget or _budget(),
        priority=priority,
        deadline=deadline,
        max_attempts=max_attempts,
        created_by="agent_supervisor",
        assigned_actor_id=assigned_actor_id,
    )


def _native(path: Path, clock: MutableClock | None = None) -> NativeEventSourcedScheduler:
    return NativeEventSourcedScheduler(SQLiteSchedulerStore(path), clock=clock or MutableClock())


@pytest.mark.asyncio
async def test_dependency_priority_deadline_and_worker_slots_drive_atomic_claims(tmp_path: Path):
    scheduler = _native(tmp_path / "scheduler.sqlite3")
    await scheduler.create_run("run_scheduler", max_concurrency=2, actor_id="agent_supervisor", mutation_id="mutation_run")
    low = await scheduler.submit(_task("low", priority=0.2), actor_id="agent_supervisor", mutation_id="mutation_low")
    high = await scheduler.submit(_task("high", priority=0.9), actor_id="agent_supervisor", mutation_id="mutation_high")
    dependent = await scheduler.submit(
        _task("dependent", dependencies=(high.task_id,), priority=1.0),
        actor_id="agent_supervisor",
        mutation_id="mutation_dependent",
    )
    assert low.envelope.status == TaskStatus.READY
    assert high.envelope.status == TaskStatus.READY
    assert dependent.envelope.status == TaskStatus.PENDING

    leases = await scheduler.claim(
        "run_scheduler",
        worker_id="agent_worker",
        limit=3,
        lease_seconds=30,
        mutation_id="mutation_claim_initial",
    )
    assert [lease.task.task_id for lease in leases] == ["task_high", "task_low"]
    assert all(lease.task.attempt == 1 for lease in leases)

    completed = await scheduler.complete(
        "task_high",
        TaskCompletion(result_id="result_high", output_artifact_ids=("artifact_high",)),
        worker_id="agent_worker",
        mutation_id="mutation_complete_high",
    )
    assert completed.envelope.status == TaskStatus.COMPLETED
    snapshot = await scheduler.snapshot("run_scheduler")
    assert snapshot.by_id["task_dependent"].envelope.status == TaskStatus.READY
    next_lease = await scheduler.claim(
        "run_scheduler",
        worker_id="agent_worker_two",
        limit=3,
        lease_seconds=30,
        mutation_id="mutation_claim_dependent",
    )
    assert [item.task.task_id for item in next_lease] == ["task_dependent"]
    await scheduler.close()


@pytest.mark.asyncio
async def test_mutations_and_claims_are_idempotent_and_conflicting_reuse_is_rejected(tmp_path: Path):
    scheduler = _native(tmp_path / "idempotent.sqlite3")
    first_run = await scheduler.create_run("run_scheduler", max_concurrency=1, actor_id="agent_supervisor", mutation_id="mutation_run")
    replay_run = await scheduler.create_run("run_scheduler", max_concurrency=1, actor_id="agent_supervisor", mutation_id="mutation_run")
    assert replay_run == first_run

    task = _task("one")
    first = await scheduler.submit(task, actor_id="agent_supervisor", mutation_id="mutation_submit")
    replay = await scheduler.submit(task, actor_id="agent_supervisor", mutation_id="mutation_submit")
    assert replay == first
    with pytest.raises(SchedulerMutationConflict):
        await scheduler.submit(_task("other"), actor_id="agent_supervisor", mutation_id="mutation_submit")

    leases = await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=30, mutation_id="mutation_claim")
    replayed = await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=30, mutation_id="mutation_claim")
    assert replayed == leases
    assert (await scheduler.snapshot("run_scheduler")).by_id[task.task_id].envelope.attempt == 1
    await scheduler.close()


@pytest.mark.asyncio
async def test_concurrent_scheduler_instances_never_double_claim_and_honor_global_slots(tmp_path: Path):
    path = tmp_path / "concurrent.sqlite3"
    first = _native(path)
    await first.create_run("run_scheduler", max_concurrency=2, actor_id="agent_supervisor", mutation_id="mutation_run")
    for index in range(5):
        await first.submit(_task(f"concurrent_{index}"), actor_id="agent_supervisor", mutation_id=f"mutation_submit_{index}")
    second = _native(path)
    left, right = await asyncio.gather(
        first.claim("run_scheduler", worker_id="agent_left", limit=3, lease_seconds=30, mutation_id="mutation_claim_left"),
        second.claim("run_scheduler", worker_id="agent_right", limit=3, lease_seconds=30, mutation_id="mutation_claim_right"),
    )
    claimed = [item.task.task_id for item in (*left, *right)]
    assert len(claimed) == 2
    assert len(set(claimed)) == 2
    snapshot = await first.snapshot("run_scheduler")
    assert sum(item.envelope.status == TaskStatus.RUNNING for item in snapshot.tasks) == 2
    await first.close()
    await second.close()


@pytest.mark.asyncio
async def test_expired_leases_recover_after_restart_and_attempt_exhaustion_fails(tmp_path: Path):
    path = tmp_path / "recovery.sqlite3"
    clock = MutableClock()
    scheduler = _native(path, clock)
    await scheduler.create_run("run_scheduler", max_concurrency=1, actor_id="agent_supervisor", mutation_id="mutation_run")
    await scheduler.submit(_task("recover", max_attempts=2), actor_id="agent_supervisor", mutation_id="mutation_submit")
    await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=5, mutation_id="mutation_claim_one")
    await scheduler.close()

    clock.advance(6)
    restarted = _native(path, clock)
    report = await restarted.recover("run_scheduler", actor_id="agent_recovery", mutation_id="mutation_recover_one")
    assert report.recovered_task_ids == ("task_recover",)
    recovered = (await restarted.snapshot("run_scheduler")).by_id["task_recover"]
    assert recovered.envelope.status == TaskStatus.READY
    assert recovered.budget_usage.retries == 1
    await restarted.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=5, mutation_id="mutation_claim_two")
    clock.advance(6)
    exhausted = await restarted.recover("run_scheduler", actor_id="agent_recovery", mutation_id="mutation_recover_two")
    assert exhausted.failed_task_ids == ("task_recover",)
    assert (await restarted.snapshot("run_scheduler")).by_id["task_recover"].envelope.status == TaskStatus.FAILED
    await restarted.close()


@pytest.mark.asyncio
async def test_dynamic_split_merge_defer_prune_and_cycle_safe_edit(tmp_path: Path):
    clock = MutableClock()
    scheduler = _native(tmp_path / "dynamic.sqlite3", clock)
    await scheduler.create_run("run_scheduler", max_concurrency=3, actor_id="agent_supervisor", mutation_id="mutation_run")
    parent = await scheduler.submit(_task("parent"), actor_id="agent_supervisor", mutation_id="mutation_parent")
    child_one = _task("child_one", parent=parent.task_id)
    child_two = _task("child_two", parent=parent.task_id, dependencies=(child_one.task_id,))
    children = await scheduler.split(parent.task_id, (child_one, child_two), actor_id="agent_supervisor", mutation_id="mutation_split")
    assert children[0].envelope.status == TaskStatus.READY
    assert children[1].envelope.status == TaskStatus.PENDING

    deferred = await scheduler.defer(
        child_one.task_id,
        until=clock() + timedelta(seconds=20),
        reason="wait for source",
        actor_id="agent_supervisor",
        mutation_id="mutation_defer",
    )
    assert deferred.envelope.status == TaskStatus.DEFERRED
    early = await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=3, lease_seconds=10, mutation_id="mutation_claim_early")
    assert child_one.task_id not in {item.task.task_id for item in early}
    clock.advance(21)
    leases = await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=3, lease_seconds=10, mutation_id="mutation_claim_due")
    assert child_one.task_id in {item.task.task_id for item in leases}

    source = await scheduler.submit(_task("merge_source"), actor_id="agent_supervisor", mutation_id="mutation_merge_source")
    target = await scheduler.submit(_task("merge_target"), actor_id="agent_supervisor", mutation_id="mutation_merge_target")
    dependent = await scheduler.submit(
        _task("merge_dependent", dependencies=(source.task_id,)), actor_id="agent_supervisor", mutation_id="mutation_merge_dependent"
    )
    merged = await scheduler.merge(source.task_id, target.task_id, actor_id="agent_supervisor", mutation_id="mutation_merge")
    snapshot = await scheduler.snapshot("run_scheduler")
    assert merged.envelope.status == TaskStatus.MERGED
    assert snapshot.by_id[dependent.task_id].envelope.dependency_task_ids == (target.task_id,)

    with pytest.raises((SchedulerStateError, ValueError), match="cycle|itself"):
        await scheduler.edit(
            target.task_id,
            TaskEdit(dependency_task_ids=(dependent.task_id,)),
            actor_id="agent_supervisor",
            mutation_id="mutation_cycle",
        )
    pruned = await scheduler.prune(target.task_id, reason="superseded", actor_id="agent_supervisor", mutation_id="mutation_prune")
    assert pruned.envelope.status == TaskStatus.PRUNED
    await scheduler.close()


@pytest.mark.asyncio
async def test_failure_retry_budget_deadline_and_lease_ownership_paths(tmp_path: Path):
    clock = MutableClock()
    scheduler = _native(tmp_path / "limits.sqlite3", clock)
    await scheduler.create_run("run_scheduler", max_concurrency=3, actor_id="agent_supervisor", mutation_id="mutation_run")
    await scheduler.submit(_task("failure", max_attempts=2), actor_id="agent_supervisor", mutation_id="mutation_failure")
    await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=30, mutation_id="mutation_claim_failure")
    with pytest.raises(SchedulerLeaseError):
        await scheduler.fail("task_failure", error_ref="error_provider", worker_id="agent_other", mutation_id="mutation_wrong_worker")
    failed = await scheduler.fail("task_failure", error_ref="error_provider", worker_id="agent_worker", mutation_id="mutation_fail")
    assert failed.envelope.status == TaskStatus.FAILED
    retried = await scheduler.retry("task_failure", actor_id="agent_supervisor", mutation_id="mutation_retry")
    assert retried.envelope.status == TaskStatus.READY
    assert retried.budget_usage.retries == 1

    await scheduler.submit(
        _task("budget", budget=_budget(max_model_calls=1)), actor_id="agent_supervisor", mutation_id="mutation_budget"
    )
    leases = await scheduler.claim("run_scheduler", worker_id="agent_budget", limit=2, lease_seconds=30, mutation_id="mutation_claim_budget")
    assert "task_budget" in {item.task.task_id for item in leases}
    budgeted = await scheduler.update_usage(
        "task_budget",
        BudgetUsage(model_calls=1),
        worker_id="agent_budget",
        mutation_id="mutation_usage",
    )
    assert budgeted.envelope.status == TaskStatus.FAILED
    assert budgeted.error_ref == "error_budget_exhausted"

    await scheduler.submit(
        _task("deadline", deadline=clock() - timedelta(seconds=1)),
        actor_id="agent_supervisor",
        mutation_id="mutation_deadline",
    )
    await scheduler.claim("run_scheduler", worker_id="agent_deadline", limit=1, lease_seconds=30, mutation_id="mutation_claim_deadline")
    deadline = (await scheduler.snapshot("run_scheduler")).by_id["task_deadline"]
    assert deadline.envelope.status == TaskStatus.FAILED
    assert deadline.error_ref == "error_deadline_reached"

    await scheduler.submit(
        _task(
            "budget_deadline",
            deadline=clock() + timedelta(seconds=100),
            budget=_budget(deadline=clock() - timedelta(seconds=1)),
        ),
        actor_id="agent_supervisor",
        mutation_id="mutation_budget_deadline",
    )
    await scheduler.claim(
        "run_scheduler",
        worker_id="agent_budget_deadline",
        limit=1,
        lease_seconds=30,
        mutation_id="mutation_claim_budget_deadline",
    )
    budget_deadline = (await scheduler.snapshot("run_scheduler")).by_id["task_budget_deadline"]
    assert budget_deadline.envelope.status == TaskStatus.FAILED
    assert budget_deadline.error_ref == "error_deadline_reached"

    await scheduler.submit(
        _task("completion_budget", budget=_budget(max_tool_calls=1)),
        actor_id="agent_supervisor",
        mutation_id="mutation_completion_budget",
    )
    await scheduler.claim(
        "run_scheduler",
        worker_id="agent_completion_budget",
        limit=1,
        lease_seconds=30,
        mutation_id="mutation_claim_completion_budget",
    )
    exhausted_completion = await scheduler.complete(
        "task_completion_budget",
        TaskCompletion(result_id="result_over_budget", usage=BudgetUsage(tool_calls=1)),
        worker_id="agent_completion_budget",
        mutation_id="mutation_complete_over_budget",
    )
    assert exhausted_completion.envelope.status == TaskStatus.FAILED
    assert exhausted_completion.result_id is None
    assert exhausted_completion.error_ref == "error_budget_exhausted"

    await scheduler.submit(_task("stale_lease"), actor_id="agent_supervisor", mutation_id="mutation_stale_lease")
    await scheduler.claim(
        "run_scheduler",
        worker_id="agent_stale",
        limit=1,
        lease_seconds=5,
        mutation_id="mutation_claim_stale",
    )
    clock.advance(6)
    with pytest.raises(SchedulerLeaseError, match="expired"):
        await scheduler.complete(
            "task_stale_lease",
            TaskCompletion(result_id="result_stale"),
            worker_id="agent_stale",
            mutation_id="mutation_complete_stale",
        )
    recovered = await scheduler.recover(
        "run_scheduler",
        actor_id="agent_recovery",
        mutation_id="mutation_recover_stale",
    )
    assert recovered.recovered_task_ids == ("task_stale_lease",)
    await scheduler.close()


@pytest.mark.asyncio
async def test_hitl_pause_approve_reject_edit_resume_and_user_cancellation(tmp_path: Path):
    scheduler = _native(tmp_path / "hitl.sqlite3")
    await scheduler.create_run("run_scheduler", max_concurrency=2, actor_id="agent_supervisor", mutation_id="mutation_run")
    await scheduler.submit(_task("approval"), actor_id="agent_supervisor", mutation_id="mutation_approval")
    await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=30, mutation_id="mutation_claim")
    waiting = await scheduler.request_approval(
        "task_approval", worker_id="agent_worker", reason="external side effect", mutation_id="mutation_request"
    )
    assert waiting.envelope.status == TaskStatus.WAITING_APPROVAL
    edited = await scheduler.edit(
        "task_approval",
        TaskEdit(priority=0.95, constraints={"scope": "approved-domain"}),
        actor_id="agent_human",
        mutation_id="mutation_edit",
    )
    assert edited.envelope.priority == 0.95
    approved = await scheduler.approve("task_approval", actor_id="agent_human", note="approved", mutation_id="mutation_approve")
    assert approved.envelope.status == TaskStatus.READY
    assert approved.approval.status.value == "approved"
    await scheduler.claim("run_scheduler", worker_id="agent_worker", limit=1, lease_seconds=30, mutation_id="mutation_reclaim")
    paused = await scheduler.pause_task("task_approval", worker_id="agent_worker", reason="operator pause", mutation_id="mutation_pause_task")
    assert paused.envelope.status == TaskStatus.PAUSED
    resumed = await scheduler.resume_task("task_approval", actor_id="agent_human", mutation_id="mutation_resume_task")
    assert resumed.envelope.status == TaskStatus.READY
    await scheduler.cancel_task(
        "task_approval", reason="approval flow verified", actor_id="agent_human", mutation_id="mutation_cancel_approval"
    )

    await scheduler.submit(_task("reject"), actor_id="agent_supervisor", mutation_id="mutation_reject_task")
    await scheduler.claim("run_scheduler", worker_id="agent_reject", limit=1, lease_seconds=30, mutation_id="mutation_claim_reject")
    await scheduler.request_approval("task_reject", worker_id="agent_reject", reason="needs review", mutation_id="mutation_request_reject")
    rejected = await scheduler.reject(
        "task_reject", actor_id="agent_human", note="denied", error_ref="error_approval_rejected", mutation_id="mutation_reject"
    )
    assert rejected.envelope.status == TaskStatus.FAILED

    await scheduler.submit(_task("run_pause"), actor_id="agent_supervisor", mutation_id="mutation_run_pause_task")
    run_lease = await scheduler.claim("run_scheduler", worker_id="agent_run", limit=1, lease_seconds=30, mutation_id="mutation_claim_run")
    assert run_lease[0].task.task_id == "task_run_pause"
    paused_run = await scheduler.pause_run("run_scheduler", actor_id="agent_human", reason="maintenance", mutation_id="mutation_pause_run")
    assert paused_run.status == RunControlStatus.PAUSED
    assert (await scheduler.snapshot("run_scheduler")).by_id["task_run_pause"].paused_by_run is True
    assert not await scheduler.claim("run_scheduler", worker_id="agent_blocked", limit=2, lease_seconds=30, mutation_id="mutation_claim_paused")
    resumed_run = await scheduler.resume_run("run_scheduler", actor_id="agent_human", mutation_id="mutation_resume_run")
    assert resumed_run.status == RunControlStatus.ACTIVE
    cancelled = await scheduler.cancel_run("run_scheduler", actor_id="agent_human", reason="user requested", mutation_id="mutation_cancel_run")
    assert cancelled.status == RunControlStatus.CANCELLED
    assert all(
        item.envelope.status in {TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.PRUNED, TaskStatus.MERGED}
        for item in (await scheduler.snapshot("run_scheduler")).tasks
    )
    await scheduler.close()


@pytest.mark.asyncio
async def test_projection_rebuild_restart_backup_and_checksum_corruption(tmp_path: Path):
    path = tmp_path / "projection.sqlite3"
    scheduler = _native(path)
    await scheduler.create_run("run_scheduler", max_concurrency=1, actor_id="agent_supervisor", mutation_id="mutation_run")
    await scheduler.submit(_task("one"), actor_id="agent_supervisor", mutation_id="mutation_submit")
    before = await scheduler.snapshot("run_scheduler")
    with scheduler.store.transaction() as connection:
        connection.execute("UPDATE scheduler_tasks SET record_json='{}' WHERE task_id='task_one'")
    with pytest.raises(SchedulerCorruption, match="checksum"):
        scheduler.store.integrity_check()
    rebuilt = await scheduler.rebuild_projection("run_scheduler")
    assert rebuilt == before.model_copy(update={"captured_at": rebuilt.captured_at})
    scheduler.store.integrity_check()
    backup = scheduler.store.backup_to(tmp_path / "backup.sqlite3")
    await scheduler.close()

    restarted = _native(path)
    assert (await restarted.snapshot("run_scheduler")).by_id["task_one"].envelope.status == TaskStatus.READY
    await restarted.close()
    restored = _native(backup)
    restored.store.integrity_check()
    await restored.close()

    connection = sqlite3.connect(path)
    connection.execute("UPDATE scheduler_events SET event_json='{}' WHERE sequence_no=1")
    connection.commit()
    connection.close()
    corrupt = _native(path)
    with pytest.raises(SchedulerCorruption, match="checksum"):
        corrupt.store.integrity_check()
    await corrupt.close()


@pytest.mark.asyncio
async def test_native_scheduler_satisfies_the_production_behavioral_contract(
    tmp_path: Path,
):
    store = SQLiteSchedulerStore(tmp_path / "native-store.sqlite3")
    scheduler = NativeEventSourcedScheduler(store)
    try:
        await scheduler.create_run("run_scheduler", max_concurrency=1, actor_id="agent_supervisor", mutation_id="mutation_run")
        await scheduler.submit(
            _task("contract", assigned_actor_id="agent_contract"),
            actor_id="agent_supervisor",
            mutation_id="mutation_submit",
        )
        assert not await scheduler.claim(
            "run_scheduler",
            worker_id="agent_wrong",
            limit=1,
            lease_seconds=30,
            mutation_id="mutation_claim_wrong",
        )
        lease = (
            await scheduler.claim(
                "run_scheduler",
                worker_id="agent_contract",
                limit=1,
                lease_seconds=5,
                mutation_id="mutation_claim",
            )
        )[0]
        heartbeat = await scheduler.heartbeat(
            lease.task.task_id,
            worker_id="agent_contract",
            lease_seconds=30,
            mutation_id="mutation_heartbeat",
        )
        assert heartbeat.lease_expires_at > lease.lease_expires_at
        await scheduler.complete(
            lease.task.task_id,
            TaskCompletion(result_id="result_contract", output_artifact_ids=("artifact_contract",)),
            worker_id="agent_contract",
            mutation_id="mutation_complete",
        )
        completed_run = await scheduler.complete_run(
            "run_scheduler", actor_id="agent_supervisor", mutation_id="mutation_complete_run"
        )
        assert completed_run.status == RunControlStatus.COMPLETED
        snapshot = await scheduler.snapshot("run_scheduler")
        assert snapshot.by_id["task_contract"].result_id == "result_contract"
        assert snapshot.control.projection_revision == len(store.list_events("run_scheduler"))
        store.integrity_check()
    finally:
        await scheduler.close()
