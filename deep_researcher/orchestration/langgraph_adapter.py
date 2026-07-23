from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import empty_checkpoint
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from deep_researcher.contracts import BudgetUsage, TaskEnvelope

from .models import (
    RecoveryReport,
    RunControl,
    SchedulerSnapshot,
    TaskCompletion,
    TaskEdit,
    TaskLease,
    TaskRecord,
    ThinRuntimeState,
    thin_state_from_snapshot,
)
from .scheduler import Scheduler


class LangGraphCheckpointBridge:
    """Persists only the compact runtime projection in a real LangGraph saver."""

    CHANNEL = "orchestration_runtime"
    NAMESPACE = "background001/orchestration"

    def __init__(self, saver: AsyncSqliteSaver) -> None:
        self.saver = saver

    async def save(self, state: ThinRuntimeState) -> dict[str, Any]:
        base_config: dict[str, Any] = {
            "configurable": {
                "thread_id": state.run_id,
                "checkpoint_ns": self.NAMESPACE,
            }
        }
        existing = await self.saver.aget_tuple(base_config)
        config = existing.config if existing is not None else base_config
        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {self.CHANNEL: state.model_dump(mode="json")}
        checkpoint["channel_versions"] = {self.CHANNEL: state.projection_revision}
        checkpoint["versions_seen"] = {}
        checkpoint["updated_channels"] = [self.CHANNEL]
        return await self.saver.aput(
            config,
            checkpoint,
            {
                "source": "update",
                "step": state.projection_revision,
                "parents": {},
                "run_id": state.run_id,
            },
            {self.CHANNEL: state.projection_revision},
        )

    async def load(self, run_id: str) -> ThinRuntimeState | None:
        checkpoint = await self.saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": run_id,
                    "checkpoint_ns": self.NAMESPACE,
                }
            }
        )
        if checkpoint is None:
            return None
        payload = checkpoint.checkpoint.get("channel_values", {}).get(self.CHANNEL)
        return ThinRuntimeState.model_validate(payload, strict=False) if payload is not None else None


class LangGraphRuntimeAdapter:
    """Scheduler adapter that mirrors event-derived thin state to LangGraph."""

    def __init__(self, scheduler: Scheduler, bridge: LangGraphCheckpointBridge) -> None:
        self.scheduler = scheduler
        self.bridge = bridge

    async def create_run(self, run_id: str, *, max_concurrency: int = 4, actor_id: str, mutation_id: str) -> RunControl:
        result = await self.scheduler.create_run(run_id, max_concurrency=max_concurrency, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def submit(self, task: TaskEnvelope, *, actor_id: str, mutation_id: str, available_at=None) -> TaskRecord:
        result = await self.scheduler.submit(task, actor_id=actor_id, mutation_id=mutation_id, available_at=available_at)
        await self._sync(task.run_id)
        return result

    async def split(self, parent_task_id: str, children: tuple[TaskEnvelope, ...], *, actor_id: str, mutation_id: str) -> tuple[TaskRecord, ...]:
        result = await self.scheduler.split(parent_task_id, children, actor_id=actor_id, mutation_id=mutation_id)
        if result:
            await self._sync(result[0].run_id)
        return result

    async def merge(self, source_task_id: str, target_task_id: str, *, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.merge(source_task_id, target_task_id, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def defer(self, task_id: str, *, until, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.defer(task_id, until=until, reason=reason, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def prune(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.prune(task_id, reason=reason, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def cancel_task(self, task_id: str, *, reason: str, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.cancel_task(task_id, reason=reason, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def claim(self, run_id: str, *, worker_id: str, limit: int = 1, lease_seconds: float = 30, mutation_id: str) -> tuple[TaskLease, ...]:
        result = await self.scheduler.claim(run_id, worker_id=worker_id, limit=limit, lease_seconds=lease_seconds, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def heartbeat(self, task_id: str, *, worker_id: str, lease_seconds: float, mutation_id: str) -> TaskLease:
        result = await self.scheduler.heartbeat(task_id, worker_id=worker_id, lease_seconds=lease_seconds, mutation_id=mutation_id)
        await self._sync(result.task.run_id)
        return result

    async def complete(self, task_id: str, completion: TaskCompletion, *, worker_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.complete(task_id, completion, worker_id=worker_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def update_usage(self, task_id: str, usage: BudgetUsage, *, worker_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.update_usage(task_id, usage, worker_id=worker_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def fail(self, task_id: str, *, error_ref: str, worker_id: str, mutation_id: str, usage: BudgetUsage | None = None) -> TaskRecord:
        result = await self.scheduler.fail(task_id, error_ref=error_ref, worker_id=worker_id, mutation_id=mutation_id, usage=usage)
        await self._sync(result.run_id)
        return result

    async def retry(self, task_id: str, *, actor_id: str, mutation_id: str, available_at=None) -> TaskRecord:
        result = await self.scheduler.retry(task_id, actor_id=actor_id, mutation_id=mutation_id, available_at=available_at)
        await self._sync(result.run_id)
        return result

    async def pause_task(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.pause_task(task_id, worker_id=worker_id, reason=reason, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def resume_task(self, task_id: str, *, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.resume_task(task_id, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def request_approval(self, task_id: str, *, worker_id: str, reason: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.request_approval(task_id, worker_id=worker_id, reason=reason, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def approve(self, task_id: str, *, actor_id: str, note: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.approve(task_id, actor_id=actor_id, note=note, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def reject(self, task_id: str, *, actor_id: str, note: str, error_ref: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.reject(task_id, actor_id=actor_id, note=note, error_ref=error_ref, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def edit(self, task_id: str, edit: TaskEdit, *, actor_id: str, mutation_id: str) -> TaskRecord:
        result = await self.scheduler.edit(task_id, edit, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(result.run_id)
        return result

    async def pause_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl:
        result = await self.scheduler.pause_run(run_id, actor_id=actor_id, reason=reason, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def resume_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl:
        result = await self.scheduler.resume_run(run_id, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def cancel_run(self, run_id: str, *, actor_id: str, reason: str, mutation_id: str) -> RunControl:
        result = await self.scheduler.cancel_run(run_id, actor_id=actor_id, reason=reason, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def complete_run(self, run_id: str, *, actor_id: str, mutation_id: str) -> RunControl:
        result = await self.scheduler.complete_run(run_id, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def snapshot(self, run_id: str) -> SchedulerSnapshot:
        return await self.scheduler.snapshot(run_id)

    async def recover(self, run_id: str, *, actor_id: str, mutation_id: str) -> RecoveryReport:
        result = await self.scheduler.recover(run_id, actor_id=actor_id, mutation_id=mutation_id)
        await self._sync(run_id)
        return result

    async def rebuild_projection(self, run_id: str) -> SchedulerSnapshot:
        result = await self.scheduler.rebuild_projection(run_id)
        await self.bridge.save(thin_state_from_snapshot(result))
        return result

    async def checkpoint_state(self, run_id: str) -> ThinRuntimeState | None:
        return await self.bridge.load(run_id)

    async def close(self) -> None:
        await self.scheduler.close()

    async def _sync(self, run_id: str) -> ThinRuntimeState:
        state = thin_state_from_snapshot(await self.scheduler.snapshot(run_id))
        await self.bridge.save(state)
        return state
