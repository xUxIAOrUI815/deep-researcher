from __future__ import annotations

import hashlib
import json
from typing import Any

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    BudgetUsage,
    ConflictSeverity,
    ConflictStatus,
    SectionCoverageStatus,
    TaskEnvelope,
    TaskResultStatus,
    TaskStatus,
    combine_usage,
    utc_now,
)
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.kernel import (
    AgentKernel,
    CancellationToken,
    KernelRunResult,
)
from deep_researcher.orchestration import (
    RunControlStatus,
    Scheduler,
    SchedulerSnapshot,
    TaskCompletion,
    TaskEdit,
    TaskLease,
)

from .models import (
    ConvergenceAction,
    ConvergenceDecision,
    ConvergencePolicy,
    ConvergenceSnapshot,
    ResearchRunOutcome,
)
from .store import SQLiteResearchCoordinationStore
from .worker import (
    CrossWorkerResultMerger,
    ResearchWorkerPool,
    ResearchWorkerResultReconciler,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


_TERMINAL_TASK_STATUSES = {
    TaskStatus.COMPLETED,
    TaskStatus.FAILED,
    TaskStatus.CANCELLED,
    TaskStatus.PRUNED,
    TaskStatus.MERGED,
}


class ConvergenceEvaluator:
    """Authoritative semantic/budget/task convergence gate for research."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        artifact_store: ArtifactStore,
        coordination: SQLiteResearchCoordinationStore,
        policy: ConvergencePolicy,
        producer_id: str = "runtime_research_convergence_evaluator",
        clock=utc_now,
    ) -> None:
        self.evidence = evidence
        self.artifact_store = artifact_store
        self.coordination = coordination
        self.policy = policy
        self.producer_id = producer_id
        self.clock = clock

    def assess(
        self,
        *,
        scheduler_snapshot: SchedulerSnapshot,
        root_task_id: str,
        cycle: int,
        latest_information_gain: float,
    ) -> ConvergenceDecision:
        run_id = scheduler_snapshot.control.run_id
        existing = {
            item.cycle: item
            for item in self.coordination.convergence_decisions(run_id)
        }.get(cycle)
        if existing is not None:
            return existing

        repository = self.evidence.knowledge.repository
        sections = {
            item.section_id: item
            for item in repository.sections.list(run_id)
        }
        required = (
            self.policy.required_section_ids
            if self.policy.required_section_ids
            else tuple(sorted(sections))
        )
        complete: list[str] = []
        gaps: list[str] = []
        for section_id in required:
            section = sections.get(section_id)
            if (
                section is not None
                and section.coverage_status == SectionCoverageStatus.COMPLETE
                and section.coverage_score
                >= self.policy.minimum_section_coverage
                and section.citation_score
                >= self.policy.minimum_citation_coverage
            ):
                complete.append(section_id)
            else:
                gaps.append(section_id)

        blocked = (
            tuple(
                item.claim_id
                for item in self.evidence.verified.blocked_high_impact_claims(
                    run_id
                )
            )
            if self.policy.require_no_high_impact_blockers
            else ()
        )
        severe = (
            tuple(
                item.conflict_id
                for item in repository.conflicts.list(run_id)
                if item.status != ConflictStatus.RESOLVED
                and item.severity
                in {ConflictSeverity.HIGH, ConflictSeverity.CRITICAL}
            )
            if self.policy.stop_on_any_severe_conflict
            else ()
        )

        active: list[str] = []
        pending: list[str] = []
        failed: list[str] = []
        waiting: list[str] = []
        for record in scheduler_snapshot.tasks:
            if record.task_id == root_task_id:
                if (
                    record.envelope.status
                    == TaskStatus.WAITING_APPROVAL
                ):
                    waiting.append(record.task_id)
                continue
            status = record.envelope.status
            if status == TaskStatus.RUNNING:
                active.append(record.task_id)
            elif status in {
                TaskStatus.PENDING,
                TaskStatus.READY,
                TaskStatus.PAUSED,
                TaskStatus.DEFERRED,
            }:
                pending.append(record.task_id)
            elif status == TaskStatus.FAILED:
                failed.append(record.task_id)
            elif status == TaskStatus.WAITING_APPROVAL:
                waiting.append(record.task_id)

        usage = combine_usage(
            record.budget_usage for record in scheduler_snapshot.tasks
        )
        budget_exhausted = bool(
            self.policy.run_budget.exceeded_dimensions(
                usage,
                now=self.clock(),
            )
        )
        prior = self.coordination.convergence_decisions(run_id)
        previous_low_gain = (
            prior[-1].snapshot.low_gain_cycles
            if prior and prior[-1].cycle == cycle - 1
            else 0
        )
        low_gain_cycles = (
            previous_low_gain + 1
            if latest_information_gain <= self.policy.low_gain_threshold
            else 0
        )
        cancelled = (
            scheduler_snapshot.control.status
            == RunControlStatus.CANCELLED
        )
        snapshot = ConvergenceSnapshot(
            run_id=run_id,
            cycle=cycle,
            required_section_ids=required,
            complete_section_ids=tuple(complete),
            coverage_gap_section_ids=tuple(gaps),
            blocked_high_impact_claim_ids=blocked,
            severe_conflict_ids=severe,
            active_task_ids=tuple(active),
            pending_task_ids=tuple(pending),
            failed_task_ids=tuple(failed),
            waiting_approval_task_ids=tuple(waiting),
            low_gain_cycles=low_gain_cycles,
            latest_information_gain=max(0.0, latest_information_gain),
            budget_usage=usage,
            budget_exhausted=budget_exhausted,
            cancelled=cancelled,
            captured_at=self.clock(),
        )
        action, reasons = self._decide(snapshot)
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "snapshot": snapshot.model_dump(mode="json"),
                    "action": action.value,
                    "reasons": reasons,
                },
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        decision_id = _stable_id(
            "convergence",
            run_id,
            str(cycle),
            fingerprint,
        )
        artifact_id = _stable_id("artifact", decision_id)
        decision = ConvergenceDecision(
            decision_id=decision_id,
            run_id=run_id,
            cycle=cycle,
            action=action,
            reasons=reasons,
            snapshot=snapshot,
            decision_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        self.artifact_store.put_json(
            {
                "schema": "ConvergenceDecision@1",
                "decision": decision.model_dump(mode="json"),
                "policy": self.policy.model_dump(mode="json"),
            },
            redact=False,
            kind=ArtifactKind.CONVERGENCE_DECISION,
            producer_id=self.producer_id,
            run_id=run_id,
            content_schema="ConvergenceDecision@1",
            artifact_id=artifact_id,
            idempotency_key=f"research-convergence:{run_id}:{cycle}",
            metadata={
                "cycle": cycle,
                "action": action.value,
            },
        )
        self.coordination.save_convergence(decision)
        return decision

    def _decide(
        self,
        snapshot: ConvergenceSnapshot,
    ) -> tuple[ConvergenceAction, tuple[str, ...]]:
        if snapshot.cancelled:
            return (
                ConvergenceAction.CANCEL,
                ("The durable scheduler run is cancelled.",),
            )
        if snapshot.budget_exhausted:
            return (
                ConvergenceAction.STOP_BUDGET,
                ("The authoritative run budget is exhausted.",),
            )
        if snapshot.waiting_approval_task_ids:
            return (
                ConvergenceAction.AWAIT_APPROVAL,
                (
                    "Research is waiting for an explicit approval decision "
                    "before it may continue.",
                ),
            )
        if snapshot.active_task_ids or snapshot.pending_task_ids:
            if snapshot.cycle + 1 >= self.policy.max_cycles:
                return (
                    ConvergenceAction.STOP_MAX_CYCLES,
                    ("The bounded supervisor cycle limit was reached.",),
                )
            return (
                ConvergenceAction.CONTINUE,
                ("Scheduled research work remains active or runnable.",),
            )
        semantically_complete = not (
            snapshot.coverage_gap_section_ids
            or snapshot.blocked_high_impact_claim_ids
            or snapshot.severe_conflict_ids
            or snapshot.failed_task_ids
        )
        if semantically_complete:
            return (
                ConvergenceAction.COMPLETE,
                (
                    "All required section, citation, high-impact claim, "
                    "conflict, and task gates passed.",
                ),
            )
        if snapshot.cycle + 1 >= self.policy.max_cycles:
            return (
                ConvergenceAction.STOP_MAX_CYCLES,
                ("The bounded supervisor cycle limit was reached.",),
            )
        if snapshot.low_gain_cycles >= self.policy.max_low_gain_cycles:
            return (
                ConvergenceAction.STOP_LOW_GAIN,
                (
                    "Repeated research cycles remained below the configured "
                    "information-gain threshold.",
                ),
            )
        return (
            ConvergenceAction.REPLAN,
            (
                "Semantic evidence gaps, blocking claims, severe conflicts, "
                "or failed work require dynamic replanning.",
            ),
        )


class ResearchSupervisorRunner:
    """Runs one leased Supervisor turn and preserves the root as a control task."""

    def __init__(
        self,
        *,
        worker_id: str,
        agent_spec_id: str,
        kernel: AgentKernel,
        scheduler: Scheduler,
        artifact_store: ArtifactStore,
    ) -> None:
        self.worker_id = worker_id
        self.agent_spec_id = agent_spec_id
        self.kernel = kernel
        self.scheduler = scheduler
        self.artifact_store = artifact_store

    async def run_lease(
        self,
        lease: TaskLease,
        *,
        cancellation: CancellationToken | None = None,
    ) -> KernelRunResult:
        if lease.worker_id != self.worker_id:
            raise ValueError("Supervisor runner received another worker's lease")
        kernel_task = lease.task.model_copy(
            update={"assigned_actor_id": self.agent_spec_id}
        )
        result = await self.kernel.run(
            agent_spec_id=self.agent_spec_id,
            task=kernel_task,
            cancellation=cancellation,
        )
        task_result = result.task_result
        prefix = _stable_id(
            "mutation",
            lease.task.run_id,
            lease.task.task_id,
            str(lease.task.attempt),
            task_result.result_id,
        )
        approval_observation = next(
            (
                item
                for item in result.observations
                if item.normalized_data.get("request_approval") is True
            ),
            None,
        )
        if approval_observation is not None:
            usage_record = await self.scheduler.update_usage(
                lease.task.task_id,
                task_result.usage,
                worker_id=self.worker_id,
                mutation_id=f"{prefix}_usage",
            )
            if usage_record.envelope.status == TaskStatus.FAILED:
                return result
            reason = str(
                approval_observation.normalized_data.get(
                    "approval_reason",
                    "Supervisor requires approval.",
                )
            )
            await self.scheduler.request_approval(
                lease.task.task_id,
                worker_id=self.worker_id,
                reason=reason,
                mutation_id=f"{prefix}_approval",
            )
        elif task_result.status in {
            TaskResultStatus.SUCCEEDED,
            TaskResultStatus.PARTIAL,
        }:
            allow_stop = bool(
                lease.task.constraints.get("supervisor_context", {}).get(
                    "allow_stop",
                    False,
                )
            )
            if (
                result.stop_decision.reason.value == "semantic_complete"
                and allow_stop
            ):
                await self.scheduler.complete(
                    lease.task.task_id,
                    TaskCompletion(
                        result_id=task_result.result_id,
                        output_artifact_ids=task_result.output_artifact_ids,
                        usage=task_result.usage,
                    ),
                    worker_id=self.worker_id,
                    mutation_id=f"{prefix}_complete",
                )
            else:
                usage_record = await self.scheduler.update_usage(
                    lease.task.task_id,
                    task_result.usage,
                    worker_id=self.worker_id,
                    mutation_id=f"{prefix}_usage",
                )
                if usage_record.envelope.status == TaskStatus.FAILED:
                    return result
                await self.scheduler.pause_task(
                    lease.task.task_id,
                    worker_id=self.worker_id,
                    reason="Supervisor turn completed; workers own execution.",
                    mutation_id=f"{prefix}_pause",
                )
        elif task_result.status == TaskResultStatus.CANCELLED:
            usage_record = await self.scheduler.update_usage(
                lease.task.task_id,
                task_result.usage,
                worker_id=self.worker_id,
                mutation_id=f"{prefix}_usage",
            )
            if usage_record.envelope.status == TaskStatus.FAILED:
                return result
            await self.scheduler.cancel_task(
                lease.task.task_id,
                actor_id=self.worker_id,
                reason=task_result.summary,
                mutation_id=f"{prefix}_cancel",
            )
        elif task_result.status == TaskResultStatus.DEFERRED:
            usage_record = await self.scheduler.update_usage(
                lease.task.task_id,
                task_result.usage,
                worker_id=self.worker_id,
                mutation_id=f"{prefix}_usage",
            )
            if usage_record.envelope.status == TaskStatus.FAILED:
                return result
            approval_command = next(
                (
                    item
                    for item in result.commands
                    if item.kind.value == "request_approval"
                ),
                None,
            )
            requested_reason = next(
                (
                    str(item.arguments.get("reason"))
                    for item in result.commands
                    if item.kind.value == "request_approval"
                    and item.arguments.get("reason")
                ),
                task_result.summary,
            )
            if approval_command is not None:
                plan = approval_command.arguments.get("plan")
                plan_id = (
                    str(plan.get("plan_id"))
                    if isinstance(plan, dict) and plan.get("plan_id")
                    else _stable_id(
                        "plan",
                        lease.task.run_id,
                        lease.task.task_id,
                        str(lease.task.attempt),
                        "approval",
                    )
                )
                self.artifact_store.put_json(
                    {
                        "schema": "SupervisorApprovalRequest@1",
                        "plan": plan,
                    },
                    redact=False,
                    kind=ArtifactKind.SUPERVISOR_PLAN,
                    producer_id=self.agent_spec_id,
                    run_id=lease.task.run_id,
                    task_id=lease.task.task_id,
                    content_schema="SupervisorApprovalRequest@1",
                    artifact_id=_stable_id("artifact", plan_id),
                    idempotency_key=f"supervisor-approval:{plan_id}",
                )
            await self.scheduler.request_approval(
                lease.task.task_id,
                worker_id=self.worker_id,
                reason=requested_reason,
                mutation_id=f"{prefix}_deferred",
            )
        else:
            assert task_result.error is not None
            await self.scheduler.fail(
                lease.task.task_id,
                error_ref=task_result.error.error_id,
                worker_id=self.worker_id,
                usage=task_result.usage,
                mutation_id=f"{prefix}_fail",
            )
            if (
                task_result.error.retryable
                and lease.task.attempt < lease.task.max_attempts
            ):
                await self.scheduler.retry(
                    lease.task.task_id,
                    actor_id=self.worker_id,
                    mutation_id=f"{prefix}_retry",
                )
        return result


class ResearchCoordinator:
    """Durable Supervisor/Worker research loop; report writing is out of scope."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        supervisor: ResearchSupervisorRunner,
        worker_pool: ResearchWorkerPool,
        reconciler: ResearchWorkerResultReconciler,
        merger: CrossWorkerResultMerger,
        convergence: ConvergenceEvaluator,
        evidence: EvidenceRuntime,
        actor_id: str = "runtime_research_coordinator",
        supervisor_lease_seconds: float = 120.0,
        finalize_scheduler_run: bool = True,
        clock=utc_now,
    ) -> None:
        if supervisor_lease_seconds <= 0:
            raise ValueError("Supervisor lease duration must be positive")
        self.scheduler = scheduler
        self.supervisor = supervisor
        self.worker_pool = worker_pool
        self.reconciler = reconciler
        self.merger = merger
        self.convergence = convergence
        self.evidence = evidence
        self.actor_id = actor_id
        self.supervisor_lease_seconds = supervisor_lease_seconds
        self.finalize_scheduler_run = finalize_scheduler_run
        self.clock = clock

    async def run(
        self,
        root_task: TaskEnvelope,
        *,
        max_concurrency: int,
        cancellation: CancellationToken | None = None,
    ) -> ResearchRunOutcome:
        if root_task.parent_task_id is not None:
            raise ValueError("Research root task cannot have a parent")
        token = cancellation or CancellationToken()
        root_task = root_task.model_copy(
            update={
                "assigned_actor_id": self.supervisor.worker_id,
                "constraints": self._supervisor_constraints(
                    root_task,
                    cycle=0,
                    previous=None,
                ),
            }
        )
        await self._ensure_run(root_task, max_concurrency=max_concurrency)
        prior = self.convergence.coordination.convergence_decisions(
            root_task.run_id
        )
        if prior and prior[-1].action == ConvergenceAction.AWAIT_APPROVAL:
            approval_snapshot = await self.scheduler.snapshot(
                root_task.run_id
            )
            if any(
                item.envelope.status == TaskStatus.WAITING_APPROVAL
                for item in approval_snapshot.tasks
            ):
                return await self._outcome(prior[-1])
        if prior and prior[-1].action in {
            ConvergenceAction.COMPLETE,
            ConvergenceAction.STOP_LOW_GAIN,
            ConvergenceAction.STOP_BUDGET,
            ConvergenceAction.STOP_MAX_CYCLES,
            ConvergenceAction.CANCEL,
        }:
            return await self._finish_or_report(root_task, prior[-1])

        cycle = prior[-1].cycle + 1 if prior else 0
        if prior and prior[-1].action in {
            ConvergenceAction.REPLAN,
            ConvergenceAction.CONTINUE,
        }:
            await self._prepare_supervisor(root_task, cycle, prior[-1])

        while True:
            if token.cancelled:
                self.worker_pool.cancel_active()
                snapshot = await self.scheduler.snapshot(root_task.run_id)
                if snapshot.control.status == RunControlStatus.ACTIVE:
                    await self.scheduler.cancel_run(
                        root_task.run_id,
                        actor_id=self.actor_id,
                        reason="Research was cancelled by the caller.",
                        mutation_id=_stable_id(
                            "mutation",
                            root_task.run_id,
                            "caller_cancel",
                        ),
                    )
                decision = self.convergence.assess(
                    scheduler_snapshot=await self.scheduler.snapshot(
                        root_task.run_id
                    ),
                    root_task_id=root_task.task_id,
                    cycle=cycle,
                    latest_information_gain=0.0,
                )
                return await self._outcome(decision)

            await self.scheduler.recover(
                root_task.run_id,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    root_task.run_id,
                    str(cycle),
                    self.clock().isoformat(),
                    "recover",
                ),
            )
            await self.reconciler.reconcile(root_task.run_id)
            snapshot = await self.scheduler.snapshot(root_task.run_id)
            root_record = snapshot.by_id[root_task.task_id]
            if root_record.envelope.status == TaskStatus.READY:
                leases = await self.scheduler.claim(
                    root_task.run_id,
                    worker_id=self.supervisor.worker_id,
                    limit=1,
                    lease_seconds=self.supervisor_lease_seconds,
                    mutation_id=_stable_id(
                        "mutation",
                        root_task.run_id,
                        str(cycle),
                        "claim_supervisor",
                    ),
                )
                root_leases = tuple(
                    item
                    for item in leases
                    if item.task.task_id == root_task.task_id
                )
                if len(root_leases) != 1:
                    raise RuntimeError(
                        "Supervisor could not claim its assigned root task"
                    )
                await self.supervisor.run_lease(
                    root_leases[0],
                    cancellation=token,
                )

            cycle_results = await self.worker_pool.drain(root_task.run_id)
            merged = self.merger.merge(root_task.run_id)
            await self.evidence.engine.verify_run(
                root_task.run_id,
                task_id=root_task.task_id,
                repair_round=cycle,
            )
            latest_gain = sum(
                item.information_gain for item in cycle_results
            )
            decision = self.convergence.assess(
                scheduler_snapshot=await self.scheduler.snapshot(
                    root_task.run_id
                ),
                root_task_id=root_task.task_id,
                cycle=cycle,
                latest_information_gain=latest_gain,
            )
            if decision.action in {
                ConvergenceAction.COMPLETE,
                ConvergenceAction.STOP_LOW_GAIN,
                ConvergenceAction.STOP_BUDGET,
                ConvergenceAction.STOP_MAX_CYCLES,
                ConvergenceAction.CANCEL,
                ConvergenceAction.AWAIT_APPROVAL,
            }:
                return await self._finish_or_report(
                    root_task,
                    decision,
                    merge_artifact_id=(
                        merged.merge_artifact_id
                        if merged is not None
                        else None
                    ),
                )
            cycle += 1
            if decision.action == ConvergenceAction.REPLAN:
                await self._prepare_supervisor(root_task, cycle, decision)

    async def _ensure_run(
        self,
        root_task: TaskEnvelope,
        *,
        max_concurrency: int,
    ) -> None:
        try:
            snapshot = await self.scheduler.snapshot(root_task.run_id)
        except KeyError:
            await self.scheduler.create_run(
                root_task.run_id,
                max_concurrency=max_concurrency,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    root_task.run_id,
                    "create",
                ),
            )
            await self.scheduler.submit(
                root_task,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    root_task.run_id,
                    root_task.task_id,
                    "submit_root",
                ),
            )
            return
        if root_task.task_id not in snapshot.by_id:
            raise ValueError(
                "Existing scheduler run does not contain the requested root task"
            )

    async def _prepare_supervisor(
        self,
        root_task: TaskEnvelope,
        cycle: int,
        previous: ConvergenceDecision,
    ) -> None:
        snapshot = await self.scheduler.snapshot(root_task.run_id)
        record = snapshot.by_id[root_task.task_id]
        if record.envelope.status == TaskStatus.PAUSED:
            await self.scheduler.edit(
                root_task.task_id,
                TaskEdit(
                    constraints=self._supervisor_constraints(
                        record.envelope,
                        cycle=cycle,
                        previous=previous,
                    )
                ),
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    root_task.run_id,
                    str(cycle),
                    "edit_supervisor_context",
                ),
            )
            await self.scheduler.resume_task(
                root_task.task_id,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    root_task.run_id,
                    str(cycle),
                    "resume_supervisor",
                ),
            )

    def _supervisor_constraints(
        self,
        root_task: TaskEnvelope,
        *,
        cycle: int,
        previous: ConvergenceDecision | None,
    ) -> dict[str, Any]:
        context = {
            "cycle": cycle,
            "objective": root_task.goal,
            "allow_stop": False,
            "max_tasks_per_plan": 32,
            "allowed_worker_ids": list(self.worker_pool.runners),
        }
        if previous is not None:
            context.update(
                {
                    "previous_action": previous.action.value,
                    "previous_reasons": list(previous.reasons),
                    "convergence_snapshot": previous.snapshot.model_dump(
                        mode="json"
                    ),
                }
            )
        return {
            **root_task.constraints,
            "supervisor_context": context,
            "research_boundary": {
                "report_writing": False,
                "provider_access_for_supervisor": False,
                "free_form_agent_chat": False,
            },
        }

    async def _finish_or_report(
        self,
        root_task: TaskEnvelope,
        decision: ConvergenceDecision,
        *,
        merge_artifact_id: str | None = None,
    ) -> ResearchRunOutcome:
        if decision.action == ConvergenceAction.AWAIT_APPROVAL:
            return await self._outcome(
                decision,
                merge_artifact_id=merge_artifact_id,
            )
        snapshot = await self.scheduler.snapshot(root_task.run_id)
        if decision.action in {
            ConvergenceAction.STOP_BUDGET,
            ConvergenceAction.CANCEL,
        }:
            if snapshot.control.status == RunControlStatus.ACTIVE:
                await self.scheduler.cancel_run(
                    root_task.run_id,
                    actor_id=self.actor_id,
                    reason="; ".join(decision.reasons),
                    mutation_id=_stable_id(
                        "mutation",
                        root_task.run_id,
                        decision.decision_id,
                        "cancel_run",
                    ),
                )
            return await self._outcome(
                decision,
                merge_artifact_id=merge_artifact_id,
            )

        if decision.action in {
            ConvergenceAction.STOP_LOW_GAIN,
            ConvergenceAction.STOP_MAX_CYCLES,
        }:
            for record in snapshot.tasks:
                if (
                    record.task_id != root_task.task_id
                    and record.envelope.status not in _TERMINAL_TASK_STATUSES
                ):
                    await self.scheduler.cancel_task(
                        record.task_id,
                        actor_id=self.actor_id,
                        reason="; ".join(decision.reasons),
                        mutation_id=_stable_id(
                            "mutation",
                            decision.decision_id,
                            record.task_id,
                            "cancel_task",
                        ),
                    )

        await self._complete_root(root_task, decision)
        current = await self.scheduler.snapshot(root_task.run_id)
        if (
            self.finalize_scheduler_run
            and current.control.status == RunControlStatus.ACTIVE
        ):
            await self.scheduler.complete_run(
                root_task.run_id,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    decision.decision_id,
                    "complete_run",
                ),
            )
        return await self._outcome(
            decision,
            merge_artifact_id=merge_artifact_id,
        )

    async def _complete_root(
        self,
        root_task: TaskEnvelope,
        decision: ConvergenceDecision,
    ) -> None:
        snapshot = await self.scheduler.snapshot(root_task.run_id)
        record = snapshot.by_id[root_task.task_id]
        if record.envelope.status == TaskStatus.COMPLETED:
            return
        if record.envelope.status == TaskStatus.PAUSED:
            await self.scheduler.resume_task(
                root_task.task_id,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    decision.decision_id,
                    "resume_root_for_completion",
                ),
            )
        leases = await self.scheduler.claim(
            root_task.run_id,
            worker_id=self.supervisor.worker_id,
            limit=1,
            lease_seconds=self.supervisor_lease_seconds,
            mutation_id=_stable_id(
                "mutation",
                decision.decision_id,
                "claim_root_for_completion",
            ),
        )
        lease = next(
            (
                item
                for item in leases
                if item.task.task_id == root_task.task_id
            ),
            None,
        )
        if lease is None:
            raise RuntimeError("Root task could not be leased for completion")
        assert decision.decision_artifact_id is not None
        await self.scheduler.complete(
            root_task.task_id,
            TaskCompletion(
                result_id=_stable_id(
                    "result",
                    decision.decision_id,
                    "research_complete",
                ),
                output_artifact_ids=(decision.decision_artifact_id,),
                usage=BudgetUsage(),
            ),
            worker_id=self.supervisor.worker_id,
            mutation_id=_stable_id(
                "mutation",
                decision.decision_id,
                "complete_root",
            ),
        )

    async def _outcome(
        self,
        decision: ConvergenceDecision,
        *,
        merge_artifact_id: str | None = None,
    ) -> ResearchRunOutcome:
        snapshot = await self.scheduler.snapshot(decision.run_id)
        assert decision.decision_artifact_id is not None
        merge_ids = tuple(
            dict.fromkeys(
                (
                    *(
                        item.merge_artifact_id
                        for item in self.convergence.coordination.merges(
                            decision.run_id
                        )
                    ),
                    *((merge_artifact_id,) if merge_artifact_id else ()),
                )
            )
        )
        return ResearchRunOutcome(
            run_id=decision.run_id,
            action=decision.action,
            cycles=decision.cycle + 1,
            decision_artifact_id=decision.decision_artifact_id,
            merged_result_artifact_ids=merge_ids,
            completed_task_ids=tuple(
                item.task_id
                for item in snapshot.tasks
                if item.envelope.status == TaskStatus.COMPLETED
            ),
            failed_task_ids=tuple(
                item.task_id
                for item in snapshot.tasks
                if item.envelope.status == TaskStatus.FAILED
            ),
            waiting_approval_task_ids=tuple(
                item.task_id
                for item in snapshot.tasks
                if item.envelope.status == TaskStatus.WAITING_APPROVAL
            ),
            usage=combine_usage(
                item.budget_usage for item in snapshot.tasks
            ),
            completed_at=self.clock(),
        )
