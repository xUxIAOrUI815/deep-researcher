from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Protocol

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    Budget,
    BudgetDimension,
    BudgetUsage,
    ReportStatus,
    TaskEnvelope,
    TaskKind,
    TaskStatus,
    combine_usage,
    utc_now,
)
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.kernel import CancellationToken
from deep_researcher.orchestration import RunControlStatus, Scheduler
from deep_researcher.research import ResearchWorkerPool

from .evidence import VerifiedWriterPacketBuilder
from .models import (
    LoopStatus,
    ReportLoopOutcome,
    ReportLoopPolicy,
    ReportRepairAction,
    ReportRevision,
    ReviewActionKind,
    ReviewerDecision,
    WriterEvidencePacket,
)
from .reviewer import ReportReviewerRunner
from .store import SQLiteReportingStore
from .writer import SynthesisWriterRunner


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


class TargetedResearchApprovalRequired(RuntimeError):
    pass


@dataclass(frozen=True)
class TargetedResearchDispatch:
    packet: WriterEvidencePacket
    task_ids: tuple[str, ...]
    usage: BudgetUsage


class TargetedResearchDispatcher(Protocol):
    async def dispatch(
        self,
        *,
        report_id: str,
        packet: WriterEvidencePacket,
        decision: ReviewerDecision,
        round_no: int,
        cancellation: CancellationToken,
    ) -> TargetedResearchDispatch: ...


class SchedulerTargetedResearchDispatcher:
    """Materializes Reviewer research repairs through the real Scheduler/pool."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        worker_pool: ResearchWorkerPool,
        evidence: EvidenceRuntime,
        packet_builder: VerifiedWriterPacketBuilder,
        research_task_budget: Budget,
        actor_id: str = "runtime_report_targeted_research",
        clock=utc_now,
    ) -> None:
        self.scheduler = scheduler
        self.worker_pool = worker_pool
        self.evidence = evidence
        self.packet_builder = packet_builder
        self.research_task_budget = research_task_budget
        self.actor_id = actor_id
        self.clock = clock

    async def dispatch(
        self,
        *,
        report_id: str,
        packet: WriterEvidencePacket,
        decision: ReviewerDecision,
        round_no: int,
        cancellation: CancellationToken,
    ) -> TargetedResearchDispatch:
        if cancellation.cancelled:
            return TargetedResearchDispatch(
                packet=packet,
                task_ids=(),
                usage=BudgetUsage(),
            )
        snapshot = await self.scheduler.snapshot(packet.run_id)
        if snapshot.control.status != RunControlStatus.ACTIVE:
            raise TargetedResearchApprovalRequired(
                "Targeted research requires an active scheduler run; the "
                "current run is terminal and must be explicitly resumed or "
                "forked by the integration layer."
            )
        actions = tuple(
            item
            for item in decision.repair_actions
            if item.kind == ReviewActionKind.TARGETED_RESEARCH
        )
        if not actions:
            raise ValueError(
                "targeted research decision has no targeted research action"
            )
        gaps = {item.claim_id: item for item in packet.gaps}
        tasks: list[TaskEnvelope] = []
        for action_index, action in enumerate(actions):
            claim_ids = action.claim_ids or tuple(
                item.claim_id for item in packet.gaps if item.high_impact
            )
            if claim_ids:
                for claim_id in claim_ids:
                    gap = gaps.get(claim_id)
                    tasks.append(
                        self._task(
                            packet=packet,
                            decision=decision,
                            action=action,
                            discriminator=f"{action_index}:claim:{claim_id}",
                            kind=TaskKind.GAP,
                            title=f"Resolve report evidence gap {claim_id}",
                            goal=(
                                (
                                    f"Find independent, authoritative evidence "
                                    f"for: {gap.statement}. Gap reason: "
                                    f"{gap.reason}"
                                )
                                if gap is not None
                                else (
                                    "Find additional independently verified "
                                    f"support for report claim {claim_id}."
                                )
                            ),
                            round_no=round_no,
                        )
                    )
            for section_id in action.section_ids:
                tasks.append(
                    self._task(
                        packet=packet,
                        decision=decision,
                        action=action,
                        discriminator=(
                            f"{action_index}:section:{section_id}"
                        ),
                        kind=TaskKind.SECTION_SUPPORT,
                        title=f"Repair report section evidence {section_id}",
                        goal=(
                            "Find verified evidence needed to satisfy the "
                            f"Reviewer findings for section {section_id}."
                        ),
                        round_no=round_no,
                    )
                )
        if not tasks:
            action = actions[0]
            tasks.append(
                self._task(
                    packet=packet,
                    decision=decision,
                    action=action,
                    discriminator="generic",
                    kind=TaskKind.GAP,
                    title="Resolve report completeness gap",
                    goal=(
                        "Find authoritative evidence that resolves the "
                        "Reviewer completeness finding."
                    ),
                    round_no=round_no,
                )
            )
        unique = {item.task_id: item for item in tasks}
        for task in unique.values():
            await self.scheduler.submit(
                task,
                actor_id=self.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    decision.review_id,
                    task.task_id,
                    "submit",
                ),
            )
        results = await self.worker_pool.drain(packet.run_id)
        if cancellation.cancelled:
            self.worker_pool.cancel_active()
        await self.evidence.engine.verify_run(
            packet.run_id,
            task_id=next(iter(unique)),
            repair_round=round_no,
        )
        refreshed = self.packet_builder.build(report_id)
        return TargetedResearchDispatch(
            packet=refreshed,
            task_ids=tuple(unique),
            usage=combine_usage(item.usage for item in results),
        )

    def _task(
        self,
        *,
        packet: WriterEvidencePacket,
        decision: ReviewerDecision,
        action: ReportRepairAction,
        discriminator: str,
        kind: TaskKind,
        title: str,
        goal: str,
        round_no: int,
    ) -> TaskEnvelope:
        return TaskEnvelope(
            task_id=_stable_id(
                "task",
                decision.review_id,
                str(round_no),
                discriminator,
            ),
            run_id=packet.run_id,
            kind=kind,
            status=TaskStatus.PENDING,
            title=title,
            goal=goal,
            constraints={
                "report_loop": True,
                "report_id": packet.report_id,
                "review_id": decision.review_id,
                "repair_action": action.model_dump(mode="json"),
                "evidence_packet_artifact_id": packet.packet_artifact_id,
                "candidate_output_only": True,
                "verification_required_before_writing": True,
            },
            input_artifact_ids=(
                (packet.packet_artifact_id,)
                if packet.packet_artifact_id is not None
                else ()
            ),
            expected_output_schema="ResearchWorkerResult@1",
            budget=self.research_task_budget,
            priority=0.98,
            max_attempts=3,
            created_by=self.actor_id,
            tags=(
                "report-repair",
                "targeted-research",
                f"round:{round_no}",
            ),
        )


class ReportLoopCoordinator:
    """Runs the bounded Writer -> Reviewer -> repair state machine."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        artifact_store: ArtifactStore,
        reporting_store: SQLiteReportingStore,
        packet_builder: VerifiedWriterPacketBuilder,
        writer: SynthesisWriterRunner,
        reviewer: ReportReviewerRunner,
        policy: ReportLoopPolicy,
        writer_budget: Budget,
        reviewer_budget: Budget,
        targeted_research: TargetedResearchDispatcher | None = None,
        actor_id: str = "runtime_report_loop",
        clock=utc_now,
    ) -> None:
        self.evidence = evidence
        self.artifact_store = artifact_store
        self.reporting_store = reporting_store
        self.packet_builder = packet_builder
        self.writer = writer
        self.reviewer = reviewer
        self.policy = policy
        self.writer_budget = writer_budget
        self.reviewer_budget = reviewer_budget
        self.targeted_research = targeted_research
        self.actor_id = actor_id
        self.clock = clock

    async def run(
        self,
        report_id: str,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ReportLoopOutcome:
        report = self.evidence.knowledge.repository.reports.require(report_id)
        existing_outcome = self.reporting_store.outcome(
            report.run_id,
            report_id,
        )
        if existing_outcome is not None:
            return existing_outcome
        token = cancellation or CancellationToken()
        packet = self.packet_builder.build(report_id)
        usage = BudgetUsage()
        targeted_rounds = 0
        repair_feedback: tuple[dict[str, object], ...] = ()
        latest_revision: ReportRevision | None = None
        latest_review: ReviewerDecision | None = None

        revisions = self.reporting_store.revisions(report.run_id, report_id)
        reviews = self.reporting_store.reviews(report.run_id, report_id)
        if revisions:
            latest_revision = revisions[-1]
            latest_review = next(
                (
                    item
                    for item in reversed(reviews)
                    if item.revision_id == latest_revision.revision_id
                ),
                None,
            )
            if latest_review is not None:
                repair_feedback = tuple(
                    item.model_dump(mode="json")
                    for item in latest_review.repair_actions
                )

        while True:
            if token.cancelled:
                return self._finish(
                    packet=packet,
                    status=LoopStatus.CANCELLED,
                    usage=usage,
                    revision=latest_revision,
                    review=latest_review,
                    summary="Report loop was cancelled by the caller.",
                )
            exhausted_set = set(
                self.policy.run_budget.exceeded_dimensions(usage)
            )
            if usage.retries == 0:
                exhausted_set.discard(BudgetDimension.RETRIES)
            if usage.errors == 0:
                exhausted_set.discard(BudgetDimension.ERRORS)
            exhausted = tuple(
                item for item in BudgetDimension if item in exhausted_set
            )
            if exhausted:
                return self._finish(
                    packet=packet,
                    status=LoopStatus.BUDGET_EXHAUSTED,
                    usage=usage,
                    revision=latest_revision,
                    review=latest_review,
                    summary=(
                        "Report loop exhausted aggregate budget dimensions: "
                        + ", ".join(item.value for item in exhausted)
                    ),
                )

            if latest_revision is None or latest_review is not None:
                next_revision = (
                    latest_revision.revision + 1
                    if latest_revision is not None
                    else 1
                )
                if next_revision > self.policy.max_revisions:
                    return self._finish(
                        packet=packet,
                        status=LoopStatus.REVISION_EXHAUSTED,
                        usage=usage,
                        revision=latest_revision,
                        review=latest_review,
                        summary=(
                            "Report loop reached the configured revision bound."
                        ),
                    )
                writer_run = await self.writer.write(
                    packet=packet,
                    title=report.title,
                    revision=next_revision,
                    budget=self.writer_budget,
                    repair_feedback=repair_feedback,
                    cancellation=token,
                )
                latest_revision = writer_run.revision
                latest_review = None
                usage = combine_usage((usage, writer_run.usage))
                continue

            reviewer_run = await self.reviewer.review(
                revision=latest_revision,
                budget=self.reviewer_budget,
                cancellation=token,
            )
            latest_review = reviewer_run.decision
            usage = combine_usage((usage, reviewer_run.usage))
            decision = latest_review.decision
            if decision == ReviewActionKind.ACCEPT:
                return self._finish(
                    packet=packet,
                    status=LoopStatus.ACCEPTED,
                    usage=usage,
                    revision=latest_revision,
                    review=latest_review,
                    summary=(
                        "Report passed all rubric and deterministic "
                        "traceability gates."
                    ),
                )
            if decision == ReviewActionKind.REJECT:
                return self._finish(
                    packet=packet,
                    status=LoopStatus.REJECTED,
                    usage=usage,
                    revision=latest_revision,
                    review=latest_review,
                    summary=latest_review.decision_summary,
                )
            repair_feedback = tuple(
                item.model_dump(mode="json")
                for item in latest_review.repair_actions
            )
            if decision == ReviewActionKind.TARGETED_RESEARCH:
                if targeted_rounds >= self.policy.max_targeted_research_rounds:
                    return self._finish(
                        packet=packet,
                        status=LoopStatus.REJECTED,
                        usage=usage,
                        revision=latest_revision,
                        review=latest_review,
                        summary=(
                            "Report still requires targeted research after the "
                            "configured research-repair bound."
                        ),
                    )
                if self.targeted_research is None:
                    return self._finish(
                        packet=packet,
                        status=LoopStatus.APPROVAL_REQUIRED,
                        usage=usage,
                        revision=latest_revision,
                        review=latest_review,
                        summary=(
                            "Reviewer requested targeted research, but no "
                            "governed dispatcher is configured."
                        ),
                    )
                targeted_rounds += 1
                try:
                    dispatched = await self.targeted_research.dispatch(
                        report_id=report_id,
                        packet=packet,
                        decision=latest_review,
                        round_no=targeted_rounds,
                        cancellation=token,
                    )
                except TargetedResearchApprovalRequired as exc:
                    return self._finish(
                        packet=packet,
                        status=LoopStatus.APPROVAL_REQUIRED,
                        usage=usage,
                        revision=latest_revision,
                        review=latest_review,
                        summary=str(exc),
                    )
                packet = dispatched.packet
                usage = combine_usage((usage, dispatched.usage))
            # Citation, local, and structural repair all return to the Writer
            # with typed feedback. The Writer remains unable to search.

    def _finish(
        self,
        *,
        packet: WriterEvidencePacket,
        status: LoopStatus,
        usage: BudgetUsage,
        revision: ReportRevision | None,
        review: ReviewerDecision | None,
        summary: str,
    ) -> ReportLoopOutcome:
        accepted = status == LoopStatus.ACCEPTED
        completed_at = (
            review.created_at
            if review is not None
            else (
                revision.created_at
                if revision is not None
                else packet.created_at
            )
        )
        outcome = ReportLoopOutcome(
            run_id=packet.run_id,
            report_id=packet.report_id,
            status=status,
            revisions=revision.revision if revision is not None else 0,
            final_revision_id=(
                revision.revision_id if revision is not None else None
            ),
            final_report_artifact_id=(
                revision.report_artifact_id if revision is not None else None
            ),
            citation_map_artifact_id=(
                revision.citation_map_artifact_id
                if revision is not None
                else None
            ),
            final_review_id=(
                review.review_id if review is not None else None
            ),
            usage=usage,
            summary=summary,
            completed_at=completed_at,
        )
        artifact_id = _stable_id(
            "artifact",
            packet.run_id,
            packet.report_id,
            status.value,
            revision.revision_id if revision is not None else "none",
            review.review_id if review is not None else "none",
        )
        sources = tuple(
            dict.fromkeys(
                (
                    *(
                        (packet.packet_artifact_id,)
                        if packet.packet_artifact_id is not None
                        else ()
                    ),
                    *(
                        (
                            revision.report_artifact_id,
                            revision.citation_map_artifact_id,
                        )
                        if revision is not None
                        else ()
                    ),
                    *(
                        (review.review_artifact_id,)
                        if review is not None
                        and review.review_artifact_id is not None
                        else ()
                    ),
                )
            )
        )
        self.artifact_store.put_json(
            {
                "schema": "ReportLoopOutcome@1",
                "outcome": outcome.model_dump(mode="json"),
                "accepted": accepted,
                "bounded": {
                    "max_revisions": self.policy.max_revisions,
                    "max_targeted_research_rounds": (
                        self.policy.max_targeted_research_rounds
                    ),
                },
            },
            redact=False,
            kind=ArtifactKind.REPORT_LOOP_DECISION,
            producer_id=self.actor_id,
            run_id=packet.run_id,
            content_schema="ReportLoopOutcome@1",
            source_artifact_ids=sources,
            artifact_id=artifact_id,
            idempotency_key=(
                f"report-loop-outcome:{packet.run_id}:{packet.report_id}"
            ),
        )
        self.reporting_store.save_outcome(outcome)
        if status in {
            LoopStatus.REJECTED,
            LoopStatus.BUDGET_EXHAUSTED,
            LoopStatus.REVISION_EXHAUSTED,
            LoopStatus.CANCELLED,
        }:
            self._fail_domain(packet.report_id)
        return outcome

    def _fail_domain(self, report_id: str) -> None:
        repository = self.evidence.knowledge.repository
        report = repository.reports.require(report_id)
        if report.status == ReportStatus.PUBLISHED:
            return
        if report.status == ReportStatus.APPROVED:
            report = report.transition(ReportStatus.REVISION_REQUIRED)
        if report.status == ReportStatus.DRAFT:
            report = report.transition(ReportStatus.FAILED)
        elif report.status == ReportStatus.VERIFYING:
            report = report.transition(ReportStatus.FAILED)
        elif report.status == ReportStatus.REVISION_REQUIRED:
            report = report.transition(ReportStatus.FAILED)
        repository.reports.save(
            report.model_copy(update={"updated_at": self.clock()})
        )
