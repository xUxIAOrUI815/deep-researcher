from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import Budget
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.kernel import (
    AgentKernel,
    AgentSpecRegistry,
    KernelConfig,
    KernelEventSink,
    ModelAdapter,
)
from deep_researcher.orchestration import Scheduler
from deep_researcher.research import ResearchWorkerPool

from .evidence import VerifiedWriterPacketBuilder
from .loop import (
    ReportLoopCoordinator,
    SchedulerTargetedResearchDispatcher,
)
from .models import ReportLoopPolicy
from .reviewer import (
    ReportReviewerActionExecutor,
    ReportReviewerModelAdapter,
    ReportReviewerRunner,
    ReviewerDecisionVerifier,
)
from .specs import build_report_reviewer_spec, build_synthesis_writer_spec
from .store import SQLiteReportingStore
from .writer import (
    SynthesisWriterActionExecutor,
    SynthesisWriterModelAdapter,
    SynthesisWriterRunner,
    WriterRevisionVerifier,
)


@dataclass
class ReportingRuntime:
    """Complete Background001 Writer/Reviewer runtime and durable report loop."""

    registry: AgentSpecRegistry
    store: SQLiteReportingStore
    packet_builder: VerifiedWriterPacketBuilder
    writer_kernel: AgentKernel
    reviewer_kernel: AgentKernel
    writer: SynthesisWriterRunner
    reviewer: ReportReviewerRunner
    loop: ReportLoopCoordinator
    targeted_research: SchedulerTargetedResearchDispatcher | None
    evidence: EvidenceRuntime

    def integrity_check(self) -> None:
        self.store.integrity_check()
        self.evidence.integrity_check()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ReportingRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_reporting_runtime(
    root: str | Path,
    *,
    evidence: EvidenceRuntime,
    writer_model: ModelAdapter,
    reviewer_model: ModelAdapter,
    event_sink: KernelEventSink,
    policy: ReportLoopPolicy,
    artifact_store: ArtifactStore | None = None,
    scheduler: Scheduler | None = None,
    worker_pool: ResearchWorkerPool | None = None,
    targeted_research_task_budget: Budget | None = None,
    writer_budget: Budget | None = None,
    reviewer_budget: Budget | None = None,
    writer_kernel_config: KernelConfig | None = None,
    reviewer_kernel_config: KernelConfig | None = None,
) -> ReportingRuntime:
    if event_sink is None:
        raise ValueError("A durable KernelEventSink is required")
    if (scheduler is None) != (worker_pool is None):
        raise ValueError(
            "scheduler and worker_pool must be configured together"
        )
    if scheduler is not None and targeted_research_task_budget is None:
        raise ValueError(
            "targeted research requires an explicit per-task budget"
        )
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    artifacts = artifact_store or evidence.knowledge.artifacts
    store = SQLiteReportingStore(root_path / "reporting.sqlite3")
    registry = AgentSpecRegistry()
    writer_spec = registry.register(build_synthesis_writer_spec())
    reviewer_spec = registry.register(build_report_reviewer_spec())
    packet_builder = VerifiedWriterPacketBuilder(
        evidence=evidence,
        artifact_store=artifacts,
    )

    writer_executor = SynthesisWriterActionExecutor(
        evidence=evidence,
        artifact_store=artifacts,
        reporting_store=store,
        policy=policy,
        actor_id=writer_spec.agent_spec_id,
    )
    writer_kernel = AgentKernel(
        registry=registry,
        model_adapter=SynthesisWriterModelAdapter(
            writer_model,
            artifact_store=artifacts,
        ),
        action_executor=writer_executor,
        verifier=WriterRevisionVerifier(
            reporting_store=store,
            artifact_store=artifacts,
        ),
        event_sink=event_sink,
        config=writer_kernel_config,
    )
    writer = SynthesisWriterRunner(
        agent_spec_id=writer_spec.agent_spec_id,
        kernel=writer_kernel,
        reporting_store=store,
    )

    reviewer_executor = ReportReviewerActionExecutor(
        evidence=evidence,
        artifact_store=artifacts,
        reporting_store=store,
        policy=policy,
        actor_id=reviewer_spec.agent_spec_id,
    )
    reviewer_kernel = AgentKernel(
        registry=registry,
        model_adapter=ReportReviewerModelAdapter(
            reviewer_model,
            artifact_store=artifacts,
            reporting_store=store,
        ),
        action_executor=reviewer_executor,
        verifier=ReviewerDecisionVerifier(
            reporting_store=store,
            artifact_store=artifacts,
        ),
        event_sink=event_sink,
        config=reviewer_kernel_config,
    )
    reviewer = ReportReviewerRunner(
        agent_spec_id=reviewer_spec.agent_spec_id,
        kernel=reviewer_kernel,
        reporting_store=store,
    )

    targeted = (
        SchedulerTargetedResearchDispatcher(
            scheduler=scheduler,
            worker_pool=worker_pool,
            evidence=evidence,
            packet_builder=packet_builder,
            research_task_budget=targeted_research_task_budget,
        )
        if scheduler is not None
        and worker_pool is not None
        and targeted_research_task_budget is not None
        else None
    )
    loop = ReportLoopCoordinator(
        evidence=evidence,
        artifact_store=artifacts,
        reporting_store=store,
        packet_builder=packet_builder,
        writer=writer,
        reviewer=reviewer,
        policy=policy,
        writer_budget=writer_budget or writer_spec.default_budget,
        reviewer_budget=reviewer_budget or reviewer_spec.default_budget,
        targeted_research=targeted,
    )
    runtime = ReportingRuntime(
        registry=registry,
        store=store,
        packet_builder=packet_builder,
        writer_kernel=writer_kernel,
        reviewer_kernel=reviewer_kernel,
        writer=writer,
        reviewer=reviewer,
        loop=loop,
        targeted_research=targeted,
        evidence=evidence,
    )
    runtime.integrity_check()
    return runtime
