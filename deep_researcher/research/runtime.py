from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.kernel import (
    AgentKernel,
    AgentSpecRegistry,
    KernelConfig,
    KernelEventSink,
    ModelAdapter,
)
from deep_researcher.orchestration import Scheduler

from .models import ConvergencePolicy
from .planning import (
    SupervisorActionExecutor,
    SupervisorPlanningModelAdapter,
    SupervisorPlanVerifier,
)
from .specs import (
    build_research_supervisor_spec,
    build_research_worker_spec,
)
from .store import SQLiteResearchCoordinationStore
from .supervisor import (
    ConvergenceEvaluator,
    ResearchCoordinator,
    ResearchSupervisorRunner,
)
from .worker import (
    CommandExecutionBoundary,
    CrossWorkerResultMerger,
    InformationGainEstimator,
    ResearchWorkerActionExecutor,
    ResearchWorkerPool,
    ResearchWorkerResultReconciler,
    ResearchWorkerRunner,
    WorkerObservationVerifier,
)


@dataclass
class ResearchRuntime:
    """Composed Background001 research runtime for one durable deployment."""

    registry: AgentSpecRegistry
    coordination: SQLiteResearchCoordinationStore
    supervisor_kernel: AgentKernel
    worker_kernel: AgentKernel
    supervisor: ResearchSupervisorRunner
    worker_pool: ResearchWorkerPool
    reconciler: ResearchWorkerResultReconciler
    merger: CrossWorkerResultMerger
    convergence: ConvergenceEvaluator
    coordinator: ResearchCoordinator

    def integrity_check(self) -> None:
        self.coordination.integrity_check()
        self.convergence.artifact_store.integrity_check()
        self.convergence.evidence.integrity_check()

    def close(self) -> None:
        self.coordination.close()

    def __enter__(self) -> "ResearchRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_research_runtime(
    root: str | Path,
    *,
    scheduler: Scheduler,
    evidence: EvidenceRuntime,
    supervisor_model: ModelAdapter,
    worker_model: ModelAdapter,
    command_executor: CommandExecutionBoundary,
    event_sink: KernelEventSink,
    convergence_policy: ConvergencePolicy,
    supervisor_worker_id: str = "worker_research_supervisor",
    worker_ids: tuple[str, ...] = (
        "worker_research_1",
        "worker_research_2",
        "worker_research_3",
        "worker_research_4",
    ),
    artifact_store: ArtifactStore | None = None,
    supervisor_kernel_config: KernelConfig | None = None,
    worker_kernel_config: KernelConfig | None = None,
    lease_seconds: float = 120.0,
    max_claim_rounds: int = 100,
    finalize_scheduler_run: bool = True,
) -> ResearchRuntime:
    """Build the full pool without introducing implicit provider/event fallbacks."""
    if not worker_ids or len(worker_ids) != len(set(worker_ids)):
        raise ValueError("worker_ids must be a non-empty unique tuple")
    if supervisor_worker_id in worker_ids:
        raise ValueError("Supervisor worker ID must be outside the worker pool")
    if event_sink is None:
        raise ValueError("A durable KernelEventSink is required")
    artifacts = artifact_store or evidence.knowledge.artifacts
    coordination_root = Path(root)
    coordination_root.mkdir(parents=True, exist_ok=True)
    coordination = SQLiteResearchCoordinationStore(
        coordination_root / "research_coordination.sqlite3"
    )
    registry = AgentSpecRegistry()
    supervisor_spec = registry.register(build_research_supervisor_spec())
    worker_spec = registry.register(build_research_worker_spec())

    supervisor_executor = SupervisorActionExecutor(
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
        actor_id=supervisor_spec.agent_spec_id,
        allowed_worker_ids=worker_ids,
    )
    supervisor_kernel = AgentKernel(
        registry=registry,
        model_adapter=SupervisorPlanningModelAdapter(supervisor_model),
        action_executor=supervisor_executor,
        verifier=SupervisorPlanVerifier(),
        event_sink=event_sink,
        config=supervisor_kernel_config,
    )
    worker_executor = ResearchWorkerActionExecutor(
        command_executor=command_executor,
        scheduler=scheduler,
        artifact_store=artifacts,
        coordination=coordination,
        allowed_worker_ids=worker_ids,
    )
    gain = InformationGainEstimator(coordination)
    worker_verifier = WorkerObservationVerifier(gain)
    worker_kernel = AgentKernel(
        registry=registry,
        model_adapter=worker_model,
        action_executor=worker_executor,
        verifier=worker_verifier,
        event_sink=event_sink,
        config=worker_kernel_config,
    )
    supervisor = ResearchSupervisorRunner(
        worker_id=supervisor_worker_id,
        agent_spec_id=supervisor_spec.agent_spec_id,
        kernel=supervisor_kernel,
        scheduler=scheduler,
        artifact_store=artifacts,
    )
    runners = {
        worker_id: ResearchWorkerRunner(
            worker_id=worker_id,
            agent_spec_id=worker_spec.agent_spec_id,
            kernel=worker_kernel,
            verifier=worker_verifier,
            scheduler=scheduler,
            artifact_store=artifacts,
            coordination=coordination,
        )
        for worker_id in worker_ids
    }
    pool = ResearchWorkerPool(
        scheduler=scheduler,
        runners=runners,
        lease_seconds=lease_seconds,
        max_claim_rounds=max_claim_rounds,
    )
    reconciler = ResearchWorkerResultReconciler(
        scheduler=scheduler,
        coordination=coordination,
    )
    merger = CrossWorkerResultMerger(
        artifact_store=artifacts,
        coordination=coordination,
    )
    convergence = ConvergenceEvaluator(
        evidence=evidence,
        artifact_store=artifacts,
        coordination=coordination,
        policy=convergence_policy,
    )
    coordinator = ResearchCoordinator(
        scheduler=scheduler,
        supervisor=supervisor,
        worker_pool=pool,
        reconciler=reconciler,
        merger=merger,
        convergence=convergence,
        evidence=evidence,
        supervisor_lease_seconds=lease_seconds,
        finalize_scheduler_run=finalize_scheduler_run,
    )
    runtime = ResearchRuntime(
        registry=registry,
        coordination=coordination,
        supervisor_kernel=supervisor_kernel,
        worker_kernel=worker_kernel,
        supervisor=supervisor,
        worker_pool=pool,
        reconciler=reconciler,
        merger=merger,
        convergence=convergence,
        coordinator=coordinator,
    )
    runtime.integrity_check()
    return runtime
