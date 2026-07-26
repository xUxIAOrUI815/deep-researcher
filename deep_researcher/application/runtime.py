from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import uuid
from typing import Any

from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    Budget,
    BudgetUsage,
    EntityProvenance,
    ErrorCategory,
    ErrorRecord,
    Report,
    ReportStatus,
    Section,
    SectionStatus,
    TaskEnvelope,
    TaskKind,
    combine_usage,
)
from deep_researcher.evidence import (
    AgentSpecSemanticVerificationAdapter,
    VerificationPolicy,
    build_evidence_runtime,
    build_evidence_verifier_spec,
)
from deep_researcher.events import (
    EventRecorder,
    RedactionPolicy,
    SQLiteEventStore,
)
from deep_researcher.kernel import CancellationToken, ModelAdapter
from deep_researcher.knowledge import KnowledgeRepository, SQLiteKnowledgeStorage
from deep_researcher.orchestration import (
    NativeEventSourcedScheduler,
    RunControlStatus,
    SQLiteSchedulerStore,
    TaskStatus,
)
from deep_researcher.providers import (
    EnvironmentModelAdapter,
    GovernedResearchToolRuntime,
    OpenAICompatibleModelAdapter,
    OpenAICompatibleModelConfig,
    WORKER_TOOL_NAMES,
    build_governed_research_tools,
)
from deep_researcher.reporting import (
    LoopStatus,
    ReportLoopPolicy,
    SQLiteReportingStore,
    build_reporting_runtime,
)
from deep_researcher.research import (
    ConvergenceAction,
    ConvergencePolicy,
    SQLiteResearchCoordinationStore,
    build_research_runtime,
)
from deep_researcher.studio import (
    KernelReplayBackend,
    SQLiteStudioAdvancedStore,
    SQLiteStudioProjectionStore,
    StudioAdvancedService,
    StudioProjectionExporter,
    StudioProjector,
    StudioV2Service,
)
from deep_researcher.version_registry import SQLiteVersionRegistryStore

from .events import (
    ApplicationRunEventController,
    application_component_versions,
)
from .knowledge_boundary import KnowledgeIngestingCommandBoundary
from .models import (
    ApplicationRunRecord,
    ApplicationRunStatus,
    ResearchCreateRequest,
)
from .store import SQLiteApplicationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:32]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class ApplicationRuntimeDependencies:
    supervisor_model: ModelAdapter
    worker_model: ModelAdapter
    verifier_model: ModelAdapter
    writer_model: ModelAdapter
    reviewer_model: ModelAdapter


@dataclass(frozen=True)
class ApplicationRuntimeConfig:
    verification_policy: VerificationPolicy = field(
        default_factory=lambda: VerificationPolicy(
            policy_version_id="policy_background001_verification_1"
        )
    )
    research_budget: Budget = field(
        default_factory=lambda: Budget(
            max_tokens=400_000,
            max_cost_usd=80.0,
            max_wall_time_seconds=7200,
            max_model_calls=160,
            max_tool_calls=240,
            max_search_calls=80,
            max_retries=40,
            max_errors=30,
        )
    )
    report_budget: Budget = field(
        default_factory=lambda: Budget(
            max_tokens=300_000,
            max_cost_usd=60.0,
            max_wall_time_seconds=5400,
            max_model_calls=100,
            max_tool_calls=40,
            max_search_calls=20,
            max_retries=30,
            max_errors=20,
        )
    )
    targeted_research_task_budget: Budget = field(
        default_factory=lambda: Budget(
            max_tokens=48_000,
            max_cost_usd=8.0,
            max_wall_time_seconds=900,
            max_model_calls=24,
            max_tool_calls=32,
            max_search_calls=12,
            max_retries=6,
            max_errors=6,
        )
    )
    max_research_cycles: int = 12
    max_report_revisions: int = 6
    max_targeted_research_rounds: int = 2
    worker_ids: tuple[str, ...] = (
        "worker_research_1",
        "worker_research_2",
        "worker_research_3",
        "worker_research_4",
    )
    max_concurrency: int = 4
    minimum_section_coverage: float = 0.85
    minimum_citation_coverage: float = 1.0


class ApplicationRuntime:
    """Production composition root for all non-RL Background001 layers."""

    def __init__(
        self,
        root: str | Path,
        *,
        dependencies: ApplicationRuntimeDependencies,
        config: ApplicationRuntimeConfig | None = None,
        tool_runtime: GovernedResearchToolRuntime | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dependencies = dependencies
        self.config = config or ApplicationRuntimeConfig()
        self.application_store = SQLiteApplicationStore(
            self.root / "application.sqlite3"
        )
        self.event_store = SQLiteEventStore(self.root / "events.sqlite3")
        self.studio_store = SQLiteStudioProjectionStore(
            self.root / "studio_projection.sqlite3"
        )
        self.event_recorder = EventRecorder(
            self.event_store,
            (StudioProjectionExporter(self.studio_store),),
        )
        self.studio_projector = StudioProjector(
            self.event_store,
            self.studio_store,
        )
        self.scheduler_store = SQLiteSchedulerStore(
            self.root / "scheduler.sqlite3"
        )
        self.scheduler = NativeEventSourcedScheduler(self.scheduler_store)
        evidence_root = self.root / "evidence"
        evidence_root.mkdir(parents=True, exist_ok=True)
        self.artifact_store = SQLiteArtifactStore(
            evidence_root / "artifacts.sqlite3"
        )
        self.knowledge_storage = SQLiteKnowledgeStorage(
            evidence_root / "knowledge.sqlite3",
            artifact_store=self.artifact_store,
        )
        self.knowledge_repository = KnowledgeRepository(self.knowledge_storage)
        self.version_store = SQLiteVersionRegistryStore(
            self.root / "version_registry.sqlite3"
        )
        (self.root / "research").mkdir(parents=True, exist_ok=True)
        (self.root / "reporting").mkdir(parents=True, exist_ok=True)
        self.research_store = SQLiteResearchCoordinationStore(
            self.root / "research" / "research_coordination.sqlite3"
        )
        self.reporting_store = SQLiteReportingStore(
            self.root / "reporting" / "reporting.sqlite3"
        )
        self.tool_runtime = tool_runtime or build_governed_research_tools(
            self.root / "tools"
        )
        self._owns_tool_runtime = tool_runtime is None
        self.studio_advanced_store = SQLiteStudioAdvancedStore(
            self.root / "studio_advanced.sqlite3"
        )
        self.studio_v2 = StudioV2Service(
            event_store=self.event_store,
            scheduler_store=self.scheduler_store,
            knowledge_repository=self.knowledge_repository,
            artifact_store=self.artifact_store,
            version_store=self.version_store,
        )
        self.studio_advanced = StudioAdvancedService(
            store=self.studio_advanced_store,
            event_store=self.event_store,
            artifact_store=self.artifact_store,
            version_store=self.version_store,
            studio_v2=self.studio_v2,
            replay_backend=KernelReplayBackend(
                event_store=self.event_store,
                recorder=self.event_recorder,
                artifact_store=self.artifact_store,
                version_store=self.version_store,
            ),
        )
        self._tokens: dict[str, CancellationToken] = {}
        self._token_lock = asyncio.Lock()
        self._closed = False
        self.studio_projector.sync_all()
        self._drain_projection_outbox()
        self.integrity_check()

    def new_run(self, request: ResearchCreateRequest) -> ApplicationRunRecord:
        suffix = uuid.uuid4().hex
        research_id = f"research_{suffix[:16]}"
        run_id = f"run_{suffix}"
        record = ApplicationRunRecord(
            research_id=research_id,
            thread_id=f"thread_{suffix}",
            session_id=f"session_{suffix}",
            run_id=run_id,
            trace_id=f"trace_{suffix}",
            root_task_id=f"task_root_{suffix}",
            report_id=f"report_{suffix}",
            query=request.query,
            instructions=request.instructions,
            depth=request.depth,
            metadata={"created_via": "background001_application_runtime"},
        )
        return self.application_store.create(record)

    async def execute(self, research_id: str) -> ApplicationRunRecord:
        record = self.application_store.get(research_id)
        if record is None:
            raise KeyError(research_id)
        if record.status in {
            ApplicationRunStatus.COMPLETED,
            ApplicationRunStatus.FAILED,
            ApplicationRunStatus.CANCELLED,
        }:
            return record
        token = CancellationToken()
        async with self._token_lock:
            if record.run_id in self._tokens:
                raise RuntimeError(f"run is already executing: {record.run_id}")
            self._tokens[record.run_id] = token
        record = self.application_store.transition(
            research_id,
            status=ApplicationRunStatus.RUNNING,
            current_stage="initializing",
            resumed=record.revision > 0,
        )
        controller = ApplicationRunEventController(
            self.event_recorder,
            run_id=record.run_id,
            thread_id=record.thread_id,
            trace_id=record.trace_id,
            correlation_id=f"correlation_{record.run_id}",
            component_versions=application_component_versions(),
            attempt=max(1, record.revision),
        )
        controller.start(query=record.query, research_id=record.research_id)
        evidence = None
        research = None
        reporting = None
        try:
            semantic = AgentSpecSemanticVerificationAdapter(
                agent_spec=build_evidence_verifier_spec(),
                model=self.dependencies.verifier_model,
            )
            evidence = build_evidence_runtime(
                self.root / "evidence",
                semantic_adapter=semantic,
                event_sink=controller.evidence_sink,
                policy=self.config.verification_policy,
            )
            section_ids = self._ensure_report_scaffold(record, evidence)
            boundary = KnowledgeIngestingCommandBoundary(
                gateway=self.tool_runtime.gateway,
                ingestion=evidence.knowledge.ingestion,
                repository=evidence.knowledge.repository,
            )
            kernel_sink = controller.kernel_sink()
            research = build_research_runtime(
                self.root / "research",
                scheduler=self.scheduler,
                evidence=evidence,
                supervisor_model=self.dependencies.supervisor_model,
                worker_model=self.dependencies.worker_model,
                command_executor=boundary,
                event_sink=kernel_sink,
                convergence_policy=ConvergencePolicy(
                    required_section_ids=(section_ids["findings"],),
                    minimum_section_coverage=(
                        self.config.minimum_section_coverage
                    ),
                    minimum_citation_coverage=(
                        self.config.minimum_citation_coverage
                    ),
                    low_gain_threshold=0.03,
                    max_low_gain_cycles=2,
                    max_cycles=self._cycles_for_depth(record.depth),
                    run_budget=self.config.research_budget,
                    stop_on_any_severe_conflict=True,
                    require_no_high_impact_blockers=True,
                ),
                worker_ids=self.config.worker_ids,
                max_claim_rounds=200,
                finalize_scheduler_run=False,
            )
            reporting = build_reporting_runtime(
                self.root / "reporting",
                evidence=evidence,
                writer_model=self.dependencies.writer_model,
                reviewer_model=self.dependencies.reviewer_model,
                event_sink=kernel_sink,
                policy=ReportLoopPolicy(
                    max_revisions=self.config.max_report_revisions,
                    max_targeted_research_rounds=(
                        self.config.max_targeted_research_rounds
                    ),
                    minimum_score=0.8,
                    minimum_support_score=1.0,
                    minimum_citation_score=1.0,
                    minimum_sources_per_statement=1,
                    high_impact_minimum_sources=2,
                    run_budget=self.config.report_budget,
                ),
                scheduler=self.scheduler,
                worker_pool=research.worker_pool,
                targeted_research_task_budget=(
                    self.config.targeted_research_task_budget
                ),
            )
            record = self.application_store.transition(
                research_id,
                status=ApplicationRunStatus.RUNNING,
                current_stage="researching",
            )
            controller.stage(
                "researching",
                payload={
                    "root_task_id": record.root_task_id,
                    "required_section_id": section_ids["findings"],
                },
            )
            research_outcome = await research.coordinator.run(
                self._root_task(record, section_ids),
                max_concurrency=self._concurrency_for_depth(record.depth),
                cancellation=token,
            )
            if (
                research_outcome.action == ConvergenceAction.AWAIT_APPROVAL
            ):
                controller.evidence_sink.close()
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.WAITING_APPROVAL,
                    current_stage="waiting_approval",
                    metadata={
                        "research_decision_artifact_id": (
                            research_outcome.decision_artifact_id
                        )
                    },
                )
            if research_outcome.action == ConvergenceAction.STOP_BUDGET:
                error = ErrorRecord(
                    category=ErrorCategory.BUDGET_EXHAUSTED,
                    code="research_budget_exhausted",
                    message="Research stopped because its governed budget was exhausted.",
                    fatal=True,
                )
                controller.fail(error=error, usage=research_outcome.usage)
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.FAILED,
                    current_stage="failed",
                    error_code=error.code,
                    error_message=error.message,
                    metadata={
                        "research_action": research_outcome.action.value,
                        "research_decision_artifact_id": (
                            research_outcome.decision_artifact_id
                        ),
                    },
                )
            if token.cancelled or research_outcome.action == ConvergenceAction.CANCEL:
                await self._cancel_scheduler(record.run_id, "Caller cancellation")
                controller.cancel(
                    reason="Research was cancelled.",
                    usage=research_outcome.usage,
                )
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.CANCELLED,
                    current_stage="cancelled",
                )

            record = self.application_store.transition(
                research_id,
                status=ApplicationRunStatus.RUNNING,
                current_stage="reporting",
                metadata={
                    "research_action": research_outcome.action.value,
                    "research_decision_artifact_id": (
                        research_outcome.decision_artifact_id
                    ),
                },
            )
            controller.stage(
                "reporting",
                output_artifact_ids=(
                    research_outcome.decision_artifact_id,
                    *research_outcome.merged_result_artifact_ids,
                ),
            )
            report_outcome = await reporting.loop.run(
                record.report_id,
                cancellation=token,
            )
            total_usage = combine_usage(
                (research_outcome.usage, report_outcome.usage)
            )
            if report_outcome.status == LoopStatus.APPROVAL_REQUIRED:
                controller.evidence_sink.close()
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.WAITING_APPROVAL,
                    current_stage="waiting_approval",
                    report_artifact_id=report_outcome.final_report_artifact_id,
                    metadata={
                        "report_status": report_outcome.status.value,
                        "report_summary": report_outcome.summary,
                    },
                )
            if token.cancelled or report_outcome.status == LoopStatus.CANCELLED:
                await self._cancel_scheduler(record.run_id, report_outcome.summary)
                controller.cancel(reason=report_outcome.summary, usage=total_usage)
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.CANCELLED,
                    current_stage="cancelled",
                    report_artifact_id=report_outcome.final_report_artifact_id,
                )
            if report_outcome.status != LoopStatus.ACCEPTED:
                error = ErrorRecord(
                    category=ErrorCategory.VERIFICATION,
                    code=f"report_{report_outcome.status.value}",
                    message=report_outcome.summary,
                    fatal=True,
                )
                await self._cancel_scheduler(record.run_id, report_outcome.summary)
                controller.fail(error=error, usage=total_usage)
                return self.application_store.transition(
                    research_id,
                    status=ApplicationRunStatus.FAILED,
                    current_stage="failed",
                    report_artifact_id=report_outcome.final_report_artifact_id,
                    error_code=error.code,
                    error_message=error.message,
                )
            await self._complete_scheduler(record.run_id)
            final_artifacts = tuple(
                item
                for item in (
                    report_outcome.final_report_artifact_id,
                    report_outcome.citation_map_artifact_id,
                )
                if item is not None
            )
            controller.complete(
                output_artifact_ids=final_artifacts,
                usage=total_usage,
            )
            completed = self.application_store.transition(
                research_id,
                status=ApplicationRunStatus.COMPLETED,
                current_stage="completed",
                report_artifact_id=report_outcome.final_report_artifact_id,
                metadata={
                    "report_status": report_outcome.status.value,
                    "report_summary": report_outcome.summary,
                    "report_revision_id": report_outcome.final_revision_id,
                    "review_id": report_outcome.final_review_id,
                },
            )
            self.studio_projector.sync_run(record.run_id)
            return completed
        except asyncio.CancelledError:
            token.cancel()
            await self._cancel_scheduler(record.run_id, "Application task cancelled")
            controller.cancel(
                reason="Application execution task was cancelled.",
                usage=BudgetUsage(),
            )
            self.application_store.transition(
                research_id,
                status=ApplicationRunStatus.CANCELLED,
                current_stage="cancelled",
            )
            raise
        except Exception as exc:
            current = self.application_store.get(research_id)
            if (
                current is not None
                and current.status == ApplicationRunStatus.CANCELLED
            ):
                event_run = self.event_store.get_run(record.run_id)
                if event_run is not None and event_run.terminal_event_id is None:
                    controller.cancel(
                        reason=current.error_message or "Run cancelled.",
                        usage=BudgetUsage(),
                    )
                return current
            error = self._error(exc)
            await self._cancel_scheduler(record.run_id, error.message)
            event_run = self.event_store.get_run(record.run_id)
            if event_run is not None and event_run.terminal_event_id is None:
                controller.fail(error=error, usage=BudgetUsage())
            return self.application_store.transition(
                research_id,
                status=ApplicationRunStatus.FAILED,
                current_stage="failed",
                error_code=error.code,
                error_message=error.message,
            )
        finally:
            if reporting is not None:
                reporting.close()
            if research is not None:
                research.close()
            if evidence is not None:
                evidence.close()
            async with self._token_lock:
                self._tokens.pop(record.run_id, None)
            self.studio_projector.sync_run(record.run_id)

    async def approve_run(
        self,
        research_id: str,
        *,
        approved_by: str,
        note: str,
    ) -> ApplicationRunRecord:
        record = self._require_record(research_id)
        snapshot = await self.scheduler.snapshot(record.run_id)
        waiting = [
            item
            for item in snapshot.tasks
            if item.envelope.status == TaskStatus.WAITING_APPROVAL
        ]
        if not waiting:
            raise RuntimeError("run has no scheduler task awaiting approval")
        for item in waiting:
            if item.approval is None:
                raise RuntimeError("waiting task has no approval record")
            await self.scheduler.approve(
                item.task_id,
                actor_id=approved_by,
                note=note,
                mutation_id=_stable_id(
                    "mutation",
                    item.approval.approval_id,
                    approved_by,
                    "approve",
                ),
            )
        return self.application_store.transition(
            research_id,
            status=ApplicationRunStatus.QUEUED,
            current_stage="queued",
            resumed=True,
            metadata={"approval_resolved_by": approved_by},
        )

    async def cancel_run(
        self,
        research_id: str,
        *,
        reason: str,
    ) -> ApplicationRunRecord:
        record = self._require_record(research_id)
        async with self._token_lock:
            token = self._tokens.get(record.run_id)
            active = token is not None
            if token is not None:
                token.cancel()
        await self._cancel_scheduler(record.run_id, reason)
        event_run = self.event_store.get_run(record.run_id)
        if (
            not active
            and (
                event_run is None
                or event_run.terminal_event_id is None
            )
        ):
            controller = ApplicationRunEventController(
                self.event_recorder,
                run_id=record.run_id,
                thread_id=record.thread_id,
                trace_id=record.trace_id,
                correlation_id=f"correlation_{record.run_id}",
                component_versions=application_component_versions(),
                attempt=max(1, record.revision + 1),
            )
            controller.start(query=record.query, research_id=record.research_id)
            controller.cancel(reason=reason, usage=BudgetUsage())
        return self.application_store.transition(
            research_id,
            status=ApplicationRunStatus.CANCELLED,
            current_stage="cancelled",
            metadata={"cancellation_reason": reason},
        )

    def recoverable_runs(self) -> tuple[ApplicationRunRecord, ...]:
        return tuple(
            item
            for item in self.application_store.list(limit=10_000)
            if item.status
            in {
                ApplicationRunStatus.QUEUED,
                ApplicationRunStatus.RUNNING,
            }
        )

    def integrity_check(self) -> None:
        self.application_store.integrity_check()
        self.event_store.integrity_check()
        self.studio_store.integrity_check()
        self.scheduler_store.integrity_check()
        self.artifact_store.integrity_check()
        self.knowledge_storage.integrity_check()
        self.version_store.integrity_check()
        self.research_store.integrity_check()
        self.reporting_store.integrity_check()
        self.tool_runtime.integrity_check()
        self.studio_advanced_store.integrity_check()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        async with self._token_lock:
            for token in self._tokens.values():
                token.cancel()
        await self.scheduler.close()
        if self._owns_tool_runtime:
            self.tool_runtime.close()
        self.reporting_store.close()
        self.research_store.close()
        self.studio_advanced_store.close()
        self.version_store.close()
        self.knowledge_storage.close()
        self.artifact_store.close()
        self.studio_store.close()
        self.event_store.close()
        self.application_store.close()

    def _ensure_report_scaffold(
        self,
        record: ApplicationRunRecord,
        evidence: Any,
    ) -> dict[str, str]:
        repository = evidence.knowledge.repository
        existing = repository.reports.get(record.report_id)
        section_ids = {
            "executive": _stable_id("section", record.run_id, "executive"),
            "findings": _stable_id("section", record.run_id, "findings"),
            "conflicts": _stable_id("section", record.run_id, "conflicts"),
            "methodology": _stable_id("section", record.run_id, "methodology"),
        }
        if existing is not None:
            if existing.run_id != record.run_id:
                raise RuntimeError("report scaffold belongs to another run")
            return section_ids
        input_artifact = evidence.knowledge.artifacts.put_json(
            {
                "schema": "ResearchRequest@1",
                "query": record.query,
                "instructions": record.instructions,
                "depth": record.depth,
            },
            redact=True,
            kind=ArtifactKind.MODEL_INPUT,
            producer_id="runtime_application",
            run_id=record.run_id,
            task_id=record.root_task_id,
            content_schema="ResearchRequest@1",
            idempotency_key=f"research-request:{record.run_id}",
        )
        provenance = EntityProvenance(
            producer_id="runtime_application",
            run_id=record.run_id,
            task_id=record.root_task_id,
            source_artifact_ids=(input_artifact.artifact_id,),
        )
        sections = (
            Section(
                section_id=section_ids["executive"],
                report_id=record.report_id,
                title="Executive Summary",
                goal="Summarize verified findings, confidence, and material gaps.",
                order=0,
                status=SectionStatus.PLANNED,
                provenance=provenance,
            ),
            Section(
                section_id=section_ids["findings"],
                report_id=record.report_id,
                title="Verified Findings",
                goal=(
                    "Answer the research question with independently verified, "
                    "citation-complete claims."
                ),
                order=1,
                status=SectionStatus.PLANNED,
                provenance=provenance,
            ),
            Section(
                section_id=section_ids["conflicts"],
                report_id=record.report_id,
                title="Conflicts, Gaps, and Uncertainty",
                goal=(
                    "Disclose unresolved conflicts, missing evidence, and bounded "
                    "uncertainty without inventing conclusions."
                ),
                order=2,
                status=SectionStatus.PLANNED,
                provenance=provenance,
            ),
            Section(
                section_id=section_ids["methodology"],
                report_id=record.report_id,
                title="Sources and Methodology",
                goal=(
                    "Explain source selection, verification boundaries, and "
                    "limitations of the research process."
                ),
                order=3,
                status=SectionStatus.PLANNED,
                provenance=provenance,
            ),
        )
        report = Report(
            report_id=record.report_id,
            thread_id=record.thread_id,
            run_id=record.run_id,
            title=record.query,
            research_question=record.query,
            section_ids=tuple(item.section_id for item in sections),
            status=ReportStatus.DRAFT,
            provenance=provenance,
        )
        repository.save_graph(report, *sections)
        return section_ids

    def _root_task(
        self,
        record: ApplicationRunRecord,
        section_ids: dict[str, str],
    ) -> TaskEnvelope:
        tool_contracts = {
            "research.search": {
                "kind": "search",
                "required_arguments": ["operation=search", "query"],
            },
            "research.read": {
                "kind": "read",
                "required_arguments": ["operation=read", "url or urls"],
            },
            "research.extract": {
                "kind": "extract",
                "required_arguments": [
                    "operation=extract",
                    "section_id",
                    "claims",
                    "atomic_facts",
                    "evidence",
                ],
                "grounding_rule": (
                    "Every evidence item must reference a persisted source and "
                    "contain an exact quote from a prior read/search passage."
                ),
            },
            "research.compare": {
                "kind": "compare",
                "required_arguments": ["operation=compare"],
            },
            "research.verify_source": {
                "kind": "verify_source",
                "required_arguments": ["operation=verify_source"],
            },
        }
        return TaskEnvelope(
            task_id=record.root_task_id,
            run_id=record.run_id,
            kind=TaskKind.ROOT,
            title=f"Research: {record.query[:250]}",
            goal=record.query,
            constraints={
                "user_instructions": record.instructions,
                "depth": record.depth,
                "report_id": record.report_id,
                "required_section_id": section_ids["findings"],
                "report_section_ids": section_ids,
                "available_worker_tools": list(WORKER_TOOL_NAMES),
                "worker_tool_contracts": tool_contracts,
                "evidence_rules": {
                    "candidate_only_until_verifier": True,
                    "exact_quote_offsets_required": True,
                    "citations_required": True,
                    "conflicts_must_be_preserved": True,
                    "high_impact_requires_independent_sources": True,
                },
            },
            expected_output_schema="ResearchRunOutcome@1",
            budget=self.config.research_budget,
            priority=1.0,
            max_attempts=3,
            created_by="runtime_application",
            tags=("background001", "research-root", record.depth),
        )

    def _cycles_for_depth(self, depth: str) -> int:
        return {
            "quick": min(6, self.config.max_research_cycles),
            "standard": min(12, self.config.max_research_cycles),
            "deep": self.config.max_research_cycles,
        }[depth]

    def _concurrency_for_depth(self, depth: str) -> int:
        return min(
            self.config.max_concurrency,
            {"quick": 2, "standard": 3, "deep": 4}[depth],
        )

    async def _complete_scheduler(self, run_id: str) -> None:
        snapshot = await self.scheduler.snapshot(run_id)
        if snapshot.control.status == RunControlStatus.ACTIVE:
            await self.scheduler.complete_run(
                run_id,
                actor_id="runtime_application",
                mutation_id=_stable_id("mutation", run_id, "complete_application"),
            )

    async def _cancel_scheduler(self, run_id: str, reason: str) -> None:
        try:
            snapshot = await self.scheduler.snapshot(run_id)
        except KeyError:
            return
        if snapshot.control.status == RunControlStatus.ACTIVE:
            await self.scheduler.cancel_run(
                run_id,
                actor_id="runtime_application",
                reason=reason[:2000] or "Application run cancelled.",
                mutation_id=_stable_id("mutation", run_id, "cancel_application"),
            )

    def _require_record(self, research_id: str) -> ApplicationRunRecord:
        record = self.application_store.get(research_id)
        if record is None:
            raise KeyError(research_id)
        return record

    def _drain_projection_outbox(self) -> None:
        while self.event_store.pending_exports(
            exporter_name="studio_projection",
            limit=1000,
        ):
            outcomes = self.event_recorder.retry_pending(
                exporter_name="studio_projection",
                limit=1000,
            )
            if not any(item.exported_to for item in outcomes):
                break

    @staticmethod
    def _error(exc: Exception) -> ErrorRecord:
        category = (
            ErrorCategory.TRANSIENT_PROVIDER
            if bool(getattr(exc, "retryable", False))
            else ErrorCategory.INTERNAL
        )
        return ErrorRecord(
            category=category,
            code=f"application_{type(exc).__name__.casefold()}",
            message=RedactionPolicy().redact_text(
                str(exc) or type(exc).__name__
            )[:2000],
            retryable=bool(getattr(exc, "retryable", False)),
            fatal=True,
        )


def build_application_runtime(
    root: str | Path,
    *,
    dependencies: ApplicationRuntimeDependencies,
    config: ApplicationRuntimeConfig | None = None,
    tool_runtime: GovernedResearchToolRuntime | None = None,
) -> ApplicationRuntime:
    return ApplicationRuntime(
        root,
        dependencies=dependencies,
        config=config,
        tool_runtime=tool_runtime,
    )


def build_live_application_runtime(
    root: str | Path,
    *,
    model_config: OpenAICompatibleModelConfig | None = None,
    config: ApplicationRuntimeConfig | None = None,
) -> ApplicationRuntime:
    model: ModelAdapter = (
        OpenAICompatibleModelAdapter(model_config)
        if model_config is not None
        else EnvironmentModelAdapter()
    )
    dependencies = ApplicationRuntimeDependencies(
        supervisor_model=model,
        worker_model=model,
        verifier_model=model,
        writer_model=model,
        reviewer_model=model,
    )
    return build_application_runtime(
        root,
        dependencies=dependencies,
        config=config,
    )
