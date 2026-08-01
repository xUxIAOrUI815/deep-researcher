from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Protocol

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    Command,
    CommandKind,
    ErrorCategory,
    ErrorRecord,
    Observation,
    ObservationStatus,
    TaskEnvelope,
    TaskKind,
    TaskResultStatus,
    TaskStatus,
    StopReason,
    utc_now,
)
from deep_researcher.kernel import (
    AgentKernel,
    CancellationToken,
    RawObservation,
    VerificationFeedback,
)
from deep_researcher.knowledge.normalization import canonicalize_url
from deep_researcher.orchestration import (
    Scheduler,
    SchedulerLeaseError,
    TaskCompletion,
    TaskLease,
)

from .models import (
    DedupDecisionKind,
    DedupKind,
    InformationGainAssessment,
    MergedResearchResult,
    ResearchWorkerResult,
)
from .store import SQLiteResearchCoordinationStore


_PROVIDER_COMMANDS = {
    CommandKind.SEARCH,
    CommandKind.READ,
    CommandKind.EXTRACT,
    CommandKind.COMPARE,
    CommandKind.VERIFY_SOURCE,
}
_DELEGATABLE_KINDS = {
    TaskKind.SOURCE_DISCOVERY,
    TaskKind.RESEARCH,
    TaskKind.GAP,
    TaskKind.CONFLICT,
    TaskKind.VERIFICATION,
    TaskKind.SECTION_SUPPORT,
    TaskKind.REPAIR,
}


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


class CommandExecutionBoundary(Protocol):
    async def execute(self, command: Command) -> Observation:
        ...


class ResearchWorkerActionExecutor:
    """Executes worker commands through a governed boundary with durable dedup."""

    def __init__(
        self,
        *,
        command_executor: CommandExecutionBoundary,
        scheduler: Scheduler,
        artifact_store: ArtifactStore,
        coordination: SQLiteResearchCoordinationStore,
        allowed_worker_ids: tuple[str, ...] = (),
        dedup_lease_seconds: float = 120.0,
        clock=utc_now,
    ) -> None:
        if dedup_lease_seconds <= 0:
            raise ValueError("dedup_lease_seconds must be positive")
        self.command_executor = command_executor
        self.scheduler = scheduler
        self.artifact_store = artifact_store
        self.coordination = coordination
        self.allowed_worker_ids = frozenset(allowed_worker_ids)
        self.dedup_lease_seconds = dedup_lease_seconds
        self.clock = clock

    async def execute(self, command: Command) -> Observation | RawObservation:
        if command.kind == CommandKind.DELEGATE:
            return await self._delegate(command)
        if command.kind not in _PROVIDER_COMMANDS:
            raise ValueError(
                f"worker command is outside the research boundary: "
                f"{command.kind.value}"
            )
        started = self.clock()
        dedup = self._dedup_target(command)
        decision = None
        if dedup is not None:
            kind, key = dedup
            decision = self.coordination.claim_key(
                run_id=command.run_id,
                kind=kind,
                normalized_key=key,
                task_id=command.task_id,
                lease_seconds=self.dedup_lease_seconds,
            )
            if decision.decision in {
                DedupDecisionKind.DUPLICATE_COMPLETED,
                DedupDecisionKind.DUPLICATE_IN_PROGRESS,
            }:
                return self._persist_observation(
                    command,
                    Observation(
                        command_id=command.command_id,
                        run_id=command.run_id,
                        task_id=command.task_id,
                        actor_id=command.actor_id,
                        status=ObservationStatus.SUCCEEDED,
                        output_artifact_ids=decision.artifact_ids,
                        normalized_data={
                            "duplicate": True,
                            "dedup_kind": decision.kind.value,
                            "dedup_key": decision.normalized_key,
                            "dedup_owner_task_id": decision.owner_task_id,
                            "dedup_decision": decision.decision.value,
                            "semantic_complete": False,
                            "_research": {
                                "duplicate": True,
                                "query_keys": (
                                    [decision.normalized_key]
                                    if decision.kind == DedupKind.QUERY
                                    else []
                                ),
                                "source_keys": (
                                    [decision.normalized_key]
                                    if decision.kind == DedupKind.SOURCE
                                    else []
                                ),
                                "meaningful_artifact_ids": list(
                                    decision.artifact_ids
                                ),
                            },
                        },
                        started_at=started,
                        completed_at=self.clock(),
                    ),
                )
        try:
            observation = await self.command_executor.execute(command)
        except Exception as exc:
            if dedup is not None:
                self.coordination.fail_key(
                    run_id=command.run_id,
                    kind=dedup[0],
                    normalized_key=dedup[1],
                    task_id=command.task_id,
                    error=str(exc) or type(exc).__name__,
                )
            raise
        data = dict(observation.normalized_data)
        source_keys = self._source_keys(data)
        query_keys = [dedup[1]] if dedup and dedup[0] == DedupKind.QUERY else []
        data["_research"] = {
            "duplicate": False,
            "query_keys": query_keys,
            "source_keys": source_keys,
            "meaningful_artifact_ids": list(observation.output_artifact_ids),
        }
        observation = observation.model_copy(update={"normalized_data": data})
        persisted = self._persist_observation(command, observation)
        if dedup is not None:
            if persisted.status == ObservationStatus.SUCCEEDED:
                self.coordination.complete_key(
                    run_id=command.run_id,
                    kind=dedup[0],
                    normalized_key=dedup[1],
                    task_id=command.task_id,
                    artifact_ids=persisted.output_artifact_ids,
                )
            else:
                self.coordination.fail_key(
                    run_id=command.run_id,
                    kind=dedup[0],
                    normalized_key=dedup[1],
                    task_id=command.task_id,
                    error=(
                        persisted.error.message
                        if persisted.error is not None
                        else persisted.status.value
                    ),
                )
        return persisted

    async def _delegate(self, command: Command) -> RawObservation:
        raw = command.arguments.get("task")
        task = TaskEnvelope.model_validate(raw, strict=False)
        if (
            task.run_id != command.run_id
            or task.parent_task_id != command.task_id
        ):
            raise ValueError(
                "delegated worker task must share the run and name its current "
                "task as parent"
            )
        if task.kind not in _DELEGATABLE_KINDS:
            raise ValueError("worker cannot delegate writing or review tasks")
        if task.status != TaskStatus.PENDING:
            raise ValueError("delegated worker task must be pending")
        if (
            task.assigned_actor_id is not None
            and task.assigned_actor_id not in self.allowed_worker_ids
        ):
            raise ValueError(
                "delegated task assignment is outside the configured Worker Pool"
            )
        fingerprint = hashlib.sha256(
            canonical_task_payload(task).encode("utf-8")
        ).hexdigest()
        reservation = self.coordination.reserve_task(
            run_id=task.run_id,
            fingerprint=fingerprint,
            task_id=task.task_id,
        )
        snapshot = await self.scheduler.snapshot(task.run_id)
        canonical_id = reservation.canonical_task_id
        new_task = canonical_id not in snapshot.by_id
        if new_task:
            if canonical_id != task.task_id:
                task = task.model_copy(update={"task_id": canonical_id})
            await self.scheduler.split(
                command.task_id,
                (task,),
                actor_id=command.actor_id,
                mutation_id=_stable_id(
                    "mutation",
                    command.command_id,
                    "worker_delegate",
                ),
            )
        artifact_id = _stable_id(
            "artifact",
            command.command_id,
            "worker_delegate",
        )
        self.artifact_store.put_json(
            {
                "schema": "WorkerDelegation@1",
                "command_id": command.command_id,
                "task": task.model_dump(mode="json"),
                "canonical_task_id": canonical_id,
                "new_task": new_task,
            },
            redact=False,
            kind=ArtifactKind.WORKER_DELEGATION,
            producer_id=command.actor_id,
            run_id=command.run_id,
            task_id=command.task_id,
            content_schema="WorkerDelegation@1",
            source_artifact_ids=command.input_artifact_ids,
            artifact_id=artifact_id,
            idempotency_key=f"worker-delegation:{command.command_id}",
        )
        now = self.clock()
        return RawObservation(
            status=ObservationStatus.SUCCEEDED.value,
            data={
                "delegated_task_id": canonical_id,
                "canonical_task_id": canonical_id,
                "new_task": new_task,
                "semantic_complete": False,
                "_research": {
                    "duplicate": not new_task,
                    "query_keys": [],
                    "source_keys": [],
                    "meaningful_artifact_ids": [artifact_id],
                },
            },
            output_artifact_ids=(artifact_id,),
            started_at=now,
            completed_at=self.clock(),
        )

    def _persist_observation(
        self,
        command: Command,
        observation: Observation,
    ) -> Observation:
        artifact_id = _stable_id(
            "artifact",
            command.command_id,
            "observation",
        )
        source_artifacts = tuple(
            dict.fromkeys(
                item
                for item in (
                    *command.input_artifact_ids,
                    *observation.output_artifact_ids,
                )
                if item != artifact_id
            )
        )
        existing = self.artifact_store.get(artifact_id)
        if existing is None:
            self.artifact_store.put_json(
                {
                    "schema": "ResearchCommandObservation@1",
                    "command": command.model_dump(mode="json"),
                    "observation": observation.model_dump(mode="json"),
                },
                redact=True,
                kind=ArtifactKind.TOOL_RESULT,
                producer_id=command.actor_id,
                run_id=command.run_id,
                task_id=command.task_id,
                content_schema="ResearchCommandObservation@1",
                source_artifact_ids=source_artifacts,
                artifact_id=artifact_id,
                idempotency_key=f"research-observation:{command.command_id}",
                metadata={
                    "command_kind": command.kind.value,
                    "command_name": command.name,
                    "status": observation.status.value,
                },
            )
        elif (
            existing.kind != ArtifactKind.TOOL_RESULT
            or existing.run_id != command.run_id
            or existing.task_id != command.task_id
            or existing.content_schema != "ResearchCommandObservation@1"
        ):
            raise RuntimeError(
                "Existing research observation artifact violates its "
                "idempotent command boundary."
            )
        return observation.model_copy(
            update={
                "output_artifact_ids": tuple(
                    dict.fromkeys(
                        (*observation.output_artifact_ids, artifact_id)
                    )
                )
            }
        )

    @staticmethod
    def _dedup_target(
        command: Command,
    ) -> tuple[DedupKind, str] | None:
        if command.kind == CommandKind.SEARCH:
            query = str(
                command.arguments.get("query")
                or command.arguments.get("q")
                or ""
            )
            normalized = " ".join(query.casefold().split())
            return (DedupKind.QUERY, normalized) if normalized else None
        if command.kind in {CommandKind.READ, CommandKind.VERIFY_SOURCE}:
            source = str(
                command.arguments.get("url")
                or command.arguments.get("source_url")
                or command.arguments.get("source_id")
                or ""
            ).strip()
            if not source:
                return None
            try:
                source = canonicalize_url(source)
            except ValueError:
                source = source.casefold()
            return DedupKind.SOURCE, source
        return None

    @classmethod
    def _source_keys(cls, value: Any) -> list[str]:
        output: list[str] = []

        def walk(item: Any) -> None:
            if isinstance(item, dict):
                for key, nested in item.items():
                    if str(key).casefold() in {
                        "url",
                        "source_url",
                        "canonical_url",
                    } and isinstance(nested, str):
                        try:
                            output.append(canonicalize_url(nested))
                        except ValueError:
                            pass
                    else:
                        walk(nested)
            elif isinstance(item, (list, tuple)):
                for nested in item:
                    walk(nested)

        walk(value)
        return list(dict.fromkeys(output))


def canonical_task_payload(task: TaskEnvelope) -> str:
    value = {
        "run_id": task.run_id,
        "parent_task_id": task.parent_task_id,
        "dependency_task_ids": list(task.dependency_task_ids),
        "kind": task.kind.value,
        "goal": task.goal,
        "constraints": task.constraints,
        "input_artifact_ids": list(task.input_artifact_ids),
        "expected_output_schema": task.expected_output_schema,
    }
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


class InformationGainEstimator:
    def __init__(
        self,
        coordination: SQLiteResearchCoordinationStore,
    ) -> None:
        self.coordination = coordination

    def assess(
        self,
        *,
        command: Command,
        observation: Observation,
    ) -> InformationGainAssessment:
        research = observation.normalized_data.get("_research", {})
        if not isinstance(research, dict):
            research = {}
        duplicate = bool(research.get("duplicate", False))
        query_keys = tuple(
            str(item)
            for item in research.get("query_keys", ())
            if str(item)
        )
        source_keys = tuple(
            str(item)
            for item in research.get("source_keys", ())
            if str(item)
        )
        artifact_ids = tuple(
            str(item)
            for item in research.get("meaningful_artifact_ids", ())
            if str(item)
        )
        knowledge_keys = self._knowledge_keys(
            observation.normalized_data
        )
        new_queries = tuple(
            item
            for item in query_keys
            if self.coordination.register_novelty(
                run_id=command.run_id,
                kind=DedupKind.QUERY,
                normalized_key=item,
                task_id=command.task_id,
            )
        )
        new_sources = tuple(
            item
            for item in source_keys
            if self.coordination.register_novelty(
                run_id=command.run_id,
                kind=DedupKind.SOURCE,
                normalized_key=item,
                task_id=command.task_id,
            )
        )
        new_artifacts = tuple(
            item
            for item in artifact_ids
            if self.coordination.register_novelty(
                run_id=command.run_id,
                kind=DedupKind.ARTIFACT,
                normalized_key=item,
                task_id=command.task_id,
            )
        )
        new_knowledge = tuple(
            item
            for item in knowledge_keys
            if self.coordination.register_novelty(
                run_id=command.run_id,
                kind=DedupKind.KNOWLEDGE,
                normalized_key=item,
                task_id=command.task_id,
            )
        )
        if duplicate:
            score = 0.0
        else:
            score = min(
                1.0,
                0.12 * len(new_queries)
                + 0.24 * len(new_sources)
                + 0.18 * len(new_artifacts)
                + 0.22 * len(new_knowledge),
            )
        return InformationGainAssessment(
            task_id=command.task_id,
            command_id=command.command_id,
            score=score,
            new_artifact_ids=new_artifacts,
            new_source_keys=new_sources,
            new_query_keys=new_queries,
            new_knowledge_keys=new_knowledge,
            duplicate=duplicate,
            decision_summary=(
                "Duplicate work produced no information gain."
                if duplicate
                else (
                    f"Novelty: {len(new_queries)} query, "
                    f"{len(new_sources)} source, "
                    f"{len(new_artifacts)} artifact, and "
                    f"{len(new_knowledge)} knowledge item(s)."
                )
            ),
        )

    @staticmethod
    def _knowledge_keys(value: Any) -> tuple[str, ...]:
        keys: list[str] = []
        if not isinstance(value, dict):
            return ()
        ingestion = value.get("ingestion")
        if isinstance(ingestion, dict):
            entity_ids = ingestion.get("entity_ids", ())
            if isinstance(entity_ids, (list, tuple)):
                persisted = tuple(
                    f"entity:{item}"
                    for item in entity_ids
                    if isinstance(item, str)
                    and item.startswith(
                        (
                            "evidence_",
                            "fact_",
                            "claim_",
                            "citation_",
                            "conflict_",
                        )
                    )
                )
                # Once an ingestion boundary has reported its durable result,
                # raw model arrays are not evidence of information gain. This
                # prevents invalid or ungrounded candidate JSON from turning a
                # budget-exhausted partial task into a completed task.
                return tuple(dict.fromkeys(persisted))
        for field in (
            "facts",
            "atomic_facts",
            "claims",
            "evidence",
            "passages",
            "extractions",
            "conflicts",
        ):
            raw = value.get(field)
            if raw is None:
                continue
            items = raw if isinstance(raw, list) else [raw]
            for item in items:
                encoded = json.dumps(
                    item,
                    ensure_ascii=False,
                    sort_keys=True,
                    default=str,
                    separators=(",", ":"),
                )
                keys.append(
                    f"{field}:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
                )
        return tuple(dict.fromkeys(keys))


class WorkerObservationVerifier:
    def __init__(self, estimator: InformationGainEstimator) -> None:
        self.estimator = estimator
        self.assessments: dict[str, InformationGainAssessment] = {}

    async def verify(
        self,
        *,
        spec: Any,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        del spec, prior_observations
        if observation.status != ObservationStatus.SUCCEEDED:
            return VerificationFeedback(
                passed=False,
                information_gain=0.0,
                summary=(
                    observation.error.message
                    if observation.error is not None
                    else "Research command failed."
                ),
                repair_feedback=(
                    "Choose a retry, governed fallback, or narrower command.",
                ),
            )
        assessment = self.estimator.assess(
            command=command,
            observation=observation,
        )
        self.assessments[command.command_id] = assessment
        research = observation.normalized_data.get("_research", {})
        duplicate = bool(
            research.get("duplicate", False)
            if isinstance(research, dict)
            else False
        )
        source_discovery_complete = bool(
            task.kind == TaskKind.SOURCE_DISCOVERY
            and command.kind == CommandKind.SEARCH
            and observation.normalized_data.get("sources")
            and not duplicate
        )
        semantic_complete = bool(
            observation.normalized_data.get("semantic_complete", False)
            or source_discovery_complete
        )
        ingestion = observation.normalized_data.get("ingestion", {})
        ingestion_issues = tuple(
            str(item)
            for item in (
                ingestion.get("issues", ())
                if isinstance(ingestion, dict)
                else ()
            )
            if str(item)
        )
        repair_feedback = tuple(
            dict.fromkeys(
                (
                    *(("Avoid this duplicate query or source.",) if assessment.duplicate else ()),
                    *ingestion_issues,
                    *(
                        (
                            "Repair the extract with verbatim quotes copied from "
                            "persisted passages and ensure every claim has a "
                            "persisted citation.",
                        )
                        if ingestion_issues
                        else ()
                    ),
                )
            )
        )
        return VerificationFeedback(
            passed=True,
            success=semantic_complete,
            semantic_complete=semantic_complete,
            information_gain=assessment.score,
            summary=(
                assessment.decision_summary
                if not ingestion_issues
                else (
                    f"{assessment.decision_summary} Ingestion rejected "
                    f"{len(ingestion_issues)} ungrounded item(s)."
                )
            ),
            repair_feedback=repair_feedback,
        )


class ResearchWorkerRunner:
    def __init__(
        self,
        *,
        worker_id: str,
        agent_spec_id: str,
        kernel: AgentKernel,
        verifier: WorkerObservationVerifier,
        scheduler: Scheduler,
        artifact_store: ArtifactStore,
        coordination: SQLiteResearchCoordinationStore,
        clock=utc_now,
    ) -> None:
        self.worker_id = worker_id
        self.agent_spec_id = agent_spec_id
        self.kernel = kernel
        self.verifier = verifier
        self.scheduler = scheduler
        self.artifact_store = artifact_store
        self.coordination = coordination
        self.clock = clock

    async def run_lease(
        self,
        lease: TaskLease,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ResearchWorkerResult:
        if lease.worker_id != self.worker_id:
            raise ValueError("worker runner received another worker's lease")
        kernel_task = lease.task.model_copy(
            update={"assigned_actor_id": self.agent_spec_id}
        )
        kernel_result = await self.kernel.run(
            agent_spec_id=self.agent_spec_id,
            task=kernel_task,
            cancellation=cancellation,
        )
        task_result = kernel_result.task_result
        assessments = tuple(
            self.verifier.assessments[item.command_id]
            for item in kernel_result.commands
            if item.command_id in self.verifier.assessments
        )
        query_keys = tuple(
            item
            for assessment in assessments
            for item in assessment.new_query_keys
        )
        source_keys = tuple(
            item
            for assessment in assessments
            for item in assessment.new_source_keys
        )
        knowledge_keys = tuple(
            item
            for assessment in assessments
            for item in assessment.new_knowledge_keys
        )
        information_gain = sum(
            item.score for item in assessments
        )
        worker_result_id = _stable_id(
            "worker_result",
            lease.task.run_id,
            lease.task.task_id,
            str(lease.task.attempt),
            task_result.result_id,
        )
        result_artifact_id = _stable_id(
            "artifact",
            worker_result_id,
        )
        source_artifacts = tuple(
            dict.fromkeys(
                (
                    *lease.task.input_artifact_ids,
                    *task_result.output_artifact_ids,
                )
            )
        )
        self.artifact_store.put_json(
            {
                "schema": "ResearchWorkerResult@1",
                "task_result": task_result.model_dump(mode="json"),
                "stop_decision": kernel_result.stop_decision.model_dump(
                    mode="json"
                ),
                "commands": [
                    item.model_dump(mode="json")
                    for item in kernel_result.commands
                ],
                "observations": [
                    item.model_dump(mode="json")
                    for item in kernel_result.observations
                ],
                "information_gain": [
                    item.model_dump(mode="json")
                    for item in assessments
                ],
            },
            redact=True,
            kind=ArtifactKind.WORKER_RESULT,
            producer_id=self.agent_spec_id,
            run_id=lease.task.run_id,
            task_id=lease.task.task_id,
            content_schema="ResearchWorkerResult@1",
            source_artifact_ids=source_artifacts,
            artifact_id=result_artifact_id,
            idempotency_key=f"worker-result:{worker_result_id}",
        )
        output_artifacts = tuple(
            dict.fromkeys(
                (*task_result.output_artifact_ids, result_artifact_id)
            )
        )
        meaningful_partial = bool(
            task_result.status == TaskResultStatus.PARTIAL
            and information_gain > 0
            and task_result.output_artifact_ids
            and (
                knowledge_keys
                or (
                    lease.task.kind == TaskKind.SOURCE_DISCOVERY
                    and source_keys
                )
            )
        )
        error = task_result.error
        if (
            task_result.status == TaskResultStatus.PARTIAL
            and error is None
            and not meaningful_partial
        ):
            error = ErrorRecord(
                category=(
                    ErrorCategory.BUDGET_EXHAUSTED
                    if kernel_result.stop_decision.reason
                    in {
                        StopReason.BUDGET_EXHAUSTED,
                        StopReason.DEADLINE_REACHED,
                    }
                    else ErrorCategory.VERIFICATION
                ),
                code=(
                    "worker_partial_"
                    f"{kernel_result.stop_decision.reason.value}"
                ),
                message=task_result.summary,
                fatal=lease.task.attempt >= lease.task.max_attempts,
                retryable=(
                    lease.task.attempt < lease.task.max_attempts
                    and kernel_result.stop_decision.reason
                    in {
                        StopReason.NO_ACTION_AVAILABLE,
                        StopReason.BUDGET_EXHAUSTED,
                        StopReason.LOW_INFORMATION_GAIN,
                    }
                ),
                actor_id=self.agent_spec_id,
                task_id=lease.task.task_id,
            )
        if task_result.status in {
            TaskResultStatus.FAILED,
            TaskResultStatus.REJECTED,
        } and error is None:
            error = ErrorRecord(
                category=ErrorCategory.INTERNAL,
                code="worker_task_failed",
                message=task_result.summary,
                fatal=True,
                retryable=False,
            )
        retry_scheduled = bool(
            error is not None
            and not meaningful_partial
            and error.retryable
            and lease.task.attempt < lease.task.max_attempts
        )
        result = ResearchWorkerResult(
            worker_result_id=worker_result_id,
            run_id=lease.task.run_id,
            task_id=lease.task.task_id,
            worker_id=self.worker_id,
            task_result_id=task_result.result_id,
            task_attempt=lease.task.attempt,
            status=task_result.status,
            output_artifact_ids=output_artifacts,
            query_keys=tuple(dict.fromkeys(query_keys)),
            source_keys=tuple(dict.fromkeys(source_keys)),
            knowledge_keys=tuple(dict.fromkeys(knowledge_keys)),
            information_gain=information_gain,
            command_count=len(kernel_result.commands),
            usage=task_result.usage,
            summary=task_result.summary,
            retry_scheduled=retry_scheduled,
            error_ref=error.error_id if error is not None else None,
            created_at=self.clock(),
        )
        # Persist the intent before crossing the scheduler store boundary. A
        # restart can reconcile this immutable attempt without rerunning tools.
        self.coordination.save_worker_result(result)
        if (
            task_result.status == TaskResultStatus.SUCCEEDED
            or meaningful_partial
        ):
            await self.scheduler.complete(
                lease.task.task_id,
                TaskCompletion(
                    result_id=task_result.result_id,
                    output_artifact_ids=output_artifacts,
                    usage=task_result.usage,
                ),
                worker_id=self.worker_id,
                mutation_id=_stable_id(
                    "mutation",
                    worker_result_id,
                    "complete",
                ),
            )
        elif task_result.status == TaskResultStatus.DEFERRED:
            usage_record = await self.scheduler.update_usage(
                lease.task.task_id,
                task_result.usage,
                worker_id=self.worker_id,
                mutation_id=_stable_id(
                    "mutation",
                    worker_result_id,
                    "usage",
                ),
            )
            if usage_record.envelope.status != TaskStatus.FAILED:
                await self.scheduler.request_approval(
                    lease.task.task_id,
                    worker_id=self.worker_id,
                    reason=task_result.summary,
                    mutation_id=_stable_id(
                        "mutation",
                        worker_result_id,
                        "approval",
                    ),
                )
        elif task_result.status == TaskResultStatus.CANCELLED:
            usage_record = await self.scheduler.update_usage(
                lease.task.task_id,
                task_result.usage,
                worker_id=self.worker_id,
                mutation_id=_stable_id(
                    "mutation",
                    worker_result_id,
                    "usage",
                ),
            )
            if usage_record.envelope.status != TaskStatus.FAILED:
                await self.scheduler.cancel_task(
                    lease.task.task_id,
                    reason=task_result.summary,
                    actor_id=self.worker_id,
                    mutation_id=_stable_id(
                        "mutation",
                        worker_result_id,
                        "cancel",
                    ),
                )
        else:
            assert error is not None
            await self.scheduler.fail(
                lease.task.task_id,
                error_ref=error.error_id,
                worker_id=self.worker_id,
                mutation_id=_stable_id(
                    "mutation",
                    worker_result_id,
                    "fail",
                ),
                usage=task_result.usage,
            )
            if retry_scheduled:
                await self.scheduler.retry(
                    lease.task.task_id,
                    actor_id=self.worker_id,
                    mutation_id=_stable_id(
                        "mutation",
                        worker_result_id,
                        "retry",
                    ),
                )
        return result


class ResearchWorkerResultReconciler:
    """Repairs a crash between durable Worker intent and scheduler mutation."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        coordination: SQLiteResearchCoordinationStore,
        actor_id: str = "runtime_research_worker_reconciler",
    ) -> None:
        self.scheduler = scheduler
        self.coordination = coordination
        self.actor_id = actor_id

    async def reconcile(self, run_id: str) -> tuple[str, ...]:
        reconciled: list[str] = []
        for result in self.coordination.worker_results(run_id):
            snapshot = await self.scheduler.snapshot(run_id)
            record = snapshot.by_id.get(result.task_id)
            if record is None:
                raise RuntimeError(
                    "Worker result references a missing scheduler task: "
                    f"{result.task_id}"
                )
            attempt = record.envelope.attempt
            if result.task_attempt > attempt:
                raise RuntimeError(
                    "Worker result attempt is ahead of scheduler state"
                )
            if result.task_attempt < attempt:
                continue
            status = record.envelope.status
            if status == TaskStatus.FAILED and result.retry_scheduled:
                await self.scheduler.retry(
                    result.task_id,
                    actor_id=self.actor_id,
                    mutation_id=self._mutation(result, "retry"),
                )
                reconciled.append(result.worker_result_id)
                continue
            if status != TaskStatus.RUNNING:
                continue
            if record.lease_owner != result.worker_id:
                raise RuntimeError(
                    "Worker result lease owner does not match scheduler state"
                )
            try:
                await self._apply_running(
                    result,
                    task_kind=record.envelope.kind,
                )
            except SchedulerLeaseError:
                # The scheduler recovery pass will fence an expired attempt;
                # its governed dedup keys make the subsequent attempt safe.
                continue
            reconciled.append(result.worker_result_id)
        return tuple(dict.fromkeys(reconciled))

    async def _apply_running(
        self,
        result: ResearchWorkerResult,
        *,
        task_kind: TaskKind,
    ) -> None:
        meaningful_partial = bool(
            result.status == TaskResultStatus.PARTIAL
            and result.information_gain > 0
            and result.output_artifact_ids
            and (
                result.knowledge_keys
                or (
                    task_kind == TaskKind.SOURCE_DISCOVERY
                    and result.source_keys
                )
            )
        )
        if result.status == TaskResultStatus.SUCCEEDED or meaningful_partial:
            await self.scheduler.complete(
                result.task_id,
                TaskCompletion(
                    result_id=result.task_result_id,
                    output_artifact_ids=result.output_artifact_ids,
                    usage=result.usage,
                ),
                worker_id=result.worker_id,
                mutation_id=self._mutation(result, "complete"),
            )
            return
        if result.status == TaskResultStatus.DEFERRED:
            usage = await self.scheduler.update_usage(
                result.task_id,
                result.usage,
                worker_id=result.worker_id,
                mutation_id=self._mutation(result, "usage"),
            )
            if usage.envelope.status != TaskStatus.FAILED:
                await self.scheduler.request_approval(
                    result.task_id,
                    worker_id=result.worker_id,
                    reason=result.summary,
                    mutation_id=self._mutation(result, "approval"),
                )
            return
        if result.status == TaskResultStatus.CANCELLED:
            usage = await self.scheduler.update_usage(
                result.task_id,
                result.usage,
                worker_id=result.worker_id,
                mutation_id=self._mutation(result, "usage"),
            )
            if usage.envelope.status != TaskStatus.FAILED:
                await self.scheduler.cancel_task(
                    result.task_id,
                    actor_id=result.worker_id,
                    reason=result.summary,
                    mutation_id=self._mutation(result, "cancel"),
                )
            return
        await self.scheduler.fail(
            result.task_id,
            error_ref=(
                result.error_ref
                or _stable_id(
                    "error",
                    result.worker_result_id,
                    "worker_failure",
                )
            ),
            worker_id=result.worker_id,
            usage=result.usage,
            mutation_id=self._mutation(result, "fail"),
        )
        if result.retry_scheduled:
            await self.scheduler.retry(
                result.task_id,
                actor_id=self.actor_id,
                mutation_id=self._mutation(result, "retry"),
            )

    @staticmethod
    def _mutation(
        result: ResearchWorkerResult,
        operation: str,
    ) -> str:
        return _stable_id(
            "mutation",
            result.worker_result_id,
            operation,
        )


class ResearchWorkerPool:
    def __init__(
        self,
        *,
        scheduler: Scheduler,
        runners: dict[str, ResearchWorkerRunner],
        lease_seconds: float = 120.0,
        max_claim_rounds: int = 100,
        clock=utc_now,
    ) -> None:
        if not runners:
            raise ValueError("worker pool requires at least one runner")
        if set(runners) != {
            runner.worker_id for runner in runners.values()
        }:
            raise ValueError("worker runner keys must match worker IDs")
        if lease_seconds <= 0 or max_claim_rounds < 1:
            raise ValueError("worker pool bounds must be positive")
        self.scheduler = scheduler
        self.runners = runners
        self.lease_seconds = lease_seconds
        self.max_claim_rounds = max_claim_rounds
        self.clock = clock
        self._tokens: dict[str, CancellationToken] = {}

    async def drain(self, run_id: str) -> tuple[ResearchWorkerResult, ...]:
        await self.scheduler.recover(
            run_id,
            actor_id="runtime_research_worker_pool",
            mutation_id=_stable_id(
                "mutation",
                run_id,
                self.clock().isoformat(),
                "recover",
            ),
        )
        results: list[ResearchWorkerResult] = []
        for round_no in range(self.max_claim_rounds):
            claims = await asyncio.gather(
                *(
                    self.scheduler.claim(
                        run_id,
                        worker_id=worker_id,
                        limit=1,
                        lease_seconds=self.lease_seconds,
                        mutation_id=_stable_id(
                            "mutation",
                            run_id,
                            worker_id,
                            self.clock().isoformat(),
                            str(round_no),
                        ),
                    )
                    for worker_id in self.runners
                )
            )
            leases = [
                lease
                for worker_claims in claims
                for lease in worker_claims
            ]
            if not leases:
                break
            tasks = []
            for lease in leases:
                token = CancellationToken()
                self._tokens[lease.task.task_id] = token
                tasks.append(
                    self.runners[lease.worker_id].run_lease(
                        lease,
                        cancellation=token,
                    )
                )
            try:
                round_results = await asyncio.gather(*tasks)
            finally:
                for lease in leases:
                    self._tokens.pop(lease.task.task_id, None)
            results.extend(round_results)
        else:
            snapshot = await self.scheduler.snapshot(run_id)
            worker_ids = set(self.runners)
            runnable = tuple(
                item.task_id
                for item in snapshot.tasks
                if item.envelope.status == TaskStatus.READY
                and (
                    item.envelope.assigned_actor_id is None
                    or item.envelope.assigned_actor_id in worker_ids
                )
            )
            if runnable:
                raise RuntimeError(
                    "Worker pool exhausted max_claim_rounds with runnable "
                    f"tasks remaining: {list(runnable)}"
                )
        return tuple(results)

    def cancel_active(self) -> tuple[str, ...]:
        task_ids = tuple(self._tokens)
        for token in self._tokens.values():
            token.cancel()
        return task_ids


class CrossWorkerResultMerger:
    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        coordination: SQLiteResearchCoordinationStore,
        producer_id: str = "runtime_research_result_merger",
        clock=utc_now,
    ) -> None:
        self.artifact_store = artifact_store
        self.coordination = coordination
        self.producer_id = producer_id
        self.clock = clock

    def merge(self, run_id: str) -> MergedResearchResult | None:
        results = self.coordination.worker_results(run_id)
        if not results:
            return None
        worker_result_ids = tuple(
            item.worker_result_id for item in results
        )
        fingerprint = hashlib.sha256(
            "\0".join(worker_result_ids).encode("utf-8")
        ).hexdigest()
        merge_id = _stable_id(
            "research_merge",
            run_id,
            fingerprint,
        )
        existing = next(
            (
                item
                for item in self.coordination.merges(run_id)
                if item.merge_id == merge_id
            ),
            None,
        )
        if existing is not None:
            if self.artifact_store.get(existing.merge_artifact_id) is None:
                raise RuntimeError(
                    "Persisted research merge references a missing artifact"
                )
            return existing
        artifact_id = _stable_id("artifact", merge_id)
        artifact_ids = tuple(
            dict.fromkeys(
                artifact_id
                for item in results
                for artifact_id in item.output_artifact_ids
            )
        )
        merged = MergedResearchResult(
            merge_id=merge_id,
            run_id=run_id,
            worker_result_ids=worker_result_ids,
            task_ids=tuple(
                dict.fromkeys(item.task_id for item in results)
            ),
            artifact_ids=artifact_ids,
            query_keys=tuple(
                dict.fromkeys(
                    key
                    for item in results
                    for key in item.query_keys
                )
            ),
            source_keys=tuple(
                dict.fromkeys(
                    key
                    for item in results
                    for key in item.source_keys
                )
            ),
            knowledge_keys=tuple(
                dict.fromkeys(
                    key
                    for item in results
                    for key in item.knowledge_keys
                )
            ),
            total_information_gain=sum(
                item.information_gain for item in results
            ),
            merge_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        self.artifact_store.put_json(
            {
                "schema": "MergedResearchResult@1",
                "result": merged.model_dump(mode="json"),
                "worker_results": [
                    item.model_dump(mode="json")
                    for item in results
                ],
            },
            redact=True,
            kind=ArtifactKind.RESEARCH_MERGE,
            producer_id=self.producer_id,
            run_id=run_id,
            content_schema="MergedResearchResult@1",
            source_artifact_ids=artifact_ids,
            artifact_id=artifact_id,
            idempotency_key=f"research-merge:{merge_id}",
        )
        self.coordination.save_merge(merged)
        return merged
