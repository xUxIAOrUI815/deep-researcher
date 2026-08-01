from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import ValidationError

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    BudgetUsage,
    Command,
    CommandKind,
    Observation,
    ObservationStatus,
    StopReason,
    TaskEnvelope,
    TaskKind,
    TaskStatus,
    utc_now,
)
from deep_researcher.kernel import (
    ModelAdapter,
    ModelInvocationError,
    ModelRequest,
    ModelResponse,
    RawObservation,
    VerificationFeedback,
)
from deep_researcher.orchestration import Scheduler

from .models import (
    SupervisorPlan,
    SupervisorPlanAction,
)
from .store import SQLiteResearchCoordinationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _usage_sum(items: list[BudgetUsage]) -> BudgetUsage:
    usage = BudgetUsage()
    for item in items:
        usage = usage.plus(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cost_usd=item.cost_usd,
            wall_time_seconds=item.wall_time_seconds,
            model_calls=max(1, item.model_calls),
            tool_calls=item.tool_calls,
            search_calls=item.search_calls,
            retries=item.retries,
            errors=item.errors,
        )
    return usage


class SupervisorPlanningModelAdapter(ModelAdapter):
    """Validates and repairs model-driven plans before AgentKernel sees them."""

    def __init__(
        self,
        model: ModelAdapter,
        *,
        max_plan_repairs: int = 2,
        clock=utc_now,
    ) -> None:
        if max_plan_repairs < 0:
            raise ValueError("max_plan_repairs cannot be negative")
        self.model = model
        self.max_plan_repairs = max_plan_repairs
        self.clock = clock

    async def complete(self, request: ModelRequest) -> ModelResponse:
        specialized = self._request(request)
        response = await self.model.complete(specialized)
        return await self._validated_response(
            request=specialized,
            response=response,
            previous_usage=[],
        )

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        specialized = self._request(request)
        response = await self.model.repair(
            specialized,
            invalid_response,
            errors,
        )
        return await self._validated_response(
            request=specialized,
            response=response,
            previous_usage=[],
        )

    async def _validated_response(
        self,
        *,
        request: ModelRequest,
        response: ModelResponse,
        previous_usage: list[BudgetUsage],
    ) -> ModelResponse:
        usage_items = [*previous_usage, response.usage]
        current = response
        for attempt in range(self.max_plan_repairs + 1):
            try:
                plan = self._parse_plan(request, current)
            except (ValidationError, TypeError, ValueError) as exc:
                if attempt >= self.max_plan_repairs:
                    raise ModelInvocationError(
                        "Supervisor plan remained invalid after bounded repair: "
                        f"{exc}",
                        retryable=False,
                    ) from exc
                current = await self.model.repair(
                    request,
                    current,
                    (str(exc),),
                )
                usage_items.append(
                    current.usage.model_copy(
                        update={
                            "retries": current.usage.retries + 1,
                        }
                    )
                )
                continue
            return self._to_kernel_response(
                plan,
                current,
                usage=_usage_sum(usage_items),
            )
        raise AssertionError("unreachable supervisor plan repair loop")

    def _parse_plan(
        self,
        request: ModelRequest,
        response: ModelResponse,
    ) -> SupervisorPlan:
        payload: Any = response.structured
        if isinstance(payload, dict) and "plan" in payload:
            payload = payload["plan"]
        if payload is None:
            text = response.content.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.casefold().startswith("json"):
                    text = text[4:]
            payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Supervisor must return a structured plan object")
        context = self._context(request)
        cycle = int(context.get("cycle", 0))
        objective = str(
            payload.get("objective")
            or context.get("objective")
            or "Resolve current research evidence gaps."
        )
        material = {
            **payload,
            "run_id": request.run_id,
            "root_task_id": request.task_id,
            "cycle": cycle,
            "objective": objective,
        }
        material.pop("plan_id", None)
        material.pop("created_at", None)
        fingerprint = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        material.update(
            plan_id=_stable_id(
                "plan",
                request.run_id,
                request.task_id,
                str(cycle),
                fingerprint,
            ),
            created_at=self.clock(),
        )
        plan = SupervisorPlan.model_validate(material, strict=False)
        allow_stop = bool(context.get("allow_stop", False))
        if (
            plan.action
            in {
                SupervisorPlanAction.CONVERGED,
                SupervisorPlanAction.STOP,
            }
            and not allow_stop
        ):
            raise ValueError(
                "Supervisor cannot stop before the authoritative convergence "
                "gate permits termination"
            )
        max_tasks = int(context.get("max_tasks_per_plan", 6))
        if len(plan.tasks) > max_tasks:
            raise ValueError(
                f"Supervisor plan exceeds max_tasks_per_plan={max_tasks}"
            )
        minimum_extraction_calls = int(
            context.get("minimum_model_calls_per_extraction_task", 3)
        )
        underfunded = {
            item.proposal_key: item.budget.max_model_calls
            for item in plan.tasks
            if item.kind != TaskKind.SOURCE_DISCOVERY
            and item.budget.max_model_calls is not None
            and item.budget.max_model_calls < minimum_extraction_calls
        }
        if underfunded:
            raise ValueError(
                "Research tasks must reserve enough model calls for read, "
                "extract, and bounded grounding repair; required "
                f"max_model_calls>={minimum_extraction_calls}: {underfunded}"
            )
        minimum_extraction_tokens = int(
            context.get("minimum_tokens_per_extraction_task", 0)
        )
        token_underfunded = {
            item.proposal_key: item.budget.max_tokens
            for item in plan.tasks
            if item.kind != TaskKind.SOURCE_DISCOVERY
            and minimum_extraction_tokens > 0
            and item.budget.max_tokens is not None
            and item.budget.max_tokens < minimum_extraction_tokens
        }
        if token_underfunded:
            raise ValueError(
                "Research tasks must reserve enough tokens for read, extract, "
                "and bounded grounding repair; required "
                f"max_tokens>={minimum_extraction_tokens}: {token_underfunded}"
            )
        allowed_workers = {
            str(item)
            for item in context.get("allowed_worker_ids", ())
            if str(item)
        }
        invalid_assignments = {
            item.assigned_actor_id
            for item in plan.tasks
            if item.assigned_actor_id is not None
            and item.assigned_actor_id not in allowed_workers
        }
        if invalid_assignments:
            raise ValueError(
                "Supervisor plan assigned tasks outside the configured Worker "
                f"Pool: {sorted(invalid_assignments)}"
            )
        return plan

    @staticmethod
    def _context(request: ModelRequest) -> dict[str, Any]:
        for message in request.messages:
            content = message.get("content")
            if not isinstance(content, dict):
                continue
            constraints = content.get("constraints")
            if isinstance(constraints, dict):
                context = constraints.get("supervisor_context")
                if isinstance(context, dict):
                    return context
        return {}

    @staticmethod
    def _request(request: ModelRequest) -> ModelRequest:
        return ModelRequest(
            run_id=request.run_id,
            task_id=request.task_id,
            actor_id=request.actor_id,
            system=(
                "Act as the Research Supervisor. Return one structured "
                "SupervisorPlan only. Dynamically decompose evidence gaps into "
                "research tasks with explicit dependencies, constraints, "
                "artifact inputs, schemas, budgets, priorities, deadlines, and "
                "attempt limits. Do not call providers, write report prose, or "
                "expose hidden reasoning. Use a short decision_summary. When "
                "the supervisor context reports persisted sources but no "
                "facts, evidence, or claims, the plan must advance those known "
                "sources into extraction/section-support work; it must not "
                "repeat broad source-discovery tasks. Extraction work should "
                "read a bounded set of known_source_candidates and then call "
                "research.extract with exact quotes. Obey the contextual "
                "max_tasks_per_plan exactly. Every non-source-discovery path "
                "must budget four turns and 64000 tokens for search, bounded "
                "source reading, exact-quote extraction, and one repair. Keep "
                "each extract to at most three claims."
            ),
            messages=request.messages,
            command_schema=SupervisorPlan.model_json_schema(),
            model_version=request.model_version,
            prompt_version=request.prompt_version,
            max_output_tokens=request.max_output_tokens,
            metadata={
                **request.metadata,
                "structured_output": "SupervisorPlan@1",
            },
        )

    @staticmethod
    def _to_kernel_response(
        plan: SupervisorPlan,
        response: ModelResponse,
        *,
        usage: BudgetUsage,
    ) -> ModelResponse:
        if plan.action in {
            SupervisorPlanAction.DECOMPOSE,
            SupervisorPlanAction.REPLAN,
        }:
            commands = [
                {
                    "kind": CommandKind.DELEGATE.value,
                    "name": "supervisor.apply_plan",
                    "arguments": {
                        "plan": plan.model_dump(mode="json"),
                    },
                    "input_artifact_ids": list(
                        dict.fromkeys(
                            artifact_id
                            for task in plan.tasks
                            for artifact_id in task.input_artifact_ids
                        )
                    ),
                    "expected_output_schema": "SupervisorPlanApplication@1",
                }
            ]
        elif plan.action == SupervisorPlanAction.REQUEST_APPROVAL:
            commands = [
                {
                    "kind": CommandKind.REQUEST_APPROVAL.value,
                    "name": "supervisor.request_approval",
                    "arguments": {
                        "reason": plan.approval_reason,
                        "plan": plan.model_dump(mode="json"),
                    },
                }
            ]
        else:
            commands = [
                {
                    "kind": CommandKind.STOP.value,
                    "name": "supervisor.stop",
                    "arguments": {
                        "reason": StopReason.SEMANTIC_COMPLETE.value,
                        "summary": plan.stop_reason,
                        "plan": plan.model_dump(mode="json"),
                    },
                }
            ]
        return ModelResponse(
            content=plan.decision_summary,
            structured={
                "summary": plan.decision_summary,
                "commands": commands,
                "supervisor_plan": plan.model_dump(mode="json"),
            },
            usage=usage,
            latency_ms=response.latency_ms,
            finish_reason=response.finish_reason,
            response_id=response.response_id,
        )


class SupervisorActionExecutor:
    """Materializes validated Supervisor plans into the durable task DAG."""

    def __init__(
        self,
        *,
        scheduler: Scheduler,
        artifact_store: ArtifactStore,
        coordination: SQLiteResearchCoordinationStore,
        actor_id: str,
        allowed_worker_ids: tuple[str, ...] = (),
        clock=utc_now,
    ) -> None:
        self.scheduler = scheduler
        self.artifact_store = artifact_store
        self.coordination = coordination
        self.actor_id = actor_id
        self.allowed_worker_ids = frozenset(allowed_worker_ids)
        self.clock = clock

    async def execute(self, command: Command) -> RawObservation:
        if (
            command.kind == CommandKind.REQUEST_APPROVAL
            and command.name == "supervisor.request_approval"
        ):
            plan = SupervisorPlan.model_validate(
                command.arguments.get("plan"),
                strict=False,
            )
            if (
                plan.run_id != command.run_id
                or plan.root_task_id != command.task_id
            ):
                raise ValueError(
                    "Supervisor approval plan identity does not match the command"
                )
            started = self.clock()
            artifact_id = _stable_id("artifact", plan.plan_id)
            self.artifact_store.put_json(
                {
                    "schema": "SupervisorApprovalRequest@1",
                    "plan": plan.model_dump(mode="json"),
                },
                redact=False,
                kind=ArtifactKind.SUPERVISOR_PLAN,
                producer_id=self.actor_id,
                run_id=plan.run_id,
                task_id=plan.root_task_id,
                content_schema="SupervisorApprovalRequest@1",
                artifact_id=artifact_id,
                idempotency_key=f"supervisor-approval:{plan.plan_id}",
            )
            return RawObservation(
                status=ObservationStatus.SUCCEEDED.value,
                data={
                    "plan_id": plan.plan_id,
                    "plan_artifact_id": artifact_id,
                    "request_approval": True,
                    "approval_reason": plan.approval_reason,
                    "semantic_complete": False,
                },
                output_artifact_ids=(artifact_id,),
                started_at=started,
                completed_at=self.clock(),
            )
        if (
            command.kind != CommandKind.DELEGATE
            or command.name != "supervisor.apply_plan"
        ):
            raise ValueError(
                "Supervisor action executor accepts only validated plan delegation"
            )
        plan = SupervisorPlan.model_validate(
            command.arguments.get("plan"),
            strict=False,
        )
        if (
            plan.run_id != command.run_id
            or plan.root_task_id != command.task_id
        ):
            raise ValueError("Supervisor plan identity does not match the command")
        invalid_assignments = {
            item.assigned_actor_id
            for item in plan.tasks
            if item.assigned_actor_id is not None
            and item.assigned_actor_id not in self.allowed_worker_ids
        }
        if invalid_assignments:
            raise ValueError(
                "Supervisor plan assigned tasks outside the configured Worker "
                f"Pool: {sorted(invalid_assignments)}"
            )
        started = self.clock()
        snapshot = await self.scheduler.snapshot(plan.run_id)
        existing_ids = set(snapshot.by_id)
        root_constraints = snapshot.by_id[plan.root_task_id].envelope.constraints
        inherited_runtime_constraints = {
            key: root_constraints[key]
            for key in (
                "user_instructions",
                "depth",
                "report_id",
                "required_section_id",
                "report_section_ids",
                "available_worker_tools",
                "worker_tool_contracts",
                "evidence_rules",
                "known_source_candidates",
            )
            if key in root_constraints
        }
        proposal_fingerprints = self._proposal_fingerprints(plan.tasks)
        proposed_task_ids = {
            key: _stable_id(
                "task",
                plan.run_id,
                plan.root_task_id,
                fingerprint,
            )
            for key, fingerprint in proposal_fingerprints.items()
        }
        reservations = {
            item.proposal_key: self.coordination.reserve_task(
                run_id=plan.run_id,
                fingerprint=proposal_fingerprints[item.proposal_key],
                task_id=proposed_task_ids[item.proposal_key],
            )
            for item in plan.tasks
        }
        canonical_ids = {
            key: reservation.canonical_task_id
            for key, reservation in reservations.items()
        }
        children: list[TaskEnvelope] = []
        now = self.clock()
        for proposal in plan.tasks:
            task_id = canonical_ids[proposal.proposal_key]
            if task_id in existing_ids:
                continue
            dependencies = tuple(
                dict.fromkeys(
                    canonical_ids[item]
                    for item in proposal.dependency_keys
                )
            )
            children.append(
                TaskEnvelope(
                    task_id=task_id,
                    run_id=plan.run_id,
                    parent_task_id=plan.root_task_id,
                    dependency_task_ids=dependencies,
                    kind=proposal.kind,
                    status=TaskStatus.PENDING,
                    title=proposal.title,
                    goal=proposal.goal,
                    constraints={
                        **inherited_runtime_constraints,
                        **proposal.constraints,
                        "supervisor_plan_id": plan.plan_id,
                        "proposal_key": proposal.proposal_key,
                    },
                    input_artifact_ids=proposal.input_artifact_ids,
                    expected_output_schema=proposal.expected_output_schema,
                    budget=proposal.budget,
                    priority=proposal.priority,
                    deadline=proposal.deadline,
                    max_attempts=proposal.max_attempts,
                    created_by=self.actor_id,
                    assigned_actor_id=proposal.assigned_actor_id,
                    tags=tuple(
                        dict.fromkeys(
                            (
                                *proposal.tags,
                                "background001",
                                "research_worker",
                            )
                        )
                    ),
                    created_at=now,
                    updated_at=now,
                )
            )
        plan_artifact_id = _stable_id("artifact", plan.plan_id)
        source_artifacts = tuple(
            dict.fromkeys(
                (
                    *command.input_artifact_ids,
                    *(
                        artifact_id
                        for item in plan.tasks
                        for artifact_id in item.input_artifact_ids
                    ),
                )
            )
        )
        application = {
            "schema": "SupervisorPlanApplication@1",
            "plan": plan.model_dump(mode="json"),
            "task_ids_by_proposal": canonical_ids,
            "new_task_ids": [item.task_id for item in children],
            "duplicate_task_ids": [
                reservation.canonical_task_id
                for reservation in reservations.values()
                if reservation.canonical_task_id
                != reservation.task_id
                or reservation.canonical_task_id in existing_ids
            ],
        }
        existing_application = self.artifact_store.get(plan_artifact_id)
        if existing_application is None:
            self.artifact_store.put_json(
                application,
                redact=False,
                kind=ArtifactKind.SUPERVISOR_PLAN,
                producer_id=self.actor_id,
                run_id=plan.run_id,
                task_id=plan.root_task_id,
                content_schema="SupervisorPlanApplication@1",
                source_artifact_ids=source_artifacts,
                artifact_id=plan_artifact_id,
                idempotency_key=f"supervisor-plan:{plan.plan_id}",
                metadata={
                    "cycle": plan.cycle,
                    "action": plan.action.value,
                    "task_count": len(plan.tasks),
                },
            )
            recorded_application = application
        else:
            try:
                recorded_application = json.loads(
                    self.artifact_store.read_bytes(
                        plan_artifact_id
                    ).decode("utf-8")
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "Existing Supervisor plan artifact is not valid JSON"
                ) from exc
            if (
                recorded_application.get("schema")
                != "SupervisorPlanApplication@1"
                or recorded_application.get("plan")
                != plan.model_dump(mode="json")
                or recorded_application.get("task_ids_by_proposal")
                != canonical_ids
            ):
                raise ValueError(
                    "Existing Supervisor plan artifact conflicts with replay"
                )
        if children:
            await self.scheduler.split(
                plan.root_task_id,
                tuple(children),
                actor_id=self.actor_id,
                mutation_id=_stable_id("mutation", plan.plan_id, "split"),
            )
        recorded_new_ids = list(
            recorded_application.get("new_task_ids", ())
        )
        return RawObservation(
            status=ObservationStatus.SUCCEEDED.value,
            data={
                "plan_id": plan.plan_id,
                "plan_artifact_id": plan_artifact_id,
                "child_task_ids": list(canonical_ids.values()),
                "new_task_ids": recorded_new_ids,
                "duplicate_count": len(plan.tasks) - len(recorded_new_ids),
                "semantic_complete": False,
            },
            output_artifact_ids=(plan_artifact_id,),
            started_at=started,
            completed_at=self.clock(),
        )

    @classmethod
    def _proposal_fingerprints(
        cls,
        proposals: tuple[Any, ...],
    ) -> dict[str, str]:
        """Fingerprint a task together with the semantics of its prerequisites."""
        by_key = {item.proposal_key: item for item in proposals}
        output: dict[str, str] = {}
        visiting: set[str] = set()

        def resolve(key: str) -> str:
            if key in output:
                return output[key]
            if key in visiting:
                raise ValueError("supervisor task dependency cycle detected")
            visiting.add(key)
            proposal = by_key[key]
            dependency_fingerprints = tuple(
                sorted(resolve(item) for item in proposal.dependency_keys)
            )
            output[key] = cls._task_fingerprint(
                proposal,
                dependency_fingerprints=dependency_fingerprints,
            )
            visiting.remove(key)
            return output[key]

        for proposal in proposals:
            resolve(proposal.proposal_key)
        return output

    @staticmethod
    def _task_fingerprint(
        proposal: Any,
        *,
        dependency_fingerprints: tuple[str, ...],
    ) -> str:
        value = {
            "kind": proposal.kind.value,
            "goal": proposal.goal,
            "constraints": proposal.constraints,
            "input_artifact_ids": list(proposal.input_artifact_ids),
            "expected_output_schema": proposal.expected_output_schema,
            "dependency_fingerprints": list(dependency_fingerprints),
        }
        return hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


class SupervisorPlanVerifier:
    async def verify(
        self,
        *,
        spec: Any,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        del spec, task, command, prior_observations
        if observation.status != ObservationStatus.SUCCEEDED:
            return VerificationFeedback(
                passed=False,
                summary="Supervisor plan could not be applied to the task DAG.",
                repair_feedback=("Produce a valid dependency-safe plan.",),
            )
        if observation.normalized_data.get("request_approval") is True:
            return VerificationFeedback(
                passed=True,
                success=True,
                information_gain=0.0,
                summary="Supervisor approval request was durably recorded.",
            )
        task_ids = tuple(
            observation.normalized_data.get("child_task_ids", ())
        )
        if not task_ids:
            return VerificationFeedback(
                passed=False,
                information_gain=0.0,
                summary="Supervisor plan produced no canonical task.",
                repair_feedback=(
                    "Replan with at least one non-duplicate research task.",
                ),
            )
        new_ids = tuple(
            observation.normalized_data.get("new_task_ids", ())
        )
        return VerificationFeedback(
            passed=True,
            success=True,
            information_gain=min(1.0, len(new_ids) / max(1, len(task_ids))),
            summary=(
                f"Applied {len(task_ids)} canonical research task(s), "
                f"including {len(new_ids)} new DAG node(s)."
            ),
        )
