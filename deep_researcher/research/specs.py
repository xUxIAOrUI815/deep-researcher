from __future__ import annotations

from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    Budget,
    CommandKind,
    ComponentKind,
    MiddlewareSpec,
    MiddlewareStage,
    VersionRef,
)


def _version(
    kind: ComponentKind,
    name: str,
    version: str,
) -> VersionRef:
    return VersionRef(
        version_id=(
            f"version_{name.replace('-', '_')}_{version.replace('.', '_')}"
        ),
        kind=kind,
        name=name,
        version=version,
    )


def _middleware() -> tuple[MiddlewareSpec, ...]:
    order = (
        MiddlewareStage.CONTEXT_TRIMMING,
        MiddlewareStage.REDACTION,
        MiddlewareStage.VERSION_INJECTION,
        MiddlewareStage.BUDGET_CHECK,
        MiddlewareStage.SCHEMA_VALIDATION,
        MiddlewareStage.SCHEMA_REPAIR,
        MiddlewareStage.COMMAND_NORMALIZATION,
        MiddlewareStage.POLICY_CHECK,
    )
    return tuple(
        MiddlewareSpec(stage=stage, order=index)
        for index, stage in enumerate(order)
    )


def build_research_supervisor_spec(
    *,
    version: str = "1.0.0",
    model: VersionRef | None = None,
    prompt: VersionRef | None = None,
    stop_policy: VersionRef | None = None,
) -> AgentSpec:
    return AgentSpec(
        agent_spec_id=(
            f"agent_spec_research_supervisor_{version.replace('.', '_')}"
        ),
        name="Research Supervisor",
        version=version,
        role=AgentRole.RESEARCH_SUPERVISOR,
        description=(
            "Dynamically decomposes and replans research work from evidence "
            "coverage, conflicts, information gain, task state, and remaining "
            "budgets. It delegates only structured TaskEnvelopes and never "
            "calls providers or writes report prose."
        ),
        input_schema="SupervisorPlanContext@1",
        output_schema="SupervisorPlan@1",
        allowed_commands=(
            CommandKind.DELEGATE,
            CommandKind.REQUEST_APPROVAL,
            CommandKind.STOP,
        ),
        model=model
        or _version(
            ComponentKind.MODEL,
            "research-supervisor-model",
            version,
        ),
        prompt=prompt
        or _version(
            ComponentKind.PROMPT,
            "research-supervisor-prompt",
            version,
        ),
        tool_policy=_version(
            ComponentKind.TOOL_POLICY,
            "supervisor-no-provider-tools",
            version,
        ),
        stop_policy=stop_policy
        or _version(
            ComponentKind.STOP_POLICY,
            "supervisor-semantic-convergence",
            version,
        ),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "supervisor-plan-validation",
            version,
        ),
        default_budget=Budget(
            max_tokens=48_000,
            max_cost_usd=8.0,
            max_wall_time_seconds=600.0,
            max_model_calls=24,
            max_tool_calls=1,
            max_search_calls=1,
            max_retries=5,
            max_errors=5,
        ),
        middleware=_middleware(),
        context_window_tokens=96_000,
        reserved_output_tokens=8_000,
        max_parallel_commands=1,
        supports_delegation=True,
        metadata={
            "provider_access": False,
            "writes_final_report": False,
            "free_form_agent_chat": False,
            "requires_dynamic_plan_schema": True,
            "records_hidden_chain_of_thought": False,
        },
    )


def build_research_worker_spec(
    *,
    version: str = "1.0.0",
    model: VersionRef | None = None,
    prompt: VersionRef | None = None,
    stop_policy: VersionRef | None = None,
    max_parallel_commands: int = 4,
) -> AgentSpec:
    return AgentSpec(
        agent_spec_id=(
            f"agent_spec_research_worker_{version.replace('.', '_')}"
        ),
        name="Research Worker",
        version=version,
        role=AgentRole.RESEARCH_WORKER,
        description=(
            "Executes bounded search, read, extract, compare, source "
            "verification, and structured delegation commands through the "
            "governed tool boundary. It returns artifact-backed research "
            "results and never writes final report prose."
        ),
        input_schema="TaskEnvelope@1",
        output_schema="ResearchWorkerResult@1",
        allowed_commands=(
            CommandKind.SEARCH,
            CommandKind.READ,
            CommandKind.EXTRACT,
            CommandKind.DELEGATE,
            CommandKind.COMPARE,
            CommandKind.VERIFY_SOURCE,
            CommandKind.REQUEST_APPROVAL,
            CommandKind.STOP,
        ),
        model=model
        or _version(
            ComponentKind.MODEL,
            "research-worker-model",
            version,
        ),
        prompt=prompt
        or _version(
            ComponentKind.PROMPT,
            "research-worker-prompt",
            version,
        ),
        tool_policy=_version(
            ComponentKind.TOOL_POLICY,
            "governed-research-tools",
            version,
        ),
        stop_policy=stop_policy
        or _version(
            ComponentKind.STOP_POLICY,
            "worker-information-gain",
            version,
        ),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "worker-observation-validation",
            version,
        ),
        default_budget=Budget(
            max_tokens=64_000,
            max_cost_usd=12.0,
            max_wall_time_seconds=900.0,
            max_model_calls=32,
            max_tool_calls=48,
            max_search_calls=16,
            max_retries=8,
            max_errors=8,
        ),
        middleware=_middleware(),
        context_window_tokens=96_000,
        reserved_output_tokens=8_000,
        max_parallel_commands=max_parallel_commands,
        supports_delegation=True,
        metadata={
            "allowed_capabilities": (
                "search",
                "read",
                "extract",
                "delegate",
                "compare",
                "verify_source",
            ),
            "writes_final_report": False,
            "free_form_agent_chat": False,
            "provider_access": "governed_gateway_only",
            "records_hidden_chain_of_thought": False,
        },
    )
