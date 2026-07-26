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

from .models import ReviewDimension


def _version(kind: ComponentKind, name: str, version: str) -> VersionRef:
    return VersionRef(
        version_id=f"version_{name}_{version.replace('.', '_')}",
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
        MiddlewareSpec(stage=item, order=index)
        for index, item in enumerate(order)
    )


def build_synthesis_writer_spec(
    *,
    version: str = "1.0.0",
) -> AgentSpec:
    return AgentSpec(
        agent_spec_id=f"agent_spec_synthesis_writer_{version.replace('.', '_')}",
        name="Synthesis Writer",
        version=version,
        role=AgentRole.SYNTHESIS_WRITER,
        description=(
            "Synthesizes report sections exclusively from a verified evidence "
            "packet. Every factual statement declares its verified claims and "
            "citations; gaps and conflicts remain explicit."
        ),
        input_schema="WriterEvidencePacket@1",
        output_schema="WriterDraftProposal@1",
        allowed_commands=(CommandKind.SYNTHESIZE, CommandKind.STOP),
        model=_version(ComponentKind.MODEL, "synthesis_writer_model", version),
        prompt=_version(ComponentKind.PROMPT, "synthesis_writer_prompt", version),
        tool_policy=_version(
            ComponentKind.TOOL_POLICY,
            "writer_no_tools_verified_only",
            version,
        ),
        stop_policy=_version(
            ComponentKind.STOP_POLICY,
            "writer_draft_complete",
            version,
        ),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "writer_traceability",
            version,
        ),
        default_budget=Budget(
            max_tokens=80_000,
            max_cost_usd=15.0,
            max_wall_time_seconds=900,
            max_model_calls=20,
            max_tool_calls=1,
            max_search_calls=1,
            max_retries=4,
            max_errors=4,
        ),
        middleware=_middleware(),
        context_window_tokens=128_000,
        reserved_output_tokens=24_000,
        max_parallel_commands=1,
        metadata={
            "verified_only": True,
            "provider_tools": False,
            "search": False,
            "candidate_evidence": False,
            "free_form_factual_prose": False,
        },
    )


def build_report_reviewer_spec(
    *,
    version: str = "1.0.0",
) -> AgentSpec:
    return AgentSpec(
        agent_spec_id=f"agent_spec_report_reviewer_{version.replace('.', '_')}",
        name="Report Reviewer",
        version=version,
        role=AgentRole.REPORT_REVIEWER,
        description=(
            "Scores every required report rubric dimension and emits bounded "
            "targeted-research, citation-repair, local-rewrite, structural-"
            "rewrite, accept, or reject decisions without changing evidence."
        ),
        input_schema="ReportReviewContext@1",
        output_schema="ReviewerDecision@1",
        allowed_commands=(CommandKind.REVIEW, CommandKind.STOP),
        model=_version(ComponentKind.MODEL, "report_reviewer_model", version),
        prompt=_version(ComponentKind.PROMPT, "report_reviewer_prompt", version),
        tool_policy=_version(
            ComponentKind.TOOL_POLICY,
            "reviewer_no_tools_read_only",
            version,
        ),
        stop_policy=_version(
            ComponentKind.STOP_POLICY,
            "reviewer_bounded_decision",
            version,
        ),
        verification_policy=_version(
            ComponentKind.VERIFICATION_POLICY,
            "report_rubric",
            version,
        ),
        default_budget=Budget(
            max_tokens=48_000,
            max_cost_usd=10.0,
            max_wall_time_seconds=600,
            max_model_calls=16,
            max_tool_calls=1,
            max_search_calls=1,
            max_retries=4,
            max_errors=4,
        ),
        middleware=_middleware(),
        context_window_tokens=128_000,
        reserved_output_tokens=12_000,
        max_parallel_commands=1,
        metadata={
            "evidence_mutation": False,
            "provider_tools": False,
            "search": False,
            "rubric_dimensions": tuple(
                item.value
                for item in ReviewDimension
            ),
        },
    )
