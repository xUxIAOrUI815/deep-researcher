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


def _version(kind: ComponentKind, name: str, version: str) -> VersionRef:
    return VersionRef(
        version_id=f"version_{name.replace('-', '_')}_{version.replace('.', '_')}",
        kind=kind,
        name=name,
        version=version,
    )


def build_evidence_verifier_spec(
    *,
    version: str = "1.0.0",
    model: VersionRef | None = None,
    prompt: VersionRef | None = None,
    verification_policy: VersionRef | None = None,
) -> AgentSpec:
    model_ref = model or _version(
        ComponentKind.MODEL, "evidence-verifier-model", version
    )
    prompt_ref = prompt or _version(
        ComponentKind.PROMPT, "evidence-verifier-prompt", version
    )
    verification_ref = verification_policy or _version(
        ComponentKind.VERIFICATION_POLICY,
        "evidence-verification-policy",
        version,
    )
    middleware_stages = (
        MiddlewareStage.CONTEXT_TRIMMING,
        MiddlewareStage.REDACTION,
        MiddlewareStage.VERSION_INJECTION,
        MiddlewareStage.BUDGET_CHECK,
        MiddlewareStage.SCHEMA_VALIDATION,
        MiddlewareStage.SCHEMA_REPAIR,
        MiddlewareStage.COMMAND_NORMALIZATION,
        MiddlewareStage.POLICY_CHECK,
    )
    return AgentSpec(
        agent_spec_id=f"agent_spec_evidence_verifier_{version.replace('.', '_')}",
        name="Evidence Verifier",
        version=version,
        role=AgentRole.EVIDENCE_VERIFIER,
        description=(
            "Independently verifies quote grounding, evidence-to-claim support, overreach, "
            "contradictions, source independence, authority/freshness, citations, conflicts, "
            "and section coverage without searching or writing report prose."
        ),
        input_schema="EvidenceVerificationInput@1",
        output_schema="VerificationResult@1",
        allowed_commands=(CommandKind.REVIEW, CommandKind.STOP),
        model=model_ref,
        prompt=prompt_ref,
        tool_policy=_version(
            ComponentKind.TOOL_POLICY, "evidence-verifier-no-tools", version
        ),
        stop_policy=_version(
            ComponentKind.STOP_POLICY, "bounded-evidence-verification", version
        ),
        verification_policy=verification_ref,
        default_budget=Budget(
            max_tokens=32_000,
            max_cost_usd=5.0,
            max_wall_time_seconds=300.0,
            max_model_calls=20,
            max_tool_calls=1,
            max_retries=3,
            max_errors=3,
        ),
        middleware=tuple(
            MiddlewareSpec(stage=stage, order=index)
            for index, stage in enumerate(middleware_stages)
        ),
        context_window_tokens=64_000,
        reserved_output_tokens=4_000,
        max_parallel_commands=1,
        supports_delegation=False,
        metadata={
            "independent_from": ("research_worker", "distiller", "synthesis_writer"),
            "forbidden_capabilities": (
                "search",
                "source_snapshot_mutation",
                "report_writing",
            ),
            "records_hidden_chain_of_thought": False,
        },
    )
