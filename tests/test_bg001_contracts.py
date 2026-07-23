from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import Field, ValidationError

from deep_researcher.contracts import (
    AgentRole,
    AgentSpec,
    ArtifactEnvelope,
    ArtifactKind,
    Budget,
    BudgetDimension,
    BudgetUsage,
    Claim,
    ClaimStatus,
    CommandKind,
    ComponentKind,
    ComponentVersionSet,
    ContractModel,
    DatasetAccessRequest,
    DatasetPurpose,
    DatasetSplit,
    EntityProvenance,
    ErrorCategory,
    ErrorRecord,
    EvaluationMetric,
    EvaluationResult,
    EventType,
    Evidence,
    EvidenceRelation,
    EvidenceStatus,
    MetricDirection,
    MiddlewareSpec,
    MiddlewareStage,
    Report,
    ReportStatus,
    RunEvent,
    RunStatus,
    SchemaMigrationRegistry,
    Section,
    SectionStatus,
    SerializedContract,
    Source,
    SourceStatus,
    SourceType,
    SpanKind,
    TaskEnvelope,
    TaskKind,
    TaskStatus,
    ToolGrant,
    VerificationCategory,
    VerificationIssue,
    VerificationResult,
    VerificationSeverity,
    VersionRef,
    canonical_contract_json,
    contract_fingerprint,
    utc_now,
)


def _version(kind: ComponentKind, name: str) -> VersionRef:
    return VersionRef(kind=kind, name=name, version="1.0.0")


def _component_versions() -> ComponentVersionSet:
    return ComponentVersionSet(
        runtime=_version(ComponentKind.RUNTIME, "runtime"),
        scheduler=_version(ComponentKind.SCHEDULER, "scheduler"),
        model=_version(ComponentKind.MODEL, "model"),
    )


def _provenance() -> EntityProvenance:
    return EntityProvenance(producer_id="agent_worker", run_id="run_contracts", task_id="task_contracts")


def _budget() -> Budget:
    return Budget(
        max_tokens=10_000,
        max_cost_usd=2.0,
        max_wall_time_seconds=60.0,
        max_model_calls=5,
        max_tool_calls=20,
        max_search_calls=10,
        max_retries=2,
        max_errors=2,
    )


def test_domain_contract_package_has_no_framework_provider_or_storage_imports():
    root = Path(__file__).resolve().parents[1] / "deep_researcher" / "contracts"
    forbidden_roots = {
        "aiosqlite",
        "fastapi",
        "httpx",
        "langchain",
        "langchain_core",
        "langgraph",
        "openai",
        "providers",
        "sqlite3",
    }
    violations: list[str] = []
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in forbidden_roots:
                    violations.append(f"{path.name}:{node.lineno}:{name}")
    assert violations == []


def test_task_contract_json_round_trip_and_invalid_transition():
    task = TaskEnvelope(
        run_id="run_contracts",
        kind=TaskKind.RESEARCH,
        title="Research a bounded question",
        goal="Collect and verify primary evidence.",
        expected_output_schema="EvidencePack@1",
        budget=_budget(),
        created_by="agent_supervisor",
    )
    decoded = TaskEnvelope.model_validate_json(task.model_dump_json())
    assert decoded == task
    ready = task.transition(TaskStatus.READY)
    running = ready.transition(TaskStatus.RUNNING)
    assert running.attempt == 1
    with pytest.raises(ValueError, match="invalid task transition"):
        running.transition(TaskStatus.PENDING)


def test_budget_enforces_all_dimensions_and_immutable_accumulation():
    budget = _budget()
    usage = BudgetUsage(input_tokens=5_000, output_tokens=5_000, tool_calls=20, search_calls=10)
    assert budget.exceeded_dimensions(usage) == (
        BudgetDimension.TOKENS,
        BudgetDimension.TOOL_CALLS,
        BudgetDimension.SEARCH_CALLS,
    )
    increased = usage.plus(retries=1, errors=1)
    assert usage.retries == 0
    assert increased.retries == 1
    with pytest.raises(ValueError, match="unknown budget"):
        usage.plus(requests=1)


def test_artifact_contract_requires_hash_provenance_and_valid_expiry():
    artifact = ArtifactEnvelope(
        kind=ArtifactKind.SOURCE_SNAPSHOT,
        content_uri="artifact://sha256/abc",
        content_hash="a" * 64,
        byte_length=128,
        media_type="text/html",
        producer_id="agent_worker",
        run_id="run_contracts",
        task_id="task_contracts",
    )
    assert ArtifactEnvelope.model_validate_json(artifact.model_dump_json()) == artifact
    with pytest.raises(ValidationError, match="expires_at"):
        ArtifactEnvelope(
            **artifact.model_dump(exclude={"expires_at"}),
            expires_at=artifact.created_at - timedelta(seconds=1),
        )


def test_event_contract_has_trace_causation_versions_and_terminal_rules():
    event = RunEvent(
        sequence_no=1,
        event_type=EventType.RUN_COMPLETED,
        status=RunStatus.SUCCEEDED,
        trace_id="trace_contracts",
        span_id="span_root",
        span_kind=SpanKind.RUN,
        correlation_id="correlation_contracts",
        run_id="run_contracts",
        thread_id="thread_contracts",
        actor_id="agent_supervisor",
        producer_id="runtime_primary",
        component_versions=_component_versions(),
    )
    assert RunEvent.model_validate_json(event.model_dump_json()) == event
    with pytest.raises(ValidationError, match="requires failed status"):
        RunEvent(**{**event.model_dump(), "event_type": EventType.RUN_FAILED})


def test_evidence_and_claim_state_machines_reject_invalid_or_unbacked_states():
    evidence = Evidence(
        passage_ids=("passage_one",),
        relation=EvidenceRelation.SUPPORTS,
        summary="The passage directly states the measured value.",
        confidence=0.9,
        relevance=0.95,
        source_quality=0.9,
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError, match="verification identity"):
        evidence.transition(EvidenceStatus.VERIFIED)
    verified = evidence.transition(
        EvidenceStatus.VERIFIED,
        verification_id="verification_contracts",
    )
    assert verified.status == EvidenceStatus.VERIFIED
    with pytest.raises(ValueError, match="invalid Evidence transition"):
        verified.transition(EvidenceStatus.PROPOSED)
    with pytest.raises(ValidationError, match="supported claims require"):
        Claim(
            statement="An unsupported statement",
            status=ClaimStatus.SUPPORTED,
            confidence=0.9,
            provenance=_provenance(),
        )


def test_source_section_and_report_terminal_content_rules():
    source = Source(
        canonical_url="https://example.test/source",
        source_type=SourceType.PRIMARY,
        provenance=_provenance(),
    )
    assert source.transition(SourceStatus.ACCESSIBLE).status == SourceStatus.ACCESSIBLE
    section = Section(
        report_id="report_contracts",
        title="Findings",
        goal="Present verified findings.",
        order=1,
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError, match="content artifact"):
        Section(**{**section.model_dump(), "status": SectionStatus.VERIFIED})
    report = Report(
        thread_id="thread_contracts",
        run_id="run_contracts",
        title="Report",
        research_question="What does the evidence support?",
        section_ids=(section.section_id,),
        provenance=_provenance(),
    )
    with pytest.raises(ValidationError, match="content artifact"):
        Report(**{**report.model_dump(), "status": ReportStatus.APPROVED})


def test_agent_spec_carries_governance_versions_and_complete_budget():
    spec = AgentSpec(
        name="Research Worker",
        version="1.0.0",
        role=AgentRole.RESEARCH_WORKER,
        description="Searches, reads, and extracts bounded evidence.",
        input_schema="TaskEnvelope@1",
        output_schema="TaskResult@1",
        allowed_commands=(CommandKind.SEARCH, CommandKind.READ, CommandKind.EXTRACT),
        tool_grants=(ToolGrant(tool_name="web", allowed_operations=("search", "open"), max_calls_per_task=20),),
        model=_version(ComponentKind.MODEL, "worker-model"),
        prompt=_version(ComponentKind.PROMPT, "worker-prompt"),
        tool_policy=_version(ComponentKind.TOOL_POLICY, "worker-tools"),
        stop_policy=_version(ComponentKind.STOP_POLICY, "worker-stop"),
        default_budget=_budget(),
        middleware=(
            MiddlewareSpec(stage=MiddlewareStage.REDACTION, order=0),
            MiddlewareSpec(stage=MiddlewareStage.BUDGET_CHECK, order=1),
            MiddlewareSpec(stage=MiddlewareStage.SCHEMA_VALIDATION, order=2),
        ),
        context_window_tokens=32_000,
        reserved_output_tokens=4_000,
    )
    assert AgentSpec.model_validate_json(spec.model_dump_json()) == spec
    with pytest.raises(ValidationError, match="supports_delegation"):
        AgentSpec(**{**spec.model_dump(), "allowed_commands": (CommandKind.DELEGATE,)})


def test_verification_result_cannot_pass_with_blocking_issue():
    issue = VerificationIssue(
        category=VerificationCategory.CITATION_ACCURACY,
        severity=VerificationSeverity.ERROR,
        code="citation_mismatch",
        message="The quote does not support the claim.",
        subject_id="citation_one",
    )
    with pytest.raises(ValidationError, match="blocking issues"):
        VerificationResult(
            run_id="run_contracts",
            verifier_id="agent_verifier",
            subject_id="claim_one",
            subject_type="claim",
            passed=True,
            score=0.9,
            threshold=0.8,
            checks_performed=(VerificationCategory.CITATION_ACCURACY,),
            issues=(issue,),
            policy_version_id="version_verification",
            started_at=utc_now(),
        )


@pytest.mark.parametrize(
    ("split", "purpose"),
    [
        (DatasetSplit.TRAIN, DatasetPurpose.TRAINING),
        (DatasetSplit.DEV, DatasetPurpose.DEVELOPMENT),
        (DatasetSplit.SELECTION, DatasetPurpose.CANDIDATE_SELECTION),
        (DatasetSplit.TEST, DatasetPurpose.FINAL_EVALUATION),
        (DatasetSplit.HIDDEN_TEST, DatasetPurpose.RELEASE_GATE),
    ],
)
def test_dataset_access_accepts_only_declared_purpose(split, purpose):
    request = DatasetAccessRequest(
        actor_id="evaluator_release",
        dataset_id="dataset_quality",
        split=split,
        purpose=purpose,
    )
    assert request.split == split


def test_dataset_access_blocks_leakage_and_evaluation_records_versions():
    with pytest.raises(ValidationError, match="cannot access"):
        DatasetAccessRequest(
            actor_id="optimizer_candidate",
            dataset_id="dataset_hidden",
            split=DatasetSplit.HIDDEN_TEST,
            purpose=DatasetPurpose.CANDIDATE_SELECTION,
        )
    metric = EvaluationMetric(
        name="citation_precision",
        value=0.95,
        direction=MetricDirection.HIGHER_IS_BETTER,
        threshold=0.9,
        passed=True,
        evaluator="deterministic-citation-checker",
    )
    result = EvaluationResult(
        run_id="run_evaluation",
        evaluator_id="evaluator_release",
        subject_version_id="version_candidate",
        dataset_id="dataset_test",
        dataset_split=DatasetSplit.TEST,
        purpose=DatasetPurpose.RELEASE_GATE,
        sample_count=100,
        metrics=(metric,),
        aggregate_score=0.95,
        passed=True,
        component_versions=_component_versions(),
        started_at=utc_now(),
    )
    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_schema_registry_migrates_explicitly_and_fingerprints_canonically():
    class V2Example(ContractModel):
        schema_version: str = "2.0.0"
        value: str = Field(min_length=1)

    registry = SchemaMigrationRegistry()
    registry.register_model(V2Example)
    registry.register_migration(
        "V2Example",
        "1.0.0",
        "2.0.0",
        lambda data: {"value": data["old_value"]},
    )
    decoded = registry.decode(
        SerializedContract(
            contract_type="V2Example",
            schema_version="1.0.0",
            data={"schema_version": "1.0.0", "old_value": "migrated"},
        )
    )
    encoded = registry.encode(decoded)
    assert decoded.value == "migrated"
    assert encoded.schema_version == "2.0.0"
    assert canonical_contract_json(decoded) == canonical_contract_json(decoded)
    assert contract_fingerprint(decoded) == contract_fingerprint(decoded)


def test_contracts_expose_no_hidden_reasoning_fields():
    forbidden = {"chain_of_thought", "cot", "hidden_reasoning", "private_reasoning"}
    contract_root = Path(__file__).resolve().parents[1] / "deep_researcher" / "contracts"
    for path in contract_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                assert node.target.id.lower() not in forbidden
    with pytest.raises(ValidationError, match="hidden reasoning"):
        RunEvent(
            sequence_no=1,
            event_type=EventType.RUN_STARTED,
            status=RunStatus.RUNNING,
            trace_id="trace_contracts",
            span_id="span_root",
            span_kind=SpanKind.RUN,
            correlation_id="correlation_contracts",
            run_id="run_contracts",
            thread_id="thread_contracts",
            actor_id="agent_supervisor",
            producer_id="runtime_primary",
            component_versions=_component_versions(),
            payload={"chain_of_thought": "must never be persisted"},
        )


def test_contracts_reject_naive_timestamps_and_unknown_fields():
    from datetime import datetime

    with pytest.raises(ValidationError, match="timezone-aware"):
        ErrorRecord(
            category=ErrorCategory.INTERNAL,
            code="bad_time",
            message="A naive timestamp was supplied.",
            occurred_at=datetime(2026, 1, 1),
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        Budget(max_tokens=10, unowned_field=True)
