from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import re

import pytest

from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    BudgetUsage,
    ClaimStatus,
    ComponentKind,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    MetricDirection,
    VersionRef,
)
from deep_researcher.evaluation_lab import (
    BlindJudgeCandidate,
    BlindJudgeRequest,
    BlindMultiJudgePanel,
    GateEvaluationEvidence,
    GateOutcome,
    GateStage,
    HumanRating,
    JudgeBallot,
    JudgeDimension,
    ModelSemanticJudgeAdapter,
    ReleaseGatePolicy,
    ReportRequirementObservation,
    RetrievalCaseObservation,
    RetrievalResultObservation,
    SemanticClaimObservation,
    SemanticEvaluationInput,
    build_evaluation_lab_runtime,
)
from deep_researcher.kernel import ModelResponse
from deep_researcher.version_registry import (
    VersionLifecycleState,
    VersionRegistryCorruption,
    build_version_registry_runtime,
)
from tests.test_bg001_evaluation_lab_core import (
    _sample_artifacts,
    _snapshot,
)


class StaticJudge:
    def __init__(self, suffix: str, bias: float) -> None:
        self._judge_version_id = f"version_judge_{suffix}"
        self._rubric_version_id = "version_rubric_semantic_1"
        self.bias = bias
        self.requests: list[BlindJudgeRequest] = []

    @property
    def judge_version_id(self) -> str:
        return self._judge_version_id

    @property
    def rubric_version_id(self) -> str:
        return self._rubric_version_id

    async def judge(self, request: BlindJudgeRequest) -> JudgeBallot:
        self.requests.append(request)
        matched = re.search(
            r"quality:([0-9.]+)",
            request.report_markdown,
        )
        if matched is None:
            raise ValueError("test candidate has no quality marker")
        quality = float(matched.group(1))
        score = max(0.0, min(1.0, quality + self.bias))
        passed = score >= 0.6
        return JudgeBallot(
            ballot_id=(
                f"judge_ballot_{self.judge_version_id}_"
                f"{request.blind_label}"
            ),
            panel_id=request.panel_id,
            blind_label=request.blind_label,
            judge_version_id=self.judge_version_id,
            rubric_version_id=self.rubric_version_id,
            scores={item: score for item in JudgeDimension},
            passed=passed,
            violations=() if passed else ("quality_below_threshold",),
            rationale_summary="Fixed-version blind rubric summary.",
            usage=BudgetUsage(input_tokens=20, output_tokens=10),
        )


class RepairingJudgeModel:
    def __init__(self) -> None:
        self.requests = []
        self.repairs = []

    async def complete(self, request):
        self.requests.append(request)
        return ModelResponse(structured={"not": "a ballot"})

    async def repair(self, request, invalid_response, errors):
        self.repairs.append((request, invalid_response, errors))
        return ModelResponse(
            structured={
                "ballot": {
                    "scores": {
                        item.value: 0.9 for item in JudgeDimension
                    },
                    "passed": True,
                    "violations": [],
                    "rationale_summary": "Concise repaired judge summary.",
                }
            },
            usage=BudgetUsage(input_tokens=30, output_tokens=15),
        )


def _version(
    artifacts: SQLiteArtifactStore,
    *,
    suffix: str,
    version: str,
) -> VersionRef:
    artifact = artifacts.put_json(
        {"skill": suffix, "version": version},
        redact=False,
        kind=ArtifactKind.SKILL,
        producer_id="test_version_author",
        run_id=f"run_version_{suffix}",
        content_schema="SkillVersion@1",
    )
    return VersionRef(
        version_id=f"version_skill_{suffix}",
        kind=ComponentKind.SKILL,
        name="research-skill",
        version=version,
        artifact_id=artifact.artifact_id,
        content_hash=artifact.content_hash,
    )


def _judge_candidates() -> tuple[BlindJudgeCandidate, ...]:
    return tuple(
        BlindJudgeCandidate(
            candidate_id=f"version_candidate_quality_{index}",
            report_markdown=f"quality:{quality} grounded report",
            instruction="Write a grounded research report.",
            evidence_summary="All listed claims have verified evidence.",
            requirement_summary="Cover facts, risks, and conclusion.",
        )
        for index, quality in enumerate((0.3, 0.6, 0.9), start=1)
    )


def _human_ratings(
    candidates: tuple[BlindJudgeCandidate, ...],
) -> tuple[HumanRating, ...]:
    output = []
    for candidate in candidates:
        quality = float(
            re.search(
                r"quality:([0-9.]+)",
                candidate.report_markdown,
            ).group(1)
        )
        output.append(
            HumanRating(
                rating_id=f"human_rating_{candidate.candidate_id}",
                candidate_id=candidate.candidate_id,
                reviewer_id="reviewer_calibration_one",
                scores={item: quality for item in JudgeDimension},
                passed=quality >= 0.6,
            )
        )
    return tuple(output)


def _semantic_input(
    *,
    subject_version_id: str,
    registered,
    access,
    snapshot,
    panel,
    candidate_id: str,
    calibration,
) -> SemanticEvaluationInput:
    claim_id = snapshot.sections[0].required_claim_ids[0]
    return SemanticEvaluationInput(
        subject_version_id=subject_version_id,
        bundle_id=registered.bundle.bundle_id,
        dataset_id=access.request.dataset_id,
        dataset_split=access.request.split,
        dataset_purpose=access.request.purpose,
        dataset_access_record_id=access.access_record_id,
        snapshot=snapshot,
        claims=(
            SemanticClaimObservation(
                claim_id=claim_id,
                status=ClaimStatus.SUPPORTED,
                support_score=0.95,
                factual=True,
                cited=True,
            ),
        ),
        retrieval_cases=(
            RetrievalCaseObservation(
                case_id=f"retrieval_case_{snapshot.snapshot_id}",
                relevant_source_ids=(
                    "source_relevant_one",
                    "source_relevant_two",
                ),
                results=(
                    RetrievalResultObservation(
                        source_id="source_relevant_one",
                        rank=1,
                        relevant=True,
                        authority_score=0.95,
                        fresh=True,
                        domain="primary.example",
                        source_type="official",
                    ),
                    RetrievalResultObservation(
                        source_id="source_relevant_two",
                        rank=2,
                        relevant=True,
                        authority_score=0.85,
                        fresh=True,
                        domain="secondary.example",
                        source_type="paper",
                    ),
                ),
            ),
        ),
        report_requirements=ReportRequirementObservation(
            required_topics=("facts", "risks"),
            covered_topics=("facts", "risks"),
            required_instructions=("cite evidence",),
            satisfied_instructions=("cite evidence",),
            word_count=1000,
            section_count=4,
            target_word_count=800,
            target_section_count=3,
        ),
        judge_panel_result_id=panel.panel_result_id,
        judge_candidate_id=candidate_id,
        calibration_id=calibration.calibration_id,
    )


def _metric(
    name: str,
    value: float,
    *,
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER,
) -> EvaluationMetric:
    return EvaluationMetric(
        name=name,
        value=value,
        direction=direction,
        evaluator="gate_test",
    )


def _gate_metrics(
    *,
    quality: float,
    cost: float,
    protocol: float = 1.0,
    safety: float = 0.98,
) -> tuple[EvaluationMetric, ...]:
    return (
        _metric("report_completeness_score", quality),
        _metric("report_readability_score", quality),
        _metric("report_safety_score", safety),
        _metric(
            "cost_usd",
            cost,
            direction=MetricDirection.LOWER_IS_BETTER,
        ),
        _metric("protocol_compliance_rate", protocol),
    )


def _gate_policy() -> ReleaseGatePolicy:
    return ReleaseGatePolicy(
        policy_id="release_policy_semantic_test",
        required_improvements={"report_completeness_score": 0.05},
        non_regression_tolerances={"report_readability_score": 0.05},
        maximum_cost_usd=1.0,
        maximum_cost_increase_fraction=0.25,
        maximum_variances={"report_completeness_score": 0.01},
        safety_minimums={"report_safety_score": 0.95},
        protocol_minimum=0.99,
        minimum_human_correlation=0.9,
        minimum_human_pass_agreement=0.9,
    )


@pytest.mark.parametrize(
    ("component_kind", "artifact_kind"),
    (
        (ComponentKind.AGENT_SPEC, ArtifactKind.POLICY),
        (ComponentKind.SKILL, ArtifactKind.SKILL),
        (ComponentKind.PROMPT, ArtifactKind.PROMPT),
        (ComponentKind.TOOL_POLICY, ArtifactKind.POLICY),
        (ComponentKind.STOP_POLICY, ArtifactKind.POLICY),
        (ComponentKind.RUBRIC, ArtifactKind.RUBRIC),
    ),
)
def test_version_registry_owns_every_releasable_component_kind(
    tmp_path,
    component_kind,
    artifact_kind,
):
    artifacts = SQLiteArtifactStore(
        tmp_path / f"artifacts-{component_kind.value}.sqlite3"
    )
    runtime = build_version_registry_runtime(
        tmp_path / f"versions-{component_kind.value}",
        artifact_store=artifacts,
    )
    try:
        content = artifacts.put_json(
            {"component": component_kind.value},
            redact=False,
            kind=artifact_kind,
            producer_id="test_component_author",
            run_id=f"run_component_{component_kind.value}",
            content_schema="ReleasableComponent@1",
        )
        ref = VersionRef(
            version_id=f"version_component_{component_kind.value}",
            kind=component_kind,
            name=f"component-{component_kind.value}",
            version="1.0.0",
            artifact_id=content.artifact_id,
            content_hash=content.content_hash,
        )
        record = runtime.registry.register(ref)
        assert record.state == VersionLifecycleState.CANDIDATE
        assert record.manifest.version_ref.kind == component_kind
        runtime.integrity_check()
    finally:
        runtime.close()
        artifacts.close()


@pytest.mark.asyncio
async def test_model_judge_is_fixed_blind_toolless_and_repairs_schema():
    model = RepairingJudgeModel()
    judge = ModelSemanticJudgeAdapter(
        model=model,
        judge_version=VersionRef(
            version_id="version_model_judge_fixed",
            kind=ComponentKind.MODEL,
            name="semantic-judge",
            version="1.0.0",
        ),
        rubric_version=VersionRef(
            version_id="version_rubric_judge_fixed",
            kind=ComponentKind.RUBRIC,
            name="semantic-rubric",
            version="1.0.0",
        ),
    )
    request = BlindJudgeRequest(
        panel_id="judge_panel_model_adapter",
        blind_label="candidate_012345abcdef",
        report_markdown="A blinded report.",
        instruction="Evaluate report quality.",
        evidence_summary="Verified evidence only.",
        requirement_summary="Required topics are listed.",
        rubric_dimensions=tuple(JudgeDimension),
        randomization_seed=7,
    )
    ballot = await judge.judge(request)

    assert ballot.passed is True
    assert len(model.requests) == 1
    assert len(model.repairs) == 1
    sent = model.requests[0]
    assert sent.metadata["blind"] is True
    assert sent.metadata["search_allowed"] is False
    assert sent.metadata["rewrite_allowed"] is False
    assert sent.command_schema["properties"]["ballot"]
    assert "subject_version" not in str(sent.messages)


@pytest.mark.asyncio
async def test_blind_randomized_multi_judge_voting_disagreement_and_calibration(
    tmp_path,
):
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    runtime = build_evaluation_lab_runtime(
        tmp_path / "evaluation",
        artifact_store=artifacts,
    )
    judges = (
        StaticJudge("one", -0.1),
        StaticJudge("two", 0.0),
        StaticJudge("three", 0.1),
    )
    try:
        candidates = _judge_candidates()
        panel = await BlindMultiJudgePanel(
            judges=judges,
            artifact_store=artifacts,
            store=runtime.store,
            disagreement_threshold=0.15,
        ).evaluate(candidates, randomization_seed=41)

        assert len(panel.ballots) == 9
        assert panel.blind_label_map
        assert all(
            not label.startswith("version_")
            for label in panel.blind_label_map
        )
        assert len(
            {
                order
                for order in panel.candidate_order_by_judge.values()
            }
        ) > 1
        assert panel.disagreements
        assert all(
            set(item.consensus_scores) == set(JudgeDimension)
            for item in panel.candidates
        )
        assert all(
            all(
                request.blind_label.startswith("candidate_")
                for request in judge.requests
            )
            for judge in judges
        )

        calibration = runtime.calibrator.calibrate(
            panel_result_ids=(panel.panel_result_id,),
            human_ratings=_human_ratings(candidates),
            minimum_correlation=0.9,
            minimum_pass_agreement=0.9,
        )
        assert calibration.accepted is True
        assert calibration.overall_human_correlation == pytest.approx(1.0)
        assert calibration.pass_agreement_rate == 1.0
        assert runtime.store.judge_panel(panel.panel_result_id) == panel
        assert (
            runtime.store.judge_calibration(calibration.calibration_id)
            == calibration
        )
        inverted_ratings = tuple(
            item.model_copy(
                update={
                    "rating_id": f"{item.rating_id}_inverted",
                    "reviewer_id": "reviewer_calibration_adversarial",
                    "scores": {
                        dimension: 1.0 - score
                        for dimension, score in item.scores.items()
                    },
                    "passed": not item.passed,
                }
            )
            for item in _human_ratings(candidates)
        )
        rejected_calibration = runtime.calibrator.calibrate(
            panel_result_ids=(panel.panel_result_id,),
            human_ratings=inverted_ratings,
            minimum_correlation=0.9,
            minimum_pass_agreement=0.9,
        )
        assert rejected_calibration.accepted is False
        assert rejected_calibration.overall_human_correlation < 0
        assert rejected_calibration.pass_agreement_rate == 0.0
        runtime.integrity_check()
    finally:
        runtime.close()
        artifacts.close()


@pytest.mark.asyncio
async def test_semantic_metrics_cover_claim_retrieval_report_and_deterministic(
    tmp_path,
):
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    runtime = build_evaluation_lab_runtime(
        tmp_path / "evaluation",
        artifact_store=artifacts,
    )
    subject = "version_candidate_quality_3"
    try:
        registered = runtime.datasets.register(
            name="semantic-dataset",
            version="1.0.0",
            sample_schema="SemanticInput@1",
            samples_by_split=_sample_artifacts(artifacts, "semantic"),
            description="Semantic evaluation dataset.",
        )
        access, _ = runtime.datasets.access(
            bundle_id=registered.bundle.bundle_id,
            split=DatasetSplit.SELECTION,
            purpose=DatasetPurpose.CANDIDATE_SELECTION,
            actor_id=subject,
            request_id="dataset_access_semantic_selection",
        )
        candidates = _judge_candidates()
        panel = await BlindMultiJudgePanel(
            judges=(
                StaticJudge("one", -0.1),
                StaticJudge("two", 0.0),
                StaticJudge("three", 0.1),
            ),
            artifact_store=artifacts,
            store=runtime.store,
        ).evaluate(candidates, randomization_seed=3)
        calibration = runtime.calibrator.calibrate(
            panel_result_ids=(panel.panel_result_id,),
            human_ratings=_human_ratings(candidates),
            minimum_correlation=0.9,
            minimum_pass_agreement=0.9,
        )
        snapshot = _snapshot("semantic_selection")
        semantic = runtime.semantic.evaluate(
            _semantic_input(
                subject_version_id=subject,
                registered=registered,
                access=access,
                snapshot=snapshot,
                panel=panel,
                candidate_id=subject,
                calibration=calibration,
            )
        )
        metrics = {item.name: item for item in semantic.metrics}
        required = {
            "claim_support_score",
            "supported_claim_rate",
            "contradicted_claim_rate",
            "unsupported_claim_rate",
            "uncited_fact_rate",
            "retrieval_precision",
            "retrieval_recall",
            "retrieval_authority_mean",
            "retrieval_freshness_rate",
            "retrieval_diversity_rate",
            "report_completeness_score",
            "report_depth_score",
            "report_instruction_following_score",
            "report_organization_score",
            "report_readability_score",
            "url_validity_rate",
            "schema_validity_rate",
            "citation_position_validity_rate",
            "cost_usd",
            "source_freshness_rate",
        }
        assert required <= set(metrics)
        assert metrics["supported_claim_rate"].value == 1.0
        assert metrics["retrieval_precision"].value == 1.0
        assert metrics["retrieval_recall"].value == 1.0
        assert metrics["citation_position_validity_rate"].value == 1.0
        assert semantic.passed is True
        assert (
            runtime.store.semantic_result(
                semantic.semantic_evaluation_id
            )
            == semantic
        )
    finally:
        runtime.close()
        artifacts.close()


@pytest.mark.asyncio
async def test_release_gates_enforce_splits_promote_reject_and_rollback(
    tmp_path,
    monkeypatch,
):
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    versions = build_version_registry_runtime(
        tmp_path / "versions",
        artifact_store=artifacts,
    )
    evaluation = build_evaluation_lab_runtime(
        tmp_path / "evaluation",
        artifact_store=artifacts,
        version_registry=versions.registry,
    )
    base_ref = _version(
        artifacts,
        suffix="base",
        version="1.0.0",
    )
    candidate_ref = _version(
        artifacts,
        suffix="candidate",
        version="2.0.0",
    )
    bad_ref = _version(
        artifacts,
        suffix="bad",
        version="3.0.0",
    )
    try:
        versions.registry.register(base_ref)
        versions.registry.register(
            candidate_ref,
            parent_version_id=base_ref.version_id,
        )
        versions.registry.register(
            bad_ref,
            parent_version_id=candidate_ref.version_id,
        )
        bootstrap = artifacts.put_json(
            {"decision": "bootstrap"},
            redact=False,
            kind=ArtifactKind.RELEASE_GATE_DECISION,
            producer_id="test_release_bootstrap",
            run_id="run_bootstrap_gate",
            content_schema="ReleaseGateDecision@1",
        )
        versions.registry.promote(
            base_ref.version_id,
            gate_decision_id="gate_decision_bootstrap",
            gate_decision_artifact_id=bootstrap.artifact_id,
            reason="Bootstrap initial active version.",
            actor_id="release_gate_service",
            occurred_at=base_ref.created_at,
        )

        registered = evaluation.datasets.register(
            name="release-dataset",
            version="1.0.0",
            sample_schema="ReleaseInput@1",
            samples_by_split=_sample_artifacts(artifacts, "release"),
            description="Release gate dataset.",
        )
        candidates = _judge_candidates()
        panel = await BlindMultiJudgePanel(
            judges=(
                StaticJudge("one", -0.1),
                StaticJudge("two", 0.0),
                StaticJudge("three", 0.1),
            ),
            artifact_store=artifacts,
            store=evaluation.store,
        ).evaluate(candidates, randomization_seed=9)
        calibration = evaluation.calibrator.calibrate(
            panel_result_ids=(panel.panel_result_id,),
            human_ratings=_human_ratings(candidates),
            minimum_correlation=0.9,
            minimum_pass_agreement=0.9,
        )
        semantic_by_split = {}
        access_by_split = {}
        purposes = {
            DatasetSplit.SELECTION:
                DatasetPurpose.CANDIDATE_SELECTION,
            DatasetSplit.TEST: DatasetPurpose.FINAL_EVALUATION,
            DatasetSplit.HIDDEN_TEST: DatasetPurpose.RELEASE_GATE,
        }
        for split, purpose in purposes.items():
            access, _ = evaluation.datasets.access(
                bundle_id=registered.bundle.bundle_id,
                split=split,
                purpose=purpose,
                actor_id=candidate_ref.version_id,
                request_id=(
                    "dataset_access_release_"
                    f"{split.value.replace('-', '_')}"
                ),
            )
            semantic = evaluation.semantic.evaluate(
                _semantic_input(
                    subject_version_id=candidate_ref.version_id,
                    registered=registered,
                    access=access,
                    snapshot=_snapshot(
                        f"release_{split.value.replace('-', '_')}"
                    ),
                    panel=panel,
                    candidate_id="version_candidate_quality_3",
                    calibration=calibration,
                )
            )
            access_by_split[split] = access
            semantic_by_split[split] = semantic

        policy = _gate_policy()
        baseline_metrics = _gate_metrics(quality=0.8, cost=0.5)
        candidate_metrics = _gate_metrics(quality=0.9, cost=0.55)
        selection_semantic = semantic_by_split[DatasetSplit.SELECTION]
        selection_evidence = GateEvaluationEvidence(
            candidate_version_id=candidate_ref.version_id,
            baseline_version_id=base_ref.version_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            candidate_variances={"report_completeness_score": 0.001},
            semantic_evaluation_ids=(
                selection_semantic.semantic_evaluation_id,
            ),
            dataset_access_record_ids=(
                access_by_split[
                    DatasetSplit.SELECTION
                ].access_record_id,
            ),
            calibration_id=calibration.calibration_id,
            evaluation_artifact_ids=(
                selection_semantic.result_artifact_id,
            ),
        )
        assert evaluation.release_gates is not None
        selection = evaluation.release_gates.evaluate(
            stage=GateStage.SELECTION,
            policy=policy,
            evidence=selection_evidence,
        )
        assert selection.outcome == GateOutcome.ADVANCE_TO_FINAL
        assert versions.store.record(
            candidate_ref.version_id
        ).state == VersionLifecycleState.CANDIDATE

        leaked = selection_evidence.model_copy(
            update={
                "semantic_evaluation_ids": (
                    semantic_by_split[
                        DatasetSplit.HIDDEN_TEST
                    ].semantic_evaluation_id,
                ),
                "dataset_access_record_ids": (
                    access_by_split[
                        DatasetSplit.HIDDEN_TEST
                    ].access_record_id,
                ),
                "evaluation_artifact_ids": (
                    semantic_by_split[
                        DatasetSplit.HIDDEN_TEST
                    ].result_artifact_id,
                ),
            }
        )
        with pytest.raises(ValueError, match="selection split"):
            evaluation.release_gates.evaluate(
                stage=GateStage.SELECTION,
                policy=policy,
                evidence=leaked,
            )

        final_semantics = (
            semantic_by_split[DatasetSplit.TEST],
            semantic_by_split[DatasetSplit.HIDDEN_TEST],
        )
        final_evidence = GateEvaluationEvidence(
            candidate_version_id=candidate_ref.version_id,
            baseline_version_id=base_ref.version_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            candidate_variances={"report_completeness_score": 0.001},
            semantic_evaluation_ids=tuple(
                item.semantic_evaluation_id for item in final_semantics
            ),
            dataset_access_record_ids=(
                access_by_split[DatasetSplit.TEST].access_record_id,
                access_by_split[
                    DatasetSplit.HIDDEN_TEST
                ].access_record_id,
            ),
            calibration_id=calibration.calibration_id,
            evaluation_artifact_ids=tuple(
                item.result_artifact_id for item in final_semantics
            ),
        )
        original_save_application = (
            evaluation.store.save_gate_application
        )
        application_attempts = 0

        def fail_first_application(value):
            nonlocal application_attempts
            application_attempts += 1
            if application_attempts == 1:
                raise RuntimeError("injected gate application interruption")
            return original_save_application(value)

        monkeypatch.setattr(
            evaluation.store,
            "save_gate_application",
            fail_first_application,
        )
        with pytest.raises(
            RuntimeError,
            match="gate application interruption",
        ):
            evaluation.release_gates.evaluate(
                stage=GateStage.FINAL_PROMOTION,
                policy=policy,
                evidence=final_evidence,
            )
        assert versions.store.record(
            candidate_ref.version_id
        ).state == VersionLifecycleState.PROMOTED
        interrupted_decision = next(
            item
            for item in evaluation.store.gate_decisions(
                candidate_ref.version_id
            )
            if item.stage == GateStage.FINAL_PROMOTION
        )
        assert evaluation.store.gate_application(
            interrupted_decision.gate_decision_id
        ) is None

        monkeypatch.setattr(
            evaluation.store,
            "save_gate_application",
            original_save_application,
        )
        promoted = evaluation.release_gates.evaluate(
            stage=GateStage.FINAL_PROMOTION,
            policy=policy,
            evidence=final_evidence,
        )
        assert promoted.outcome == GateOutcome.PROMOTE
        assert evaluation.store.gate_application(
            promoted.gate_decision_id
        ) is not None
        assert versions.store.record(
            candidate_ref.version_id
        ).state == VersionLifecycleState.PROMOTED
        assert versions.store.record(
            base_ref.version_id
        ).state == VersionLifecycleState.SUPERSEDED

        rollback_evidence = final_evidence.model_copy(
            update={
                "candidate_metrics": _gate_metrics(
                    quality=0.6,
                    cost=1.5,
                    protocol=0.8,
                    safety=0.7,
                ),
                "rollback_target_version_id": base_ref.version_id,
            }
        )
        rollback = evaluation.release_gates.evaluate(
            stage=GateStage.POST_RELEASE,
            policy=policy,
            evidence=rollback_evidence,
        )
        assert rollback.outcome == GateOutcome.ROLLBACK
        assert versions.store.record(
            candidate_ref.version_id
        ).state == VersionLifecycleState.ROLLED_BACK
        assert versions.store.record(
            base_ref.version_id
        ).state == VersionLifecycleState.PROMOTED
        assert evaluation.release_gates.evaluate(
            stage=GateStage.POST_RELEASE,
            policy=policy,
            evidence=rollback_evidence,
        ).gate_decision_id == rollback.gate_decision_id
        versions.store.rebuild_projections()
        assert versions.store.record(
            candidate_ref.version_id
        ).state == VersionLifecycleState.ROLLED_BACK
        assert versions.store.record(
            base_ref.version_id
        ).state == VersionLifecycleState.PROMOTED

        bad_access, _ = evaluation.datasets.access(
            bundle_id=registered.bundle.bundle_id,
            split=DatasetSplit.SELECTION,
            purpose=DatasetPurpose.CANDIDATE_SELECTION,
            actor_id=bad_ref.version_id,
            request_id="dataset_access_bad_candidate",
        )
        # Reuse a valid semantic body under an explicitly persisted bad-subject
        # result by evaluating the same calibrated candidate.
        bad_semantic = evaluation.semantic.evaluate(
            _semantic_input(
                subject_version_id=bad_ref.version_id,
                registered=registered,
                access=bad_access,
                snapshot=_snapshot("release_bad"),
                panel=panel,
                candidate_id="version_candidate_quality_3",
                calibration=calibration,
            )
        )
        rejected = evaluation.release_gates.evaluate(
            stage=GateStage.SELECTION,
            policy=policy,
            evidence=GateEvaluationEvidence(
                candidate_version_id=bad_ref.version_id,
                baseline_version_id=base_ref.version_id,
                baseline_metrics=baseline_metrics,
                candidate_metrics=_gate_metrics(
                    quality=0.7,
                    cost=2.0,
                    protocol=0.5,
                    safety=0.5,
                ),
                candidate_variances={
                    "report_completeness_score": 1.0
                },
                semantic_evaluation_ids=(
                    bad_semantic.semantic_evaluation_id,
                ),
                dataset_access_record_ids=(
                    bad_access.access_record_id,
                ),
                calibration_id=calibration.calibration_id,
                evaluation_artifact_ids=(
                    bad_semantic.result_artifact_id,
                ),
            ),
        )
        assert rejected.outcome == GateOutcome.REJECT
        assert versions.store.record(
            bad_ref.version_id
        ).state == VersionLifecycleState.REJECTED
        versions.integrity_check()
        evaluation.integrity_check()
    finally:
        evaluation.close()
        versions.close()
        artifacts.close()


def test_version_registry_restart_rebuild_backup_concurrency_and_corruption(
    tmp_path,
):
    artifacts = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    root = tmp_path / "versions"
    first = build_version_registry_runtime(root, artifact_store=artifacts)
    second = build_version_registry_runtime(root, artifact_store=artifacts)
    version_ref = _version(
        artifacts,
        suffix="concurrent",
        version="1.0.0",
    )

    def register(index: int):
        runtime = first if index % 2 == 0 else second
        return runtime.registry.register(version_ref)

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(register, range(20)))
        assert all(item == records[0] for item in records)
        first.store.rebuild_projections()
        first.integrity_check()
        backup = first.store.backup_to(tmp_path / "versions-backup.sqlite3")
        assert backup.is_file()
    finally:
        second.close()
        first.close()

    reopened = build_version_registry_runtime(
        root,
        artifact_store=artifacts,
    )
    try:
        assert reopened.store.record(version_ref.version_id) == records[0]
        reopened.store._connection.execute(
            """
            UPDATE version_manifests
            SET payload='{}' WHERE version_id=?
            """,
            (version_ref.version_id,),
        )
        with pytest.raises(VersionRegistryCorruption, match="checksum"):
            reopened.integrity_check()
    finally:
        reopened.close()
        artifacts.close()
