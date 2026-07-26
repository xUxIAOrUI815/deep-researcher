from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import timedelta

import pytest

from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentKind,
    ComponentVersionSet,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    MetricDirection,
    VersionRef,
)
from deep_researcher.evaluation_lab import (
    BlindMultiJudgePanel,
    GateEvaluationEvidence,
    build_evaluation_lab_runtime,
)
from deep_researcher.evolution_lab import (
    AddPlacement,
    CandidateStatus,
    CrossTaskExperience,
    EvolutionCampaignStatus,
    EvolutionConflict,
    EvolutionCorruption,
    ExperienceEditSuggestion,
    HumanGateOutcome,
    OfflineGeneratorResult,
    OfflineEvolutionLab,
    OptimizationTarget,
    PatchBudgetExceeded,
    PatchOperationKind,
    PoolEntryStatus,
    PoolSourceKind,
    SQLiteEvolutionStore,
    TextEditBudget,
    TextPatchApplier,
    TextPatchOperation,
    TraceSignalPatchGenerator,
    target_accepts_component,
)
from deep_researcher.studio import StudioBadcase
from deep_researcher.version_registry import (
    VersionLifecycleState,
    build_version_registry_runtime,
)
from tests.test_bg001_evaluation_lab_core import (
    _sample_artifacts,
    _snapshot,
)
from tests.test_bg001_evaluation_semantic_gates import (
    StaticJudge,
    _gate_metrics,
    _gate_policy,
    _human_ratings,
    _judge_candidates,
    _semantic_input,
)


@dataclass
class EvolutionHarness:
    artifacts: SQLiteArtifactStore
    versions: object
    evaluation: object
    store: SQLiteEvolutionStore
    lab: OfflineEvolutionLab
    base_ref: VersionRef
    registered_dataset: object
    panel: object
    calibration: object

    def close(self) -> None:
        self.store.close()
        self.evaluation.close()
        self.versions.close()
        self.artifacts.close()


def _ref(kind: ComponentKind, name: str, suffix: str) -> VersionRef:
    return VersionRef(
        version_id=f"version_{suffix}",
        kind=kind,
        name=name,
        version="1.0.0",
    )


def _score(name: str, value: float) -> EvaluationMetric:
    return EvaluationMetric(
        name=name,
        value=value,
        direction=MetricDirection.HIGHER_IS_BETTER,
        evaluator="offline_evolution_test",
    )


async def _harness(tmp_path) -> EvolutionHarness:
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
    store = SQLiteEvolutionStore(tmp_path / "evolution.sqlite3")
    content = artifacts.put_text(
        "\n".join(
            (
                "# Research Skill",
                "Use broad searches.",
                "Cite sources.",
                "Stop when done.",
                "Preserve explicit uncertainty.",
                "Prefer primary evidence.",
                "Track coverage by section.",
                "Keep conflicts visible.",
            )
        )
        + "\n",
        kind=ArtifactKind.SKILL,
        producer_id="test_skill_author",
        run_id="run_evolution_base_skill",
        content_schema="SkillText@1",
    )
    base_ref = VersionRef(
        version_id="version_evolution_base_skill",
        kind=ComponentKind.SKILL,
        name="research-skill",
        version="1.0.0",
        artifact_id=content.artifact_id,
        content_hash=content.content_hash,
    )
    versions.registry.register(base_ref)
    bootstrap = artifacts.put_json(
        {"decision": "bootstrap"},
        redact=False,
        kind=ArtifactKind.RELEASE_GATE_DECISION,
        producer_id="test_release_bootstrap",
        run_id="run_evolution_bootstrap",
        content_schema="ReleaseGateDecision@1",
    )
    versions.registry.promote(
        base_ref.version_id,
        gate_decision_id="gate_decision_evolution_bootstrap",
        gate_decision_artifact_id=bootstrap.artifact_id,
        reason="Bootstrap the immutable base Skill.",
        actor_id="release_gate_service",
        occurred_at=base_ref.created_at,
    )
    registered = evaluation.datasets.register(
        name="offline-evolution-dataset",
        version="1.0.0",
        sample_schema="EvolutionInput@1",
        samples_by_split=_sample_artifacts(artifacts, "evolution"),
        description="Offline evolution release dataset.",
    )
    candidates = _judge_candidates()
    panel = await BlindMultiJudgePanel(
        judges=(
            StaticJudge("evolution_one", -0.1),
            StaticJudge("evolution_two", 0.0),
            StaticJudge("evolution_three", 0.1),
        ),
        artifact_store=artifacts,
        store=evaluation.store,
    ).evaluate(candidates, randomization_seed=17)
    calibration = evaluation.calibrator.calibrate(
        panel_result_ids=(panel.panel_result_id,),
        human_ratings=_human_ratings(candidates),
        minimum_correlation=0.9,
        minimum_pass_agreement=0.9,
    )
    lab = OfflineEvolutionLab(
        store=store,
        artifact_store=artifacts,
        version_registry=versions.registry,
        release_gate=evaluation.release_gates,
    )
    return EvolutionHarness(
        artifacts=artifacts,
        versions=versions,
        evaluation=evaluation,
        store=store,
        lab=lab,
        base_ref=base_ref,
        registered_dataset=registered,
        panel=panel,
        calibration=calibration,
    )


def _badcase_artifact(harness: EvolutionHarness):
    artifacts = harness.artifacts
    evaluation = artifacts.put_json(
        {"evaluation_id": "evaluation_badcase_failure"},
        redact=False,
        kind=ArtifactKind.EVALUATION_RESULT,
        producer_id="test_evaluator",
        run_id="run_evolution_badcase",
        content_schema="EvaluationResult@1",
    )
    sample = artifacts.put_json(
        {
            "sample_id": "sample_evolution_badcase",
            "dataset_split": DatasetSplit.TRAIN.value,
        },
        redact=False,
        kind=ArtifactKind.DATASET_SAMPLE,
        producer_id="test_dataset",
        run_id="run_evolution_badcase",
        content_schema="DatasetSample@1",
    )
    input_artifact = artifacts.put_json(
        {"question": "Why did this research run fail?"},
        redact=False,
        kind=ArtifactKind.OTHER,
        producer_id="test_input",
        run_id="run_evolution_badcase",
    )
    artifact_id = "artifact_evolution_badcase"
    component_versions = ComponentVersionSet(
        runtime=_ref(
            ComponentKind.RUNTIME,
            "evolution-runtime",
            "runtime_evolution",
        ),
        scheduler=_ref(
            ComponentKind.SCHEDULER,
            "evolution-scheduler",
            "scheduler_evolution",
        ),
        skill=harness.base_ref,
    )
    badcase = StudioBadcase(
        badcase_id="badcase_evolution_failure",
        source_run_id="run_evolution_badcase",
        source_span_id="span_evolution_badcase",
        source_event_ids=("event_evolution_badcase",),
        input_artifact_ids=(input_artifact.artifact_id,),
        component_versions=component_versions,
        component_version_ids=tuple(
            item.version_id
            for item in (
                component_versions.runtime,
                component_versions.scheduler,
                component_versions.skill,
            )
            if item is not None
        ),
        evaluation_ids=("evaluation_badcase_failure",),
        evaluation_artifact_ids=(evaluation.artifact_id,),
        dataset_sample_artifact_id=sample.artifact_id,
        human_note="The run used an over-broad query and stopped too early.",
        created_by="reviewer_badcase",
        artifact_id=artifact_id,
        triggers_change=False,
    )
    envelope = artifacts.put_json(
        {
            "schema": "StudioBadcase@1",
            "badcase": badcase.model_dump(mode="json"),
            "triggers_change": False,
            "optimizer_invoked": False,
        },
        redact=False,
        kind=ArtifactKind.BADCASE,
        producer_id="test_studio_badcase",
        run_id="run_evolution_badcase",
        content_schema="StudioBadcase@1",
        artifact_id=artifact_id,
    )
    return envelope, evaluation


def _reviewed_pool(harness: EvolutionHarness):
    artifacts = harness.artifacts
    evaluation = artifacts.put_json(
        {
            "evaluation_id": "evaluation_evolution_pool",
            "dataset_split": DatasetSplit.DEV.value,
        },
        redact=False,
        kind=ArtifactKind.EVALUATION_RESULT,
        producer_id="test_evaluator",
        run_id="run_evolution_pool_evaluation",
        content_schema="EvaluationResult@1",
    )
    success = artifacts.put_json(
        {"outcome": "success", "score": 0.9},
        redact=False,
        kind=ArtifactKind.TRACE_EXPORT,
        producer_id="test_trace_export",
        run_id="run_evolution_success",
        content_schema="TraceExport@1",
    )
    failure = artifacts.put_json(
        {"outcome": "failure", "score": 0.2},
        redact=False,
        kind=ArtifactKind.TRACE_EXPORT,
        producer_id="test_trace_export",
        run_id="run_evolution_failure",
        content_schema="TraceExport@1",
    )
    badcase, badcase_evaluation = _badcase_artifact(harness)
    entries = (
        harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.SCORED_SUCCESS_TRACE,
            source_artifact_id=success.artifact_id,
            source_run_id=success.run_id,
            evaluation_artifact_ids=(evaluation.artifact_id,),
            score_metrics=(_score("quality", 0.9),),
            production_trace=True,
            submitted_by="collector_production",
        ),
        harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.SCORED_FAILURE_TRACE,
            source_artifact_id=failure.artifact_id,
            source_run_id=failure.run_id,
            evaluation_artifact_ids=(evaluation.artifact_id,),
            score_metrics=(_score("quality", 0.2),),
            production_trace=True,
            submitted_by="collector_production",
        ),
        harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.BADCASE,
            source_artifact_id=badcase.artifact_id,
            evaluation_artifact_ids=(badcase_evaluation.artifact_id,),
            submitted_by="studio_reviewer",
        ),
        harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.EVALUATION,
            source_artifact_id=evaluation.artifact_id,
            submitted_by="evaluation_lab",
        ),
    )
    assert all(item.status == PoolEntryStatus.PENDING_REVIEW for item in entries)
    assert harness.store.list_campaigns().items == ()
    reviewed = tuple(
        harness.lab.review_candidate_source(
            item.entry.pool_entry_id,
            approved=True,
            assigned_split=(
                item.entry.observed_dataset_splits[0]
                if item.entry.observed_dataset_splits
                else DatasetSplit.TRAIN
            ),
            reviewer_id="reviewer_candidate_pool",
            review_note="Approved only for bounded offline analysis.",
            reviewed_at=item.entry.submitted_at + timedelta(seconds=1),
        )
        for item in entries
    )
    assert all(item.status == PoolEntryStatus.APPROVED for item in reviewed)
    assert harness.store.list_campaigns().items == ()
    return reviewed


def _experiences(harness: EvolutionHarness, reviewed):
    ids = tuple(item.entry.pool_entry_id for item in reviewed)
    first = harness.lab.create_cross_task_experience(
        target=OptimizationTarget.PLANNING,
        summary="Broad planning caused redundant searches",
        recommendation="Require explicit subquestion decomposition.",
        suggestion=ExperienceEditSuggestion(
            operation=PatchOperationKind.REPLACE,
            match_lines=("Use broad searches.",),
            new_lines=(
                "Decompose the question into bounded evidence subquestions.",
            ),
        ),
        confidence=0.99,
        impact=0.95,
        pool_entry_ids=ids,
        created_by="offline_experience_curator",
    )
    second = harness.lab.create_cross_task_experience(
        target=OptimizationTarget.PLANNING,
        summary="Single-source plans failed verification",
        recommendation="Add independent-source planning requirement.",
        suggestion=ExperienceEditSuggestion(
            operation=PatchOperationKind.ADD,
            match_lines=("Cite sources.",),
            new_lines=(
                "Plan at least two independent sources for high-impact claims.",
            ),
            add_placement=AddPlacement.AFTER,
        ),
        confidence=0.95,
        impact=0.9,
        pool_entry_ids=ids,
        created_by="offline_experience_curator",
    )
    return first, second


def _campaign(harness: EvolutionHarness, reviewed, experiences):
    return harness.lab.create_campaign(
        target=OptimizationTarget.PLANNING,
        component_kind=ComponentKind.SKILL,
        component_name=harness.base_ref.name,
        base_version_id=harness.base_ref.version_id,
        pool_entry_ids=tuple(
            item.entry.pool_entry_id for item in reviewed
        ),
        experience_ids=tuple(item.experience_id for item in experiences),
        edit_budget=TextEditBudget(
            max_rounds=3,
            max_operations_per_round=1,
            max_added_lines_per_round=3,
            max_deleted_lines_per_round=3,
            max_changed_characters_per_round=300,
            max_edit_fraction_per_round=1.0,
        ),
        offline_environment_id="environment_offline_evolution_test",
        created_by="operator_offline_evolution",
    )


def _evidence(
    harness: EvolutionHarness,
    *,
    candidate_version_id: str,
    splits: tuple[DatasetSplit, ...],
    quality: float,
    cost: float,
    safety: float = 0.98,
    rollback_target_version_id: str | None = None,
    tag: str,
) -> GateEvaluationEvidence:
    purposes = {
        DatasetSplit.SELECTION: DatasetPurpose.CANDIDATE_SELECTION,
        DatasetSplit.TEST: DatasetPurpose.FINAL_EVALUATION,
        DatasetSplit.HIDDEN_TEST: DatasetPurpose.RELEASE_GATE,
    }
    accesses = []
    semantics = []
    for split in splits:
        access, _ = harness.evaluation.datasets.access(
            bundle_id=harness.registered_dataset.bundle.bundle_id,
            split=split,
            purpose=purposes[split],
            actor_id=candidate_version_id,
            request_id=(
                f"dataset_access_{tag}_{split.value.replace('-', '_')}"
            ),
        )
        semantic = harness.evaluation.semantic.evaluate(
            _semantic_input(
                subject_version_id=candidate_version_id,
                registered=harness.registered_dataset,
                access=access,
                snapshot=_snapshot(
                    f"{tag}_{split.value.replace('-', '_')}"
                ),
                panel=harness.panel,
                candidate_id="version_candidate_quality_3",
                calibration=harness.calibration,
            )
        )
        accesses.append(access)
        semantics.append(semantic)
    return GateEvaluationEvidence(
        candidate_version_id=candidate_version_id,
        baseline_version_id=harness.base_ref.version_id,
        rollback_target_version_id=rollback_target_version_id,
        baseline_metrics=_gate_metrics(quality=0.8, cost=0.5),
        candidate_metrics=_gate_metrics(
            quality=quality,
            cost=cost,
            safety=safety,
        ),
        candidate_variances={
            "report_completeness_score": 0.001
        },
        semantic_evaluation_ids=tuple(
            item.semantic_evaluation_id for item in semantics
        ),
        dataset_access_record_ids=tuple(
            item.access_record_id for item in accesses
        ),
        calibration_id=harness.calibration.calibration_id,
        evaluation_artifact_ids=tuple(
            item.result_artifact_id for item in semantics
        ),
    )


@pytest.mark.parametrize(
    ("target", "accepted", "rejected"),
    (
        (
            OptimizationTarget.PLANNING,
            ComponentKind.SKILL,
            ComponentKind.RUBRIC,
        ),
        (
            OptimizationTarget.QUERY_GENERATION,
            ComponentKind.PROMPT,
            ComponentKind.STOP_POLICY,
        ),
        (
            OptimizationTarget.SOURCE_SELECTION,
            ComponentKind.TOOL_POLICY,
            ComponentKind.RUBRIC,
        ),
        (
            OptimizationTarget.EXTRACTION_CITATION,
            ComponentKind.SKILL,
            ComponentKind.TOOL_POLICY,
        ),
        (
            OptimizationTarget.REPORT_WRITING,
            ComponentKind.PROMPT,
            ComponentKind.RUBRIC,
        ),
        (
            OptimizationTarget.TOOL_ROUTING,
            ComponentKind.TOOL_POLICY,
            ComponentKind.SKILL,
        ),
        (
            OptimizationTarget.STOP_POLICY,
            ComponentKind.STOP_POLICY,
            ComponentKind.PROMPT,
        ),
        (
            OptimizationTarget.GRADER_RUBRIC,
            ComponentKind.RUBRIC,
            ComponentKind.SKILL,
        ),
    ),
)
def test_all_eight_targets_have_explicit_component_boundaries(
    target,
    accepted,
    rejected,
):
    assert target_accepts_component(target, accepted)
    assert not target_accepts_component(target, rejected)
    experience = CrossTaskExperience(
        experience_id=f"evolution_experience_{target.value}",
        target=target,
        summary="Observed scored failure.",
        recommendation="Add a bounded corrective directive.",
        suggestion=ExperienceEditSuggestion(
            operation=PatchOperationKind.ADD,
            new_lines=(f"Directive for {target.value}.",),
            add_placement=AddPlacement.END,
        ),
        confidence=0.9,
        impact=0.8,
        pool_entry_ids=("evolution_pool_target_boundary",),
        source_artifact_ids=("artifact_target_boundary",),
        created_by="reviewer_target_boundary",
        artifact_id=f"artifact_experience_{target.value}",
    )
    result = TraceSignalPatchGenerator().generate(
        target=target,
        base_text=("Base directive.\n" * 20),
        experiences=(experience,),
        budget=TextEditBudget(
            max_rounds=2,
            max_operations_per_round=2,
            max_added_lines_per_round=2,
            max_deleted_lines_per_round=2,
            max_changed_characters_per_round=200,
            max_edit_fraction_per_round=0.5,
        ),
        rejected_operation_fingerprints=frozenset(),
    )
    assert result.operations[0].operation == PatchOperationKind.ADD
    assert result.online_inference_count == 0
    assert result.network_accessed is False


@pytest.mark.asyncio
async def test_rejected_edit_memory_changes_next_round_and_release_rolls_back(
    tmp_path,
):
    harness = await _harness(tmp_path)
    try:
        reviewed = _reviewed_pool(harness)
        experiences = _experiences(harness, reviewed)
        campaign = _campaign(harness, reviewed, experiences)
        assert campaign.status == EvolutionCampaignStatus.READY
        assert campaign.input_snapshot is not None
        assert campaign.input_snapshot.runtime_memory_used is False
        assert campaign.input_snapshot.online_inference_count == 0

        first = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_offline_evolution",
        )
        first_candidate = first.candidates[-1]
        assert first_candidate.patch.operations[0].operation == (
            PatchOperationKind.REPLACE
        )
        first_version_id = first_candidate.candidate.version_ref.version_id
        failed = harness.lab.evaluate_selection(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=first_version_id,
                splits=(DatasetSplit.SELECTION,),
                quality=0.9,
                cost=0.55,
                safety=0.5,
                tag="first_selection_rejected",
            ),
        )
        assert failed.status == EvolutionCampaignStatus.READY
        assert failed.candidates[-1].status == (
            CandidateStatus.SELECTION_REJECTED
        )
        rejection = failed.candidates[-1].rejection
        assert rejection is not None
        assert harness.versions.store.record(first_version_id).state == (
            VersionLifecycleState.REJECTED
        )

        second = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_offline_evolution",
        )
        second_candidate = second.candidates[-1]
        assert second_candidate.patch.operations[0].operation == (
            PatchOperationKind.ADD
        )
        assert (
            first_candidate.patch.operation_fingerprints[0]
            in second_candidate.patch.rejected_operation_fingerprints_skipped
        )
        assert (
            second_candidate.patch.patch_fingerprint
            != first_candidate.patch.patch_fingerprint
        )
        second_version_id = second_candidate.candidate.version_ref.version_id
        selected = harness.lab.evaluate_selection(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=second_version_id,
                splits=(DatasetSplit.SELECTION,),
                quality=0.9,
                cost=0.55,
                tag="second_selection_passed",
            ),
        )
        assert selected.status == EvolutionCampaignStatus.AWAITING_HUMAN
        assert harness.versions.store.record(second_version_id).state == (
            VersionLifecycleState.CANDIDATE
        )

        approved = harness.lab.decide_human_gate(
            campaign.request.campaign_id,
            approved=True,
            reviewer_id="reviewer_release_owner",
            note="Selection evidence is auditable and the edit is bounded.",
        )
        assert approved.status == EvolutionCampaignStatus.READY_FOR_FINAL
        assert (
            approved.candidates[-1].human_decision.outcome
            == HumanGateOutcome.APPROVED
        )
        assert harness.versions.store.record(second_version_id).state == (
            VersionLifecycleState.CANDIDATE
        )

        promoted = harness.lab.evaluate_final_promotion(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=second_version_id,
                splits=(
                    DatasetSplit.TEST,
                    DatasetSplit.HIDDEN_TEST,
                ),
                quality=0.9,
                cost=0.55,
                tag="second_final_passed",
            ),
        )
        assert promoted.status == EvolutionCampaignStatus.PROMOTED
        assert promoted.candidates[-1].status == CandidateStatus.PROMOTED
        assert harness.versions.store.record(second_version_id).state == (
            VersionLifecycleState.PROMOTED
        )
        best = harness.store.best_skill(
            target=OptimizationTarget.PLANNING,
            component_name="research-skill",
        )
        assert best is not None
        assert best.skill_version_id == second_version_id
        assert best.static_versioned is True
        assert best.online_inference_count == 0

        rolled_back = harness.lab.evaluate_post_release(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=second_version_id,
                splits=(
                    DatasetSplit.TEST,
                    DatasetSplit.HIDDEN_TEST,
                ),
                quality=0.6,
                cost=1.5,
                safety=0.7,
                rollback_target_version_id=harness.base_ref.version_id,
                tag="second_post_release_regression",
            ),
        )
        assert rolled_back.status == EvolutionCampaignStatus.ROLLED_BACK
        assert rolled_back.candidates[-1].status == (
            CandidateStatus.ROLLED_BACK
        )
        assert harness.versions.store.record(second_version_id).state == (
            VersionLifecycleState.ROLLED_BACK
        )
        assert harness.versions.store.record(
            harness.base_ref.version_id
        ).state == VersionLifecycleState.PROMOTED
        restored = harness.store.best_skill(
            target=OptimizationTarget.PLANNING,
            component_name="research-skill",
        )
        assert restored is not None
        assert restored.skill_version_id == harness.base_ref.version_id
        assert restored.previous_best_skill_id == best.best_skill_id
        assert len(
            harness.store.best_skill_history(
                target=OptimizationTarget.PLANNING,
                component_name="research-skill",
            )
        ) == 2
        harness.store.integrity_check()
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_human_rejection_is_audited_and_never_runs_final_gate(
    tmp_path,
):
    harness = await _harness(tmp_path)
    try:
        reviewed = _reviewed_pool(harness)
        campaign = _campaign(
            harness,
            reviewed,
            _experiences(harness, reviewed),
        )
        generated = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_human_rejection",
        )
        version_id = generated.candidates[-1].candidate.version_ref.version_id
        harness.lab.evaluate_selection(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=version_id,
                splits=(DatasetSplit.SELECTION,),
                quality=0.9,
                cost=0.55,
                tag="human_rejection_selection",
            ),
        )
        rejected = harness.lab.decide_human_gate(
            campaign.request.campaign_id,
            approved=False,
            reviewer_id="reviewer_human_rejection",
            note="The edit is semantically valid but conflicts with policy.",
        )
        assert rejected.status == EvolutionCampaignStatus.READY
        latest = rejected.candidates[-1]
        assert latest.status == CandidateStatus.HUMAN_REJECTED
        assert latest.rejection is not None
        assert harness.versions.store.record(version_id).state == (
            VersionLifecycleState.REJECTED
        )
        decision_artifact = harness.artifacts.get(
            latest.human_decision.decision_artifact_id
        )
        assert decision_artifact.kind == (
            ArtifactKind.EVOLUTION_HUMAN_DECISION
        )
        with pytest.raises(ValueError, match="human approval"):
            harness.lab.evaluate_final_promotion(
                campaign.request.campaign_id,
                policy=_gate_policy(),
                evidence=_evidence(
                    harness,
                    candidate_version_id=version_id,
                    splits=(
                        DatasetSplit.TEST,
                        DatasetSplit.HIDDEN_TEST,
                    ),
                    quality=0.9,
                    cost=0.55,
                    tag="human_rejection_illegal_final",
                ),
            )
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_generation_rejects_split_leakage_and_non_strict_selection(
    tmp_path,
):
    harness = await _harness(tmp_path)
    try:
        hidden = harness.artifacts.put_json(
            {
                "evaluation_id": "evaluation_hidden_leak",
                "dataset_split": DatasetSplit.HIDDEN_TEST.value,
            },
            redact=False,
            kind=ArtifactKind.EVALUATION_RESULT,
            producer_id="test_hidden_evaluator",
            run_id="run_hidden_evaluation",
            content_schema="EvaluationResult@1",
        )
        pending = harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.EVALUATION,
            source_artifact_id=hidden.artifact_id,
            submitted_by="evaluation_lab",
        )
        assert pending.entry.observed_dataset_splits == (
            DatasetSplit.HIDDEN_TEST,
        )
        with pytest.raises(ValueError, match="hidden-test"):
            harness.lab.review_candidate_source(
                pending.entry.pool_entry_id,
                approved=True,
                assigned_split=DatasetSplit.DEV,
                reviewer_id="reviewer_leakage",
                review_note="Attempted split relabel.",
            )

        reviewed = _reviewed_pool(harness)
        campaign = _campaign(
            harness,
            reviewed,
            _experiences(harness, reviewed),
        )
        generated = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_strict_selection",
        )
        version_id = generated.candidates[-1].candidate.version_ref.version_id
        with pytest.raises(ValueError, match="strictly improve"):
            harness.lab.evaluate_selection(
                campaign.request.campaign_id,
                policy=_gate_policy(),
                evidence=_evidence(
                    harness,
                    candidate_version_id=version_id,
                    splits=(DatasetSplit.SELECTION,),
                    quality=0.8,
                    cost=0.5,
                    tag="selection_no_strict_gain",
                ),
            )
        with pytest.raises(ValueError, match="selection split"):
            harness.lab.evaluate_selection(
                campaign.request.campaign_id,
                policy=_gate_policy(),
                evidence=_evidence(
                    harness,
                    candidate_version_id=version_id,
                    splits=(DatasetSplit.HIDDEN_TEST,),
                    quality=0.9,
                    cost=0.55,
                    tag="selection_hidden_leak",
                ),
            )
        unchanged = harness.store.campaign(campaign.request.campaign_id)
        assert unchanged.status == EvolutionCampaignStatus.CANDIDATE_READY
        assert harness.versions.store.record(version_id).state == (
            VersionLifecycleState.CANDIDATE
        )
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_final_rejection_records_memory_and_exhausts_round_budget(
    tmp_path,
):
    harness = await _harness(tmp_path)
    try:
        reviewed = _reviewed_pool(harness)
        experiences = _experiences(harness, reviewed)
        campaign = harness.lab.create_campaign(
            target=OptimizationTarget.PLANNING,
            component_kind=ComponentKind.SKILL,
            component_name=harness.base_ref.name,
            base_version_id=harness.base_ref.version_id,
            pool_entry_ids=tuple(
                item.entry.pool_entry_id for item in reviewed
            ),
            experience_ids=tuple(
                item.experience_id for item in experiences
            ),
            edit_budget=TextEditBudget(
                max_rounds=1,
                max_operations_per_round=1,
                max_added_lines_per_round=3,
                max_deleted_lines_per_round=3,
                max_changed_characters_per_round=300,
                max_edit_fraction_per_round=1.0,
            ),
            offline_environment_id="environment_final_rejection",
            created_by="operator_final_rejection",
        )
        generated = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_final_rejection",
        )
        version_id = generated.candidates[-1].candidate.version_ref.version_id
        harness.lab.evaluate_selection(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=version_id,
                splits=(DatasetSplit.SELECTION,),
                quality=0.9,
                cost=0.55,
                tag="final_rejection_selection",
            ),
        )
        harness.lab.decide_human_gate(
            campaign.request.campaign_id,
            approved=True,
            reviewer_id="reviewer_final_rejection",
            note="Approved for final test and hidden-test evaluation only.",
        )
        rejected = harness.lab.evaluate_final_promotion(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=version_id,
                splits=(
                    DatasetSplit.TEST,
                    DatasetSplit.HIDDEN_TEST,
                ),
                quality=0.9,
                cost=0.55,
                safety=0.5,
                tag="final_rejection_gate",
            ),
        )
        assert rejected.status == EvolutionCampaignStatus.EXHAUSTED
        assert rejected.candidates[-1].status == (
            CandidateStatus.FINAL_REJECTED
        )
        assert rejected.candidates[-1].rejection is not None
        assert harness.versions.store.record(version_id).state == (
            VersionLifecycleState.REJECTED
        )
        with pytest.raises(EvolutionConflict, match="ready campaign"):
            harness.store.claim_generation(
                campaign.request.campaign_id,
                worker_id="worker_over_round_budget",
            )
    finally:
        harness.close()


@pytest.mark.asyncio
async def test_store_concurrency_recovery_rebuild_backup_and_immutability(
    tmp_path,
):
    harness = await _harness(tmp_path)
    second = None
    reopened = None
    try:
        reviewed = _reviewed_pool(harness)
        campaign = _campaign(
            harness,
            reviewed,
            _experiences(harness, reviewed),
        )
        second = SQLiteEvolutionStore(tmp_path / "evolution.sqlite3")

        def claim(store_and_worker):
            store, worker = store_and_worker
            try:
                return store.claim_generation(
                    campaign.request.campaign_id,
                    worker_id=worker,
                )
            except EvolutionConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            attempts = list(
                pool.map(
                    claim,
                    (
                        (harness.store, "worker_concurrent_one"),
                        (second, "worker_concurrent_two"),
                    ),
                )
            )
        claimed = [item for item in attempts if item is not None]
        assert len(claimed) == 1
        recovered = second.recover_interrupted()
        assert len(recovered) == 1
        assert recovered[0].status == EvolutionCampaignStatus.READY
        assert recovered[0].recovery_count == 1
        next_attempt = harness.store.claim_generation(
            campaign.request.campaign_id,
            worker_id="worker_after_recovery",
        )
        assert next_attempt.attempt_id != claimed[0].attempt_id
        harness.store.abandon_generation(
            campaign.request.campaign_id,
            attempt_id=next_attempt.attempt_id,
            error="deliberate durability test",
        )

        first_pool_page = harness.store.list_pool(limit=2)
        assert len(first_pool_page.items) == 2
        assert first_pool_page.next_cursor is not None
        second_pool_page = harness.store.list_pool(
            cursor=first_pool_page.next_cursor,
            limit=2,
        )
        assert len(second_pool_page.items) == 2
        assert not (
            {
                item.entry.pool_entry_id
                for item in first_pool_page.items
            }
            & {
                item.entry.pool_entry_id
                for item in second_pool_page.items
            }
        )
        backup = harness.store.backup_to(
            tmp_path / "evolution-backup.sqlite3"
        )
        assert backup.is_file()
        harness.store._connection.execute(
            "DELETE FROM evolution_projection"
        )
        with pytest.raises(EvolutionCorruption, match="projection"):
            harness.store.campaign(campaign.request.campaign_id)
        harness.store.rebuild_projections()
        assert harness.store.campaign(
            campaign.request.campaign_id
        ).recovery_count == 2

        with pytest.raises(Exception, match="immutable"):
            harness.store._connection.execute(
                "UPDATE evolution_campaigns SET target='other'"
            )
        harness.store._connection.execute(
            "UPDATE evolution_projection SET checksum='bad'"
        )
        with pytest.raises(EvolutionCorruption, match="checksum"):
            harness.store.integrity_check()
        harness.store.rebuild_projections()
        harness.store.integrity_check()

        second.close()
        second = None
        harness.store.close()
        reopened = SQLiteEvolutionStore(tmp_path / "evolution.sqlite3")
        assert reopened.campaign(
            campaign.request.campaign_id
        ).recovery_count == 2
        reopened.integrity_check()
    finally:
        if reopened is not None:
            reopened.close()
        if second is not None:
            second.close()
        harness.evaluation.close()
        harness.versions.close()
        harness.artifacts.close()


@pytest.mark.asyncio
async def test_cross_store_interruption_recovery_is_idempotent(
    tmp_path,
    monkeypatch,
):
    harness = await _harness(tmp_path)
    try:
        extra_evaluation = harness.artifacts.put_json(
            {
                "evaluation_id": "evaluation_review_recovery",
                "dataset_split": DatasetSplit.TRAIN.value,
            },
            redact=False,
            kind=ArtifactKind.EVALUATION_RESULT,
            producer_id="test_recovery_evaluator",
            run_id="run_review_recovery",
            content_schema="EvaluationResult@1",
        )
        pending = harness.lab.submit_candidate_source(
            source_kind=PoolSourceKind.EVALUATION,
            source_artifact_id=extra_evaluation.artifact_id,
            submitted_by="evaluation_recovery",
        )
        original_review = harness.store.review_pool_entry
        review_calls = 0

        def fail_review_once(review):
            nonlocal review_calls
            review_calls += 1
            if review_calls == 1:
                raise RuntimeError("injected review-store interruption")
            return original_review(review)

        monkeypatch.setattr(
            harness.store,
            "review_pool_entry",
            fail_review_once,
        )
        with pytest.raises(RuntimeError, match="review-store"):
            harness.lab.review_candidate_source(
                pending.entry.pool_entry_id,
                approved=True,
                assigned_split=DatasetSplit.TRAIN,
                reviewer_id="reviewer_recovery",
                review_note="Recover the artifact/store commit gap.",
            )
        assert harness.store.pool_entry(
            pending.entry.pool_entry_id
        ).review is None
        monkeypatch.setattr(
            harness.store,
            "review_pool_entry",
            original_review,
        )
        recovered_review = harness.lab.review_candidate_source(
            pending.entry.pool_entry_id,
            approved=True,
            assigned_split=DatasetSplit.TRAIN,
            reviewer_id="reviewer_recovery",
            review_note="Recover the artifact/store commit gap.",
        )
        assert recovered_review.status == PoolEntryStatus.APPROVED

        reviewed = _reviewed_pool(harness)
        experiences = _experiences(harness, reviewed)
        original_seal = harness.store.seal_inputs
        seal_calls = 0

        def fail_seal_once(snapshot):
            nonlocal seal_calls
            seal_calls += 1
            if seal_calls == 1:
                raise RuntimeError("injected input-seal interruption")
            return original_seal(snapshot)

        monkeypatch.setattr(harness.store, "seal_inputs", fail_seal_once)
        with pytest.raises(RuntimeError, match="input-seal"):
            _campaign(harness, reviewed, experiences)
        draft = harness.store.list_campaigns(
            statuses=(EvolutionCampaignStatus.DRAFT,)
        ).items
        assert len(draft) == 1
        monkeypatch.setattr(harness.store, "seal_inputs", original_seal)
        campaign = _campaign(harness, reviewed, experiences)
        assert campaign.status == EvolutionCampaignStatus.READY

        generated = harness.lab.generate_candidate(
            campaign.request.campaign_id,
            worker_id="worker_human_gap_recovery",
        )
        version_id = generated.candidates[-1].candidate.version_ref.version_id
        harness.lab.evaluate_selection(
            campaign.request.campaign_id,
            policy=_gate_policy(),
            evidence=_evidence(
                harness,
                candidate_version_id=version_id,
                splits=(DatasetSplit.SELECTION,),
                quality=0.9,
                cost=0.55,
                tag="human_gap_selection",
            ),
        )
        original_human = harness.store.record_human_decision
        human_calls = 0

        def fail_human_once(decision, *, rejection=None):
            nonlocal human_calls
            human_calls += 1
            if human_calls == 1:
                raise RuntimeError("injected human-store interruption")
            return original_human(decision, rejection=rejection)

        monkeypatch.setattr(
            harness.store,
            "record_human_decision",
            fail_human_once,
        )
        with pytest.raises(RuntimeError, match="human-store"):
            harness.lab.decide_human_gate(
                campaign.request.campaign_id,
                approved=False,
                reviewer_id="reviewer_human_gap",
                note="Reject and recover the cross-store commit gap.",
            )
        assert harness.versions.store.record(version_id).state == (
            VersionLifecycleState.REJECTED
        )
        assert harness.store.campaign(
            campaign.request.campaign_id
        ).status == EvolutionCampaignStatus.AWAITING_HUMAN
        monkeypatch.setattr(
            harness.store,
            "record_human_decision",
            original_human,
        )
        recovered_human = harness.lab.decide_human_gate(
            campaign.request.campaign_id,
            approved=False,
            reviewer_id="reviewer_human_gap",
            note="Reject and recover the cross-store commit gap.",
        )
        assert recovered_human.status == EvolutionCampaignStatus.READY
        assert recovered_human.candidates[-1].status == (
            CandidateStatus.HUMAN_REJECTED
        )
        harness.store.integrity_check()
    finally:
        harness.close()


def test_patch_learning_rate_stale_content_and_overlap_are_fail_closed():
    base = "\n".join(f"line {index}" for index in range(20)) + "\n"
    applier = TextPatchApplier()
    budget = TextEditBudget(
        max_rounds=2,
        max_operations_per_round=2,
        max_added_lines_per_round=2,
        max_deleted_lines_per_round=2,
        max_changed_characters_per_round=200,
        max_edit_fraction_per_round=0.05,
    )
    with pytest.raises(PatchBudgetExceeded, match="fraction"):
        applier.apply(
            base,
            (
                TextPatchOperation(
                    operation=PatchOperationKind.REPLACE,
                    start_line=0,
                    end_line=1,
                    old_lines=("line 0",),
                    new_lines=("a substantially longer replacement line",),
                ),
            ),
            budget,
        )
    with pytest.raises(ValueError, match="immutable base"):
        applier.apply(
            base,
            (
                TextPatchOperation(
                    operation=PatchOperationKind.REPLACE,
                    start_line=1,
                    end_line=2,
                    old_lines=("stale line",),
                    new_lines=("replacement",),
                ),
            ),
            budget.model_copy(
                update={"max_edit_fraction_per_round": 1.0}
            ),
        )
    with pytest.raises(ValueError, match="overlap"):
        applier.apply(
            base,
            (
                TextPatchOperation(
                    operation=PatchOperationKind.REPLACE,
                    start_line=1,
                    end_line=3,
                    old_lines=("line 1", "line 2"),
                    new_lines=("one",),
                ),
                TextPatchOperation(
                    operation=PatchOperationKind.DELETE,
                    start_line=2,
                    end_line=3,
                    old_lines=("line 2",),
                ),
            ),
            budget.model_copy(
                update={
                    "max_changed_characters_per_round": 200,
                    "max_edit_fraction_per_round": 1.0,
                }
            ),
        )


@pytest.mark.asyncio
async def test_online_generator_report_is_rejected_and_campaign_recovers(
    tmp_path,
):
    harness = await _harness(tmp_path)

    class NetworkReportingGenerator:
        def generate(self, **kwargs):
            return OfflineGeneratorResult.model_construct(
                operations=(
                    TextPatchOperation(
                        operation=PatchOperationKind.ADD,
                        start_line=0,
                        end_line=0,
                        new_lines=("Unsafe online edit.",),
                    ),
                ),
                rationale=("Untrusted generator.",),
                consulted_experience_ids=(
                    kwargs["experiences"][0].experience_id,
                ),
                skipped_rejected_operation_fingerprints=(),
                online_inference_count=1,
                network_accessed=True,
            )

    try:
        reviewed = _reviewed_pool(harness)
        campaign = _campaign(
            harness,
            reviewed,
            _experiences(harness, reviewed),
        )
        harness.lab.generator = NetworkReportingGenerator()
        with pytest.raises(RuntimeError, match="online inference"):
            harness.lab.generate_candidate(
                campaign.request.campaign_id,
                worker_id="worker_untrusted_generator",
            )
        recovered = harness.store.campaign(campaign.request.campaign_id)
        assert recovered.status == EvolutionCampaignStatus.READY
        assert recovered.recovery_count == 1
        assert recovered.attempts[-1].status.value == "abandoned"
        assert recovered.candidates == ()
        assert harness.versions.store.active(
            ComponentKind.SKILL.value,
            "research-skill",
        ).manifest.version_ref.version_id == harness.base_ref.version_id
    finally:
        harness.close()
