from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    BudgetUsage,
    ComponentKind,
    ComponentVersionSet,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    FrozenReplay,
    MetricDirection,
    VersionRef,
    utc_now,
)
from deep_researcher.evaluation_lab import (
    CitationObservation,
    DatasetLeakageError,
    DatasetSampleSpec,
    DeterministicEvaluatorSuite,
    EnvironmentDescriptor,
    EvaluationMode,
    EvaluationSnapshot,
    EvaluationStoreConflict,
    EvaluationStoreCorruption,
    ExecutionCounters,
    ExperimentRunStatus,
    FrozenReplayRunner,
    FrozenReplayViolation,
    LiveWebRunner,
    ReplayExecutionResult,
    SectionObservation,
    SourceObservation,
    SQLiteEvaluationStore,
    SystemBaseline,
    ToolCallObservation,
    build_evaluation_lab_runtime,
    capture_environment,
)


def _canonical_fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _versions(suffix: str = "test") -> ComponentVersionSet:
    return ComponentVersionSet(
        runtime=VersionRef(
            version_id=f"version_runtime_{suffix}",
            kind=ComponentKind.RUNTIME,
            name=f"runtime-{suffix}",
            version="1.0.0",
        ),
        scheduler=VersionRef(
            version_id=f"version_scheduler_{suffix}",
            kind=ComponentKind.SCHEDULER,
            name=f"scheduler-{suffix}",
            version="1.0.0",
        ),
    )


def _put_json(
    artifacts: SQLiteArtifactStore,
    value: Any,
    suffix: str,
):
    return artifacts.put_json(
        value,
        redact=False,
        kind=ArtifactKind.DATASET_SAMPLE,
        producer_id="test_evaluation_lab",
        run_id=f"run_{suffix}",
        content_schema="EvaluationTestData@1",
    )


def _sample_artifacts(
    artifacts: SQLiteArtifactStore,
    prefix: str,
) -> dict[DatasetSplit, tuple[DatasetSampleSpec, ...]]:
    output = {}
    for split in DatasetSplit:
        item = _put_json(
            artifacts,
            {"question": f"{prefix}-{split.value}"},
            f"{prefix}_{split.value.replace('-', '_')}",
        )
        expected = _put_json(
            artifacts,
            {"answer": f"{prefix}-{split.value}-answer"},
            f"{prefix}_{split.value.replace('-', '_')}_expected",
        )
        output[split] = (
            DatasetSampleSpec(
                input_artifact_id=item.artifact_id,
                expected_artifact_id=expected.artifact_id,
                tags=(split.value,),
            ),
        )
    return output


def _snapshot(
    suffix: str,
    *,
    authority: float = 0.9,
    content_hash: str | None = None,
    source_url: str = "https://official.example/research",
    cost: float = 0.25,
) -> EvaluationSnapshot:
    now = utc_now()
    statement = "The evaluation snapshot is grounded."
    source_id = f"source_{suffix}"
    claim_id = f"claim_{suffix}"
    citation_id = f"citation_{suffix}"
    return EvaluationSnapshot(
        snapshot_id=f"evaluation_snapshot_{suffix}",
        run_id=f"run_{suffix}",
        report_id=f"report_{suffix}",
        report_revision_id=f"report_revision_{suffix}",
        report_markdown=f"{statement}[1]",
        sources=(
            SourceObservation(
                source_id=source_id,
                canonical_url=source_url,
                source_type="official_documentation",
                source_level="primary",
                authority_score=authority,
                publisher="Official Publisher",
                domain="official.example",
                published_at=now - timedelta(days=10),
                fetched_at=now - timedelta(days=1),
                content_hash=content_hash or hashlib.sha256(
                    suffix.encode()
                ).hexdigest(),
            ),
        ),
        citations=(
            CitationObservation(
                citation_id=citation_id,
                claim_id=claim_id,
                evidence_id=f"evidence_{suffix}",
                source_id=source_id,
                marker="[1]",
                canonical_url=source_url,
                quote=statement,
                passage_text=f"Context. {statement} More context.",
                locator="chars:9-45",
                verified=True,
                used_in_report=True,
            ),
        ),
        sections=(
            SectionObservation(
                section_id=f"section_{suffix}",
                required_claim_ids=(claim_id,),
                supported_claim_ids=(claim_id,),
                unsupported_claim_ids=(),
                citation_ids=(citation_id,),
            ),
        ),
        tool_calls=(
            ToolCallObservation(
                call_id=f"call_{suffix}_one",
                tool_name="search",
                operation="search",
                request_key="evaluation query",
                is_search=True,
                evidence_count=1,
            ),
            ToolCallObservation(
                call_id=f"call_{suffix}_two",
                tool_name="search",
                operation="search",
                request_key="evaluation query",
                is_search=True,
                evidence_count=1,
                recovered_after_retry=True,
            ),
        ),
        counters=ExecutionCounters(
            task_count=4,
            failed_task_count=1,
            retried_operation_count=1,
            recovered_operation_count=1,
            idempotent_operation_count=3,
            idempotency_violation_count=0,
            protocol_operation_count=5,
            invalid_protocol_operation_count=0,
            convergence_turns=3,
            budget_violation_count=0,
        ),
        usage=BudgetUsage(
            input_tokens=100,
            output_tokens=50,
            cost_usd=cost,
            wall_time_seconds=1.5,
            model_calls=2,
            tool_calls=2,
            search_calls=2,
            retries=1,
            errors=1,
        ),
        latency_ms=1500.0,
        source_fingerprint=hashlib.sha256(
            f"{source_url}:{content_hash or suffix}".encode()
        ).hexdigest(),
        created_at=now,
    )


@pytest.fixture
def artifact_store(tmp_path):
    store = SQLiteArtifactStore(tmp_path / "artifacts.sqlite3")
    try:
        yield store
    finally:
        store.close()


def test_dataset_registry_seals_all_splits_and_enforces_access(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab",
        artifact_store=artifact_store,
    )
    try:
        samples = _sample_artifacts(artifact_store, "dataset_v1")
        registered = runtime.datasets.register(
            name="research-quality",
            version="1.0.0",
            sample_schema="ResearchEvaluationInput@1",
            samples_by_split=samples,
            description="Five-way sealed evaluation dataset.",
        )
        assert set(registered.bundle.split_dataset_ids) == set(DatasetSplit)
        assert len(registered.definitions) == 5
        assert len(registered.samples) == 5
        assert artifact_store.get(
            registered.bundle.manifest_artifact_id
        ).kind == ArtifactKind.DATASET_MANIFEST

        allowed = (
            (DatasetSplit.TRAIN, DatasetPurpose.TRAINING),
            (DatasetSplit.DEV, DatasetPurpose.DEVELOPMENT),
            (
                DatasetSplit.SELECTION,
                DatasetPurpose.CANDIDATE_SELECTION,
            ),
            (DatasetSplit.TEST, DatasetPurpose.FINAL_EVALUATION),
            (DatasetSplit.HIDDEN_TEST, DatasetPurpose.RELEASE_GATE),
        )
        for index, (split, purpose) in enumerate(allowed):
            record, accessed = runtime.datasets.access(
                bundle_id=registered.bundle.bundle_id,
                split=split,
                purpose=purpose,
                actor_id="actor_evaluation_test",
                request_id=f"dataset_access_request_{index}",
            )
            assert record.request.split == split
            assert len(accessed) == 1
            assert accessed[0].split == split
        with pytest.raises(ValueError):
            runtime.datasets.access(
                bundle_id=registered.bundle.bundle_id,
                split=DatasetSplit.HIDDEN_TEST,
                purpose=DatasetPurpose.CANDIDATE_SELECTION,
                actor_id="actor_tuning_forbidden",
            )
        runtime.integrity_check()
    finally:
        runtime.close()

    reopened = build_evaluation_lab_runtime(
        tmp_path / "lab",
        artifact_store=artifact_store,
    )
    try:
        restored = reopened.datasets.registered(
            registered.bundle.bundle_id
        )
        assert restored == registered
        assert len(reopened.store.dataset_accesses()) == 5
    finally:
        reopened.close()


def test_dataset_registry_rejects_cross_split_leakage_and_version_reuse(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-leakage",
        artifact_store=artifact_store,
    )
    try:
        samples = _sample_artifacts(artifact_store, "leakage_v1")
        registered = runtime.datasets.register(
            name="leakage-check",
            version="1.0.0",
            sample_schema="Input@1",
            samples_by_split=samples,
            description="Leakage test.",
        )
        changed = _sample_artifacts(artifact_store, "leakage_changed")
        changed[DatasetSplit.DEV] = samples[DatasetSplit.TRAIN]
        with pytest.raises(DatasetLeakageError):
            runtime.datasets.register(
                name="leakage-check",
                version="2.0.0",
                sample_schema="Input@1",
                samples_by_split=changed,
                description="Leakage test.",
                parent_bundle_id=registered.bundle.bundle_id,
            )
        different = _sample_artifacts(artifact_store, "different")
        with pytest.raises(ValueError, match="different content"):
            runtime.datasets.register(
                name="leakage-check",
                version="1.0.0",
                sample_schema="Input@1",
                samples_by_split=different,
                description="Changed immutable version.",
            )
    finally:
        runtime.close()


def test_dataset_access_concurrency_is_idempotent(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-concurrent",
        artifact_store=artifact_store,
    )
    try:
        registered = runtime.datasets.register(
            name="concurrent-access",
            version="1.0.0",
            sample_schema="Input@1",
            samples_by_split=_sample_artifacts(
                artifact_store,
                "concurrent",
            ),
            description="Concurrent access.",
        )

        def access(_):
            return runtime.datasets.access(
                bundle_id=registered.bundle.bundle_id,
                split=DatasetSplit.DEV,
                purpose=DatasetPurpose.DEVELOPMENT,
                actor_id="actor_concurrent",
                request_id="dataset_access_request_concurrent",
            )[0]

        with ThreadPoolExecutor(max_workers=8) as pool:
            records = list(pool.map(access, range(20)))
        assert len({item.access_record_id for item in records}) == 1
        assert len(runtime.store.dataset_accesses()) == 1
    finally:
        runtime.close()


def test_dataset_registration_is_idempotent_across_runtime_connections(
    tmp_path,
    artifact_store,
):
    root = tmp_path / "lab-concurrent-registration"
    first_runtime = build_evaluation_lab_runtime(
        root,
        artifact_store=artifact_store,
    )
    second_runtime = build_evaluation_lab_runtime(
        root,
        artifact_store=artifact_store,
    )
    samples = _sample_artifacts(
        artifact_store,
        "concurrent_registration",
    )

    def register(index: int):
        runtime = first_runtime if index % 2 == 0 else second_runtime
        return runtime.datasets.register(
            name="concurrent-registration",
            version="1.0.0",
            sample_schema="ConcurrentDataset@1",
            samples_by_split=samples,
            description="Concurrent immutable registration.",
        )

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            registered = list(pool.map(register, range(24)))
        assert all(item == registered[0] for item in registered)
        first_runtime.integrity_check()
        second_runtime.integrity_check()
    finally:
        second_runtime.close()
        first_runtime.close()


def test_deterministic_evaluator_covers_quality_resource_and_trace_metrics(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-evaluator",
        artifact_store=artifact_store,
        freshness_days=30,
    )
    try:
        report = runtime.evaluator.evaluate(_snapshot("metrics"))
        metrics = {item.name: item.value for item in report.metrics}
        required = {
            "url_validity_rate",
            "citation_reference_integrity_rate",
            "citation_quote_grounding_rate",
            "citation_completeness_rate",
            "schema_validity_rate",
            "section_coverage_rate",
            "source_type_count",
            "source_authority_mean",
            "source_freshness_rate",
            "source_publisher_diversity_rate",
            "source_domain_diversity_rate",
            "primary_source_share",
            "total_tokens",
            "cost_usd",
            "latency_ms",
            "failure_rate",
            "recovery_rate",
            "idempotency_compliance_rate",
            "protocol_compliance_rate",
            "evidence_per_tool_call",
            "redundant_search_rate",
            "trace_recovery_rate",
            "convergence_turns",
            "invalid_tool_call_rate",
            "budget_violation_count",
        }
        assert required <= set(metrics)
        assert metrics["url_validity_rate"] == 1.0
        assert metrics["citation_quote_grounding_rate"] == 1.0
        assert metrics["section_coverage_rate"] == 1.0
        assert metrics["redundant_search_rate"] == 0.5
        assert metrics["total_tokens"] == 150.0
        assert metrics["failure_rate"] == 0.25
        assert report.aggregate_score > 0.8
        assert runtime.store.evaluation_report(report.report_id) == report
        assert artifact_store.get(report.details_artifact_id) is not None
    finally:
        runtime.close()


class FrozenExecutor:
    def __init__(
        self,
        artifacts: SQLiteArtifactStore,
        output: Any,
        events: Any,
        *,
        network_calls: int = 0,
        vary: bool = False,
    ) -> None:
        self.artifacts = artifacts
        self.output = output
        self.events = events
        self.network_calls = network_calls
        self.vary = vary

    async def execute(
        self,
        *,
        input_artifact_id: str,
        deterministic_seed: int,
        repetition: int,
        subject_version_id: str,
    ) -> ReplayExecutionResult:
        output = (
            {**self.output, "vary": repetition}
            if self.vary
            else self.output
        )
        output_artifact = self.artifacts.put_json(
            output,
            redact=False,
            kind=ArtifactKind.EVALUATION_RESULT,
            producer_id=subject_version_id,
            run_id=f"run_frozen_{repetition}",
            content_schema="FrozenOutput@1",
        )
        event_artifact = self.artifacts.put_json(
            self.events,
            redact=False,
            kind=ArtifactKind.TRACE_EXPORT,
            producer_id=subject_version_id,
            run_id=f"run_frozen_{repetition}",
            content_schema="FrozenEvents@1",
        )
        snapshot = _snapshot(f"frozen_{repetition}")
        # Determinism excludes run identity, but the evaluator snapshot itself
        # must be identical across repeats for metric stability.
        snapshot = snapshot.model_copy(
            update={
                "snapshot_id": "evaluation_snapshot_frozen_stable",
                "run_id": "run_frozen_stable",
            }
        )
        return ReplayExecutionResult(
            execution_id=f"replay_execution_{repetition}",
            run_id=f"run_frozen_{repetition}",
            output_artifact_id=output_artifact.artifact_id,
            event_artifact_id=event_artifact.artifact_id,
            snapshot=snapshot,
            network_calls=self.network_calls,
        )


@pytest.mark.asyncio
async def test_frozen_replay_is_repeated_reproducible_and_network_free(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-frozen",
        artifact_store=artifact_store,
    )
    fixture = {"question": "frozen input", "seed": 17}
    output = {"answer": "stable"}
    events = [{"type": "completed"}]
    input_artifact = _put_json(artifact_store, fixture, "frozen_input")
    output_artifact = _put_json(
        artifact_store,
        output,
        "frozen_expected_output",
    )
    event_artifact = _put_json(
        artifact_store,
        events,
        "frozen_expected_events",
    )
    replay = FrozenReplay(
        replay_id="replay_frozen_core",
        name="Frozen core replay",
        version="1.0.0",
        input_artifact_id=input_artifact.artifact_id,
        expected_event_artifact_id=event_artifact.artifact_id,
        expected_output_artifact_id=output_artifact.artifact_id,
        fixture_fingerprint=_canonical_fingerprint(fixture),
        deterministic_seed=17,
    )
    try:
        runner = FrozenReplayRunner(
            artifact_store=artifact_store,
            store=runtime.store,
            evaluator=runtime.evaluator,
            executor=FrozenExecutor(
                artifact_store,
                output,
                events,
            ),
        )
        result = await runner.run(
            replay,
            subject_version_id="version_subject_frozen",
            repeat_count=3,
        )
        assert result.expected_output_match is True
        assert result.expected_event_match is True
        assert result.deterministic is True
        assert result.network_free is True
        assert len(runtime.store.frozen_results(replay.replay_id)) == 1

        network_runner = runtime.frozen_replay(
            FrozenExecutor(
                artifact_store,
                output,
                events,
                network_calls=1,
            )
        )
        with pytest.raises(FrozenReplayViolation, match="network"):
            await network_runner.run(
                replay,
                subject_version_id="version_subject_network_violation",
            )

        varying = await runtime.frozen_replay(
            FrozenExecutor(
                artifact_store,
                output,
                events,
                vary=True,
            )
        ).run(
            replay,
            subject_version_id="version_subject_nondeterministic",
            repeat_count=3,
        )
        assert varying.deterministic is False
        assert varying.expected_output_match is False
    finally:
        runtime.close()


class LiveExecutor:
    def __init__(self, artifacts: SQLiteArtifactStore) -> None:
        self.artifacts = artifacts

    async def execute(
        self,
        *,
        sample,
        deterministic_seed: int,
        repetition: int,
        subject_version_id: str,
    ) -> ReplayExecutionResult:
        output = self.artifacts.put_json(
            {"seed": deterministic_seed, "repetition": repetition},
            redact=False,
            kind=ArtifactKind.EVALUATION_RESULT,
            producer_id=subject_version_id,
            run_id=f"run_live_{repetition}",
            content_schema="LiveOutput@1",
        )
        events = self.artifacts.put_json(
            [{"type": "live", "repetition": repetition}],
            redact=False,
            kind=ArtifactKind.TRACE_EXPORT,
            producer_id=subject_version_id,
            run_id=f"run_live_{repetition}",
            content_schema="LiveEvents@1",
        )
        snapshot = _snapshot(
            "live_shared",
            authority=0.8 + repetition * 0.05,
            content_hash=hashlib.sha256(
                f"live-source-{repetition}".encode()
            ).hexdigest(),
            cost=0.1 + repetition * 0.1,
        ).model_copy(
            update={
                "snapshot_id": f"evaluation_snapshot_live_{repetition}",
                "run_id": f"run_live_{repetition}",
            }
        )
        if repetition == 0:
            transient = SourceObservation(
                source_id="source_live_transient",
                canonical_url="https://transient.example/source",
                source_type="news",
                source_level="secondary",
                authority_score=0.6,
                publisher="Transient Publisher",
                domain="transient.example",
                fetched_at=utc_now(),
                content_hash=hashlib.sha256(b"transient").hexdigest(),
            )
            snapshot = snapshot.model_copy(
                update={"sources": (*snapshot.sources, transient)}
            )
        return ReplayExecutionResult(
            execution_id=f"replay_execution_live_{repetition}",
            run_id=f"run_live_{repetition}",
            output_artifact_id=output.artifact_id,
            event_artifact_id=events.artifact_id,
            snapshot=snapshot,
            network_calls=2,
        )


@pytest.mark.asyncio
async def test_live_web_reports_metric_variance_and_source_change(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-live",
        artifact_store=artifact_store,
    )
    try:
        registered = runtime.datasets.register(
            name="live-evaluation",
            version="1.0.0",
            sample_schema="LiveInput@1",
            samples_by_split=_sample_artifacts(
                artifact_store,
                "live_dataset",
            ),
            description="Live Web evaluation dataset.",
        )
        access, samples = runtime.datasets.access(
            bundle_id=registered.bundle.bundle_id,
            split=DatasetSplit.TEST,
            purpose=DatasetPurpose.FINAL_EVALUATION,
            actor_id="version_subject_live",
            request_id="dataset_access_live_subject",
        )
        runner = LiveWebRunner(
            artifact_store=artifact_store,
            store=runtime.store,
            evaluator=runtime.evaluator,
            executor=LiveExecutor(artifact_store),
        )
        with pytest.raises(ValueError, match="does not authorize"):
            await runner.run(
                dataset_id=samples[0].dataset_id,
                sample=samples[0],
                subject_version_id="version_subject_unauthorized",
                dataset_access_record_id=access.access_record_id,
                repeat_count=2,
                deterministic_seed=100,
            )
        result = await runner.run(
            dataset_id=samples[0].dataset_id,
            sample=samples[0],
            subject_version_id="version_subject_live",
            dataset_access_record_id=access.access_record_id,
            repeat_count=3,
            deterministic_seed=100,
        )
        assert result.repeat_count == 3
        assert result.metric_means["cost_usd"] == pytest.approx(0.2)
        assert result.metric_variances["cost_usd"] > 0
        assert result.dataset_fingerprint == registered.bundle.fingerprint
        assert result.dataset_access_record_id == access.access_record_id
        assert result.changed_source_ids == (
            "source_live_shared",
            "source_live_transient",
        )
        assert result.source_change_rate == 1.0
        assert runtime.store.live_results(samples[0].dataset_id) == (
            result,
        )
    finally:
        runtime.close()


def test_experiment_registry_compares_aligned_three_baselines(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-experiments",
        artifact_store=artifact_store,
    )
    try:
        registered = runtime.datasets.register(
            name="baseline-comparison",
            version="1.0.0",
            sample_schema="ExperimentInput@1",
            samples_by_split=_sample_artifacts(
                artifact_store,
                "experiments",
            ),
            description="Aligned comparison dataset.",
        )
        selection_samples = tuple(
            item
            for item in registered.samples
            if item.split == DatasetSplit.SELECTION
        )
        environment = capture_environment(
            dependency_files=(Path("requirements.txt"),),
            configuration={"mode": "offline"},
            network_mode="offline",
        )
        assert environment.metadata["dependency_manifests"][0]["name"] == (
            "requirements.txt"
        )
        output_artifact = _put_json(
            artifact_store,
            {"result": "experiment"},
            "experiment_result",
        )
        run_ids = []
        values = {
            SystemBaseline.LEGACY: 0.4,
            SystemBaseline.FIXED_WORKFLOW: 0.6,
            SystemBaseline.NEW_RUNTIME: 0.9,
        }
        for baseline, value in values.items():
            subject_version_id = (
                f"version_subject_{baseline.value.replace('-', '_')}"
            )
            access, accessed_samples = runtime.datasets.access(
                bundle_id=registered.bundle.bundle_id,
                split=DatasetSplit.SELECTION,
                purpose=DatasetPurpose.CANDIDATE_SELECTION,
                actor_id=subject_version_id,
                request_id=(
                    "dataset_access_experiment_"
                    f"{baseline.value.replace('-', '_')}"
                ),
            )
            assert accessed_samples == selection_samples
            definition = runtime.experiments.define(
                name=f"{baseline.value} comparison",
                baseline=baseline,
                subject_version_id=subject_version_id,
                mode=EvaluationMode.FROZEN_REPLAY,
                bundle_id=registered.bundle.bundle_id,
                dataset_id=selection_samples[0].dataset_id,
                dataset_split=DatasetSplit.SELECTION,
                dataset_purpose=DatasetPurpose.CANDIDATE_SELECTION,
                dataset_access_record_id=access.access_record_id,
                dataset_fingerprint=registered.bundle.fingerprint,
                component_versions=_versions(
                    baseline.value.replace("-", "_")
                ),
                environment=environment,
                input_artifact_ids=(
                    selection_samples[0].input_artifact_id,
                ),
                deterministic_seed=7,
                repeat_count=2,
            )
            metric = EvaluationMetric(
                name="citation_completeness_rate",
                value=value,
                direction=MetricDirection.HIGHER_IS_BETTER,
                evaluator="comparison_test",
            )
            run = runtime.experiments.record_run(
                experiment_id=definition.experiment_id,
                metrics=(metric,),
                evaluation_artifact_ids=(output_artifact.artifact_id,),
                output_artifact_ids=(output_artifact.artifact_id,),
                started_at=access.granted_at,
                completed_at=access.granted_at,
            )
            run_ids.append(run.experiment_run_id)
        comparison = runtime.experiments.compare(tuple(run_ids))
        assert len(comparison.metrics) == 1
        metric = comparison.metrics[0]
        assert metric.new_vs_legacy_delta == pytest.approx(0.5)
        assert metric.new_vs_fixed_delta == pytest.approx(0.3)
        assert comparison.unavailable_metrics == {
            item: () for item in SystemBaseline
        }
        assert runtime.store.comparison(comparison.comparison_id) == comparison
        assert artifact_store.get(comparison.result_artifact_id) is not None
    finally:
        runtime.close()


@pytest.mark.asyncio
async def test_snapshot_builder_projects_verified_report_stores_end_to_end(
    tmp_path,
):
    from deep_researcher.evidence import (
        DeterministicSemanticVerificationAdapter,
        RecordingEvidenceEventSink,
        build_evidence_runtime,
    )
    from deep_researcher.reporting import (
        ReviewActionKind,
        build_reporting_runtime,
    )
    from tests.test_bg001_synthesis_writer_reviewer import (
        EventCollector,
        QueueModel,
        _evidence_policy,
        _report_policy,
        _review_response,
        _seed,
        _writer_response,
    )

    evidence = build_evidence_runtime(
        tmp_path / "snapshot-evidence",
        semantic_adapter=DeterministicSemanticVerificationAdapter(),
        event_sink=RecordingEvidenceEventSink(),
        policy=_evidence_policy(),
    )
    reporting = None
    lab = None
    try:
        seed = _seed(evidence, run_id="run_evaluation_projection")
        await evidence.engine.verify_run(seed.run_id)
        reporting = build_reporting_runtime(
            tmp_path / "snapshot-reporting",
            evidence=evidence,
            writer_model=QueueModel([_writer_response(seed, revision=1)]),
            reviewer_model=QueueModel(
                [_review_response(decision=ReviewActionKind.ACCEPT)]
            ),
            event_sink=EventCollector(),
            policy=_report_policy(),
        )
        await reporting.loop.run(seed.report_id)
        lab = build_evaluation_lab_runtime(
            tmp_path / "snapshot-lab",
            artifact_store=evidence.knowledge.artifacts,
            evidence=evidence,
            reporting_store=reporting.store,
        )
        assert lab.snapshots is not None
        snapshot = lab.snapshots.build(
            run_id=seed.run_id,
            report_id=seed.report_id,
            counter_overrides={
                "task_count": 1,
                "idempotent_operation_count": 1,
                "protocol_operation_count": 1,
                "convergence_turns": 1,
            },
        )
        result = lab.evaluator.evaluate(snapshot)
        metrics = {item.name: item.value for item in result.metrics}

        assert snapshot.report_markdown
        assert len(snapshot.sources) == 2
        assert len(snapshot.citations) == 2
        assert all(item.verified for item in snapshot.citations)
        assert all(item.used_in_report for item in snapshot.citations)
        assert metrics["citation_reference_integrity_rate"] == 1.0
        assert metrics["citation_quote_grounding_rate"] == 1.0
        assert metrics["citation_completeness_rate"] == 1.0
        assert metrics["section_coverage_rate"] == 1.0
        lab.integrity_check()
    finally:
        if lab is not None:
            lab.close()
        if reporting is not None:
            reporting.close()
        evidence.close()


def test_evaluation_store_backup_conflict_and_corruption(
    tmp_path,
    artifact_store,
):
    runtime = build_evaluation_lab_runtime(
        tmp_path / "lab-store",
        artifact_store=artifact_store,
    )
    report = runtime.evaluator.evaluate(_snapshot("store"))
    backup = runtime.store.backup_to(
        tmp_path / "evaluation-backup.sqlite3"
    )
    runtime.close()

    reopened = SQLiteEvaluationStore(
        tmp_path / "lab-store" / "evaluation_lab.sqlite3"
    )
    try:
        assert reopened.evaluation_report(report.report_id) == report
        changed = report.model_copy(update={"aggregate_score": 0.0})
        with pytest.raises(EvaluationStoreConflict):
            reopened.save_evaluation_report(changed, run_id="run_store")
        with reopened.transaction() as connection:
            connection.execute(
                """
                UPDATE evaluation_records SET checksum='bad'
                WHERE record_type='evaluation_report' AND record_id=?
                """,
                (report.report_id,),
            )
        with pytest.raises(EvaluationStoreCorruption):
            reopened.integrity_check()
    finally:
        reopened.close()
    backup_store = SQLiteEvaluationStore(backup)
    try:
        assert backup_store.evaluation_report(report.report_id) == report
        backup_store.integrity_check()
    finally:
        backup_store.close()
