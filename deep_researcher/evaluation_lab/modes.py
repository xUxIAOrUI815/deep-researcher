from __future__ import annotations

import hashlib
import json
from statistics import fmean, pvariance
from typing import Protocol

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    DatasetSample,
    FrozenReplay,
    utc_now,
)

from .evaluators import DeterministicEvaluatorSuite
from .models import (
    FrozenReplayEvaluation,
    LiveWebEvaluation,
    LiveWebExecution,
    ReplayExecutionResult,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _canonical_fingerprint(content: bytes) -> str:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return hashlib.sha256(content).hexdigest()
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fixture_fingerprint(content: bytes) -> str:
    try:
        value = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return hashlib.sha256(content).hexdigest()
    if isinstance(value, dict):
        value = dict(value)
        value.pop("fixture_fingerprint", None)
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ReplayExecutor(Protocol):
    async def execute(
        self,
        *,
        input_artifact_id: str,
        deterministic_seed: int,
        repetition: int,
        subject_version_id: str,
    ) -> ReplayExecutionResult: ...


class LiveWebExecutor(Protocol):
    async def execute(
        self,
        *,
        sample: DatasetSample,
        deterministic_seed: int,
        repetition: int,
        subject_version_id: str,
    ) -> ReplayExecutionResult: ...


class FrozenReplayViolation(RuntimeError):
    pass


class FrozenReplayRunner:
    """Network-free repeated replay with expected and repeat determinism checks."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        evaluator: DeterministicEvaluatorSuite,
        executor: ReplayExecutor,
        producer_id: str = "runtime_frozen_replay",
        clock=utc_now,
    ) -> None:
        self.artifact_store = artifact_store
        self.store = store
        self.evaluator = evaluator
        self.executor = executor
        self.producer_id = producer_id
        self.clock = clock

    async def run(
        self,
        replay: FrozenReplay,
        *,
        subject_version_id: str,
        repeat_count: int = 2,
    ) -> FrozenReplayEvaluation:
        if repeat_count < 2 or repeat_count > 100:
            raise ValueError("Frozen Replay repeats must be between 2 and 100")
        required = (
            replay.input_artifact_id,
            replay.expected_event_artifact_id,
            replay.expected_output_artifact_id,
        )
        if any(self.artifact_store.get(item) is None for item in required):
            raise ValueError("Frozen Replay references a missing artifact")
        input_content = self.artifact_store.read_bytes(
            replay.input_artifact_id
        )
        if _fixture_fingerprint(input_content) != replay.fixture_fingerprint:
            raise FrozenReplayViolation(
                "Frozen Replay fixture fingerprint does not match input artifact"
            )
        expected_output = _canonical_fingerprint(
            self.artifact_store.read_bytes(
                replay.expected_output_artifact_id
            )
        )
        expected_events = _canonical_fingerprint(
            self.artifact_store.read_bytes(
                replay.expected_event_artifact_id
            )
        )
        executions: list[ReplayExecutionResult] = []
        evaluations = []
        output_fingerprints: list[str] = []
        event_fingerprints: list[str] = []
        for repetition in range(repeat_count):
            execution = await self.executor.execute(
                input_artifact_id=replay.input_artifact_id,
                deterministic_seed=replay.deterministic_seed,
                repetition=repetition,
                subject_version_id=subject_version_id,
            )
            if execution.network_calls != 0:
                raise FrozenReplayViolation(
                    "Frozen Replay attempted network access"
                )
            if (
                self.artifact_store.get(execution.output_artifact_id) is None
                or self.artifact_store.get(execution.event_artifact_id) is None
            ):
                raise FrozenReplayViolation(
                    "Frozen Replay executor returned missing artifacts"
                )
            executions.append(execution)
            output_fingerprints.append(
                _canonical_fingerprint(
                    self.artifact_store.read_bytes(
                        execution.output_artifact_id
                    )
                )
            )
            event_fingerprints.append(
                _canonical_fingerprint(
                    self.artifact_store.read_bytes(
                        execution.event_artifact_id
                    )
                )
            )
            evaluations.append(
                self.evaluator.evaluate(execution.snapshot)
            )
        metric_fingerprints = {
            hashlib.sha256(
                json.dumps(
                    [
                        item.model_dump(mode="json")
                        for item in evaluation.metrics
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            for evaluation in evaluations
        }
        deterministic = (
            len(set(output_fingerprints)) == 1
            and len(set(event_fingerprints)) == 1
            and len(metric_fingerprints) == 1
        )
        material = {
            "replay_id": replay.replay_id,
            "subject_version_id": subject_version_id,
            "executions": [item.execution_id for item in executions],
            "output_fingerprints": output_fingerprints,
            "event_fingerprints": event_fingerprints,
        }
        result_id = _stable_id(
            "frozen_result",
            replay.replay_id,
            subject_version_id,
            hashlib.sha256(
                json.dumps(
                    material,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
        )
        artifact_id = _stable_id("artifact", result_id)
        result = FrozenReplayEvaluation(
            frozen_result_id=result_id,
            replay=replay,
            subject_version_id=subject_version_id,
            repeat_count=repeat_count,
            executions=tuple(executions),
            evaluations=tuple(evaluations),
            expected_output_match=all(
                item == expected_output for item in output_fingerprints
            ),
            expected_event_match=all(
                item == expected_events for item in event_fingerprints
            ),
            deterministic=deterministic,
            network_free=True,
            result_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        self.artifact_store.put_json(
            {
                "schema": "FrozenReplayEvaluation@1",
                "result": result.model_dump(mode="json"),
                "output_fingerprints": output_fingerprints,
                "event_fingerprints": event_fingerprints,
                "expected_output_fingerprint": expected_output,
                "expected_event_fingerprint": expected_events,
            },
            redact=False,
            kind=ArtifactKind.FROZEN_REPLAY_RESULT,
            producer_id=self.producer_id,
            run_id=executions[0].run_id,
            content_schema="FrozenReplayEvaluation@1",
            # A replay result aggregates fixture and execution runs. Their
            # immutable IDs remain in the payload; cross-run ArtifactStore
            # provenance edges are deliberately not fabricated.
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"frozen-result:{result_id}",
        )
        self.store.save_frozen_result(result)
        return result


class LiveWebRunner:
    """Repeated live execution with metric variance and source-change report."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        evaluator: DeterministicEvaluatorSuite,
        executor: LiveWebExecutor,
        producer_id: str = "runtime_live_web_evaluation",
        clock=utc_now,
    ) -> None:
        self.artifact_store = artifact_store
        self.store = store
        self.evaluator = evaluator
        self.executor = executor
        self.producer_id = producer_id
        self.clock = clock

    async def run(
        self,
        *,
        dataset_id: str,
        sample: DatasetSample,
        subject_version_id: str,
        dataset_access_record_id: str,
        repeat_count: int,
        deterministic_seed: int,
    ) -> LiveWebEvaluation:
        if repeat_count < 2 or repeat_count > 100:
            raise ValueError("Live Web repeats must be between 2 and 100")
        if sample.dataset_id != dataset_id:
            raise ValueError("Live Web sample belongs to another dataset")
        access = self.store.dataset_access(dataset_access_record_id)
        if access is None:
            raise ValueError(
                "Live Web evaluation requires a recorded dataset access"
            )
        if (
            access.request.actor_id != subject_version_id
            or access.request.dataset_id != dataset_id
            or sample.sample_id not in access.sample_ids
        ):
            raise ValueError(
                "Live Web dataset access does not authorize this subject "
                "and sample"
            )
        bundle = self.store.dataset_bundle(access.bundle_id)
        if bundle is None:
            raise ValueError("Live Web dataset bundle is missing")
        if self.artifact_store.get(sample.input_artifact_id) is None:
            raise ValueError("Live Web sample input artifact is missing")
        executions: list[LiveWebExecution] = []
        source_versions: dict[str, set[tuple[str, str | None]]] = {}
        source_presence: dict[str, int] = {}
        for repetition in range(repeat_count):
            started = self.clock()
            result = await self.executor.execute(
                sample=sample,
                deterministic_seed=deterministic_seed + repetition,
                repetition=repetition,
                subject_version_id=subject_version_id,
            )
            evaluation = self.evaluator.evaluate(result.snapshot)
            completed = self.clock()
            executions.append(
                LiveWebExecution(
                    execution_id=result.execution_id,
                    repetition=repetition,
                    seed=deterministic_seed + repetition,
                    snapshot=result.snapshot,
                    evaluation=evaluation,
                    started_at=started,
                    completed_at=completed,
                )
            )
            for source in result.snapshot.sources:
                source_versions.setdefault(source.source_id, set()).add(
                    (source.canonical_url, source.content_hash)
                )
                source_presence[source.source_id] = (
                    source_presence.get(source.source_id, 0) + 1
                )
        metric_names = {
            item.name
            for execution in executions
            for item in execution.evaluation.metrics
        }
        metric_values: dict[str, list[float]] = {
            name: [] for name in metric_names
        }
        for execution in executions:
            by_name = {
                item.name: item.value
                for item in execution.evaluation.metrics
            }
            for name in metric_names:
                if name in by_name:
                    metric_values[name].append(by_name[name])
        means = {
            name: fmean(values)
            for name, values in sorted(metric_values.items())
            if len(values) == repeat_count
        }
        variances = {
            name: pvariance(values)
            for name, values in sorted(metric_values.items())
            if len(values) == repeat_count
        }
        changed_source_ids = tuple(
            sorted(
                source_id
                for source_id, versions in source_versions.items()
                if (
                    len(versions) > 1
                    or source_presence[source_id] != repeat_count
                )
            )
        )
        source_change_rate = (
            len(changed_source_ids) / len(source_versions)
            if source_versions
            else 0.0
        )
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "subject_version_id": subject_version_id,
                    "bundle_id": access.bundle_id,
                    "dataset_id": dataset_id,
                    "dataset_access_record_id": dataset_access_record_id,
                    "dataset_fingerprint": bundle.fingerprint,
                    "sample_id": sample.sample_id,
                    "execution_ids": [
                        item.execution_id for item in executions
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result_id = _stable_id(
            "live_result",
            dataset_id,
            sample.sample_id,
            subject_version_id,
            fingerprint,
        )
        artifact_id = _stable_id("artifact", result_id)
        result = LiveWebEvaluation(
            live_result_id=result_id,
            subject_version_id=subject_version_id,
            bundle_id=access.bundle_id,
            dataset_id=dataset_id,
            dataset_access_record_id=dataset_access_record_id,
            dataset_fingerprint=bundle.fingerprint,
            sample_id=sample.sample_id,
            repeat_count=repeat_count,
            executions=tuple(executions),
            metric_means=means,
            metric_variances=variances,
            changed_source_ids=changed_source_ids,
            source_change_rate=source_change_rate,
            result_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        self.artifact_store.put_json(
            {
                "schema": "LiveWebEvaluation@1",
                "result": result.model_dump(mode="json"),
                "source_versions": {
                    source_id: sorted(versions, key=lambda item: str(item))
                    for source_id, versions in source_versions.items()
                },
                "source_presence_counts": dict(sorted(source_presence.items())),
                "population_variance": True,
            },
            redact=False,
            kind=ArtifactKind.LIVE_WEB_RESULT,
            producer_id=self.producer_id,
            run_id=executions[0].snapshot.run_id,
            content_schema="LiveWebEvaluation@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"live-result:{result_id}",
        )
        self.store.save_live_result(result)
        return result
