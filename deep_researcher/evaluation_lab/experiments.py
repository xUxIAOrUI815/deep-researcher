from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentVersionSet,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    utc_now,
)

from .models import (
    EnvironmentDescriptor,
    EvaluationMode,
    ExperimentComparison,
    ExperimentDefinition,
    ExperimentRun,
    ExperimentRunStatus,
    MetricComparison,
    SystemBaseline,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def capture_environment(
    *,
    dependency_files: tuple[str | Path, ...] = (),
    configuration: dict[str, Any] | None = None,
    network_mode: str,
    metadata: dict[str, Any] | None = None,
) -> EnvironmentDescriptor:
    dependency_material = []
    for value in dependency_files:
        path = Path(value)
        if not path.is_file():
            raise ValueError(f"dependency manifest is missing: {path}")
        dependency_material.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    dependency_material.sort(
        key=lambda item: (item["path"], item["name"])
    )
    dependency_fingerprint = _fingerprint(
        [
            {
                "name": item["name"],
                "size_bytes": item["size_bytes"],
                "sha256": item["sha256"],
            }
            for item in dependency_material
        ]
    )
    configuration_fingerprint = _fingerprint(configuration or {})
    environment_id = _stable_id(
        "environment",
        platform.platform(),
        platform.python_version(),
        dependency_fingerprint,
        configuration_fingerprint,
        network_mode,
    )
    return EnvironmentDescriptor(
        environment_id=environment_id,
        platform=platform.platform(),
        python_version=platform.python_version(),
        dependency_fingerprint=dependency_fingerprint,
        configuration_fingerprint=configuration_fingerprint,
        network_mode=network_mode,
        metadata={
            **(metadata or {}),
            "implementation": sys.implementation.name,
            "dependency_manifests": tuple(dependency_material),
            "configuration_values_persisted": False,
        },
    )


class ExperimentRegistry:
    """Immutable experiment provenance and aligned three-baseline comparison."""

    def __init__(
        self,
        *,
        store: SQLiteEvaluationStore,
        artifact_store: ArtifactStore,
        producer_id: str = "runtime_experiment_registry",
        clock=utc_now,
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.producer_id = producer_id
        self.clock = clock

    def define(
        self,
        *,
        name: str,
        baseline: SystemBaseline,
        subject_version_id: str,
        mode: EvaluationMode,
        bundle_id: str,
        dataset_id: str,
        dataset_split: DatasetSplit,
        dataset_purpose: DatasetPurpose,
        dataset_access_record_id: str,
        dataset_fingerprint: str,
        component_versions: ComponentVersionSet,
        environment: EnvironmentDescriptor,
        input_artifact_ids: tuple[str, ...],
        configuration_artifact_ids: tuple[str, ...] = (),
        deterministic_seed: int,
        repeat_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> ExperimentDefinition:
        access = self.store.dataset_access(dataset_access_record_id)
        if access is None:
            raise ValueError("experiment requires a recorded dataset access")
        if (
            access.bundle_id != bundle_id
            or access.request.dataset_id != dataset_id
            or access.request.split != dataset_split
            or access.request.purpose != dataset_purpose
        ):
            raise ValueError(
                "experiment dataset access does not match its definition"
            )
        if access.request.actor_id != subject_version_id:
            raise ValueError(
                "experiment subject must be the audited dataset access actor"
            )
        bundle = self.store.dataset_bundle(bundle_id)
        if bundle is None or bundle.fingerprint != dataset_fingerprint:
            raise ValueError(
                "experiment dataset fingerprint does not match the registry"
            )
        normalized_inputs = tuple(dict.fromkeys(input_artifact_ids))
        normalized_configuration = tuple(
            dict.fromkeys(configuration_artifact_ids)
        )
        artifact_ids = tuple(
            dict.fromkeys((*normalized_inputs, *normalized_configuration))
        )
        missing = [
            item
            for item in artifact_ids
            if self.artifact_store.get(item) is None
        ]
        if missing:
            raise ValueError(
                f"experiment provenance artifacts are missing: {missing}"
            )
        component_artifact_ids = tuple(
            item
            for item in self._component_artifacts(component_versions)
            if item is not None
        )
        missing_components = [
            item
            for item in component_artifact_ids
            if self.artifact_store.get(item) is None
        ]
        if missing_components:
            raise ValueError(
                "component version artifacts are missing: "
                f"{missing_components}"
            )
        material = {
            "name": name,
            "baseline": baseline.value,
            "subject_version_id": subject_version_id,
            "mode": mode.value,
            "bundle_id": bundle_id,
            "dataset_id": dataset_id,
            "dataset_split": dataset_split.value,
            "dataset_purpose": dataset_purpose.value,
            "dataset_access_record_id": dataset_access_record_id,
            "dataset_fingerprint": dataset_fingerprint,
            "component_versions": component_versions.model_dump(mode="json"),
            "environment": environment.model_dump(mode="json"),
            "input_artifact_ids": normalized_inputs,
            "configuration_artifact_ids": normalized_configuration,
            "deterministic_seed": deterministic_seed,
            "repeat_count": repeat_count,
            "metadata": metadata or {},
        }
        fingerprint = _fingerprint(material)
        experiment_id = _stable_id(
            "experiment",
            name,
            baseline.value,
            subject_version_id,
            fingerprint,
        )
        existing = self.store.experiment_definition(experiment_id)
        if existing is not None:
            return existing
        definition = ExperimentDefinition(
            experiment_id=experiment_id,
            name=name,
            baseline=baseline,
            subject_version_id=subject_version_id,
            mode=mode,
            bundle_id=bundle_id,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            dataset_purpose=dataset_purpose,
            dataset_access_record_id=dataset_access_record_id,
            dataset_fingerprint=dataset_fingerprint,
            component_versions=component_versions,
            environment=environment,
            input_artifact_ids=normalized_inputs,
            configuration_artifact_ids=normalized_configuration,
            deterministic_seed=deterministic_seed,
            repeat_count=repeat_count,
            created_at=access.granted_at,
            metadata={**(metadata or {}), "definition_fingerprint": fingerprint},
        )
        self.store.save_experiment_definition(definition)
        return definition

    def record_run(
        self,
        *,
        experiment_id: str,
        metrics: tuple[EvaluationMetric, ...],
        evaluation_artifact_ids: tuple[str, ...],
        output_artifact_ids: tuple[str, ...],
        status: ExperimentRunStatus = ExperimentRunStatus.SUCCEEDED,
        failure_summary: str | None = None,
        started_at=None,
        completed_at=None,
    ) -> ExperimentRun:
        definition = self.store.experiment_definition(experiment_id)
        if definition is None:
            raise KeyError(f"unknown experiment definition: {experiment_id}")
        all_artifacts = tuple(
            dict.fromkeys(
                (*evaluation_artifact_ids, *output_artifact_ids)
            )
        )
        missing = [
            item
            for item in all_artifacts
            if self.artifact_store.get(item) is None
        ]
        if missing:
            raise ValueError(
                f"experiment result artifacts are missing: {missing}"
            )
        start = started_at or self.clock()
        end = completed_at or self.clock()
        result_fingerprint = _fingerprint(
            {
                "experiment_id": experiment_id,
                "metrics": [
                    item.model_dump(mode="json") for item in metrics
                ],
                "evaluation_artifact_ids": evaluation_artifact_ids,
                "output_artifact_ids": output_artifact_ids,
                "status": status.value,
                "failure_summary": failure_summary,
            }
        )
        run_id = _stable_id(
            "experiment_run",
            experiment_id,
            result_fingerprint,
        )
        existing = self.store.experiment_run(run_id)
        if existing is not None:
            return existing
        run = ExperimentRun(
            experiment_run_id=run_id,
            experiment_id=experiment_id,
            baseline=definition.baseline,
            subject_version_id=definition.subject_version_id,
            dataset_id=definition.dataset_id,
            dataset_split=definition.dataset_split,
            dataset_fingerprint=definition.dataset_fingerprint,
            component_versions=definition.component_versions,
            environment_id=definition.environment.environment_id,
            status=status,
            metrics=metrics,
            evaluation_artifact_ids=evaluation_artifact_ids,
            output_artifact_ids=output_artifact_ids,
            failure_summary=failure_summary,
            started_at=start,
            completed_at=end,
        )
        self.store.save_experiment_run(run)
        return run

    def compare(
        self,
        experiment_run_ids: tuple[str, str, str],
    ) -> ExperimentComparison:
        runs = tuple(
            self.store.experiment_run(item)
            for item in experiment_run_ids
        )
        if any(item is None for item in runs):
            raise KeyError("comparison references an unknown experiment run")
        concrete = tuple(item for item in runs if item is not None)
        if any(
            item.status != ExperimentRunStatus.SUCCEEDED
            for item in concrete
        ):
            raise ValueError("only successful experiment runs can be compared")
        by_baseline = {item.baseline: item for item in concrete}
        if set(by_baseline) != set(SystemBaseline):
            raise ValueError(
                "comparison requires exactly legacy, fixed-workflow, and "
                "new-runtime baselines"
            )
        identities = {
            (
                item.dataset_id,
                item.dataset_split,
                item.dataset_fingerprint,
            )
            for item in concrete
        }
        if len(identities) != 1:
            raise ValueError(
                "baseline comparison requires the same sealed dataset split"
            )
        definitions = tuple(
            self.store.experiment_definition(item.experiment_id)
            for item in concrete
        )
        if any(item is None for item in definitions):
            raise ValueError(
                "baseline comparison is missing experiment definitions"
            )
        aligned_definitions = tuple(
            item for item in definitions if item is not None
        )
        conditions = {
            (
                item.mode,
                item.dataset_purpose,
                item.deterministic_seed,
                item.repeat_count,
                item.input_artifact_ids,
                item.configuration_artifact_ids,
            )
            for item in aligned_definitions
        }
        if len(conditions) != 1:
            raise ValueError(
                "baseline comparison requires aligned mode, purpose, seed, "
                "repeats, and input/configuration artifacts"
            )
        metric_maps = {
            baseline: {item.name: item for item in run.metrics}
            for baseline, run in by_baseline.items()
        }
        common = set.intersection(
            *(set(values) for values in metric_maps.values())
        )
        if not common:
            raise ValueError(
                "baseline comparison has no common deterministic metrics"
            )
        comparisons: list[MetricComparison] = []
        for name in sorted(common):
            legacy = metric_maps[SystemBaseline.LEGACY][name]
            fixed = metric_maps[SystemBaseline.FIXED_WORKFLOW][name]
            new = metric_maps[SystemBaseline.NEW_RUNTIME][name]
            if len({legacy.direction, fixed.direction, new.direction}) != 1:
                raise ValueError(
                    f"metric direction differs across baselines: {name}"
                )
            comparisons.append(
                MetricComparison(
                    metric_name=name,
                    legacy_value=legacy.value,
                    fixed_workflow_value=fixed.value,
                    new_runtime_value=new.value,
                    new_vs_legacy_delta=new.value - legacy.value,
                    new_vs_fixed_delta=new.value - fixed.value,
                    direction=new.direction.value,
                )
            )
        unavailable = {
            baseline: tuple(sorted(set.union(
                *(set(values) for values in metric_maps.values())
            ) - set(values)))
            for baseline, values in metric_maps.items()
        }
        dataset_id, dataset_split, dataset_fingerprint = next(iter(identities))
        canonical_ids = tuple(
            by_baseline[item].experiment_run_id for item in SystemBaseline
        )
        fingerprint = _fingerprint(
            {
                "runs": canonical_ids,
                "metrics": [
                    item.model_dump(mode="json") for item in comparisons
                ],
                "unavailable": {
                    key.value: list(value)
                    for key, value in unavailable.items()
                },
            }
        )
        comparison_id = _stable_id(
            "experiment_comparison",
            dataset_id,
            fingerprint,
        )
        artifact_id = _stable_id("artifact", comparison_id)
        created_at = max(item.completed_at for item in concrete)
        comparison = ExperimentComparison(
            comparison_id=comparison_id,
            experiment_run_ids=canonical_ids,
            dataset_id=dataset_id,
            dataset_split=dataset_split,
            dataset_fingerprint=dataset_fingerprint,
            metrics=tuple(comparisons),
            unavailable_metrics=unavailable,
            result_artifact_id=artifact_id,
            created_at=created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "ExperimentComparison@1",
                "comparison": comparison.model_dump(mode="json"),
                "aligned_dataset": True,
                "release_decision": None,
            },
            redact=False,
            kind=ArtifactKind.EXPERIMENT_RESULT,
            producer_id=self.producer_id,
            run_id=_stable_id("run", comparison_id),
            content_schema="ExperimentComparison@1",
            # Comparisons aggregate distinct experiment runs. All referenced
            # artifact IDs are sealed in the comparison payload.
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"experiment-comparison:{comparison_id}",
        )
        self.store.save_comparison(comparison)
        return comparison

    @staticmethod
    def _component_artifacts(
        versions: ComponentVersionSet,
    ) -> tuple[str | None, ...]:
        values = (
            versions.runtime,
            versions.scheduler,
            versions.model,
            versions.agent_spec,
            versions.prompt,
            versions.skill,
            versions.tool_policy,
            versions.stop_policy,
            versions.verification_policy,
            versions.rubric,
            *versions.tools,
        )
        return tuple(
            item.artifact_id if item is not None else None for item in values
        )
