from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.events import EventStore
from deep_researcher.reporting import SQLiteReportingStore

from .datasets import DatasetRegistry
from .evaluators import DeterministicEvaluatorSuite
from .experiments import ExperimentRegistry
from .modes import (
    FrozenReplayRunner,
    LiveWebExecutor,
    LiveWebRunner,
    ReplayExecutor,
)
from .snapshot import EvaluationSnapshotBuilder
from .store import SQLiteEvaluationStore


@dataclass
class EvaluationLabRuntime:
    store: SQLiteEvaluationStore
    artifact_store: ArtifactStore
    datasets: DatasetRegistry
    evaluator: DeterministicEvaluatorSuite
    experiments: ExperimentRegistry
    snapshots: EvaluationSnapshotBuilder | None = None

    def frozen_replay(self, executor: ReplayExecutor) -> FrozenReplayRunner:
        return FrozenReplayRunner(
            artifact_store=self.artifact_store,
            store=self.store,
            evaluator=self.evaluator,
            executor=executor,
        )

    def live_web(self, executor: LiveWebExecutor) -> LiveWebRunner:
        return LiveWebRunner(
            artifact_store=self.artifact_store,
            store=self.store,
            evaluator=self.evaluator,
            executor=executor,
        )

    def integrity_check(self) -> None:
        self.store.integrity_check()
        self.artifact_store.integrity_check()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "EvaluationLabRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_evaluation_lab_runtime(
    root: str | Path,
    *,
    artifact_store: ArtifactStore,
    evidence: EvidenceRuntime | None = None,
    reporting_store: SQLiteReportingStore | None = None,
    event_store: EventStore | None = None,
    freshness_days: int = 730,
) -> EvaluationLabRuntime:
    if artifact_store is None:
        raise ValueError("Evaluation Lab requires an immutable ArtifactStore")
    if (evidence is None) != (reporting_store is None):
        raise ValueError(
            "snapshot projection requires both EvidenceRuntime and "
            "SQLiteReportingStore"
        )
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteEvaluationStore(root_path / "evaluation_lab.sqlite3")
    datasets = DatasetRegistry(
        store=store,
        artifact_store=artifact_store,
    )
    evaluator = DeterministicEvaluatorSuite(
        artifact_store=artifact_store,
        store=store,
        freshness_days=freshness_days,
    )
    experiments = ExperimentRegistry(
        store=store,
        artifact_store=artifact_store,
    )
    snapshots = (
        EvaluationSnapshotBuilder(
            evidence=evidence,
            reporting_store=reporting_store,
            event_store=event_store,
        )
        if evidence is not None and reporting_store is not None
        else None
    )
    runtime = EvaluationLabRuntime(
        store=store,
        artifact_store=artifact_store,
        datasets=datasets,
        evaluator=evaluator,
        experiments=experiments,
        snapshots=snapshots,
    )
    runtime.integrity_check()
    return runtime
