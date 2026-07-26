from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.evaluation_lab import ReleaseGateService
from deep_researcher.version_registry import VersionRegistry

from .lab import OfflineEvolutionLab
from .patching import OfflinePatchGenerator
from .store import SQLiteEvolutionStore


@dataclass
class EvolutionLabRuntime:
    store: SQLiteEvolutionStore
    lab: OfflineEvolutionLab
    artifact_store: ArtifactStore
    version_registry: VersionRegistry
    release_gate: ReleaseGateService

    def integrity_check(self) -> None:
        self.store.integrity_check()
        self.artifact_store.integrity_check()
        self.version_registry.store.integrity_check()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "EvolutionLabRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_evolution_lab_runtime(
    root: str | Path,
    *,
    artifact_store: ArtifactStore,
    version_registry: VersionRegistry,
    release_gate: ReleaseGateService,
    generator: OfflinePatchGenerator | None = None,
) -> EvolutionLabRuntime:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteEvolutionStore(
        root_path / "offline_evolution.sqlite3"
    )
    runtime = EvolutionLabRuntime(
        store=store,
        lab=OfflineEvolutionLab(
            store=store,
            artifact_store=artifact_store,
            version_registry=version_registry,
            release_gate=release_gate,
            generator=generator,
        ),
        artifact_store=artifact_store,
        version_registry=version_registry,
        release_gate=release_gate,
    )
    runtime.integrity_check()
    return runtime
