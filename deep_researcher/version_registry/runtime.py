from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from deep_researcher.artifacts.store import ArtifactStore

from .registry import VersionRegistry
from .store import SQLiteVersionRegistryStore


@dataclass
class VersionRegistryRuntime:
    store: SQLiteVersionRegistryStore
    registry: VersionRegistry
    artifact_store: ArtifactStore

    def integrity_check(self) -> None:
        self.store.integrity_check()
        self.artifact_store.integrity_check()

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "VersionRegistryRuntime":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_version_registry_runtime(
    root: str | Path,
    *,
    artifact_store: ArtifactStore,
) -> VersionRegistryRuntime:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    store = SQLiteVersionRegistryStore(
        root_path / "version_registry.sqlite3"
    )
    runtime = VersionRegistryRuntime(
        store=store,
        registry=VersionRegistry(
            store=store,
            artifact_store=artifact_store,
        ),
        artifact_store=artifact_store,
    )
    runtime.integrity_check()
    return runtime
