from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from deep_researcher.artifacts import SQLiteArtifactStore

from .ingestion import KnowledgeIngestionService
from .repository import KnowledgeRepository
from .retrieval import KnowledgeRetrievalService, VectorRetrievalAdapter
from .sqlite_storage import SQLiteKnowledgeStorage


@dataclass
class KnowledgeRuntime:
    root: Path
    artifacts: SQLiteArtifactStore
    storage: SQLiteKnowledgeStorage
    repository: KnowledgeRepository
    ingestion: KnowledgeIngestionService
    retrieval: KnowledgeRetrievalService

    def integrity_check(self) -> None:
        self.artifacts.integrity_check()
        self.storage.integrity_check()

    @staticmethod
    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.mkdir(parents=True, exist_ok=True)
        artifact_path = target / "artifacts.sqlite3"
        knowledge_path = target / "knowledge.sqlite3"
        self.artifacts.backup_to(artifact_path)
        self.storage.backup_to(knowledge_path)
        manifest = {
            "schema": "KnowledgeRuntimeBackup@1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "files": {
                artifact_path.name: self._hash(artifact_path),
                knowledge_path.name: self._hash(knowledge_path),
            },
        }
        manifest_path = target / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return manifest_path

    def close(self) -> None:
        self.storage.close()
        self.artifacts.close()

    def __enter__(self) -> "KnowledgeRuntime":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()


def build_knowledge_runtime(
    root: str | Path,
    *,
    vector_adapter: VectorRetrievalAdapter | None = None,
) -> KnowledgeRuntime:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    artifacts = SQLiteArtifactStore(root_path / "artifacts.sqlite3")
    storage = SQLiteKnowledgeStorage(root_path / "knowledge.sqlite3", artifact_store=artifacts)
    repository = KnowledgeRepository(storage)
    runtime = KnowledgeRuntime(
        root=root_path,
        artifacts=artifacts,
        storage=storage,
        repository=repository,
        ingestion=KnowledgeIngestionService(artifacts, repository),
        retrieval=KnowledgeRetrievalService(repository, vector_adapter=vector_adapter),
    )
    runtime.integrity_check()
    return runtime


def restore_knowledge_runtime(backup: str | Path, destination: str | Path) -> KnowledgeRuntime:
    source = Path(backup)
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    for name, digest in manifest.get("files", {}).items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"backup manifest checksum mismatch: {name}")
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    artifacts = SQLiteArtifactStore.restore_backup(source / "artifacts.sqlite3", target / "artifacts.sqlite3")
    storage = SQLiteKnowledgeStorage.restore_backup(
        source / "knowledge.sqlite3", target / "knowledge.sqlite3", artifact_store=artifacts
    )
    repository = KnowledgeRepository(storage)
    return KnowledgeRuntime(
        root=target,
        artifacts=artifacts,
        storage=storage,
        repository=repository,
        ingestion=KnowledgeIngestionService(artifacts, repository),
        retrieval=KnowledgeRetrievalService(repository),
    )
