from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import sqlite3

import pytest

from deep_researcher.artifacts import (
    ArtifactConflict,
    ArtifactCorruption,
    ArtifactNotFound,
    ArtifactQuery,
    SQLiteArtifactStore,
)
from deep_researcher.contracts import ArtifactKind, ArtifactLink, ArtifactLinkRelation


def _put(store: SQLiteArtifactStore, value: str, *, run_id: str = "run_artifacts", **kwargs):
    return store.put_text(
        value,
        kind=ArtifactKind.SOURCE_SNAPSHOT,
        producer_id="producer_test",
        run_id=run_id,
        **kwargs,
    )


def test_content_addressing_keeps_immutable_envelopes_and_deduplicates_blobs(tmp_path):
    with SQLiteArtifactStore(tmp_path / "artifacts.sqlite3") as store:
        first = _put(store, "exact token=source-value")
        second = _put(store, "exact token=source-value")
        assert first.artifact_id != second.artifact_id
        assert first.content_hash == second.content_hash
        assert store.read_bytes(first.artifact_id).decode() == "exact token=source-value"
        assert store._connection.execute("SELECT COUNT(*) FROM blobs").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            store._connection.execute("UPDATE artifacts SET kind='other'")


def test_redaction_is_explicit_for_governed_structured_artifacts(tmp_path):
    with SQLiteArtifactStore(tmp_path / "artifacts.sqlite3") as store:
        artifact = store.put_json(
            {"api_key": "super-secret", "safe": True},
            kind=ArtifactKind.MODEL_OUTPUT,
            producer_id="producer_model",
            run_id="run_artifacts",
        )
        assert store.read_json(artifact.artifact_id) == {"api_key": "[REDACTED]", "safe": True}


def test_idempotency_referential_integrity_links_and_run_isolation(tmp_path):
    with SQLiteArtifactStore(tmp_path / "artifacts.sqlite3") as store:
        source = _put(store, "source", idempotency_key="source")
        assert _put(store, "source", idempotency_key="source") == source
        with pytest.raises(ArtifactConflict, match="idempotency"):
            _put(store, "changed", idempotency_key="source")
        child = store.put_text(
            "child", kind=ArtifactKind.CLEANED_CONTENT, producer_id="producer_test",
            run_id="run_artifacts", source_artifact_ids=(source.artifact_id,),
        )
        store.link(ArtifactLink(source_artifact_id=child.artifact_id, target_artifact_id=source.artifact_id, relation=ArtifactLinkRelation.DERIVED_FROM))
        assert store.links(child.artifact_id)[0].target_artifact_id == source.artifact_id
        assert store.links(source.artifact_id, incoming=True, relation=ArtifactLinkRelation.DERIVED_FROM)[0].source_artifact_id == child.artifact_id
        with pytest.raises(ArtifactNotFound):
            store.put_text("bad", kind=ArtifactKind.OTHER, producer_id="producer_test", run_id="run_artifacts", source_artifact_ids=("artifact_missing",))
        other = _put(store, "other", run_id="run_other")
        with pytest.raises(ArtifactConflict, match="cross runs"):
            store.link(ArtifactLink(source_artifact_id=source.artifact_id, target_artifact_id=other.artifact_id, relation=ArtifactLinkRelation.REPRESENTS))
        store.integrity_check()


def test_pagination_restart_backup_restore_and_migration(tmp_path):
    database = tmp_path / "artifacts.sqlite3"
    store = SQLiteArtifactStore(database)
    for index in range(5):
        _put(store, f"body-{index}", artifact_id=f"artifact_page_{index}")
    first = store.list(ArtifactQuery("run_artifacts", limit=2))
    second = store.list(ArtifactQuery("run_artifacts", after_created_at=first.next_cursor[0], after_artifact_id=first.next_cursor[1], limit=3))
    assert len(first.items) + len(second.items) == 5
    store.backup_to(tmp_path / "backup.sqlite3")
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("DROP TABLE artifact_links")
    for name in ("artifacts_no_update", "artifacts_no_delete", "blobs_no_update", "blobs_no_delete"):
        connection.execute(f"DROP TRIGGER {name}")
    connection.execute("PRAGMA user_version=1")
    connection.commit()
    connection.close()
    migrated = SQLiteArtifactStore(database)
    assert migrated._connection.execute("PRAGMA user_version").fetchone()[0] == 2
    migrated.integrity_check()
    migrated.close()
    restored = SQLiteArtifactStore.restore_backup(tmp_path / "backup.sqlite3", tmp_path / "restored.sqlite3")
    assert len(restored.list(ArtifactQuery("run_artifacts")).items) == 5
    restored.close()


def test_concurrent_connections_share_idempotency_and_isolate_runs(tmp_path):
    database = tmp_path / "artifacts.sqlite3"
    stores = [SQLiteArtifactStore(database) for _ in range(6)]
    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            values = list(executor.map(lambda store: _put(store, "same", idempotency_key="same"), stores))
        assert len({item.artifact_id for item in values}) == 1
        for index, store in enumerate(stores):
            _put(store, str(index), run_id=f"run_{index}")
        assert len(stores[0].list(ArtifactQuery("run_0")).items) == 1
        stores[0].integrity_check()
    finally:
        for store in stores:
            store.close()


def test_corruption_is_detected_on_read_list_and_audit(tmp_path):
    database = tmp_path / "artifacts.sqlite3"
    store = SQLiteArtifactStore(database)
    artifact = _put(store, "body")
    store.close()
    connection = sqlite3.connect(database)
    connection.execute("DROP TRIGGER blobs_no_update")
    connection.execute("UPDATE blobs SET content=? WHERE content_hash=?", (b"evil", artifact.content_hash))
    connection.commit()
    connection.close()
    reopened = SQLiteArtifactStore(database)
    try:
        with pytest.raises(ArtifactCorruption):
            reopened.read_bytes(artifact.artifact_id)
        with pytest.raises(ArtifactCorruption):
            reopened.integrity_check()
    finally:
        reopened.close()
