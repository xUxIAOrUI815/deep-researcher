from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.contracts import (
    ArtifactEnvelope,
    ArtifactKind,
    ArtifactLink,
    ArtifactLinkRelation,
    Sensitivity,
    canonical_contract_json,
    new_id,
)
from deep_researcher.events.redaction import RedactionPolicy

from .store import (
    ArtifactConflict,
    ArtifactCorruption,
    ArtifactNotFound,
    ArtifactPage,
    ArtifactQuery,
)


CURRENT_ARTIFACT_SCHEMA_VERSION = 2


class SQLiteArtifactStore:
    """Immutable, content-addressed artifact bodies and provenance envelopes."""

    def __init__(self, path: str | Path, *, redaction_policy: RedactionPolicy | None = None) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.redaction_policy = redaction_policy or RedactionPolicy()
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path), timeout=30.0, isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=30000")
        if str(path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
            else:
                self._connection.execute("COMMIT")

    def _migrate(self) -> None:
        version = int(self._connection.execute("PRAGMA user_version").fetchone()[0])
        if version > CURRENT_ARTIFACT_SCHEMA_VERSION:
            raise ArtifactCorruption("artifact database schema is newer than this runtime")
        if version < 1:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE blobs (
                    content_hash TEXT PRIMARY KEY,
                    byte_length INTEGER NOT NULL,
                    media_type TEXT NOT NULL,
                    content BLOB NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE artifacts (
                    artifact_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    producer_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    content_schema TEXT,
                    idempotency_key TEXT,
                    envelope_json TEXT NOT NULL,
                    envelope_checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(content_hash) REFERENCES blobs(content_hash),
                    UNIQUE(run_id, producer_id, idempotency_key)
                );
                CREATE INDEX idx_artifacts_run_created ON artifacts(run_id, created_at, artifact_id);
                CREATE INDEX idx_artifacts_run_kind_created ON artifacts(run_id, kind, created_at, artifact_id);
                CREATE INDEX idx_artifacts_hash ON artifacts(content_hash);
                PRAGMA user_version=1;
                COMMIT;
                """
            )
            version = 1
        if version < 2:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE artifact_links (
                    source_artifact_id TEXT NOT NULL,
                    target_artifact_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    link_json TEXT NOT NULL,
                    link_checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(source_artifact_id, target_artifact_id, relation),
                    FOREIGN KEY(source_artifact_id) REFERENCES artifacts(artifact_id) ON DELETE RESTRICT,
                    FOREIGN KEY(target_artifact_id) REFERENCES artifacts(artifact_id) ON DELETE RESTRICT
                );
                CREATE INDEX idx_artifact_links_target ON artifact_links(target_artifact_id, relation);
                CREATE TRIGGER artifacts_no_update BEFORE UPDATE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END;
                CREATE TRIGGER artifacts_no_delete BEFORE DELETE ON artifacts BEGIN SELECT RAISE(ABORT, 'artifacts are immutable'); END;
                CREATE TRIGGER blobs_no_update BEFORE UPDATE ON blobs BEGIN SELECT RAISE(ABORT, 'artifact blobs are immutable'); END;
                CREATE TRIGGER blobs_no_delete BEFORE DELETE ON blobs BEGIN SELECT RAISE(ABORT, 'artifact blobs are immutable'); END;
                PRAGMA user_version=2;
                COMMIT;
                """
            )

    @staticmethod
    def _sha256(content: bytes) -> str:
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _decode_envelope(value: str) -> ArtifactEnvelope:
        return ArtifactEnvelope.model_validate_json(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Enum):
            return value.value
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
        raise TypeError(f"value is not JSON serializable: {type(value).__name__}")

    @staticmethod
    def _equivalent(left: ArtifactEnvelope, right: ArtifactEnvelope) -> bool:
        fields = (
            "artifact_id",
            "kind",
            "content_hash",
            "byte_length",
            "media_type",
            "content_schema",
            "producer_id",
            "run_id",
            "task_id",
            "source_artifact_ids",
            "sensitivity",
            "metadata",
        )
        return all(getattr(left, field_name) == getattr(right, field_name) for field_name in fields)

    def put_json(self, value: Any, *, redact: bool = True, **kwargs: Any) -> ArtifactEnvelope:
        redacted = self.redaction_policy.redact(value) if redact else value
        content = json.dumps(
            redacted,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=self._json_default,
        ).encode("utf-8")
        return self.put_bytes(content, media_type="application/json", **kwargs)

    def put_text(self, value: str, *, redact: bool = False, **kwargs: Any) -> ArtifactEnvelope:
        content = (self.redaction_policy.redact_text(value) if redact else value).encode("utf-8")
        return self.put_bytes(content, media_type="text/plain; charset=utf-8", **kwargs)

    def put_bytes(
        self,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        producer_id: str,
        run_id: str,
        task_id: str | None = None,
        content_schema: str | None = None,
        source_artifact_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        idempotency_key: str | None = None,
        sensitivity: Sensitivity = Sensitivity.INTERNAL,
    ) -> ArtifactEnvelope:
        if not isinstance(content, bytes):
            raise TypeError("artifact content must be bytes")
        content_hash = self._sha256(content)
        resolved_id = artifact_id or new_id("artifact")
        envelope = ArtifactEnvelope(
            artifact_id=resolved_id,
            kind=kind,
            content_uri=f"artifact+sqlite://sha256/{content_hash}",
            content_hash=content_hash,
            byte_length=len(content),
            media_type=media_type,
            content_schema=content_schema,
            producer_id=producer_id,
            run_id=run_id,
            task_id=task_id,
            source_artifact_ids=source_artifact_ids,
            sensitivity=sensitivity,
            metadata=self.redaction_policy.redact(metadata or {}),
        )
        serialized = canonical_contract_json(envelope)
        envelope_checksum = self._sha256(serialized.encode("utf-8"))
        with self._transaction() as connection:
            existing = connection.execute(
                "SELECT envelope_json FROM artifacts WHERE artifact_id=?", (resolved_id,)
            ).fetchone()
            if existing is not None:
                stored = self._decode_envelope(existing["envelope_json"])
                if not self._equivalent(stored, envelope):
                    raise ArtifactConflict(f"artifact ID reused with different content: {resolved_id}")
                return stored
            if idempotency_key is not None:
                idem = connection.execute(
                    "SELECT envelope_json FROM artifacts WHERE run_id=? AND producer_id=? AND idempotency_key=?",
                    (run_id, producer_id, idempotency_key),
                ).fetchone()
                if idem is not None:
                    stored = self._decode_envelope(idem["envelope_json"])
                    if (
                        stored.content_hash != content_hash
                        or stored.kind != kind
                        or stored.media_type != media_type
                        or stored.task_id != task_id
                        or stored.content_schema != content_schema
                        or stored.source_artifact_ids != source_artifact_ids
                        or stored.sensitivity != sensitivity
                        or stored.metadata != self.redaction_policy.redact(metadata or {})
                    ):
                        raise ArtifactConflict("idempotency key reused for a different artifact")
                    return stored
            for source_id in source_artifact_ids:
                source = connection.execute(
                    "SELECT run_id FROM artifacts WHERE artifact_id=?", (source_id,)
                ).fetchone()
                if source is None:
                    raise ArtifactNotFound(f"source artifact does not exist: {source_id}")
                if source["run_id"] != run_id:
                    raise ArtifactConflict("artifact provenance cannot cross runs")
            blob = connection.execute(
                "SELECT byte_length, media_type, content FROM blobs WHERE content_hash=?",
                (content_hash,),
            ).fetchone()
            if blob is None:
                connection.execute(
                    "INSERT INTO blobs(content_hash, byte_length, media_type, content, created_at) VALUES (?, ?, ?, ?, ?)",
                    (content_hash, len(content), media_type, content, envelope.created_at.isoformat()),
                )
            elif bytes(blob["content"]) != content or int(blob["byte_length"]) != len(content):
                raise ArtifactCorruption(f"content-address collision or corrupt blob: {content_hash}")
            connection.execute(
                """INSERT INTO artifacts(
                    artifact_id, run_id, task_id, producer_id, kind, content_hash,
                    content_schema, idempotency_key, envelope_json,
                    envelope_checksum, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    resolved_id,
                    run_id,
                    task_id,
                    producer_id,
                    kind.value,
                    content_hash,
                    content_schema,
                    idempotency_key,
                    serialized,
                    envelope_checksum,
                    envelope.created_at.isoformat(),
                ),
            )
        return envelope

    def get(self, artifact_id: str) -> ArtifactEnvelope | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT envelope_json, envelope_checksum FROM artifacts WHERE artifact_id=?",
                (artifact_id,),
            ).fetchone()
        if row is None:
            return None
        if self._sha256(row["envelope_json"].encode("utf-8")) != row["envelope_checksum"]:
            raise ArtifactCorruption(f"envelope checksum mismatch: {artifact_id}")
        return self._decode_envelope(row["envelope_json"])

    def read_bytes(self, artifact_id: str) -> bytes:
        with self._lock:
            row = self._connection.execute(
                """SELECT a.content_hash, b.byte_length, b.content
                   FROM artifacts a JOIN blobs b ON b.content_hash=a.content_hash
                   WHERE a.artifact_id=?""",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise ArtifactNotFound(artifact_id)
        content = bytes(row["content"])
        if len(content) != int(row["byte_length"]) or self._sha256(content) != row["content_hash"]:
            raise ArtifactCorruption(f"artifact body checksum mismatch: {artifact_id}")
        return content

    def read_json(self, artifact_id: str) -> Any:
        return json.loads(self.read_bytes(artifact_id).decode("utf-8"))

    def link(self, link: ArtifactLink) -> None:
        serialized = canonical_contract_json(link)
        checksum = self._sha256(serialized.encode("utf-8"))
        with self._transaction() as connection:
            runs = []
            for artifact_id in (link.source_artifact_id, link.target_artifact_id):
                row = connection.execute(
                    "SELECT run_id FROM artifacts WHERE artifact_id=?", (artifact_id,)
                ).fetchone()
                if row is None:
                    raise ArtifactNotFound(artifact_id)
                runs.append(row["run_id"])
            if runs[0] != runs[1]:
                raise ArtifactConflict("artifact links cannot cross runs")
            existing = connection.execute(
                "SELECT link_json FROM artifact_links WHERE source_artifact_id=? AND target_artifact_id=? AND relation=?",
                (link.source_artifact_id, link.target_artifact_id, link.relation.value),
            ).fetchone()
            if existing is not None:
                if ArtifactLink.model_validate_json(existing["link_json"]) != link:
                    raise ArtifactConflict("artifact link identity reused with different content")
                return
            connection.execute(
                "INSERT INTO artifact_links VALUES (?, ?, ?, ?, ?, ?)",
                (
                    link.source_artifact_id,
                    link.target_artifact_id,
                    link.relation.value,
                    serialized,
                    checksum,
                    link.created_at.isoformat(),
                ),
            )

    def list(self, query: ArtifactQuery) -> ArtifactPage:
        clauses = ["run_id=?"]
        params: list[Any] = [query.run_id]
        if query.kinds:
            placeholders = ",".join("?" for _ in query.kinds)
            clauses.append(f"kind IN ({placeholders})")
            params.extend(kind.value for kind in query.kinds)
        for column, value in (("producer_id", query.producer_id), ("task_id", query.task_id)):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        if query.after_created_at is not None and query.after_artifact_id is not None:
            timestamp = query.after_created_at.isoformat()
            clauses.append("(created_at>? OR (created_at=? AND artifact_id>?))")
            params.extend((timestamp, timestamp, query.after_artifact_id))
        params.append(query.limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT envelope_json, envelope_checksum FROM artifacts WHERE {' AND '.join(clauses)} ORDER BY created_at, artifact_id LIMIT ?",
                params,
            ).fetchall()
        has_more = len(rows) > query.limit
        items_list: list[ArtifactEnvelope] = []
        for row in rows[: query.limit]:
            if self._sha256(row["envelope_json"].encode("utf-8")) != row["envelope_checksum"]:
                raise ArtifactCorruption("artifact envelope checksum mismatch during listing")
            items_list.append(self._decode_envelope(row["envelope_json"]))
        items = tuple(items_list)
        cursor = (items[-1].created_at, items[-1].artifact_id) if has_more and items else None
        return ArtifactPage(items=items, next_cursor=cursor)

    def links(
        self,
        artifact_id: str,
        *,
        incoming: bool = False,
        relation: ArtifactLinkRelation | None = None,
    ) -> tuple[ArtifactLink, ...]:
        if self.get(artifact_id) is None:
            raise ArtifactNotFound(artifact_id)
        column = "target_artifact_id" if incoming else "source_artifact_id"
        clauses = [f"{column}=?"]
        params: list[Any] = [artifact_id]
        if relation is not None:
            clauses.append("relation=?")
            params.append(relation.value)
        with self._lock:
            rows = self._connection.execute(
                f"SELECT link_json, link_checksum FROM artifact_links WHERE {' AND '.join(clauses)} ORDER BY created_at, source_artifact_id, target_artifact_id, relation",
                params,
            ).fetchall()
        links: list[ArtifactLink] = []
        for row in rows:
            if self._sha256(row["link_json"].encode("utf-8")) != row["link_checksum"]:
                raise ArtifactCorruption("artifact link checksum mismatch during traversal")
            links.append(ArtifactLink.model_validate_json(row["link_json"]))
        return tuple(links)

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()[0]
            if result != "ok":
                raise ArtifactCorruption(f"SQLite integrity check failed: {result}")
            foreign_keys = self._connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise ArtifactCorruption(f"artifact foreign key failure: {foreign_keys[0]}")
            for row in self._connection.execute("SELECT * FROM blobs").fetchall():
                content = bytes(row["content"])
                if len(content) != int(row["byte_length"]) or self._sha256(content) != row["content_hash"]:
                    raise ArtifactCorruption(f"blob checksum mismatch: {row['content_hash']}")
            for row in self._connection.execute("SELECT * FROM artifacts").fetchall():
                if self._sha256(row["envelope_json"].encode("utf-8")) != row["envelope_checksum"]:
                    raise ArtifactCorruption(f"envelope checksum mismatch: {row['artifact_id']}")
                envelope = self._decode_envelope(row["envelope_json"])
                if (
                    envelope.artifact_id != row["artifact_id"]
                    or envelope.content_hash != row["content_hash"]
                    or envelope.run_id != row["run_id"]
                ):
                    raise ArtifactCorruption(f"artifact index disagrees with envelope: {row['artifact_id']}")
            for row in self._connection.execute("SELECT * FROM artifact_links").fetchall():
                if self._sha256(row["link_json"].encode("utf-8")) != row["link_checksum"]:
                    raise ArtifactCorruption("artifact link checksum mismatch")

    def backup_to(self, destination: str | Path) -> None:
        destination_path = Path(destination).resolve()
        if str(self.path) != ":memory:" and destination_path == self.path.resolve():
            raise ValueError("backup destination must differ from active database")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            target = sqlite3.connect(destination_path)
            try:
                self._connection.backup(target)
            finally:
                target.close()
        verification = SQLiteArtifactStore(destination_path)
        try:
            verification.integrity_check()
        finally:
            verification.close()

    @classmethod
    def restore_backup(cls, backup: str | Path, destination: str | Path) -> "SQLiteArtifactStore":
        backup_path = Path(backup).resolve()
        destination_path = Path(destination).resolve()
        if backup_path == destination_path:
            raise ValueError("backup and destination must differ")
        source = sqlite3.connect(f"file:{backup_path.as_posix()}?mode=ro", uri=True)
        try:
            if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ArtifactCorruption("backup SQLite integrity check failed")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            target = sqlite3.connect(destination_path)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        restored = cls(destination_path)
        restored.integrity_check()
        return restored

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteArtifactStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
