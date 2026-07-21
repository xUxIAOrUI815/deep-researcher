from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
from enum import Enum
import hashlib
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    AtomicFact,
    Citation,
    Claim,
    Conflict,
    Evidence,
    Passage,
    Report,
    Section,
    Source,
    SourceSnapshot,
    canonical_contract_json,
)

from .storage import (
    KnowledgeConflict,
    KnowledgeCorruption,
    KnowledgeEntity,
    KnowledgeNotFound,
    KnowledgePage,
    KnowledgeQuery,
    KnowledgeRelation,
    SavedRevision,
)


CURRENT_KNOWLEDGE_SCHEMA_VERSION = 2
_ENTITY_MODELS: dict[str, type[KnowledgeEntity]] = {
    model.__name__: model
    for model in (Source, SourceSnapshot, Passage, Evidence, AtomicFact, Claim, Citation, Conflict, Section, Report)
}
_ID_FIELDS = {
    "Source": "source_id",
    "SourceSnapshot": "snapshot_id",
    "Passage": "passage_id",
    "Evidence": "evidence_id",
    "AtomicFact": "fact_id",
    "Claim": "claim_id",
    "Citation": "citation_id",
    "Conflict": "conflict_id",
    "Section": "section_id",
    "Report": "report_id",
}
_RELATION_TARGET_TYPES: dict[KnowledgeRelation, str] = {
    KnowledgeRelation.SNAPSHOT_SOURCE: "Source",
    KnowledgeRelation.PASSAGE_SNAPSHOT: "SourceSnapshot",
    KnowledgeRelation.EVIDENCE_PASSAGE: "Passage",
    KnowledgeRelation.FACT_EVIDENCE: "Evidence",
    KnowledgeRelation.CLAIM_FACT: "AtomicFact",
    KnowledgeRelation.CLAIM_EVIDENCE: "Evidence",
    KnowledgeRelation.CITATION_CLAIM: "Claim",
    KnowledgeRelation.CITATION_EVIDENCE: "Evidence",
    KnowledgeRelation.CITATION_PASSAGE: "Passage",
    KnowledgeRelation.CITATION_SNAPSHOT: "SourceSnapshot",
    KnowledgeRelation.CITATION_SOURCE: "Source",
    KnowledgeRelation.CONFLICT_CLAIM: "Claim",
    KnowledgeRelation.CONFLICT_FACT: "AtomicFact",
    KnowledgeRelation.CONFLICT_RESOLUTION_EVIDENCE: "Evidence",
    KnowledgeRelation.SECTION_REPORT: "Report",
    KnowledgeRelation.SECTION_PARENT: "Section",
    KnowledgeRelation.SECTION_CLAIM: "Claim",
    KnowledgeRelation.SECTION_CITATION: "Citation",
    KnowledgeRelation.REPORT_SECTION: "Section",
}
_RELATION_SOURCE_TYPES: dict[KnowledgeRelation, str] = {
    KnowledgeRelation.SNAPSHOT_SOURCE: "SourceSnapshot",
    KnowledgeRelation.PASSAGE_SNAPSHOT: "Passage",
    KnowledgeRelation.EVIDENCE_PASSAGE: "Evidence",
    KnowledgeRelation.FACT_EVIDENCE: "AtomicFact",
    KnowledgeRelation.CLAIM_FACT: "Claim",
    KnowledgeRelation.CLAIM_EVIDENCE: "Claim",
    KnowledgeRelation.CITATION_CLAIM: "Citation",
    KnowledgeRelation.CITATION_EVIDENCE: "Citation",
    KnowledgeRelation.CITATION_PASSAGE: "Citation",
    KnowledgeRelation.CITATION_SNAPSHOT: "Citation",
    KnowledgeRelation.CITATION_SOURCE: "Citation",
    KnowledgeRelation.CONFLICT_CLAIM: "Conflict",
    KnowledgeRelation.CONFLICT_FACT: "Conflict",
    KnowledgeRelation.CONFLICT_RESOLUTION_EVIDENCE: "Conflict",
    KnowledgeRelation.SECTION_REPORT: "Section",
    KnowledgeRelation.SECTION_PARENT: "Section",
    KnowledgeRelation.SECTION_CLAIM: "Section",
    KnowledgeRelation.SECTION_CITATION: "Section",
    KnowledgeRelation.REPORT_SECTION: "Report",
}


def entity_id(entity: KnowledgeEntity) -> str:
    return str(getattr(entity, _ID_FIELDS[type(entity).__name__]))


def entity_run_id(entity: KnowledgeEntity) -> str:
    provenance = getattr(entity, "provenance", None)
    return str(provenance.run_id if provenance is not None else entity.run_id)


def _status(entity: KnowledgeEntity) -> str:
    value = getattr(entity, "status")
    return value.value if isinstance(value, Enum) else str(value)


def _created_at(entity: KnowledgeEntity) -> datetime:
    for field_name in ("created_at", "discovered_at", "fetched_at"):
        value = getattr(entity, field_name, None)
        if isinstance(value, datetime):
            return value
    raise ValueError(f"entity has no creation timestamp: {type(entity).__name__}")


def _artifact_ids(entity: KnowledgeEntity) -> tuple[str, ...]:
    values: list[str] = []
    provenance = getattr(entity, "provenance", None)
    if provenance is not None:
        values.extend(provenance.source_artifact_ids)
    for field_name in ("artifact_id", "text_artifact_id", "content_artifact_id"):
        value = getattr(entity, field_name, None)
        if value:
            values.append(value)
    return tuple(dict.fromkeys(values))


def _relationships(entity: KnowledgeEntity) -> tuple[tuple[KnowledgeRelation, str], ...]:
    pairs: list[tuple[KnowledgeRelation, str]] = []

    def add(relation: KnowledgeRelation, values: str | tuple[str, ...] | None) -> None:
        if isinstance(values, str):
            pairs.append((relation, values))
        elif values:
            pairs.extend((relation, value) for value in values)

    if isinstance(entity, SourceSnapshot):
        add(KnowledgeRelation.SNAPSHOT_SOURCE, entity.source_id)
    elif isinstance(entity, Passage):
        add(KnowledgeRelation.PASSAGE_SNAPSHOT, entity.snapshot_id)
    elif isinstance(entity, Evidence):
        add(KnowledgeRelation.EVIDENCE_PASSAGE, entity.passage_ids)
    elif isinstance(entity, AtomicFact):
        add(KnowledgeRelation.FACT_EVIDENCE, entity.evidence_ids)
    elif isinstance(entity, Claim):
        add(KnowledgeRelation.CLAIM_FACT, entity.fact_ids)
        add(KnowledgeRelation.CLAIM_EVIDENCE, entity.evidence_ids)
    elif isinstance(entity, Citation):
        add(KnowledgeRelation.CITATION_CLAIM, entity.claim_id)
        add(KnowledgeRelation.CITATION_EVIDENCE, entity.evidence_id)
        add(KnowledgeRelation.CITATION_PASSAGE, entity.passage_id)
        add(KnowledgeRelation.CITATION_SNAPSHOT, entity.snapshot_id)
        add(KnowledgeRelation.CITATION_SOURCE, entity.source_id)
    elif isinstance(entity, Conflict):
        add(KnowledgeRelation.CONFLICT_CLAIM, entity.claim_ids)
        add(KnowledgeRelation.CONFLICT_FACT, entity.fact_ids)
        add(KnowledgeRelation.CONFLICT_RESOLUTION_EVIDENCE, entity.resolution_evidence_ids)
    elif isinstance(entity, Section):
        add(KnowledgeRelation.SECTION_REPORT, entity.report_id)
        add(KnowledgeRelation.SECTION_PARENT, entity.parent_section_id)
        add(KnowledgeRelation.SECTION_CLAIM, entity.claim_ids)
        add(KnowledgeRelation.SECTION_CITATION, entity.citation_ids)
    elif isinstance(entity, Report):
        add(KnowledgeRelation.REPORT_SECTION, entity.section_ids)
    return tuple(pairs)


def _natural_keys(entity: KnowledgeEntity) -> tuple[tuple[str, str], ...]:
    if isinstance(entity, Source):
        return (("canonical_url", entity.canonical_url),)
    if isinstance(entity, SourceSnapshot):
        return ((f"source_version:{entity.source_id}", str(entity.source_version)),)
    if isinstance(entity, Passage):
        return ((f"snapshot_ordinal:{entity.snapshot_id}", str(entity.ordinal)),)
    if isinstance(entity, Section):
        return ((f"report_order:{entity.report_id}", str(entity.order)),)
    if isinstance(entity, Report):
        return ((f"report_version:{entity.report_id}", str(entity.version)),)
    return ()


class SQLiteKnowledgeStorage:
    """Immutable revisions and typed relationships for the evidence domain."""

    def __init__(self, path: str | Path, *, artifact_store: ArtifactStore | None = None) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_store = artifact_store
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
        if version > CURRENT_KNOWLEDGE_SCHEMA_VERSION:
            raise KnowledgeCorruption("knowledge database schema is newer than this runtime")
        if version < 1:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TABLE entities (
                    entity_id TEXT PRIMARY KEY,
                    entity_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE entity_revisions (
                    entity_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    payload_checksum TEXT NOT NULL,
                    producer_id TEXT NOT NULL,
                    task_id TEXT,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(entity_id, revision),
                    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
                );
                CREATE TABLE relationships (
                    source_entity_id TEXT NOT NULL,
                    source_revision INTEGER NOT NULL,
                    relation TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    ordinal INTEGER NOT NULL,
                    PRIMARY KEY(source_entity_id, source_revision, relation, target_entity_id),
                    FOREIGN KEY(source_entity_id, source_revision) REFERENCES entity_revisions(entity_id, revision) ON DELETE RESTRICT,
                    FOREIGN KEY(target_entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
                );
                CREATE TABLE natural_keys (
                    entity_type TEXT NOT NULL,
                    run_id TEXT NOT NULL,
                    scope_id TEXT NOT NULL,
                    key_value TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    PRIMARY KEY(entity_type, run_id, scope_id, key_value),
                    FOREIGN KEY(entity_id) REFERENCES entities(entity_id) ON DELETE RESTRICT
                );
                CREATE INDEX idx_entities_run_type_created ON entities(run_id, entity_type, created_at, entity_id);
                CREATE INDEX idx_revisions_status ON entity_revisions(status, entity_id, revision);
                CREATE INDEX idx_relationships_target ON relationships(target_entity_id, relation);
                PRAGMA user_version=1;
                COMMIT;
                """
            )
            version = 1
        if version < 2:
            self._connection.executescript(
                """
                BEGIN IMMEDIATE;
                CREATE TRIGGER entities_no_update BEFORE UPDATE ON entities BEGIN SELECT RAISE(ABORT, 'entity identities are immutable'); END;
                CREATE TRIGGER entities_no_delete BEFORE DELETE ON entities BEGIN SELECT RAISE(ABORT, 'entity identities are immutable'); END;
                CREATE TRIGGER revisions_no_update BEFORE UPDATE ON entity_revisions BEGIN SELECT RAISE(ABORT, 'entity revisions are immutable'); END;
                CREATE TRIGGER revisions_no_delete BEFORE DELETE ON entity_revisions BEGIN SELECT RAISE(ABORT, 'entity revisions are immutable'); END;
                CREATE TRIGGER relationships_no_update BEFORE UPDATE ON relationships BEGIN SELECT RAISE(ABORT, 'relationships are immutable'); END;
                CREATE TRIGGER relationships_no_delete BEFORE DELETE ON relationships BEGIN SELECT RAISE(ABORT, 'relationships are immutable'); END;
                CREATE TRIGGER natural_keys_no_update BEFORE UPDATE ON natural_keys BEGIN SELECT RAISE(ABORT, 'natural keys are immutable'); END;
                CREATE TRIGGER natural_keys_no_delete BEFORE DELETE ON natural_keys BEGIN SELECT RAISE(ABORT, 'natural keys are immutable'); END;
                PRAGMA user_version=2;
                COMMIT;
                """
            )

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _decode(entity_type: str, payload: str) -> KnowledgeEntity:
        model = _ENTITY_MODELS.get(entity_type)
        if model is None:
            raise KnowledgeCorruption(f"unknown entity type: {entity_type}")
        return model.model_validate_json(payload)

    def _validate_artifacts(self, entity: KnowledgeEntity) -> None:
        artifact_ids = _artifact_ids(entity)
        if not artifact_ids:
            return
        if self.artifact_store is None:
            raise KnowledgeConflict("artifact-backed entities require an ArtifactStore")
        run_id = entity_run_id(entity)
        for artifact_id in artifact_ids:
            envelope = self.artifact_store.get(artifact_id)
            if envelope is None:
                raise KnowledgeNotFound(f"artifact does not exist: {artifact_id}")
            if envelope.run_id != run_id:
                raise KnowledgeConflict(f"artifact belongs to a different run: {artifact_id}")

    def save_batch(self, entities: tuple[KnowledgeEntity, ...]) -> tuple[SavedRevision, ...]:
        if not entities:
            return ()
        unique: dict[str, KnowledgeEntity] = {}
        for entity in entities:
            if type(entity).__name__ not in _ENTITY_MODELS:
                raise TypeError(f"unsupported knowledge entity: {type(entity).__name__}")
            identifier = entity_id(entity)
            existing = unique.get(identifier)
            if existing is not None and existing != entity:
                raise KnowledgeConflict(f"batch contains conflicting entity identity: {identifier}")
            unique[identifier] = entity
            self._validate_artifacts(entity)

        saved: list[SavedRevision] = []
        with self._transaction() as connection:
            for entity in unique.values():
                identifier = entity_id(entity)
                entity_type = type(entity).__name__
                run_id = entity_run_id(entity)
                identity = connection.execute(
                    "SELECT entity_type, run_id FROM entities WHERE entity_id=?", (identifier,)
                ).fetchone()
                if identity is None:
                    connection.execute(
                        "INSERT INTO entities VALUES (?, ?, ?, ?)",
                        (identifier, entity_type, run_id, _created_at(entity).isoformat()),
                    )
                elif identity["entity_type"] != entity_type or identity["run_id"] != run_id:
                    raise KnowledgeConflict(f"entity identity changed type or run: {identifier}")
                for scope_id, key_value in _natural_keys(entity):
                    natural = connection.execute(
                        "SELECT entity_id FROM natural_keys WHERE entity_type=? AND run_id=? AND scope_id=? AND key_value=?",
                        (entity_type, run_id, scope_id, key_value),
                    ).fetchone()
                    if natural is not None and natural["entity_id"] != identifier:
                        raise KnowledgeConflict(
                            f"natural key already belongs to {natural['entity_id']}: {entity_type}/{scope_id}/{key_value}"
                        )
                    connection.execute(
                        "INSERT OR IGNORE INTO natural_keys VALUES (?, ?, ?, ?, ?)",
                        (entity_type, run_id, scope_id, key_value, identifier),
                    )

            for entity in unique.values():
                identifier = entity_id(entity)
                entity_type = type(entity).__name__
                run_id = entity_run_id(entity)
                payload = canonical_contract_json(entity)
                checksum = self._checksum(payload)
                latest = connection.execute(
                    "SELECT * FROM entity_revisions WHERE entity_id=? ORDER BY revision DESC LIMIT 1",
                    (identifier,),
                ).fetchone()
                if latest is not None and latest["payload_checksum"] == checksum and latest["payload_json"] == payload:
                    saved.append(SavedRevision(entity=self._decode(entity_type, latest["payload_json"]), revision=int(latest["revision"]), inserted=False))
                    continue
                revision = int(latest["revision"]) + 1 if latest is not None else 1
                provenance = getattr(entity, "provenance", None)
                producer_id = provenance.producer_id if provenance is not None else "producer_domain"
                task_id = provenance.task_id if provenance is not None else None
                for relation, target_id in _relationships(entity):
                    target = connection.execute(
                        "SELECT entity_type, run_id FROM entities WHERE entity_id=?", (target_id,)
                    ).fetchone()
                    if target is None:
                        raise KnowledgeNotFound(f"relationship target does not exist: {target_id}")
                    expected_type = _RELATION_TARGET_TYPES[relation]
                    if target["entity_type"] != expected_type:
                        raise KnowledgeConflict(
                            f"{relation.value} requires {expected_type}, got {target['entity_type']}"
                        )
                    if target["run_id"] != run_id:
                        raise KnowledgeConflict("knowledge relationships cannot cross runs")
                created_at = getattr(entity, "updated_at", None) or _created_at(entity)
                connection.execute(
                    "INSERT INTO entity_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        identifier,
                        revision,
                        _status(entity),
                        payload,
                        checksum,
                        producer_id,
                        task_id,
                        created_at.isoformat(),
                    ),
                )
                for ordinal, (relation, target_id) in enumerate(_relationships(entity)):
                    connection.execute(
                        "INSERT INTO relationships VALUES (?, ?, ?, ?, ?)",
                        (identifier, revision, relation.value, target_id, ordinal),
                    )
                saved.append(SavedRevision(entity=entity, revision=revision, inserted=True))
        return tuple(saved)

    def _saved_from_row(self, row: sqlite3.Row) -> SavedRevision:
        payload = row["payload_json"]
        if self._checksum(payload) != row["payload_checksum"]:
            raise KnowledgeCorruption(f"knowledge payload checksum mismatch: {row['entity_id']}")
        return SavedRevision(
            entity=self._decode(row["entity_type"], payload),
            revision=int(row["revision"]),
            inserted=False,
        )

    def get_latest(self, entity_id_value: str) -> SavedRevision | None:
        with self._lock:
            row = self._connection.execute(
                """SELECT e.entity_type, r.* FROM entities e JOIN entity_revisions r ON r.entity_id=e.entity_id
                   WHERE e.entity_id=? ORDER BY r.revision DESC LIMIT 1""",
                (entity_id_value,),
            ).fetchone()
        return self._saved_from_row(row) if row is not None else None

    def get_history(self, entity_id_value: str) -> tuple[SavedRevision, ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT e.entity_type, r.* FROM entities e JOIN entity_revisions r ON r.entity_id=e.entity_id
                   WHERE e.entity_id=? ORDER BY r.revision""",
                (entity_id_value,),
            ).fetchall()
        return tuple(self._saved_from_row(row) for row in rows)

    def list_latest(self, query: KnowledgeQuery) -> KnowledgePage:
        clauses = ["e.run_id=?"]
        params: list[Any] = [query.run_id]
        if query.entity_types:
            placeholders = ",".join("?" for _ in query.entity_types)
            clauses.append(f"e.entity_type IN ({placeholders})")
            params.extend(query.entity_types)
        if query.statuses:
            placeholders = ",".join("?" for _ in query.statuses)
            clauses.append(f"r.status IN ({placeholders})")
            params.extend(query.statuses)
        if query.after_created_at is not None and query.after_entity_id is not None:
            timestamp = query.after_created_at.isoformat()
            clauses.append("(e.created_at>? OR (e.created_at=? AND e.entity_id>?))")
            params.extend((timestamp, timestamp, query.after_entity_id))
        params.append(query.limit + 1)
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT e.entity_type, e.created_at AS entity_created_at, r.*
                    FROM entities e JOIN entity_revisions r ON r.entity_id=e.entity_id
                    WHERE r.revision=(SELECT MAX(r2.revision) FROM entity_revisions r2 WHERE r2.entity_id=e.entity_id)
                      AND {' AND '.join(clauses)}
                    ORDER BY e.created_at, e.entity_id LIMIT ?""",
                params,
            ).fetchall()
        has_more = len(rows) > query.limit
        selected = rows[: query.limit]
        items = tuple(self._saved_from_row(row) for row in selected)
        cursor = (
            datetime.fromisoformat(selected[-1]["entity_created_at"]),
            entity_id(items[-1].entity),
        ) if has_more and selected else None
        return KnowledgePage(items=items, next_cursor=cursor)

    def related(
        self,
        entity_id_value: str,
        relation: KnowledgeRelation,
        *,
        incoming: bool = False,
    ) -> tuple[SavedRevision, ...]:
        if incoming:
            join = "rel.source_entity_id"
            where = "rel.target_entity_id=? AND rel.source_revision=(SELECT MAX(revision) FROM entity_revisions WHERE entity_id=rel.source_entity_id)"
        else:
            join = "rel.target_entity_id"
            where = "rel.source_entity_id=? AND rel.source_revision=(SELECT MAX(revision) FROM entity_revisions WHERE entity_id=rel.source_entity_id)"
        with self._lock:
            rows = self._connection.execute(
                f"""SELECT DISTINCT e.entity_type, r.* FROM relationships rel
                    JOIN entities e ON e.entity_id={join}
                    JOIN entity_revisions r ON r.entity_id=e.entity_id
                    WHERE {where} AND rel.relation=?
                      AND r.revision=(SELECT MAX(r2.revision) FROM entity_revisions r2 WHERE r2.entity_id=e.entity_id)
                    ORDER BY rel.ordinal, e.entity_id""",
                (entity_id_value, relation.value),
            ).fetchall()
        return tuple(self._saved_from_row(row) for row in rows)

    def find_by_natural_key(
        self,
        *,
        entity_type: str,
        run_id: str,
        scope_id: str,
        key_value: str,
    ) -> SavedRevision | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT entity_id FROM natural_keys WHERE entity_type=? AND run_id=? AND scope_id=? AND key_value=?",
                (entity_type, run_id, scope_id, key_value),
            ).fetchone()
        return self.get_latest(row["entity_id"]) if row is not None else None

    def integrity_check(self) -> None:
        with self._lock:
            if self._connection.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise KnowledgeCorruption("knowledge SQLite integrity check failed")
            foreign_keys = self._connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_keys:
                raise KnowledgeCorruption(f"knowledge foreign key failure: {foreign_keys[0]}")
            for identity in self._connection.execute("SELECT * FROM entities").fetchall():
                revisions = self._connection.execute(
                    "SELECT * FROM entity_revisions WHERE entity_id=? ORDER BY revision",
                    (identity["entity_id"],),
                ).fetchall()
                if [int(row["revision"]) for row in revisions] != list(range(1, len(revisions) + 1)):
                    raise KnowledgeCorruption(f"non-contiguous revisions: {identity['entity_id']}")
                for revision in revisions:
                    if self._checksum(revision["payload_json"]) != revision["payload_checksum"]:
                        raise KnowledgeCorruption(f"payload checksum mismatch: {identity['entity_id']}")
                    entity = self._decode(identity["entity_type"], revision["payload_json"])
                    if entity_id(entity) != identity["entity_id"] or entity_run_id(entity) != identity["run_id"]:
                        raise KnowledgeCorruption(f"identity index mismatch: {identity['entity_id']}")
                    self._validate_artifacts(entity)
            for relation_row in self._connection.execute(
                """SELECT rel.*, source.entity_type AS source_type, source.run_id AS source_run,
                          target.entity_type AS target_type, target.run_id AS target_run
                   FROM relationships rel
                   JOIN entities source ON source.entity_id=rel.source_entity_id
                   JOIN entities target ON target.entity_id=rel.target_entity_id"""
            ).fetchall():
                try:
                    relation = KnowledgeRelation(relation_row["relation"])
                except ValueError as exc:
                    raise KnowledgeCorruption(f"unknown relationship: {relation_row['relation']}") from exc
                if relation_row["target_type"] != _RELATION_TARGET_TYPES[relation]:
                    raise KnowledgeCorruption(f"relationship target type mismatch: {relation.value}")
                if relation_row["source_type"] != _RELATION_SOURCE_TYPES[relation]:
                    raise KnowledgeCorruption(f"relationship source type mismatch: {relation.value}")
                if relation_row["source_run"] != relation_row["target_run"]:
                    raise KnowledgeCorruption("cross-run knowledge relationship detected")

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
        verification = SQLiteKnowledgeStorage(destination_path, artifact_store=self.artifact_store)
        try:
            verification.integrity_check()
        finally:
            verification.close()

    @classmethod
    def restore_backup(
        cls,
        backup: str | Path,
        destination: str | Path,
        *,
        artifact_store: ArtifactStore | None = None,
    ) -> "SQLiteKnowledgeStorage":
        backup_path = Path(backup).resolve()
        destination_path = Path(destination).resolve()
        if backup_path == destination_path:
            raise ValueError("backup and destination must differ")
        source = sqlite3.connect(f"file:{backup_path.as_posix()}?mode=ro", uri=True)
        try:
            if source.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise KnowledgeCorruption("backup SQLite integrity check failed")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            target = sqlite3.connect(destination_path)
            try:
                source.backup(target)
            finally:
                target.close()
        finally:
            source.close()
        restored = cls(destination_path, artifact_store=artifact_store)
        restored.integrity_check()
        return restored

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteKnowledgeStorage":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()
