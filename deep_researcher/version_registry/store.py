from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator

from .models import (
    VersionLifecycleState,
    VersionManifest,
    VersionRecord,
    VersionTransition,
)


class VersionRegistryStoreError(RuntimeError):
    pass


class VersionRegistryConflict(VersionRegistryStoreError):
    pass


class VersionRegistryCorruption(VersionRegistryStoreError):
    pass


class SQLiteVersionRegistryStore:
    """Append-only version lifecycle journal with rebuildable projections."""

    CURRENT_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            self.path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA busy_timeout=30000")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._migrate()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS version_registry_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO version_registry_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS version_manifests(
                    version_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    semantic_version TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(kind, name, semantic_version)
                );
                CREATE INDEX IF NOT EXISTS idx_version_manifest_catalog
                ON version_manifests(kind, name, semantic_version);

                CREATE TABLE IF NOT EXISTS version_transition_journal(
                    journal_order INTEGER PRIMARY KEY AUTOINCREMENT,
                    transition_id TEXT NOT NULL UNIQUE,
                    version_id TEXT NOT NULL,
                    sequence INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    UNIQUE(version_id, sequence),
                    FOREIGN KEY(version_id)
                      REFERENCES version_manifests(version_id)
                );

                CREATE TABLE IF NOT EXISTS version_state_projection(
                    version_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    checksum TEXT NOT NULL,
                    FOREIGN KEY(version_id)
                      REFERENCES version_manifests(version_id)
                );

                CREATE TABLE IF NOT EXISTS active_version_projection(
                    kind TEXT NOT NULL,
                    name TEXT NOT NULL,
                    version_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(kind, name),
                    FOREIGN KEY(version_id)
                      REFERENCES version_manifests(version_id)
                );
                """
            )
            row = connection.execute(
                "SELECT version FROM version_registry_schema "
                "WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != self.CURRENT_VERSION:
                raise VersionRegistryCorruption(
                    "unsupported Version Registry store schema"
                )

    def save_manifest(self, manifest: VersionManifest) -> None:
        payload = self._json(manifest.model_dump(mode="json"))
        checksum = self._checksum(payload)
        version_id = manifest.version_ref.version_id
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT payload, checksum FROM version_manifests "
                "WHERE version_id=?",
                (version_id,),
            ).fetchone()
            if existing is not None:
                value = self._verified(existing, "version manifest")
                if value != manifest.model_dump(mode="json"):
                    raise VersionRegistryConflict(
                        "version ID was reused for different content"
                    )
                return
            natural = connection.execute(
                """
                SELECT version_id, payload, checksum
                FROM version_manifests
                WHERE kind=? AND name=? AND semantic_version=?
                """,
                (
                    manifest.version_ref.kind.value,
                    manifest.version_ref.name,
                    manifest.version_ref.version,
                ),
            ).fetchone()
            if natural is not None:
                self._verified(natural, "version manifest")
                raise VersionRegistryConflict(
                    "component name/version already belongs to another ID"
                )
            connection.execute(
                """
                INSERT INTO version_manifests(
                    version_id, kind, name, semantic_version,
                    payload, checksum
                ) VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    manifest.version_ref.kind.value,
                    manifest.version_ref.name,
                    manifest.version_ref.version,
                    payload,
                    checksum,
                ),
            )
            state_checksum = self._state_checksum(
                version_id,
                VersionLifecycleState.CANDIDATE,
                0,
            )
            connection.execute(
                """
                INSERT INTO version_state_projection(
                    version_id, state, revision, checksum
                ) VALUES(?, ?, 0, ?)
                """,
                (
                    version_id,
                    VersionLifecycleState.CANDIDATE.value,
                    state_checksum,
                ),
            )

    def manifest(self, version_id: str) -> VersionManifest | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload, checksum FROM version_manifests "
                "WHERE version_id=?",
                (version_id,),
            ).fetchone()
        if row is None:
            return None
        return VersionManifest.model_validate(
            self._verified(row, "version manifest"),
            strict=False,
        )

    def transitions(
        self,
        version_id: str,
    ) -> tuple[VersionTransition, ...]:
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT payload, checksum
                FROM version_transition_journal
                WHERE version_id=? ORDER BY sequence
                """,
                (version_id,),
            ).fetchall()
        return tuple(
            VersionTransition.model_validate(
                self._verified(row, "version transition"),
                strict=False,
            )
            for row in rows
        )

    def record(self, version_id: str) -> VersionRecord | None:
        manifest = self.manifest(version_id)
        if manifest is None:
            return None
        with self._lock:
            row = self._connection.execute(
                """
                SELECT state, revision, checksum
                FROM version_state_projection WHERE version_id=?
                """,
                (version_id,),
            ).fetchone()
        if row is None:
            raise VersionRegistryCorruption(
                "version state projection is missing"
            )
        state = VersionLifecycleState(str(row["state"]))
        revision = int(row["revision"])
        expected = self._state_checksum(version_id, state, revision)
        if str(row["checksum"]) != expected:
            raise VersionRegistryCorruption(
                "version state projection checksum mismatch"
            )
        return VersionRecord(
            manifest=manifest,
            state=state,
            revision=revision,
            transitions=self.transitions(version_id),
        )

    def records(
        self,
        *,
        kind: str | None = None,
        name: str | None = None,
    ) -> tuple[VersionRecord, ...]:
        clauses = []
        parameters: list[Any] = []
        if kind is not None:
            clauses.append("kind=?")
            parameters.append(kind)
        if name is not None:
            clauses.append("name=?")
            parameters.append(name)
        query = "SELECT version_id FROM version_manifests"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY kind, name, semantic_version, version_id"
        with self._lock:
            rows = self._connection.execute(
                query,
                tuple(parameters),
            ).fetchall()
        return tuple(
            record
            for row in rows
            if (record := self.record(str(row["version_id"]))) is not None
        )

    def active(self, kind: str, name: str) -> VersionRecord | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT version_id FROM active_version_projection
                WHERE kind=? AND name=?
                """,
                (kind, name),
            ).fetchone()
        return self.record(str(row["version_id"])) if row is not None else None

    def append_batch(
        self,
        transitions: tuple[VersionTransition, ...],
    ) -> None:
        if not transitions:
            raise ValueError("version transition batch cannot be empty")
        with self.transaction() as connection:
            existing_count = 0
            for transition in transitions:
                row = connection.execute(
                    """
                    SELECT payload, checksum
                    FROM version_transition_journal
                    WHERE transition_id=?
                    """,
                    (transition.transition_id,),
                ).fetchone()
                if row is None:
                    continue
                existing_count += 1
                if self._verified(
                    row,
                    "version transition",
                ) != transition.model_dump(mode="json"):
                    raise VersionRegistryConflict(
                        "transition ID was reused for different content"
                    )
            if existing_count:
                if existing_count != len(transitions):
                    raise VersionRegistryCorruption(
                        "partial transition batch exists"
                    )
                return
            for transition in transitions:
                self._append_transition(connection, transition)

    def _append_transition(
        self,
        connection: sqlite3.Connection,
        transition: VersionTransition,
    ) -> None:
        payload = self._json(transition.model_dump(mode="json"))
        connection.execute(
            """
            INSERT INTO version_transition_journal(
                transition_id, version_id, sequence, payload, checksum
            ) VALUES(?, ?, ?, ?, ?)
            """,
            (
                transition.transition_id,
                transition.version_id,
                transition.sequence,
                payload,
                self._checksum(payload),
            ),
        )
        self._project_transition(connection, transition)

    def _project_transition(
        self,
        connection: sqlite3.Connection,
        transition: VersionTransition,
    ) -> None:
        manifest_row = connection.execute(
            """
            SELECT kind, name FROM version_manifests WHERE version_id=?
            """,
            (transition.version_id,),
        ).fetchone()
        state_row = connection.execute(
            """
            SELECT state, revision, checksum
            FROM version_state_projection WHERE version_id=?
            """,
            (transition.version_id,),
        ).fetchone()
        if manifest_row is None or state_row is None:
            raise VersionRegistryConflict("transition version is not registered")
        current = VersionLifecycleState(str(state_row["state"]))
        revision = int(state_row["revision"])
        if str(state_row["checksum"]) != self._state_checksum(
            transition.version_id,
            current,
            revision,
        ):
            raise VersionRegistryCorruption(
                "version state projection checksum mismatch"
            )
        if (
            transition.from_state != current
            or transition.sequence != revision + 1
        ):
            raise VersionRegistryConflict(
                "version transition expected state/revision is stale"
            )
        connection.execute(
            """
            UPDATE version_state_projection
            SET state=?, revision=?, checksum=? WHERE version_id=?
            """,
            (
                transition.to_state.value,
                transition.sequence,
                self._state_checksum(
                    transition.version_id,
                    transition.to_state,
                    transition.sequence,
                ),
                transition.version_id,
            ),
        )
        kind = str(manifest_row["kind"])
        name = str(manifest_row["name"])
        if (
            transition.from_state == VersionLifecycleState.PROMOTED
            and transition.to_state != VersionLifecycleState.PROMOTED
        ):
            connection.execute(
                """
                DELETE FROM active_version_projection
                WHERE kind=? AND name=? AND version_id=?
                """,
                (kind, name, transition.version_id),
            )
        if transition.to_state == VersionLifecycleState.PROMOTED:
            active = connection.execute(
                """
                SELECT version_id FROM active_version_projection
                WHERE kind=? AND name=?
                """,
                (kind, name),
            ).fetchone()
            if (
                active is not None
                and str(active["version_id"]) != transition.version_id
            ):
                raise VersionRegistryConflict(
                    "another component version is still active"
                )
            connection.execute(
                """
                INSERT INTO active_version_projection(kind, name, version_id)
                VALUES(?, ?, ?)
                ON CONFLICT(kind, name)
                DO UPDATE SET version_id=excluded.version_id
                """,
                (kind, name, transition.version_id),
            )

    def rebuild_projections(self) -> None:
        with self.transaction() as connection:
            manifests = connection.execute(
                "SELECT version_id FROM version_manifests ORDER BY version_id"
            ).fetchall()
            events = connection.execute(
                """
                SELECT payload, checksum
                FROM version_transition_journal ORDER BY journal_order
                """
            ).fetchall()
            connection.execute("DELETE FROM active_version_projection")
            connection.execute("DELETE FROM version_state_projection")
            for row in manifests:
                version_id = str(row["version_id"])
                connection.execute(
                    """
                    INSERT INTO version_state_projection(
                        version_id, state, revision, checksum
                    ) VALUES(?, ?, 0, ?)
                    """,
                    (
                        version_id,
                        VersionLifecycleState.CANDIDATE.value,
                        self._state_checksum(
                            version_id,
                            VersionLifecycleState.CANDIDATE,
                            0,
                        ),
                    ),
                )
            for row in events:
                transition = VersionTransition.model_validate(
                    self._verified(row, "version transition"),
                    strict=False,
                )
                self._project_transition(connection, transition)

    def integrity_check(self) -> None:
        with self._lock:
            row = self._connection.execute("PRAGMA quick_check").fetchone()
            if row is None or str(row[0]).casefold() != "ok":
                raise VersionRegistryCorruption(
                    "Version Registry SQLite integrity failed"
                )
            manifests = self._connection.execute(
                """
                SELECT version_id, kind, name, semantic_version,
                       payload, checksum
                FROM version_manifests
                """
            ).fetchall()
            events = self._connection.execute(
                """
                SELECT transition_id, version_id, sequence, payload, checksum
                FROM version_transition_journal ORDER BY journal_order
                """
            ).fetchall()
            states = self._connection.execute(
                """
                SELECT version_id, state, revision, checksum
                FROM version_state_projection
                """
            ).fetchall()
        for item in manifests:
            manifest = VersionManifest.model_validate(
                self._verified(item, "version manifest"),
                strict=False,
            )
            ref = manifest.version_ref
            if (
                str(item["version_id"]) != ref.version_id
                or str(item["kind"]) != ref.kind.value
                or str(item["name"]) != ref.name
                or str(item["semantic_version"]) != ref.version
            ):
                raise VersionRegistryCorruption(
                    "version manifest index disagrees with payload"
                )
        histories: dict[str, list[VersionTransition]] = {}
        for item in events:
            transition = VersionTransition.model_validate(
                self._verified(item, "version transition"),
                strict=False,
            )
            if (
                str(item["transition_id"]) != transition.transition_id
                or str(item["version_id"]) != transition.version_id
                or int(item["sequence"]) != transition.sequence
            ):
                raise VersionRegistryCorruption(
                    "version transition index disagrees with payload"
                )
            histories.setdefault(transition.version_id, []).append(transition)
        state_map = {str(item["version_id"]): item for item in states}
        if len(state_map) != len(manifests):
            raise VersionRegistryCorruption(
                "version state projection cardinality is inconsistent"
            )
        for item in manifests:
            manifest = VersionManifest.model_validate(
                self._verified(item, "version manifest"),
                strict=False,
            )
            version_id = manifest.version_ref.version_id
            row = state_map.get(version_id)
            if row is None:
                raise VersionRegistryCorruption(
                    "version state projection is missing"
                )
            history = tuple(histories.get(version_id, ()))
            record = VersionRecord(
                manifest=manifest,
                state=VersionLifecycleState(str(row["state"])),
                revision=int(row["revision"]),
                transitions=history,
            )
            if str(row["checksum"]) != self._state_checksum(
                version_id,
                record.state,
                record.revision,
            ):
                raise VersionRegistryCorruption(
                    "version state projection checksum mismatch"
                )
        promoted = [
            record
            for record in self.records()
            if record.state == VersionLifecycleState.PROMOTED
        ]
        active_keys = [
            (
                record.manifest.version_ref.kind.value,
                record.manifest.version_ref.name,
            )
            for record in promoted
        ]
        if len(active_keys) != len(set(active_keys)):
            raise VersionRegistryCorruption(
                "multiple promoted versions exist for one component"
            )
        active_expected = {
            key: record.manifest.version_ref.version_id
            for key, record in zip(active_keys, promoted, strict=True)
        }
        with self._lock:
            active_rows = self._connection.execute(
                "SELECT kind, name, version_id FROM active_version_projection"
            ).fetchall()
        active_actual = {
            (str(row["kind"]), str(row["name"])): str(row["version_id"])
            for row in active_rows
        }
        if active_actual != active_expected:
            raise VersionRegistryCorruption(
                "active version projection is inconsistent"
            )

    def backup_to(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            output = sqlite3.connect(target)
            try:
                self._connection.backup(output)
            finally:
                output.close()
        return target

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @staticmethod
    def _json(value: Any) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _checksum(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @classmethod
    def _state_checksum(
        cls,
        version_id: str,
        state: VersionLifecycleState,
        revision: int,
    ) -> str:
        return cls._checksum(f"{version_id}\0{state.value}\0{revision}")

    @classmethod
    def _verified(cls, row: sqlite3.Row, label: str) -> dict[str, Any]:
        payload = str(row["payload"])
        if cls._checksum(payload) != str(row["checksum"]):
            raise VersionRegistryCorruption(f"{label} checksum mismatch")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise VersionRegistryCorruption(
                f"{label} contains invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise VersionRegistryCorruption(f"{label} is not an object")
        return value
