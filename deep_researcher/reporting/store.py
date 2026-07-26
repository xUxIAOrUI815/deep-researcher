from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Iterator, TypeVar

from pydantic import BaseModel

from .models import (
    CitationMap,
    ReportLoopOutcome,
    ReportRevision,
    ReviewerDecision,
)


class ReportingStoreError(RuntimeError):
    pass


class ReportingStoreConflict(ReportingStoreError):
    pass


class ReportingStoreCorruption(ReportingStoreError):
    pass


_T = TypeVar("_T", bound=BaseModel)


class SQLiteReportingStore:
    """Checksummed immutable report revision/review journal."""

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

    def _migrate(self) -> None:
        with self.transaction() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reporting_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO reporting_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS report_revisions(
                    revision_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, report_id, revision)
                );
                CREATE INDEX IF NOT EXISTS idx_report_revisions
                ON report_revisions(run_id, report_id, revision);

                CREATE TABLE IF NOT EXISTS citation_maps(
                    citation_map_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, report_id, revision)
                );

                CREATE TABLE IF NOT EXISTS report_reviews(
                    review_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    revision_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, report_id, revision_id)
                );
                CREATE INDEX IF NOT EXISTS idx_report_reviews
                ON report_reviews(run_id, report_id, created_at, review_id);

                CREATE TABLE IF NOT EXISTS report_outcomes(
                    run_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    completed_at TEXT NOT NULL,
                    PRIMARY KEY(run_id, report_id)
                );
                """
            )
            row = connection.execute(
                "SELECT version FROM reporting_schema WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != self.CURRENT_VERSION:
                raise ReportingStoreCorruption(
                    "unsupported reporting store schema"
                )

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

    def save_revision(self, value: ReportRevision) -> None:
        self._save(
            table="report_revisions",
            id_column="revision_id",
            identity=value.revision_id,
            value=value,
            extra_columns=("run_id", "report_id", "revision", "created_at"),
            extra_values=(
                value.run_id,
                value.report_id,
                value.revision,
                value.created_at.isoformat(),
            ),
        )

    def revisions(
        self,
        run_id: str,
        report_id: str,
    ) -> tuple[ReportRevision, ...]:
        return self._list(
            """
            SELECT payload, checksum FROM report_revisions
            WHERE run_id=? AND report_id=? ORDER BY revision
            """,
            (run_id, report_id),
            ReportRevision,
            "report revision",
        )

    def revision(self, revision_id: str) -> ReportRevision | None:
        row = self._connection.execute(
            """
            SELECT payload, checksum FROM report_revisions
            WHERE revision_id=?
            """,
            (revision_id,),
        ).fetchone()
        return (
            ReportRevision.model_validate(
                self._verified(row, "report revision"),
                strict=False,
            )
            if row is not None
            else None
        )

    def save_citation_map(self, value: CitationMap) -> None:
        self._save(
            table="citation_maps",
            id_column="citation_map_id",
            identity=value.citation_map_id,
            value=value,
            extra_columns=("run_id", "report_id", "revision", "created_at"),
            extra_values=(
                value.run_id,
                value.report_id,
                value.revision,
                value.created_at.isoformat(),
            ),
        )

    def citation_map(
        self,
        run_id: str,
        report_id: str,
        revision: int,
    ) -> CitationMap | None:
        row = self._connection.execute(
            """
            SELECT payload, checksum FROM citation_maps
            WHERE run_id=? AND report_id=? AND revision=?
            """,
            (run_id, report_id, revision),
        ).fetchone()
        return (
            CitationMap.model_validate(
                self._verified(row, "citation map"),
                strict=False,
            )
            if row is not None
            else None
        )

    def save_review(self, value: ReviewerDecision) -> None:
        self._save(
            table="report_reviews",
            id_column="review_id",
            identity=value.review_id,
            value=value,
            extra_columns=(
                "run_id",
                "report_id",
                "revision_id",
                "created_at",
            ),
            extra_values=(
                value.run_id,
                value.report_id,
                value.revision_id,
                value.created_at.isoformat(),
            ),
        )

    def reviews(
        self,
        run_id: str,
        report_id: str,
    ) -> tuple[ReviewerDecision, ...]:
        return self._list(
            """
            SELECT payload, checksum FROM report_reviews
            WHERE run_id=? AND report_id=?
            ORDER BY created_at, review_id
            """,
            (run_id, report_id),
            ReviewerDecision,
            "report review",
        )

    def save_outcome(self, value: ReportLoopOutcome) -> None:
        encoded = self._json(value.model_dump(mode="json"))
        checksum = self._checksum(encoded)
        with self.transaction() as connection:
            row = connection.execute(
                """
                SELECT payload, checksum FROM report_outcomes
                WHERE run_id=? AND report_id=?
                """,
                (value.run_id, value.report_id),
            ).fetchone()
            if row is not None:
                existing = self._verified(row, "report outcome")
                if existing != value.model_dump(mode="json"):
                    raise ReportingStoreConflict(
                        "terminal report outcome is immutable"
                    )
                return
            connection.execute(
                """
                INSERT INTO report_outcomes(
                    run_id, report_id, payload, checksum, completed_at
                ) VALUES(?, ?, ?, ?, ?)
                """,
                (
                    value.run_id,
                    value.report_id,
                    encoded,
                    checksum,
                    value.completed_at.isoformat(),
                ),
            )

    def outcome(
        self,
        run_id: str,
        report_id: str,
    ) -> ReportLoopOutcome | None:
        row = self._connection.execute(
            """
            SELECT payload, checksum FROM report_outcomes
            WHERE run_id=? AND report_id=?
            """,
            (run_id, report_id),
        ).fetchone()
        return (
            ReportLoopOutcome.model_validate(
                self._verified(row, "report outcome"),
                strict=False,
            )
            if row is not None
            else None
        )

    def integrity_check(self) -> None:
        row = self._connection.execute("PRAGMA quick_check").fetchone()
        if row is None or str(row[0]).casefold() != "ok":
            raise ReportingStoreCorruption(
                f"reporting SQLite integrity failed: {row[0] if row else 'missing'}"
            )
        for table, label in (
            ("report_revisions", "report revision"),
            ("citation_maps", "citation map"),
            ("report_reviews", "report review"),
            ("report_outcomes", "report outcome"),
        ):
            for item in self._connection.execute(
                f"SELECT payload, checksum FROM {table}"
            ).fetchall():
                self._verified(item, label)

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

    def __enter__(self) -> "SQLiteReportingStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _save(
        self,
        *,
        table: str,
        id_column: str,
        identity: str,
        value: BaseModel,
        extra_columns: tuple[str, ...],
        extra_values: tuple[Any, ...],
    ) -> None:
        encoded = self._json(value.model_dump(mode="json"))
        checksum = self._checksum(encoded)
        with self.transaction() as connection:
            row = connection.execute(
                f"SELECT payload, checksum FROM {table} WHERE {id_column}=?",
                (identity,),
            ).fetchone()
            if row is not None:
                existing = self._verified(row, table)
                if existing != value.model_dump(mode="json"):
                    raise ReportingStoreConflict(
                        f"{id_column} was reused for different content"
                    )
                return
            columns = (id_column, *extra_columns, "payload", "checksum")
            placeholders = ",".join("?" for _ in columns)
            try:
                connection.execute(
                    f"INSERT INTO {table}({','.join(columns)}) "
                    f"VALUES({placeholders})",
                    (identity, *extra_values, encoded, checksum),
                )
            except sqlite3.IntegrityError as exc:
                raise ReportingStoreConflict(
                    f"{table} revision identity already exists"
                ) from exc

    def _list(
        self,
        query: str,
        parameters: tuple[Any, ...],
        model: type[_T],
        label: str,
    ) -> tuple[_T, ...]:
        rows = self._connection.execute(query, parameters).fetchall()
        return tuple(
            model.model_validate(self._verified(row, label), strict=False)
            for row in rows
        )

    @classmethod
    def _verified(cls, row: sqlite3.Row, label: str) -> dict[str, Any]:
        payload = str(row["payload"])
        if cls._checksum(payload) != str(row["checksum"]):
            raise ReportingStoreCorruption(f"{label} checksum mismatch")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ReportingStoreCorruption(
                f"{label} payload is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise ReportingStoreCorruption(f"{label} payload is not an object")
        return value

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
