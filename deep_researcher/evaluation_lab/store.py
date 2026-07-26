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
    DatasetAccessRecord,
    DatasetBundle,
    DeterministicEvaluationReport,
    ExperimentComparison,
    ExperimentDefinition,
    ExperimentRun,
    FrozenReplayEvaluation,
    LiveWebEvaluation,
    RegisteredDataset,
)
from .semantic_models import (
    JudgeCalibrationRecord,
    JudgePanelResult,
    SemanticEvaluationResult,
)
from .gate_models import GateApplicationRecord, ReleaseGateDecision
from deep_researcher.contracts import DatasetDefinition, DatasetSample


class EvaluationStoreError(RuntimeError):
    pass


class EvaluationStoreConflict(EvaluationStoreError):
    pass


class EvaluationStoreCorruption(EvaluationStoreError):
    pass


class DatasetLeakageError(EvaluationStoreError):
    pass


_T = TypeVar("_T", bound=BaseModel)


class SQLiteEvaluationStore:
    """Checksummed immutable Evaluation Lab registry and audit journal."""

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
                CREATE TABLE IF NOT EXISTS evaluation_schema(
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO evaluation_schema(singleton, version)
                VALUES(1, 1);

                CREATE TABLE IF NOT EXISTS evaluation_records(
                    record_type TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    scope_one TEXT,
                    scope_two TEXT,
                    payload TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(record_type, record_id)
                );
                CREATE INDEX IF NOT EXISTS idx_evaluation_records_scope
                ON evaluation_records(
                    record_type, scope_one, scope_two, created_at, record_id
                );

                CREATE TABLE IF NOT EXISTS dataset_versions(
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    bundle_id TEXT NOT NULL UNIQUE,
                    fingerprint TEXT NOT NULL,
                    PRIMARY KEY(name, version)
                );

                CREATE TABLE IF NOT EXISTS dataset_sample_split_index(
                    sample_fingerprint TEXT PRIMARY KEY,
                    split TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS dataset_samples_by_bundle(
                    bundle_id TEXT NOT NULL,
                    dataset_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    sample_fingerprint TEXT NOT NULL,
                    PRIMARY KEY(bundle_id, sample_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dataset_samples_definition
                ON dataset_samples_by_bundle(bundle_id, dataset_id, sample_id);
                """
            )
            row = connection.execute(
                "SELECT version FROM evaluation_schema WHERE singleton=1"
            ).fetchone()
            if row is None or int(row["version"]) != self.CURRENT_VERSION:
                raise EvaluationStoreCorruption(
                    "unsupported Evaluation Lab store schema"
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

    def save_registered_dataset(
        self,
        value: RegisteredDataset,
        *,
        sample_fingerprints: dict[str, str],
    ) -> None:
        if set(sample_fingerprints) != {
            item.sample_id for item in value.samples
        }:
            raise ValueError("sample fingerprints do not match registration")
        with self.transaction() as connection:
            existing_version = connection.execute(
                """
                SELECT bundle_id, fingerprint FROM dataset_versions
                WHERE name=? AND version=?
                """,
                (value.bundle.name, value.bundle.version),
            ).fetchone()
            if existing_version is not None:
                if (
                    str(existing_version["bundle_id"])
                    != value.bundle.bundle_id
                    or str(existing_version["fingerprint"])
                    != value.bundle.fingerprint
                ):
                    raise EvaluationStoreConflict(
                        "dataset name/version is immutable"
                    )
            else:
                connection.execute(
                    """
                    INSERT INTO dataset_versions(
                        name, version, bundle_id, fingerprint
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        value.bundle.name,
                        value.bundle.version,
                        value.bundle.bundle_id,
                        value.bundle.fingerprint,
                    ),
                )
            for sample in value.samples:
                fingerprint = sample_fingerprints[sample.sample_id]
                split_row = connection.execute(
                    """
                    SELECT split FROM dataset_sample_split_index
                    WHERE sample_fingerprint=?
                    """,
                    (fingerprint,),
                ).fetchone()
                if (
                    split_row is not None
                    and str(split_row["split"]) != sample.split.value
                ):
                    raise DatasetLeakageError(
                        "the same sample content cannot appear in different "
                        f"dataset splits ({split_row['split']} and "
                        f"{sample.split.value})"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO dataset_sample_split_index(
                        sample_fingerprint, split
                    ) VALUES(?, ?)
                    """,
                    (fingerprint, sample.split.value),
                )
                row = connection.execute(
                    """
                    SELECT sample_fingerprint FROM dataset_samples_by_bundle
                    WHERE bundle_id=? AND sample_id=?
                    """,
                    (value.bundle.bundle_id, sample.sample_id),
                ).fetchone()
                if row is not None and str(row["sample_fingerprint"]) != fingerprint:
                    raise EvaluationStoreConflict(
                        "dataset sample ID was reused for different content"
                    )
                connection.execute(
                    """
                    INSERT OR IGNORE INTO dataset_samples_by_bundle(
                        bundle_id, dataset_id, sample_id, sample_fingerprint
                    ) VALUES(?, ?, ?, ?)
                    """,
                    (
                        value.bundle.bundle_id,
                        sample.dataset_id,
                        sample.sample_id,
                        fingerprint,
                    ),
                )
            records: list[tuple[str, str, str | None, str | None, BaseModel]] = [
                (
                    "dataset_bundle",
                    value.bundle.bundle_id,
                    value.bundle.name,
                    value.bundle.version,
                    value.bundle,
                ),
                *[
                    (
                        "dataset_definition",
                        item.dataset_id,
                        value.bundle.bundle_id,
                        item.split.value,
                        item,
                    )
                    for item in value.definitions
                ],
                *[
                    (
                        "dataset_sample",
                        item.sample_id,
                        value.bundle.bundle_id,
                        item.dataset_id,
                        item,
                    )
                    for item in value.samples
                ],
            ]
            for record in records:
                self._save_record(*record, connection=connection)

    def dataset_bundle(self, bundle_id: str) -> DatasetBundle | None:
        return self._get("dataset_bundle", bundle_id, DatasetBundle)

    def find_dataset_bundle(
        self,
        name: str,
        version: str,
    ) -> DatasetBundle | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT bundle_id FROM dataset_versions
                WHERE name=? AND version=?
                """,
                (name, version),
            ).fetchone()
        return (
            self.dataset_bundle(str(row["bundle_id"]))
            if row is not None
            else None
        )

    def dataset_definitions(
        self,
        bundle_id: str,
    ) -> tuple[DatasetDefinition, ...]:
        return self._list(
            "dataset_definition",
            DatasetDefinition,
            scope_one=bundle_id,
        )

    def dataset_samples(
        self,
        *,
        bundle_id: str,
        dataset_id: str | None = None,
    ) -> tuple[DatasetSample, ...]:
        return self._list(
            "dataset_sample",
            DatasetSample,
            scope_one=bundle_id,
            scope_two=dataset_id,
        )

    def save_dataset_access(self, value: DatasetAccessRecord) -> None:
        self._save(
            "dataset_access",
            value.access_record_id,
            value.bundle_id,
            value.request.dataset_id,
            value,
        )

    def dataset_access(
        self,
        access_record_id: str,
    ) -> DatasetAccessRecord | None:
        return self._get(
            "dataset_access",
            access_record_id,
            DatasetAccessRecord,
        )

    def dataset_accesses(
        self,
        *,
        bundle_id: str | None = None,
    ) -> tuple[DatasetAccessRecord, ...]:
        return self._list(
            "dataset_access",
            DatasetAccessRecord,
            scope_one=bundle_id,
        )

    def save_evaluation_report(
        self,
        value: DeterministicEvaluationReport,
        *,
        run_id: str,
    ) -> None:
        self._save(
            "evaluation_report",
            value.report_id,
            run_id,
            value.snapshot_id,
            value,
        )

    def evaluation_report(
        self,
        report_id: str,
    ) -> DeterministicEvaluationReport | None:
        return self._get(
            "evaluation_report",
            report_id,
            DeterministicEvaluationReport,
        )

    def save_frozen_result(self, value: FrozenReplayEvaluation) -> None:
        self._save(
            "frozen_result",
            value.frozen_result_id,
            value.replay.replay_id,
            value.subject_version_id,
            value,
        )

    def frozen_results(
        self,
        replay_id: str,
    ) -> tuple[FrozenReplayEvaluation, ...]:
        return self._list(
            "frozen_result",
            FrozenReplayEvaluation,
            scope_one=replay_id,
        )

    def save_live_result(self, value: LiveWebEvaluation) -> None:
        self._save(
            "live_result",
            value.live_result_id,
            value.dataset_id,
            value.subject_version_id,
            value,
        )

    def live_results(
        self,
        dataset_id: str,
    ) -> tuple[LiveWebEvaluation, ...]:
        return self._list(
            "live_result",
            LiveWebEvaluation,
            scope_one=dataset_id,
        )

    def save_experiment_definition(
        self,
        value: ExperimentDefinition,
    ) -> None:
        self._save(
            "experiment_definition",
            value.experiment_id,
            value.bundle_id,
            value.baseline.value,
            value,
        )

    def experiment_definition(
        self,
        experiment_id: str,
    ) -> ExperimentDefinition | None:
        return self._get(
            "experiment_definition",
            experiment_id,
            ExperimentDefinition,
        )

    def save_experiment_run(self, value: ExperimentRun) -> None:
        self._save(
            "experiment_run",
            value.experiment_run_id,
            value.experiment_id,
            value.baseline.value,
            value,
        )

    def experiment_run(self, run_id: str) -> ExperimentRun | None:
        return self._get("experiment_run", run_id, ExperimentRun)

    def experiment_runs(
        self,
        experiment_id: str | None = None,
    ) -> tuple[ExperimentRun, ...]:
        return self._list(
            "experiment_run",
            ExperimentRun,
            scope_one=experiment_id,
        )

    def save_comparison(self, value: ExperimentComparison) -> None:
        self._save(
            "experiment_comparison",
            value.comparison_id,
            value.dataset_id,
            value.dataset_fingerprint,
            value,
        )

    def comparison(
        self,
        comparison_id: str,
    ) -> ExperimentComparison | None:
        return self._get(
            "experiment_comparison",
            comparison_id,
            ExperimentComparison,
        )

    def save_judge_panel(self, value: JudgePanelResult) -> None:
        self._save(
            "judge_panel",
            value.panel_result_id,
            value.panel_id,
            value.rubric_version_id,
            value,
        )

    def judge_panel(self, panel_result_id: str) -> JudgePanelResult | None:
        return self._get(
            "judge_panel",
            panel_result_id,
            JudgePanelResult,
        )

    def save_judge_calibration(
        self,
        value: JudgeCalibrationRecord,
    ) -> None:
        self._save(
            "judge_calibration",
            value.calibration_id,
            value.rubric_version_id,
            "accepted" if value.accepted else "rejected",
            value,
        )

    def judge_calibration(
        self,
        calibration_id: str,
    ) -> JudgeCalibrationRecord | None:
        return self._get(
            "judge_calibration",
            calibration_id,
            JudgeCalibrationRecord,
        )

    def save_semantic_result(
        self,
        value: SemanticEvaluationResult,
    ) -> None:
        self._save(
            "semantic_evaluation",
            value.semantic_evaluation_id,
            value.dataset_id,
            value.subject_version_id,
            value,
        )

    def semantic_result(
        self,
        semantic_evaluation_id: str,
    ) -> SemanticEvaluationResult | None:
        return self._get(
            "semantic_evaluation",
            semantic_evaluation_id,
            SemanticEvaluationResult,
        )

    def semantic_results(
        self,
        *,
        dataset_id: str | None = None,
        subject_version_id: str | None = None,
    ) -> tuple[SemanticEvaluationResult, ...]:
        return self._list(
            "semantic_evaluation",
            SemanticEvaluationResult,
            scope_one=dataset_id,
            scope_two=subject_version_id,
        )

    def save_gate_decision(self, value: ReleaseGateDecision) -> None:
        self._save(
            "release_gate_decision",
            value.gate_decision_id,
            value.candidate_version_id,
            value.stage.value,
            value,
        )

    def gate_decision(
        self,
        gate_decision_id: str,
    ) -> ReleaseGateDecision | None:
        return self._get(
            "release_gate_decision",
            gate_decision_id,
            ReleaseGateDecision,
        )

    def gate_decisions(
        self,
        candidate_version_id: str | None = None,
    ) -> tuple[ReleaseGateDecision, ...]:
        return self._list(
            "release_gate_decision",
            ReleaseGateDecision,
            scope_one=candidate_version_id,
        )

    def save_gate_application(
        self,
        value: GateApplicationRecord,
    ) -> None:
        self._save(
            "release_gate_application",
            value.application_id,
            value.gate_decision_id,
            value.candidate_version_id,
            value,
        )

    def gate_application(
        self,
        gate_decision_id: str,
    ) -> GateApplicationRecord | None:
        values = self._list(
            "release_gate_application",
            GateApplicationRecord,
            scope_one=gate_decision_id,
        )
        if len(values) > 1:
            raise EvaluationStoreCorruption(
                "gate decision has multiple application records"
            )
        return values[0] if values else None

    def integrity_check(self) -> None:
        with self._lock:
            row = self._connection.execute("PRAGMA quick_check").fetchone()
            if row is None or str(row[0]).casefold() != "ok":
                raise EvaluationStoreCorruption(
                    f"Evaluation Lab SQLite integrity failed: "
                    f"{row[0] if row else 'missing'}"
                )
            for row in self._connection.execute(
                "SELECT payload, checksum FROM evaluation_records"
            ).fetchall():
                self._verified(row, "evaluation record")
            leakage = self._connection.execute(
                """
                SELECT sample_fingerprint,
                       COUNT(DISTINCT split) AS split_count
                FROM dataset_sample_split_index
                GROUP BY sample_fingerprint HAVING split_count > 1
                """
            ).fetchall()
        if leakage:
            raise EvaluationStoreCorruption(
                "dataset split leakage index is inconsistent"
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

    def __enter__(self) -> "SQLiteEvaluationStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _save(
        self,
        record_type: str,
        identity: str,
        scope_one: str | None,
        scope_two: str | None,
        value: BaseModel,
    ) -> None:
        with self.transaction() as connection:
            self._save_record(
                record_type,
                identity,
                scope_one,
                scope_two,
                value,
                connection=connection,
            )

    def _save_record(
        self,
        record_type: str,
        identity: str,
        scope_one: str | None,
        scope_two: str | None,
        value: BaseModel,
        *,
        connection: sqlite3.Connection,
    ) -> None:
        encoded = self._json(value.model_dump(mode="json"))
        checksum = self._checksum(encoded)
        row = connection.execute(
            """
            SELECT payload, checksum FROM evaluation_records
            WHERE record_type=? AND record_id=?
            """,
            (record_type, identity),
        ).fetchone()
        if row is not None:
            existing = self._verified(row, record_type)
            if existing != value.model_dump(mode="json"):
                raise EvaluationStoreConflict(
                    f"{record_type} identity was reused for different content"
                )
            return
        connection.execute(
            """
            INSERT INTO evaluation_records(
                record_type, record_id, scope_one, scope_two,
                payload, checksum, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record_type,
                identity,
                scope_one,
                scope_two,
                encoded,
                checksum,
                str(value.model_dump(mode="json").get("created_at", "")),
            ),
        )

    def _get(
        self,
        record_type: str,
        identity: str,
        model: type[_T],
    ) -> _T | None:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT payload, checksum FROM evaluation_records
                WHERE record_type=? AND record_id=?
                """,
                (record_type, identity),
            ).fetchone()
        return (
            model.model_validate(
                self._verified(row, record_type),
                strict=False,
            )
            if row is not None
            else None
        )

    def _list(
        self,
        record_type: str,
        model: type[_T],
        *,
        scope_one: str | None = None,
        scope_two: str | None = None,
    ) -> tuple[_T, ...]:
        clauses = ["record_type=?"]
        parameters: list[Any] = [record_type]
        if scope_one is not None:
            clauses.append("scope_one=?")
            parameters.append(scope_one)
        if scope_two is not None:
            clauses.append("scope_two=?")
            parameters.append(scope_two)
        with self._lock:
            rows = self._connection.execute(
                "SELECT payload, checksum FROM evaluation_records WHERE "
                + " AND ".join(clauses)
                + " ORDER BY created_at, record_id",
                tuple(parameters),
            ).fetchall()
        return tuple(
            model.model_validate(
                self._verified(row, record_type),
                strict=False,
            )
            for row in rows
        )

    @classmethod
    def _verified(cls, row: sqlite3.Row, label: str) -> dict[str, Any]:
        payload = str(row["payload"])
        if cls._checksum(payload) != str(row["checksum"]):
            raise EvaluationStoreCorruption(f"{label} checksum mismatch")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise EvaluationStoreCorruption(
                f"{label} payload is invalid JSON"
            ) from exc
        if not isinstance(value, dict):
            raise EvaluationStoreCorruption(
                f"{label} payload is not an object"
            )
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
