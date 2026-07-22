from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any

from .models import CircuitBreakerPolicy, CircuitState, ToolExecutionResult


@dataclass(frozen=True)
class IdempotencyClaim:
    acquired: bool
    result: ToolExecutionResult | None = None
    busy: bool = False


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    retry_after_seconds: float = 0.0


@dataclass(frozen=True)
class CircuitDecision:
    allowed: bool
    state: CircuitState


class SQLiteToolStateStore:
    """Restart-safe runtime state for idempotency, cache, limits and circuits."""

    CURRENT_SCHEMA_VERSION = 1

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA busy_timeout=5000")
        if self.path != ":memory:":
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=FULL")
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS gateway_schema (
                    singleton INTEGER PRIMARY KEY CHECK(singleton=1),
                    version INTEGER NOT NULL
                );
                INSERT OR IGNORE INTO gateway_schema(singleton, version) VALUES (1, 1);
                CREATE TABLE IF NOT EXISTS tool_idempotency (
                    idempotency_key TEXT PRIMARY KEY,
                    fingerprint TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('in_progress','completed')),
                    result_json TEXT,
                    result_checksum TEXT,
                    lease_updated REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_cache (
                    cache_key TEXT PRIMARY KEY,
                    result_json TEXT NOT NULL,
                    result_checksum TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_cache_expiry ON tool_cache(expires_at);
                CREATE TABLE IF NOT EXISTS tool_rate_events (
                    rate_key TEXT NOT NULL,
                    occurred_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_rate_window ON tool_rate_events(rate_key, occurred_at);
                CREATE TABLE IF NOT EXISTS tool_circuits (
                    tool_identity TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    consecutive_failures INTEGER NOT NULL,
                    opened_at REAL,
                    probe_active INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tool_audit_events (
                    sequence_no INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    command_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    occurred_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tool_audit_command ON tool_audit_events(command_id, sequence_no);
                """
            )
            row = self._connection.execute("SELECT version FROM gateway_schema WHERE singleton=1").fetchone()
            if row is None or int(row["version"]) != self.CURRENT_SCHEMA_VERSION:
                raise RuntimeError("unsupported tool state schema version")

    def claim_idempotency(
        self,
        key: str,
        fingerprint: str,
        *,
        lease_seconds: float = 60.0,
    ) -> IdempotencyClaim:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM tool_idempotency WHERE idempotency_key=?", (key,)
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO tool_idempotency VALUES (?, ?, 'in_progress', NULL, NULL, ?, ?)",
                    (key, fingerprint, now, self._timestamp()),
                )
                return IdempotencyClaim(acquired=True)
            if row["fingerprint"] != fingerprint:
                raise ValueError("idempotency key was reused with different tool input")
            if row["status"] == "completed":
                return IdempotencyClaim(acquired=False, result=self._decode_result(row["result_json"], row["result_checksum"]))
            if now - float(row["lease_updated"]) < lease_seconds:
                return IdempotencyClaim(acquired=False, busy=True)
            connection.execute(
                "UPDATE tool_idempotency SET lease_updated=? WHERE idempotency_key=?",
                (now, key),
            )
            return IdempotencyClaim(acquired=True)

    def complete_idempotency(self, key: str, result: ToolExecutionResult) -> None:
        payload = self._encode_result(result)
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE tool_idempotency SET status='completed', result_json=?, result_checksum=?, lease_updated=? "
                "WHERE idempotency_key=? AND status='in_progress'",
                (payload, self._checksum(payload), time.time(), key),
            )
            if cursor.rowcount != 1:
                raise ValueError("idempotency claim is missing or already completed")

    def get_cache(self, key: str) -> ToolExecutionResult | None:
        now = time.time()
        with self._transaction() as connection:
            connection.execute("DELETE FROM tool_cache WHERE expires_at<=?", (now,))
            row = connection.execute("SELECT * FROM tool_cache WHERE cache_key=?", (key,)).fetchone()
            if row is None:
                return None
            return self._decode_result(row["result_json"], row["result_checksum"])

    def put_cache(self, key: str, result: ToolExecutionResult, *, ttl_seconds: float) -> None:
        payload = self._encode_result(result)
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO tool_cache VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET result_json=excluded.result_json, "
                "result_checksum=excluded.result_checksum, expires_at=excluded.expires_at, created_at=excluded.created_at",
                (key, payload, self._checksum(payload), time.time() + ttl_seconds, self._timestamp()),
            )

    def acquire_rate_limit(self, key: str, *, calls: int, window_seconds: float) -> RateLimitDecision:
        now = time.time()
        cutoff = now - window_seconds
        with self._transaction() as connection:
            connection.execute("DELETE FROM tool_rate_events WHERE rate_key=? AND occurred_at<=?", (key, cutoff))
            rows = connection.execute(
                "SELECT occurred_at FROM tool_rate_events WHERE rate_key=? ORDER BY occurred_at", (key,)
            ).fetchall()
            if len(rows) >= calls:
                retry_after = max(0.0, float(rows[0]["occurred_at"]) + window_seconds - now)
                return RateLimitDecision(False, retry_after)
            connection.execute("INSERT INTO tool_rate_events(rate_key, occurred_at) VALUES (?, ?)", (key, now))
            return RateLimitDecision(True)

    def before_circuit(self, identity: str, policy: CircuitBreakerPolicy) -> CircuitDecision:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM tool_circuits WHERE tool_identity=?", (identity,)).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO tool_circuits VALUES (?, 'closed', 0, NULL, 0, ?)",
                    (identity, self._timestamp()),
                )
                return CircuitDecision(True, CircuitState.CLOSED)
            state = CircuitState(row["state"])
            if state == CircuitState.OPEN:
                opened_at = float(row["opened_at"] or now)
                if now - opened_at < policy.recovery_seconds:
                    return CircuitDecision(False, state)
                connection.execute(
                    "UPDATE tool_circuits SET state='half_open', probe_active=1, updated_at=? WHERE tool_identity=?",
                    (self._timestamp(), identity),
                )
                return CircuitDecision(True, CircuitState.HALF_OPEN)
            if state == CircuitState.HALF_OPEN:
                if bool(row["probe_active"]):
                    return CircuitDecision(False, state)
                connection.execute(
                    "UPDATE tool_circuits SET probe_active=1, updated_at=? WHERE tool_identity=?",
                    (self._timestamp(), identity),
                )
                return CircuitDecision(True, state)
            return CircuitDecision(True, state)

    def record_circuit_success(self, identity: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "INSERT INTO tool_circuits VALUES (?, 'closed', 0, NULL, 0, ?) "
                "ON CONFLICT(tool_identity) DO UPDATE SET state='closed', consecutive_failures=0, "
                "opened_at=NULL, probe_active=0, updated_at=excluded.updated_at",
                (identity, self._timestamp()),
            )

    def record_circuit_failure(self, identity: str, policy: CircuitBreakerPolicy) -> CircuitState:
        now = time.time()
        with self._transaction() as connection:
            row = connection.execute("SELECT * FROM tool_circuits WHERE tool_identity=?", (identity,)).fetchone()
            failures = int(row["consecutive_failures"]) + 1 if row else 1
            prior_state = CircuitState(row["state"]) if row else CircuitState.CLOSED
            should_open = failures >= policy.failure_threshold or prior_state == CircuitState.HALF_OPEN
            state = CircuitState.OPEN if should_open else CircuitState.CLOSED
            connection.execute(
                "INSERT INTO tool_circuits VALUES (?, ?, ?, ?, 0, ?) "
                "ON CONFLICT(tool_identity) DO UPDATE SET state=excluded.state, "
                "consecutive_failures=excluded.consecutive_failures, opened_at=excluded.opened_at, "
                "probe_active=0, updated_at=excluded.updated_at",
                (identity, state.value, failures, now if should_open else None, self._timestamp()),
            )
            return state

    def append_audit(
        self,
        *,
        run_id: str,
        task_id: str,
        command_id: str,
        event_type: str,
        payload: dict[str, Any],
    ) -> int:
        event_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        with self._transaction() as connection:
            cursor = connection.execute(
                "INSERT INTO tool_audit_events(run_id, task_id, command_id, event_type, event_json, checksum, occurred_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, task_id, command_id, event_type, event_json, self._checksum(event_json), self._timestamp()),
            )
            return int(cursor.lastrowid)

    def list_audit(self, command_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM tool_audit_events WHERE command_id=? ORDER BY sequence_no", (command_id,)
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            if self._checksum(row["event_json"]) != row["checksum"]:
                raise RuntimeError(f"tool audit checksum mismatch at sequence {row['sequence_no']}")
            output.append({
                "sequence_no": int(row["sequence_no"]),
                "event_type": row["event_type"],
                "payload": json.loads(row["event_json"]),
                "occurred_at": row["occurred_at"],
            })
        return tuple(output)

    def integrity_check(self) -> None:
        with self._lock:
            result = self._connection.execute("PRAGMA integrity_check").fetchone()
            if result is None or result[0] != "ok":
                raise RuntimeError(f"tool state SQLite integrity failure: {result[0] if result else 'missing'}")
            rows = self._connection.execute(
                "SELECT result_json, result_checksum FROM tool_idempotency WHERE status='completed' "
                "UNION ALL SELECT result_json, result_checksum FROM tool_cache"
            ).fetchall()
            audits = self._connection.execute("SELECT event_json, checksum FROM tool_audit_events").fetchall()
        for row in rows:
            if self._checksum(row["result_json"]) != row["result_checksum"]:
                raise RuntimeError("tool result checksum mismatch")
        for row in audits:
            if self._checksum(row["event_json"]) != row["checksum"]:
                raise RuntimeError("tool audit checksum mismatch")

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> "SQLiteToolStateStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    class _Transaction:
        def __init__(self, store: "SQLiteToolStateStore") -> None:
            self.store = store

        def __enter__(self) -> sqlite3.Connection:
            self.store._lock.acquire()
            self.store._connection.execute("BEGIN IMMEDIATE")
            return self.store._connection

        def __exit__(self, exc_type, exc, traceback) -> None:
            try:
                self.store._connection.execute("ROLLBACK" if exc_type else "COMMIT")
            finally:
                self.store._lock.release()

    def _transaction(self) -> "SQLiteToolStateStore._Transaction":
        return self._Transaction(self)

    @staticmethod
    def _checksum(payload: str) -> str:
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _encode_result(result: ToolExecutionResult) -> str:
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def _decode_result(cls, payload: str | None, checksum: str | None) -> ToolExecutionResult:
        if payload is None or checksum is None or cls._checksum(payload) != checksum:
            raise RuntimeError("tool result checksum mismatch")
        return ToolExecutionResult.model_validate_json(payload)
