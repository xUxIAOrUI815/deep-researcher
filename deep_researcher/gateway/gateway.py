from __future__ import annotations

import asyncio
from contextlib import suppress
from datetime import datetime
import hashlib
import ipaddress
import json
import re
import time
from typing import Any, Awaitable, Callable, TypeVar
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator, ValidationError

from deep_researcher.contracts import (
    BudgetUsage,
    Command,
    ErrorCategory,
    ErrorRecord,
    Observation,
    ObservationStatus,
    utc_now,
)

from .models import (
    GatewayCancelled,
    GatewayTimedOut,
    SafetyDecision,
    ToolAdapterError,
    ToolAdapterResult,
    ToolDefinition,
    ToolErrorKind,
    ToolExecutionResult,
    ToolInvocationContext,
)
from .registry import ToolRegistry
from .state import SQLiteToolStateStore


_T = TypeVar("_T")
_SENSITIVE_KEY = re.compile(r"(authorization|api[_-]?key|access[_-]?token|password|secret|cookie|^token$)", re.I)
_FORBIDDEN_KEY = {"chain_of_thought", "cot", "hidden_reasoning", "private_reasoning", "raw_model_response"}


def redact_gateway_value(value: Any) -> Any:
    if isinstance(value, dict):
        output: dict[str, Any] = {}
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_KEY:
                continue
            output[str(key)] = "[REDACTED]" if _SENSITIVE_KEY.search(str(key)) else redact_gateway_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [redact_gateway_value(item) for item in value]
    if isinstance(value, str):
        value = re.sub(r"(?i)bearer\s+[A-Za-z0-9._~+/=-]+", "Bearer [REDACTED]", value)
        return re.sub(r"(?i)(api[_-]?key|token|password|secret)=([^\s&]+)", r"\1=[REDACTED]", value)
    return value


class PatternSafetyScanner:
    """Deterministic structural and SSRF-oriented tool input/output scanner."""

    def __init__(self, *, max_payload_bytes: int = 1_000_000, max_depth: int = 20) -> None:
        self.max_payload_bytes = max_payload_bytes
        self.max_depth = max_depth

    def scan(self, value: Any, definition: ToolDefinition, *, phase: str) -> SafetyDecision:
        try:
            encoded = json.dumps(value, ensure_ascii=False, default=str).encode("utf-8")
        except Exception as exc:
            return SafetyDecision(allowed=False, reason=f"payload is not serializable: {exc}", checks=("serialization",))
        if len(encoded) > self.max_payload_bytes:
            return SafetyDecision(allowed=False, reason="payload exceeds the configured safety size limit", checks=("payload_size",))
        allow_private = bool(definition.metadata.get("allow_private_network", False))
        violation = self._walk(value, depth=0, allow_private=allow_private)
        if violation:
            return SafetyDecision(allowed=False, reason=f"{phase} safety scan rejected payload: {violation}", checks=("structure", "ssrf", "control_characters"))
        return SafetyDecision(allowed=True, reason=f"{phase} safety scan passed", checks=("structure", "ssrf", "control_characters"))

    def _walk(self, value: Any, *, depth: int, allow_private: bool) -> str | None:
        if depth > self.max_depth:
            return "payload nesting is too deep"
        if isinstance(value, dict):
            for key, item in value.items():
                if "\x00" in str(key):
                    return "object key contains a NUL byte"
                violation = self._walk(item, depth=depth + 1, allow_private=allow_private)
                if violation:
                    return violation
            return None
        if isinstance(value, (list, tuple)):
            for item in value:
                violation = self._walk(item, depth=depth + 1, allow_private=allow_private)
                if violation:
                    return violation
            return None
        if not isinstance(value, str):
            return None
        if any(ord(character) < 32 and character not in "\r\n\t" for character in value):
            return "string contains disallowed control characters"
        parsed = urlsplit(value)
        if not parsed.scheme:
            return None
        if parsed.scheme not in {"http", "https"}:
            return f"URL scheme is not allowed: {parsed.scheme}"
        if allow_private or not parsed.hostname:
            return None
        hostname = parsed.hostname.casefold()
        if hostname in {"localhost", "localhost.localdomain"} or hostname.endswith(".localhost"):
            return "loopback host is not allowed"
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            return None
        if not address.is_global:
            return "private, loopback, link-local, or reserved address is not allowed"
        return None


class ProtocolToolGateway:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        state_store: SQLiteToolStateStore,
        safety_scanner: PatternSafetyScanner | None = None,
        context_factory: Callable[[Command], ToolInvocationContext] | None = None,
        retry_base_seconds: float = 0.1,
    ) -> None:
        self.registry = registry
        self.state_store = state_store
        self.safety_scanner = safety_scanner or PatternSafetyScanner()
        self.context_factory = context_factory or (lambda command: ToolInvocationContext())
        self.retry_base_seconds = max(0.0, retry_base_seconds)

    async def execute(self, command: Command, context: ToolInvocationContext | None = None) -> Observation:
        started_at = utc_now()
        result = await self.execute_tool(command, context or self.context_factory(command))
        if result.success:
            status = ObservationStatus.SUCCEEDED
        elif result.error and result.error.category == ErrorCategory.CANCELLED:
            status = ObservationStatus.CANCELLED
        elif result.error and result.error.code in {ToolErrorKind.TIMEOUT.value, "tool_timeout"}:
            status = ObservationStatus.TIMEOUT
        elif result.error and result.error.category in {ErrorCategory.POLICY_DENIED, ErrorCategory.APPROVAL_REQUIRED}:
            status = ObservationStatus.REJECTED
        else:
            status = ObservationStatus.FAILED
        normalized_data = result.data if isinstance(result.data, dict) else {"value": result.data}
        return Observation(
            command_id=command.command_id,
            run_id=command.run_id,
            task_id=command.task_id,
            actor_id=command.actor_id,
            status=status,
            output_artifact_ids=result.output_artifact_ids,
            normalized_data={
                **redact_gateway_value(normalized_data),
                "_tool": {
                    "name": result.tool_name,
                    "version": result.tool_version,
                    "protocol": result.protocol.value,
                    "cached": result.cached,
                    "idempotent_replay": result.idempotent_replay,
                    "fallback_chain": list(result.fallback_chain),
                },
            },
            usage=result.usage,
            error=result.error,
            attempt=max(1, result.attempts),
            started_at=started_at,
            completed_at=utc_now(),
        )

    async def execute_tool(self, command: Command, context: ToolInvocationContext) -> ToolExecutionResult:
        requested_version = str(command.metadata.get("tool_version") or "") or None
        try:
            definition, _ = self.registry.resolve(command.name, requested_version)
        except KeyError as exc:
            return self._failure(command, command.name, requested_version or "unknown", ToolErrorKind.PERMANENT, str(exc), fatal=True)

        preflight = self._authorize(command, definition, context)
        if preflight is not None:
            self._audit(command, "tool.rejected", {"reason": preflight.error.message if preflight.error else "rejected"})
            return preflight

        fingerprint = self._fingerprint(definition, command.arguments)
        try:
            claim = self.state_store.claim_idempotency(command.idempotency_key, fingerprint)
        except ValueError as exc:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.IDEMPOTENCY, str(exc), fatal=True)
        if claim.result is not None:
            replay = claim.result.model_copy(update={"idempotent_replay": True})
            self._audit(command, "tool.idempotent_replay", {"tool": definition.identity})
            return replay
        if claim.busy:
            return self._failure(
                command,
                definition.name,
                definition.version,
                ToolErrorKind.IDEMPOTENCY,
                "an identical invocation is still in progress",
                retryable=True,
            )

        cache_key = self._cache_key(definition, command.arguments)
        if definition.idempotent and definition.cache_ttl_seconds is not None:
            cached = self.state_store.get_cache(cache_key)
            if cached is not None:
                cached = cached.model_copy(update={"cached": True})
                self.state_store.complete_idempotency(command.idempotency_key, cached)
                self._audit(command, "tool.cache_hit", {"tool": definition.identity})
                return cached

        self._audit(
            command,
            "tool.started",
            {
                "tool": definition.identity,
                "protocol": definition.protocol.value,
                "principal_id": context.principal_id,
                "correlation_id": context.correlation_id,
                "trace_id": context.trace_id,
                "arguments": redact_gateway_value(command.arguments),
            },
        )
        candidates = (definition.name, *definition.fallback_tools)
        total_usage = BudgetUsage()
        total_attempts = 0
        fallback_chain: list[str] = []
        final: ToolExecutionResult | None = None
        for index, candidate_name in enumerate(candidates):
            candidate_version = definition.version if index == 0 else None
            try:
                candidate, adapter = self.registry.resolve(candidate_name, candidate_version)
            except KeyError as exc:
                final = self._failure(command, candidate_name, "unknown", ToolErrorKind.PERMANENT, str(exc), fatal=True)
                continue
            candidate_preflight = self._authorize(command, candidate, context, allow_name_mismatch=index > 0)
            if candidate_preflight is not None:
                final = candidate_preflight
                if index == 0:
                    break
                continue
            if index > 0:
                fallback_chain.append(candidate.identity)
                self._audit(command, "tool.fallback_started", {"tool": candidate.identity, "index": index})
            candidate_result = await self._execute_candidate(command, candidate, adapter, context)
            total_attempts += candidate_result.attempts
            total_usage = self._combine_usage(total_usage, candidate_result.usage)
            final = candidate_result.model_copy(
                update={
                    "usage": total_usage,
                    "attempts": total_attempts,
                    "fallback_chain": tuple(fallback_chain),
                }
            )
            if final.success:
                break

        assert final is not None
        if final.success and definition.idempotent and definition.cache_ttl_seconds is not None:
            self.state_store.put_cache(cache_key, final, ttl_seconds=definition.cache_ttl_seconds)
        self.state_store.complete_idempotency(command.idempotency_key, final)
        self._audit(
            command,
            "tool.completed" if final.success else "tool.failed",
            {
                "tool": f"{final.tool_name}@{final.tool_version}",
                "success": final.success,
                "attempts": final.attempts,
                "fallback_chain": list(final.fallback_chain),
                "usage": final.usage.model_dump(mode="json"),
                "error": final.error.model_dump(mode="json") if final.error else None,
            },
        )
        return final

    def _authorize(
        self,
        command: Command,
        definition: ToolDefinition,
        context: ToolInvocationContext,
        *,
        allow_name_mismatch: bool = False,
    ) -> ToolExecutionResult | None:
        if not allow_name_mismatch and command.name != definition.name:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.PERMISSION, "command tool identity mismatch", fatal=True)
        operation = str(command.arguments.get("operation") or command.name)
        if operation not in definition.operations:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.PERMISSION, f"operation is not allowed: {operation}", fatal=True)
        missing = set(definition.permission_scopes).difference(context.permissions)
        if missing:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.PERMISSION, f"missing permission scopes: {sorted(missing)}", fatal=True)
        if (definition.requires_approval or command.requires_approval) and not context.approved:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.APPROVAL, "tool invocation requires approval", fatal=False)
        try:
            Draft202012Validator(definition.input_schema).validate(command.arguments)
        except ValidationError as exc:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.VALIDATION, f"input schema validation failed: {exc.message}", fatal=True)
        decision = self.safety_scanner.scan(command.arguments, definition, phase="input")
        if not decision.allowed:
            return self._failure(command, definition.name, definition.version, ToolErrorKind.SAFETY, decision.reason, fatal=True)
        return None

    async def _execute_candidate(self, command: Command, definition: ToolDefinition, adapter: Any, context: ToolInvocationContext) -> ToolExecutionResult:
        usage = BudgetUsage()
        last: ToolExecutionResult | None = None
        for attempt in range(1, definition.max_attempts + 1):
            circuit = self.state_store.before_circuit(definition.identity, definition.circuit_breaker)
            if not circuit.allowed:
                blocked = self._failure(
                    command,
                    definition.name,
                    definition.version,
                    ToolErrorKind.CIRCUIT_OPEN,
                    "tool circuit breaker is open",
                    retryable=True,
                )
                return blocked.model_copy(update={"usage": usage, "attempts": attempt - 1})
            rate = self.state_store.acquire_rate_limit(
                f"{context.principal_id}:{definition.identity}",
                calls=definition.rate_limit.calls,
                window_seconds=definition.rate_limit.window_seconds,
            )
            if not rate.allowed:
                blocked = self._failure(
                    command,
                    definition.name,
                    definition.version,
                    ToolErrorKind.RATE_LIMIT,
                    f"tool rate limit exceeded; retry after {rate.retry_after_seconds:.3f}s",
                    retryable=True,
                    metadata={"retry_after_seconds": rate.retry_after_seconds},
                )
                return blocked.model_copy(update={"usage": usage, "attempts": attempt - 1})
            started = time.monotonic()
            try:
                adapter_result = await self._await_controlled(
                    adapter.execute(command.arguments, context),
                    cancellation=context.cancellation,
                    timeout_seconds=definition.timeout_seconds,
                )
                if not isinstance(adapter_result, ToolAdapterResult):
                    raise ToolAdapterError("tool adapter returned an invalid result", kind=ToolErrorKind.PROTOCOL)
            except GatewayCancelled:
                adapter_result = ToolAdapterResult(
                    success=False,
                    error=self._error(command, ToolErrorKind.CANCELLED, "tool invocation was cancelled", fatal=True, attempt=attempt),
                )
            except GatewayTimedOut:
                adapter_result = ToolAdapterResult(
                    success=False,
                    error=self._error(command, ToolErrorKind.TIMEOUT, "tool invocation timed out", retryable=True, attempt=attempt),
                    retryable=True,
                )
            except Exception as exc:
                kind, retryable, status_code = self._classify_exception(exc)
                adapter_result = ToolAdapterResult(
                    success=False,
                    error=self._error(command, kind, str(exc) or type(exc).__name__, retryable=retryable, fatal=not retryable, attempt=attempt),
                    retryable=retryable,
                    status_code=status_code,
                )

            item_usage = adapter_result.usage.model_copy(
                update={
                    "tool_calls": max(1, adapter_result.usage.tool_calls),
                    "cost_usd": adapter_result.usage.cost_usd or definition.estimated_cost_usd,
                    "wall_time_seconds": max(adapter_result.usage.wall_time_seconds, time.monotonic() - started),
                    "errors": max(1, adapter_result.usage.errors) if not adapter_result.success else adapter_result.usage.errors,
                }
            )
            usage = self._combine_usage(usage, item_usage)
            if adapter_result.success:
                try:
                    if definition.output_schema is not None:
                        Draft202012Validator(definition.output_schema).validate(adapter_result.data)
                except ValidationError as exc:
                    adapter_result = ToolAdapterResult(
                        success=False,
                        error=self._error(command, ToolErrorKind.PROTOCOL, f"output schema validation failed: {exc.message}", fatal=True, attempt=attempt),
                    )
                    usage = usage.plus(errors=1)
                else:
                    safety = self.safety_scanner.scan(adapter_result.data, definition, phase="output")
                    if not safety.allowed:
                        adapter_result = ToolAdapterResult(
                            success=False,
                            error=self._error(command, ToolErrorKind.SAFETY, safety.reason, fatal=True, attempt=attempt),
                        )
                        usage = usage.plus(errors=1)
                    else:
                        self.state_store.record_circuit_success(definition.identity)
                        return ToolExecutionResult(
                            success=True,
                            tool_name=definition.name,
                            tool_version=definition.version,
                            protocol=definition.protocol,
                            data=adapter_result.data,
                            output_artifact_ids=adapter_result.output_artifact_ids,
                            usage=usage,
                            attempts=attempt,
                            metadata=redact_gateway_value(adapter_result.metadata),
                        )

            assert adapter_result.error is not None
            last = ToolExecutionResult(
                success=False,
                tool_name=definition.name,
                tool_version=definition.version,
                protocol=definition.protocol,
                usage=usage,
                error=adapter_result.error,
                attempts=attempt,
                metadata={"status_code": adapter_result.status_code} if adapter_result.status_code else {},
            )
            retryable = adapter_result.retryable or adapter_result.error.retryable
            if retryable:
                self.state_store.record_circuit_failure(definition.identity, definition.circuit_breaker)
            else:
                self.state_store.record_circuit_success(definition.identity)
            if not retryable or attempt >= definition.max_attempts or adapter_result.error.category == ErrorCategory.CANCELLED:
                return last
            usage = usage.plus(retries=1)
            retry_payload = {
                "tool": definition.identity,
                "attempt": attempt + 1,
                "error": adapter_result.error.model_dump(mode="json"),
            }
            self._audit(command, "tool.retry_scheduled", retry_payload)
            if context.on_retry is not None:
                context.on_retry(retry_payload)
            if self.retry_base_seconds:
                await asyncio.sleep(self.retry_base_seconds * (2 ** (attempt - 1)))
        assert last is not None
        return last

    async def _await_controlled(
        self,
        awaitable: Awaitable[_T],
        *,
        cancellation: Any,
        timeout_seconds: float,
    ) -> _T:
        if cancellation is not None and cancellation.cancelled:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise GatewayCancelled()
        operation = asyncio.ensure_future(awaitable)
        cancellation_wait = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        waiters = {operation, *({cancellation_wait} if cancellation_wait else set())}
        try:
            done, _ = await asyncio.wait(waiters, timeout=timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
            if cancellation_wait is not None and cancellation_wait in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise GatewayCancelled()
            if operation not in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise GatewayTimedOut()
            return await operation
        finally:
            if cancellation_wait is not None:
                cancellation_wait.cancel()
                with suppress(asyncio.CancelledError):
                    await cancellation_wait

    def _failure(
        self,
        command: Command,
        tool_name: str,
        tool_version: str,
        kind: ToolErrorKind,
        message: str,
        *,
        retryable: bool = False,
        fatal: bool = False,
        metadata: dict[str, Any] | None = None,
    ) -> ToolExecutionResult:
        try:
            definition, _ = self.registry.resolve(tool_name, None if tool_version == "unknown" else tool_version)
            protocol = definition.protocol
        except KeyError:
            from .models import ToolProtocol

            protocol = ToolProtocol.NATIVE
        return ToolExecutionResult(
            success=False,
            tool_name=tool_name,
            tool_version=tool_version,
            protocol=protocol,
            error=self._error(command, kind, message, retryable=retryable, fatal=fatal),
            attempts=0,
            metadata=metadata or {},
        )

    @staticmethod
    def _error(
        command: Command,
        kind: ToolErrorKind,
        message: str,
        *,
        retryable: bool = False,
        fatal: bool = False,
        attempt: int = 1,
    ) -> ErrorRecord:
        category = {
            ToolErrorKind.PERMISSION: ErrorCategory.POLICY_DENIED,
            ToolErrorKind.APPROVAL: ErrorCategory.APPROVAL_REQUIRED,
            ToolErrorKind.CANCELLED: ErrorCategory.CANCELLED,
            ToolErrorKind.TIMEOUT: ErrorCategory.TRANSIENT_PROVIDER,
            ToolErrorKind.TRANSIENT: ErrorCategory.TRANSIENT_PROVIDER,
            ToolErrorKind.PROTOCOL: ErrorCategory.PROTOCOL,
            ToolErrorKind.VALIDATION: ErrorCategory.SCHEMA_VALIDATION,
            ToolErrorKind.SAFETY: ErrorCategory.POLICY_DENIED,
            ToolErrorKind.RATE_LIMIT: ErrorCategory.TRANSIENT_PROVIDER,
            ToolErrorKind.CIRCUIT_OPEN: ErrorCategory.TRANSIENT_PROVIDER,
            ToolErrorKind.IDEMPOTENCY: ErrorCategory.PROTOCOL,
        }.get(kind, ErrorCategory.PERMANENT_PROVIDER)
        return ErrorRecord(
            category=category,
            code=kind.value,
            message=str(redact_gateway_value(message))[:2000],
            retryable=retryable,
            fatal=fatal,
            attempt=attempt,
            actor_id=command.actor_id,
            task_id=command.task_id,
            command_id=command.command_id,
        )

    @staticmethod
    def _classify_exception(exc: Exception) -> tuple[ToolErrorKind, bool, int | None]:
        if isinstance(exc, ToolAdapterError):
            return exc.kind, exc.retryable, exc.status_code
        if isinstance(exc, (httpx.TimeoutException, TimeoutError)):
            return ToolErrorKind.TIMEOUT, True, None
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)):
            return ToolErrorKind.TRANSIENT, True, None
        if isinstance(exc, httpx.HTTPStatusError):
            status = exc.response.status_code
            return (
                ToolErrorKind.TRANSIENT if status in {408, 409, 425, 429, 500, 502, 503, 504} else ToolErrorKind.PERMANENT,
                status in {408, 409, 425, 429, 500, 502, 503, 504},
                status,
            )
        return ToolErrorKind.PERMANENT, False, None

    def _audit(self, command: Command, event_type: str, payload: dict[str, Any]) -> None:
        self.state_store.append_audit(
            run_id=command.run_id,
            task_id=command.task_id,
            command_id=command.command_id,
            event_type=event_type,
            payload=redact_gateway_value(payload),
        )

    @staticmethod
    def _fingerprint(definition: ToolDefinition, arguments: dict[str, Any]) -> str:
        payload = json.dumps([definition.identity, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()

    @staticmethod
    def _cache_key(definition: ToolDefinition, arguments: dict[str, Any]) -> str:
        return f"cache-{ProtocolToolGateway._fingerprint(definition, arguments)}"

    @staticmethod
    def _combine_usage(first: BudgetUsage, second: BudgetUsage) -> BudgetUsage:
        return BudgetUsage(
            input_tokens=first.input_tokens + second.input_tokens,
            output_tokens=first.output_tokens + second.output_tokens,
            cost_usd=first.cost_usd + second.cost_usd,
            wall_time_seconds=first.wall_time_seconds + second.wall_time_seconds,
            model_calls=first.model_calls + second.model_calls,
            tool_calls=first.tool_calls + second.tool_calls,
            search_calls=first.search_calls + second.search_calls,
            retries=first.retries + second.retries,
            errors=first.errors + second.errors,
        )
