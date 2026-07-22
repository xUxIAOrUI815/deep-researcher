from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import re
from typing import Any, Awaitable, Callable, Protocol

from jsonschema import Draft202012Validator
from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import BudgetUsage, Command, ContractModel, ErrorRecord


MCP_PROTOCOL_VERSION = "2025-11-25"
A2A_PROTOCOL_VERSION = "1.0"


class ToolProtocol(str, Enum):
    NATIVE = "native"
    FUNCTION_CALLING = "function_calling"
    MCP = "mcp"
    A2A = "a2a"


class ToolRiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ToolHealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CLOSED = "closed"


class CircuitState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ToolErrorKind(str, Enum):
    VALIDATION = "validation"
    PERMISSION = "permission"
    APPROVAL = "approval"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    CIRCUIT_OPEN = "circuit_open"
    SAFETY = "safety"
    PROTOCOL = "protocol"
    IDEMPOTENCY = "idempotency"


class RateLimit(ContractModel):
    calls: int = Field(gt=0)
    window_seconds: float = Field(gt=0.0)


class CircuitBreakerPolicy(ContractModel):
    failure_threshold: int = Field(default=3, gt=0)
    recovery_seconds: float = Field(default=30.0, gt=0.0)


class ToolDefinition(ContractModel):
    name: str = Field(min_length=1, max_length=200, pattern=r"^[A-Za-z0-9_.:-]+$")
    version: str = Field(pattern=r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?$")
    description: str = Field(min_length=1, max_length=4000)
    input_schema: dict[str, Any]
    output_schema: dict[str, Any] | None = None
    operations: tuple[str, ...]
    permission_scopes: tuple[str, ...] = ()
    risk_level: ToolRiskLevel = ToolRiskLevel.LOW
    requires_approval: bool = False
    side_effecting: bool = False
    idempotent: bool = True
    timeout_seconds: float = Field(default=30.0, gt=0.0)
    max_attempts: int = Field(default=3, gt=0)
    rate_limit: RateLimit = Field(default_factory=lambda: RateLimit(calls=60, window_seconds=60.0))
    cache_ttl_seconds: float | None = Field(default=None, gt=0.0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    fallback_tools: tuple[str, ...] = ()
    circuit_breaker: CircuitBreakerPolicy = Field(default_factory=CircuitBreakerPolicy)
    protocol: ToolProtocol = ToolProtocol.NATIVE
    provider: str = Field(default="local", min_length=1, max_length=200)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("operations", "permission_scopes", "fallback_tools")
    @classmethod
    def _unique_strings(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(item.strip() for item in values if item.strip())
        if len(normalized) != len(set(normalized)):
            raise ValueError("tool string lists must contain unique values")
        return normalized

    @model_validator(mode="after")
    def _valid_definition(self) -> "ToolDefinition":
        if not self.operations:
            raise ValueError("a tool requires at least one operation")
        Draft202012Validator.check_schema(self.input_schema)
        if self.output_schema is not None:
            Draft202012Validator.check_schema(self.output_schema)
        if self.risk_level in {ToolRiskLevel.HIGH, ToolRiskLevel.CRITICAL} and not self.requires_approval:
            raise ValueError("high-risk tools must require approval")
        if self.side_effecting and not self.idempotent and not self.requires_approval:
            raise ValueError("non-idempotent side-effecting tools must require approval")
        if self.name in self.fallback_tools:
            raise ValueError("a tool cannot fall back to itself")
        return self

    @property
    def identity(self) -> str:
        return f"{self.name}@{self.version}"


class ToolAdapterResult(ContractModel):
    success: bool
    data: Any = None
    output_artifact_ids: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    error: ErrorRecord | None = None
    retryable: bool = False
    status_code: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ToolAdapterResult":
        if self.success and self.error is not None:
            raise ValueError("successful tool result cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("failed tool result requires an error")
        return self


class ToolExecutionResult(ContractModel):
    success: bool
    tool_name: str
    tool_version: str
    protocol: ToolProtocol
    data: Any = None
    output_artifact_ids: tuple[str, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    error: ErrorRecord | None = None
    attempts: int = Field(default=1, ge=0)
    cached: bool = False
    idempotent_replay: bool = False
    fallback_chain: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _consistent(self) -> "ToolExecutionResult":
        if self.success and self.error is not None:
            raise ValueError("successful execution cannot contain an error")
        if not self.success and self.error is None:
            raise ValueError("failed execution requires an error")
        return self


class SafetyDecision(ContractModel):
    allowed: bool
    reason: str = Field(min_length=1, max_length=2000)
    checks: tuple[str, ...] = ()


class ToolAdapter(Protocol):
    async def execute(self, arguments: dict[str, Any], context: "ToolInvocationContext") -> ToolAdapterResult:
        ...

    async def health(self) -> ToolHealthStatus:
        ...


class CancellationSignal(Protocol):
    @property
    def cancelled(self) -> bool:
        ...

    async def wait(self) -> None:
        ...


RetryCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True)
class ToolInvocationContext:
    principal_id: str = "principal_runtime"
    permissions: frozenset[str] = frozenset()
    approved: bool = False
    correlation_id: str = "correlation_unknown"
    trace_id: str = "trace_unknown"
    cancellation: CancellationSignal | None = None
    on_retry: RetryCallback | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolAdapterError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: ToolErrorKind = ToolErrorKind.PERMANENT,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code


class GatewayConfigurationError(RuntimeError):
    pass


class GatewayCancelled(RuntimeError):
    pass


class GatewayTimedOut(RuntimeError):
    pass


def validate_tool_name(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9_.:-]+", value) is None:
        raise ValueError(f"invalid tool name: {value}")
    return value
