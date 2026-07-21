from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from deep_researcher.contracts import (
    AgentSpec,
    BudgetUsage,
    Command,
    ErrorRecord,
    Observation,
    TaskEnvelope,
    utc_now,
)


@dataclass(frozen=True)
class ModelRequest:
    run_id: str
    task_id: str
    actor_id: str
    system: str
    messages: tuple[dict[str, Any], ...]
    command_schema: dict[str, Any]
    model_version: str
    prompt_version: str
    max_output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    content: str = ""
    structured: Any = None
    usage: BudgetUsage = field(default_factory=BudgetUsage)
    latency_ms: float = 0.0
    finish_reason: str = "stop"
    response_id: str = ""


class ModelAdapter(Protocol):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        ...

    async def repair(self, request: ModelRequest, invalid_response: ModelResponse, errors: tuple[str, ...]) -> ModelResponse:
        ...


@dataclass(frozen=True)
class RawObservation:
    status: str
    data: Any = None
    output_artifact_ids: tuple[str, ...] = ()
    usage: BudgetUsage = field(default_factory=BudgetUsage)
    error: ErrorRecord | None = None
    started_at: datetime = field(default_factory=utc_now)
    completed_at: datetime = field(default_factory=utc_now)


class ActionExecutor(Protocol):
    async def execute(self, command: Command) -> RawObservation | Observation:
        ...


@dataclass(frozen=True)
class VerificationFeedback:
    passed: bool
    success: bool = False
    semantic_complete: bool = False
    information_gain: float = 0.0
    summary: str = "Verification completed."
    repair_feedback: tuple[str, ...] = ()
    usage: BudgetUsage = field(default_factory=BudgetUsage)


class KernelVerifier(Protocol):
    async def verify(
        self,
        *,
        spec: AgentSpec,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        ...


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    reason: str
    approval_required: bool = False
    checks: tuple[str, ...] = ()


@dataclass(frozen=True)
class KernelEvent:
    event_type: str
    run_id: str
    task_id: str
    actor_id: str
    payload: dict[str, Any] = field(default_factory=dict)
    occurred_at: datetime = field(default_factory=utc_now)


class KernelEventSink(Protocol):
    def emit(self, event: KernelEvent) -> None:
        ...


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()


class KernelError(RuntimeError):
    pass


class CommandSchemaError(KernelError):
    def __init__(self, errors: tuple[str, ...]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors


class ActionExecutionError(KernelError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ModelInvocationError(KernelError):
    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class KernelCancelled(KernelError):
    pass


class KernelTimedOut(KernelError):
    pass
