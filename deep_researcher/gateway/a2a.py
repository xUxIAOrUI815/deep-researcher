from __future__ import annotations

import asyncio
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Literal

import httpx
from a2a.client import A2ACardResolver, ClientCallContext, ClientConfig, ClientFactory
from a2a.client.client import Client
from a2a.client.client_factory import TransportProtocol
from a2a.client.errors import A2AClientError, A2AClientTimeoutError, AgentCardResolutionError
from a2a.types import (
    AgentCard,
    Artifact,
    CancelTaskRequest,
    GetTaskRequest,
    Message,
    Part,
    Role,
    SendMessageConfiguration,
    SendMessageRequest,
    StreamResponse,
    Task,
    TaskState,
)
from google.protobuf.json_format import MessageToDict
from google.protobuf.struct_pb2 import Struct

from deep_researcher.contracts import TaskEnvelope

from .gateway import redact_gateway_value
from .models import A2A_PROTOCOL_VERSION, CancellationSignal, ToolHealthStatus


class A2AFailureKind(str, Enum):
    DISCOVERY = "discovery"
    VERSION = "version"
    AUTHENTICATION = "authentication"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    TRANSIENT = "transient"
    PERMANENT = "permanent"
    INVALID_RESPONSE = "invalid_response"
    UNSUPPORTED = "unsupported"


class A2AProtocolError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        kind: A2AFailureKind,
        retryable: bool = False,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable
        self.status_code = status_code


@dataclass(frozen=True)
class A2AClientConfig:
    base_url: str
    protocol_version: str = A2A_PROTOCOL_VERSION
    binding: Literal["JSONRPC", "HTTP+JSON"] = "JSONRPC"
    timeout_seconds: float = 30.0
    streaming: bool = False
    polling: bool = True
    accepted_output_modes: tuple[str, ...] = ("application/json", "text/plain")
    headers: dict[str, str] = field(default_factory=dict)
    extensions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.protocol_version != A2A_PROTOCOL_VERSION:
            raise ValueError(f"A2A protocol must be pinned to {A2A_PROTOCOL_VERSION}")
        if self.timeout_seconds <= 0:
            raise ValueError("A2A timeout must be positive")
        if not self.base_url.startswith(("http://", "https://")):
            raise ValueError("A2A base URL must use HTTP or HTTPS")


@dataclass(frozen=True)
class A2AArtifactHandoff:
    artifact_id: str
    name: str
    media_type: str
    text: str | None = None
    data: dict[str, Any] | None = None
    raw: bytes | None = None
    uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = [self.text is not None, self.data is not None, self.raw is not None, self.uri is not None]
        if sum(values) != 1:
            raise ValueError("A2A artifact handoff requires exactly one content representation")


@dataclass(frozen=True)
class A2ARemoteArtifact:
    artifact_id: str
    name: str
    description: str
    parts: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class A2ARemoteTask:
    remote_task_id: str
    context_id: str
    state: str
    terminal: bool
    artifacts: tuple[A2ARemoteArtifact, ...]
    metadata: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class A2ADiscovery:
    name: str
    description: str
    version: str
    protocol_version: str
    binding: str
    endpoint: str
    streaming: bool
    skills: tuple[dict[str, Any], ...]
    card: AgentCard


class A2AProtocolClient:
    """Official A2A 1.0 SDK client with domain contract mapping."""

    def __init__(
        self,
        config: A2AClientConfig,
        *,
        http_client: httpx.AsyncClient | None = None,
        signature_verifier: Callable[[AgentCard], None] | None = None,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self._owns_http_client = http_client is None
        self.signature_verifier = signature_verifier
        self._client: Client | None = None
        self._card: AgentCard | None = None
        self._discovery: A2ADiscovery | None = None
        self._local_to_remote: dict[str, str] = {}
        self.health_status = ToolHealthStatus.CLOSED
        self.last_error: str | None = None

    async def connect(self) -> A2ADiscovery:
        if self._discovery is not None:
            return self._discovery
        client = self._http_client
        if client is None:
            client = httpx.AsyncClient(headers=self.config.headers, timeout=self.config.timeout_seconds)
            self._http_client = client
        try:
            card = await asyncio.wait_for(
                A2ACardResolver(client, self.config.base_url).get_agent_card(signature_verifier=self.signature_verifier),
                timeout=self.config.timeout_seconds,
            )
            selected = next(
                (
                    interface
                    for interface in card.supported_interfaces
                    if interface.protocol_binding == self.config.binding
                    and interface.protocol_version == self.config.protocol_version
                ),
                None,
            )
            if selected is None:
                offered = [f"{item.protocol_binding}@{item.protocol_version}" for item in card.supported_interfaces]
                raise A2AProtocolError(
                    f"agent does not offer {self.config.binding}@{self.config.protocol_version}; offered={offered}",
                    kind=A2AFailureKind.VERSION,
                )
            protocol = TransportProtocol.JSONRPC if self.config.binding == "JSONRPC" else TransportProtocol.HTTP_JSON
            sdk_config = ClientConfig(
                streaming=self.config.streaming,
                polling=self.config.polling,
                httpx_client=client,
                supported_protocol_bindings=[protocol],
                accepted_output_modes=list(self.config.accepted_output_modes),
            )
            sdk_client = ClientFactory(sdk_config).create(card)
        except Exception as exc:
            error = self._classify(exc, discovery=True)
            self.health_status = ToolHealthStatus.UNHEALTHY
            self.last_error = str(error)
            if self._owns_http_client and self._http_client is not None:
                await self._http_client.aclose()
                self._http_client = None
            raise error from exc
        self._card = card
        self._client = sdk_client
        self._discovery = A2ADiscovery(
            name=card.name,
            description=card.description,
            version=card.version,
            protocol_version=selected.protocol_version,
            binding=selected.protocol_binding,
            endpoint=selected.url,
            streaming=bool(card.capabilities.streaming),
            skills=tuple(MessageToDict(skill) for skill in card.skills),
            card=card,
        )
        self.health_status = ToolHealthStatus.HEALTHY
        self.last_error = None
        return self._discovery

    async def discover(self) -> A2ADiscovery:
        return await self.connect()

    async def close(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()
        if self._owns_http_client and self._http_client is not None:
            if not self._http_client.is_closed:
                await self._http_client.aclose()
            self._http_client = None
        self._card = None
        self._discovery = None
        self.health_status = ToolHealthStatus.CLOSED

    async def __aenter__(self) -> "A2AProtocolClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def submit_task(
        self,
        task: TaskEnvelope,
        *,
        artifacts: tuple[A2AArtifactHandoff, ...] = (),
        correlation_id: str,
        trace_id: str,
        cancellation: CancellationSignal | None = None,
    ) -> A2ARemoteTask:
        await self.connect()
        message = self._task_message(task, artifacts, correlation_id=correlation_id, trace_id=trace_id)
        metadata = self._struct({
            "localTaskId": task.task_id,
            "localRunId": task.run_id,
            "correlationId": correlation_id,
            "traceId": trace_id,
        })
        request = SendMessageRequest(
            message=message,
            configuration=SendMessageConfiguration(
                accepted_output_modes=list(self.config.accepted_output_modes),
                return_immediately=False,
            ),
            metadata=metadata,
        )
        context = self._call_context(correlation_id, trace_id)
        latest: A2ARemoteTask | None = None
        accumulated: Task | None = None
        iterator = self._require_client().send_message(request, context=context).__aiter__()
        try:
            while True:
                try:
                    event = await self._next_controlled(iterator, cancellation)
                except StopAsyncIteration:
                    break
                task_message = self._apply_stream_event(accumulated, event)
                if task_message is not None:
                    accumulated = task_message
                    latest = self._snapshot(task_message)
        except Exception as exc:
            error = self._classify(exc)
            self.health_status = ToolHealthStatus.DEGRADED if error.retryable else ToolHealthStatus.UNHEALTHY
            self.last_error = str(error)
            raise error from exc
        if latest is None:
            error = A2AProtocolError("A2A submission returned no task", kind=A2AFailureKind.INVALID_RESPONSE)
            self.last_error = str(error)
            raise error
        self._local_to_remote[task.task_id] = latest.remote_task_id
        self.health_status = ToolHealthStatus.HEALTHY
        return latest

    async def get_task(
        self,
        task_id: str,
        *,
        correlation_id: str,
        trace_id: str,
        history_length: int = 0,
    ) -> A2ARemoteTask:
        await self.connect()
        remote_id = self._local_to_remote.get(task_id, task_id)
        try:
            task = await asyncio.wait_for(
                self._require_client().get_task(
                    GetTaskRequest(id=remote_id, history_length=max(0, history_length)),
                    context=self._call_context(correlation_id, trace_id),
                ),
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise self._classify(exc) from exc
        return self._snapshot(task)

    async def cancel_task(
        self,
        task_id: str,
        *,
        correlation_id: str,
        trace_id: str,
    ) -> A2ARemoteTask:
        await self.connect()
        remote_id = self._local_to_remote.get(task_id, task_id)
        try:
            task = await asyncio.wait_for(
                self._require_client().cancel_task(
                    CancelTaskRequest(
                        id=remote_id,
                        metadata=self._struct({"correlationId": correlation_id, "traceId": trace_id}),
                    ),
                    context=self._call_context(correlation_id, trace_id),
                ),
                timeout=self.config.timeout_seconds,
            )
        except Exception as exc:
            raise self._classify(exc) from exc
        return self._snapshot(task)

    def _task_message(
        self,
        task: TaskEnvelope,
        artifacts: tuple[A2AArtifactHandoff, ...],
        *,
        correlation_id: str,
        trace_id: str,
    ) -> Message:
        payload = redact_gateway_value(task.model_dump(mode="json"))
        parts = [
            Part(
                data={"struct_value": self._value_struct({"taskEnvelope": payload})},
                media_type="application/json",
                metadata=self._struct({"contract": "TaskEnvelope@1"}),
            )
        ]
        parts.extend(self._artifact_part(item) for item in artifacts)
        digest = hashlib.sha256(f"{task.run_id}\0{task.task_id}\0{correlation_id}".encode()).hexdigest()
        return Message(
            message_id=f"message_{digest[:32]}",
            context_id=correlation_id,
            role=Role.ROLE_USER,
            parts=parts,
            metadata=self._struct({
                "localTaskId": task.task_id,
                "localRunId": task.run_id,
                "correlationId": correlation_id,
                "traceId": trace_id,
            }),
        )

    def _artifact_part(self, artifact: A2AArtifactHandoff) -> Part:
        common = {
            "filename": artifact.name,
            "media_type": artifact.media_type,
            "metadata": self._struct({"artifactId": artifact.artifact_id, **redact_gateway_value(artifact.metadata)}),
        }
        if artifact.text is not None:
            return Part(text=artifact.text, **common)
        if artifact.data is not None:
            return Part(data={"struct_value": self._value_struct(redact_gateway_value(artifact.data))}, **common)
        if artifact.raw is not None:
            return Part(raw=artifact.raw, **common)
        assert artifact.uri is not None
        return Part(url=artifact.uri, **common)

    def _call_context(self, correlation_id: str, trace_id: str) -> ClientCallContext:
        parameters = {"A2A-Version": self.config.protocol_version}
        if self.config.extensions:
            parameters["A2A-Extensions"] = ",".join(self.config.extensions)
        return ClientCallContext(
            timeout=self.config.timeout_seconds,
            service_parameters=parameters,
            state={"correlation_id": correlation_id, "trace_id": trace_id},
        )

    async def _next_controlled(self, iterator: Any, cancellation: CancellationSignal | None) -> StreamResponse:
        if cancellation is not None and cancellation.cancelled:
            raise A2AProtocolError("A2A submission was cancelled", kind=A2AFailureKind.CANCELLED)
        operation = asyncio.create_task(anext(iterator))
        cancel_wait = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        waiters = {operation, *({cancel_wait} if cancel_wait else set())}
        try:
            done, _ = await asyncio.wait(waiters, timeout=self.config.timeout_seconds, return_when=asyncio.FIRST_COMPLETED)
            if cancel_wait is not None and cancel_wait in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise A2AProtocolError("A2A submission was cancelled", kind=A2AFailureKind.CANCELLED)
            if operation not in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise A2AProtocolError("A2A request timed out", kind=A2AFailureKind.TIMEOUT, retryable=True)
            return await operation
        finally:
            if cancel_wait is not None:
                cancel_wait.cancel()
                with suppress(asyncio.CancelledError):
                    await cancel_wait

    @staticmethod
    def _apply_stream_event(current: Task | None, event: StreamResponse) -> Task | None:
        if event.HasField("task"):
            task = Task()
            task.CopyFrom(event.task)
            return task
        if event.HasField("status_update"):
            update = event.status_update
            task = A2AProtocolClient._copy_or_create_task(current, update.task_id, update.context_id)
            task.status.CopyFrom(update.status)
            if update.HasField("metadata"):
                task.metadata.update(MessageToDict(update.metadata))
            return task
        if event.HasField("artifact_update"):
            update = event.artifact_update
            task = A2AProtocolClient._copy_or_create_task(current, update.task_id, update.context_id)
            existing_index = next(
                (index for index, artifact in enumerate(task.artifacts) if artifact.artifact_id == update.artifact.artifact_id),
                None,
            )
            if update.append and existing_index is not None:
                target = task.artifacts[existing_index]
                target.parts.extend(update.artifact.parts)
                if update.artifact.name:
                    target.name = update.artifact.name
                if update.artifact.description:
                    target.description = update.artifact.description
                if update.artifact.HasField("metadata"):
                    target.metadata.update(MessageToDict(update.artifact.metadata))
            elif existing_index is not None:
                task.artifacts[existing_index].CopyFrom(update.artifact)
            else:
                task.artifacts.add().CopyFrom(update.artifact)
            if update.HasField("metadata"):
                task.metadata.update(MessageToDict(update.metadata))
            return task
        return None

    @staticmethod
    def _copy_or_create_task(current: Task | None, task_id: str, context_id: str) -> Task:
        if current is None:
            return Task(id=task_id, context_id=context_id)
        if current.id != task_id or current.context_id != context_id:
            raise A2AProtocolError(
                "A2A stream update changed task or context identity",
                kind=A2AFailureKind.INVALID_RESPONSE,
            )
        task = Task()
        task.CopyFrom(current)
        return task

    @staticmethod
    def _snapshot(task: Task) -> A2ARemoteTask:
        state_name = TaskState.Name(task.status.state).removeprefix("TASK_STATE_").casefold()
        terminal = task.status.state in {
            TaskState.TASK_STATE_COMPLETED,
            TaskState.TASK_STATE_FAILED,
            TaskState.TASK_STATE_CANCELED,
            TaskState.TASK_STATE_REJECTED,
        }
        return A2ARemoteTask(
            remote_task_id=task.id,
            context_id=task.context_id,
            state=state_name,
            terminal=terminal,
            artifacts=tuple(A2AProtocolClient._artifact_snapshot(item) for item in task.artifacts),
            metadata=MessageToDict(task.metadata) if task.HasField("metadata") else {},
            raw=MessageToDict(task),
        )

    @staticmethod
    def _artifact_snapshot(artifact: Artifact) -> A2ARemoteArtifact:
        return A2ARemoteArtifact(
            artifact_id=artifact.artifact_id,
            name=artifact.name,
            description=artifact.description,
            parts=tuple(MessageToDict(part) for part in artifact.parts),
            metadata=MessageToDict(artifact.metadata) if artifact.HasField("metadata") else {},
        )

    def _require_client(self) -> Client:
        if self._client is None:
            raise A2AProtocolError("A2A client is not connected", kind=A2AFailureKind.DISCOVERY)
        return self._client

    @staticmethod
    def _struct(value: dict[str, Any]) -> Struct:
        structure = Struct()
        structure.update(json.loads(json.dumps(value, ensure_ascii=False, default=str)))
        return structure

    @staticmethod
    def _value_struct(value: dict[str, Any]) -> Struct:
        """Build the Struct nested inside the A2A Part.data protobuf Value."""
        return A2AProtocolClient._struct(value)

    @staticmethod
    def _classify(exc: Exception, *, discovery: bool = False) -> A2AProtocolError:
        if isinstance(exc, A2AProtocolError):
            return exc
        if isinstance(exc, AgentCardResolutionError):
            status = getattr(exc, "status_code", None)
            return A2AProtocolError(
                str(exc),
                kind=A2AFailureKind.AUTHENTICATION if status in {401, 403} else A2AFailureKind.DISCOVERY,
                retryable=status is None or status in {408, 425, 429, 500, 502, 503, 504},
                status_code=status,
            )
        if isinstance(exc, (A2AClientTimeoutError, httpx.TimeoutException, TimeoutError)):
            return A2AProtocolError(str(exc) or "A2A request timed out", kind=A2AFailureKind.TIMEOUT, retryable=True)
        if isinstance(exc, (httpx.ConnectError, httpx.RemoteProtocolError)):
            return A2AProtocolError(str(exc), kind=A2AFailureKind.TRANSIENT, retryable=True)
        if isinstance(exc, A2AClientError):
            message = str(exc)
            kind = A2AFailureKind.UNSUPPORTED if "Unsupported" in type(exc).__name__ else A2AFailureKind.PERMANENT
            return A2AProtocolError(message, kind=kind)
        return A2AProtocolError(
            f"{type(exc).__name__}: {exc}",
            kind=A2AFailureKind.DISCOVERY if discovery else A2AFailureKind.PERMANENT,
        )
