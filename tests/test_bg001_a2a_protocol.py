from __future__ import annotations

import asyncio
from datetime import timedelta
import json
from typing import Any

import httpx
import pytest
from a2a.types import (
    Artifact,
    Part,
    StreamResponse,
    Task,
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)

from deep_researcher.contracts import Budget, TaskEnvelope, TaskKind, utc_now
from deep_researcher.gateway import (
    A2AArtifactHandoff,
    A2AClientConfig,
    A2AFailureKind,
    A2AProtocolClient,
    A2AProtocolError,
    A2A_PROTOCOL_VERSION,
)
from deep_researcher.kernel import CancellationToken


def _task(suffix: str = "one") -> TaskEnvelope:
    return TaskEnvelope(
        task_id=f"task_{suffix}",
        run_id=f"run_{suffix}",
        kind=TaskKind.RESEARCH,
        title="Remote research",
        goal="Produce a structured remote result.",
        expected_output_schema="ArtifactBundle@1",
        budget=Budget(
            max_tokens=1000,
            max_cost_usd=1,
            max_wall_time_seconds=30,
            max_model_calls=3,
            max_tool_calls=5,
            max_search_calls=3,
            max_retries=1,
            max_errors=1,
        ),
        created_by="agent_supervisor",
    )


def _card(*, protocol_version: str = "1.0", binding: str = "JSONRPC") -> dict[str, Any]:
    return {
        "name": "Remote Research Agent",
        "description": "Performs bounded remote research.",
        "supportedInterfaces": [
            {
                "url": "https://agent.test/a2a",
                "protocolBinding": binding,
                "protocolVersion": protocol_version,
            }
        ],
        "version": "2.0.0",
        "capabilities": {"streaming": False},
        "defaultInputModes": ["application/json", "text/plain"],
        "defaultOutputModes": ["application/json", "text/plain"],
        "skills": [
            {
                "id": "research",
                "name": "Research",
                "description": "Research a question.",
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            }
        ],
    }


def _remote_task(*, state: str = "TASK_STATE_WORKING") -> dict[str, Any]:
    return {
        "id": "remote-task-123",
        "contextId": "correlation-a2a",
        "status": {"state": state},
        "artifacts": [
            {
                "artifactId": "remote-artifact-1",
                "name": "Remote Result",
                "description": "Structured result",
                "parts": [{"text": "remote body", "mediaType": "text/plain"}],
                "metadata": {"source": "remote"},
            }
        ],
        "metadata": {"remoteRun": "r-1"},
    }


class A2AServerMock:
    def __init__(self, *, card: dict[str, Any] | None = None, delay: float = 0.0, message_only: bool = False) -> None:
        self.card = card or _card()
        self.delay = delay
        self.message_only = message_only
        self.requests: list[httpx.Request] = []
        self.jsonrpc: list[dict[str, Any]] = []

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "GET" and request.url.path == "/.well-known/agent-card.json":
            return httpx.Response(200, json=self.card)
        if self.delay:
            await asyncio.sleep(self.delay)
        body = json.loads(request.content)
        self.jsonrpc.append(body)
        method = body["method"]
        if method == "SendMessage":
            if self.message_only:
                result = {
                    "message": {
                        "messageId": "message-only",
                        "contextId": "correlation-a2a",
                        "role": "ROLE_AGENT",
                        "parts": [{"text": "not a task"}],
                    }
                }
            else:
                result = {"task": _remote_task()}
        elif method == "GetTask":
            result = _remote_task(state="TASK_STATE_COMPLETED")
        elif method == "CancelTask":
            result = _remote_task(state="TASK_STATE_CANCELED")
        else:
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "error": {"code": -32601, "message": "unsupported"}})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result})


@pytest.mark.asyncio
async def test_a2a_discovery_submit_status_cancel_artifact_handoff_and_correlation():
    server = A2AServerMock()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    client = A2AProtocolClient(
        A2AClientConfig(base_url="https://agent.test", extensions=("https://example.test/citations/v1",)),
        http_client=http_client,
    )
    async with client:
        discovery = await client.discover()
        assert discovery.protocol_version == A2A_PROTOCOL_VERSION
        assert discovery.binding == "JSONRPC"
        assert discovery.endpoint == "https://agent.test/a2a"
        assert discovery.skills[0]["id"] == "research"

        artifact = A2AArtifactHandoff(
            artifact_id="artifact_local_input",
            name="input.txt",
            media_type="text/plain",
            text="local artifact body",
            metadata={"classification": "public"},
        )
        submitted = await client.submit_task(
            _task(),
            artifacts=(artifact,),
            correlation_id="correlation-a2a",
            trace_id="trace-a2a",
        )
        assert submitted.remote_task_id == "remote-task-123"
        assert submitted.state == "working"
        assert not submitted.terminal
        assert submitted.artifacts[0].parts[0]["text"] == "remote body"

        completed = await client.get_task(
            "task_one",
            correlation_id="correlation-a2a",
            trace_id="trace-a2a",
        )
        assert completed.state == "completed"
        assert completed.terminal
        cancelled = await client.cancel_task(
            "task_one",
            correlation_id="correlation-a2a",
            trace_id="trace-a2a",
        )
        assert cancelled.state == "canceled"
        assert cancelled.terminal

    methods = [item["method"] for item in server.jsonrpc]
    assert methods == ["SendMessage", "GetTask", "CancelTask"]
    send = server.jsonrpc[0]["params"]
    assert send["message"]["contextId"] == "correlation-a2a"
    assert send["message"]["metadata"]["traceId"] == "trace-a2a"
    assert send["message"]["parts"][0]["data"]["taskEnvelope"]["task_id"] == "task_one"
    assert send["message"]["parts"][1]["text"] == "local artifact body"
    assert send["message"]["parts"][1]["metadata"]["artifactId"] == "artifact_local_input"
    assert server.jsonrpc[1]["params"]["id"] == "remote-task-123"
    assert server.jsonrpc[2]["params"]["id"] == "remote-task-123"
    for request in server.requests[1:]:
        assert request.headers["A2A-Version"] == "1.0"
        assert request.headers["A2A-Extensions"] == "https://example.test/citations/v1"


@pytest.mark.parametrize(
    "handoff",
    [
        {"artifact_id": "artifact_x", "name": "x", "media_type": "text/plain"},
        {"artifact_id": "artifact_x", "name": "x", "media_type": "text/plain", "text": "a", "uri": "https://example.test/x"},
    ],
)
def test_a2a_artifact_handoff_requires_exactly_one_representation(handoff: dict[str, Any]):
    with pytest.raises(ValueError, match="exactly one"):
        A2AArtifactHandoff(**handoff)


@pytest.mark.asyncio
async def test_a2a_strict_version_and_binding_negotiation():
    for card in (_card(protocol_version="0.3"), _card(binding="HTTP+JSON")):
        transport = httpx.MockTransport(A2AServerMock(card=card))
        http_client = httpx.AsyncClient(transport=transport)
        client = A2AProtocolClient(A2AClientConfig(base_url="https://agent.test"), http_client=http_client)
        with pytest.raises(A2AProtocolError) as captured:
            await client.connect()
        assert captured.value.kind == A2AFailureKind.VERSION
        await http_client.aclose()
    with pytest.raises(ValueError, match="pinned"):
        A2AClientConfig(base_url="https://agent.test", protocol_version="0.3")


@pytest.mark.asyncio
async def test_a2a_discovery_auth_failure_is_classified():
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = A2AProtocolClient(A2AClientConfig(base_url="https://agent.test"), http_client=http_client)
    with pytest.raises(A2AProtocolError) as captured:
        await client.connect()
    assert captured.value.kind == A2AFailureKind.AUTHENTICATION
    assert captured.value.status_code == 401
    await http_client.aclose()


@pytest.mark.asyncio
async def test_a2a_timeout_and_active_cancellation_are_classified_and_interrupt_requests():
    timeout_server = A2AServerMock(delay=1)
    timeout_http = httpx.AsyncClient(transport=httpx.MockTransport(timeout_server))
    timeout_client = A2AProtocolClient(
        A2AClientConfig(base_url="https://agent.test", timeout_seconds=0.02),
        http_client=timeout_http,
    )
    with pytest.raises(A2AProtocolError) as timeout_error:
        await timeout_client.submit_task(_task("timeout"), correlation_id="correlation-timeout", trace_id="trace-timeout")
    assert timeout_error.value.kind == A2AFailureKind.TIMEOUT
    assert timeout_error.value.retryable
    await timeout_client.close()
    await timeout_http.aclose()

    cancel_server = A2AServerMock(delay=1)
    cancel_http = httpx.AsyncClient(transport=httpx.MockTransport(cancel_server))
    cancel_client = A2AProtocolClient(A2AClientConfig(base_url="https://agent.test"), http_client=cancel_http)
    token = CancellationToken()
    running = asyncio.create_task(
        cancel_client.submit_task(
            _task("cancel"),
            correlation_id="correlation-cancel",
            trace_id="trace-cancel",
            cancellation=token,
        )
    )
    await asyncio.sleep(0.02)
    token.cancel()
    with pytest.raises(A2AProtocolError) as cancel_error:
        await asyncio.wait_for(running, timeout=1)
    assert cancel_error.value.kind == A2AFailureKind.CANCELLED
    await cancel_client.close()
    await cancel_http.aclose()


@pytest.mark.asyncio
async def test_a2a_message_only_response_is_invalid_for_task_submission():
    server = A2AServerMock(message_only=True)
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server))
    client = A2AProtocolClient(A2AClientConfig(base_url="https://agent.test"), http_client=http_client)
    with pytest.raises(A2AProtocolError) as captured:
        await client.submit_task(_task("message_only"), correlation_id="correlation-message", trace_id="trace-message")
    assert captured.value.kind == A2AFailureKind.INVALID_RESPONSE
    await client.close()
    await http_client.aclose()


def test_a2a_stream_updates_accumulate_status_and_artifact_chunks_without_losing_identity():
    initial = StreamResponse(
        task=Task(
            id="remote-stream",
            context_id="context-stream",
            status=TaskStatus(state=TaskState.TASK_STATE_WORKING),
            artifacts=[
                Artifact(
                    artifact_id="artifact-stream",
                    name="stream.txt",
                    parts=[Part(text="first")],
                )
            ],
        )
    )
    current = A2AProtocolClient._apply_stream_event(None, initial)
    appended = A2AProtocolClient._apply_stream_event(
        current,
        StreamResponse(
            artifact_update=TaskArtifactUpdateEvent(
                task_id="remote-stream",
                context_id="context-stream",
                artifact=Artifact(
                    artifact_id="artifact-stream",
                    name="stream.txt",
                    parts=[Part(text="second")],
                ),
                append=True,
                last_chunk=True,
            )
        ),
    )
    completed = A2AProtocolClient._apply_stream_event(
        appended,
        StreamResponse(
            status_update=TaskStatusUpdateEvent(
                task_id="remote-stream",
                context_id="context-stream",
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        ),
    )

    snapshot = A2AProtocolClient._snapshot(completed)
    assert snapshot.remote_task_id == "remote-stream"
    assert snapshot.state == "completed"
    assert snapshot.terminal is True
    assert [part["text"] for part in snapshot.artifacts[0].parts] == ["first", "second"]

    with pytest.raises(A2AProtocolError) as captured:
        A2AProtocolClient._apply_stream_event(
            completed,
            StreamResponse(
                status_update=TaskStatusUpdateEvent(
                    task_id="remote-other",
                    context_id="context-stream",
                    status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
                )
            ),
        )
    assert captured.value.kind == A2AFailureKind.INVALID_RESPONSE
