from __future__ import annotations

import asyncio
import os
from pathlib import Path
import socket
import sys
from typing import Any

from mcp import types as mcp_types
import pytest
import uvicorn

from deep_researcher.contracts import Command, CommandKind
from deep_researcher.gateway import (
    GatewayMCPHost,
    MCPClientConfig,
    MCPPromptRegistration,
    MCPProtocolClient,
    MCPProtocolError,
    MCPRequestCancelled,
    MCPRequestTimeout,
    MCPResourceRegistration,
    MCP_PROTOCOL_VERSION,
    ProtocolToolGateway,
    RateLimit,
    SQLiteToolStateStore,
    ToolAdapterResult,
    ToolDefinition,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
    ToolRegistry,
    register_discovered_mcp_tools,
)
from deep_researcher.kernel import CancellationToken


class EchoAdapter:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        return ToolAdapterResult(success=True, data={"echo": arguments["message"]})

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


def _definition() -> ToolDefinition:
    return ToolDefinition(
        name="echo",
        version="1.0.0",
        description="Echo a message.",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
        output_schema={
            "type": "object",
            "properties": {"echo": {"type": "string"}},
            "required": ["echo"],
            "additionalProperties": False,
        },
        operations=("echo",),
        max_attempts=1,
        rate_limit=RateLimit(calls=100, window_seconds=60),
        protocol=ToolProtocol.MCP,
        provider="test-mcp",
    )


def _build_host(tmp_path: Path, *, delay: float = 0.0):
    registry = ToolRegistry()
    adapter = EchoAdapter(delay=delay)
    registry.register(_definition(), adapter)
    store = SQLiteToolStateStore(tmp_path / "host-state.sqlite3")
    gateway = ProtocolToolGateway(registry=registry, state_store=store, retry_base_seconds=0)
    host = GatewayMCPHost(gateway=gateway, registry=registry, name="test-mcp-host", version="1.0.0")

    async def resource() -> str:
        return "MCP resource body"

    async def prompt(arguments: dict[str, str]) -> mcp_types.GetPromptResult:
        return mcp_types.GetPromptResult(
            description="Research prompt",
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=f"Research {arguments['topic']}"),
                )
            ],
        )

    host.register_resource(MCPResourceRegistration("memory://guide", "guide", "Guide", "text/plain", resource))
    host.register_prompt(
        MCPPromptRegistration(
            "research",
            "Research a topic",
            (mcp_types.PromptArgument(name="topic", required=True),),
            prompt,
        )
    )
    return host, adapter, store


@pytest.mark.asyncio
async def test_real_mcp_stdio_lifecycle_discovery_tools_resources_prompts_and_remote_adapter(tmp_path):
    fixture = Path(__file__).parent / "fixtures" / "mcp_gateway_server.py"
    repository = Path(__file__).resolve().parents[1]
    python_path = str(repository)
    if os.environ.get("PYTHONPATH"):
        python_path = f"{python_path}{os.pathsep}{os.environ['PYTHONPATH']}"
    env = {
        **os.environ,
        "MCP_FIXTURE_STATE": str(tmp_path / "stdio-server.sqlite3"),
        "PYTHONPATH": python_path,
    }
    config = MCPClientConfig(
        transport="stdio",
        command=sys.executable,
        args=(str(fixture),),
        env=env,
        cwd=repository,
        timeout_seconds=10,
    )
    client = MCPProtocolClient(config)
    async with client:
        discovery = await client.discover()
        assert discovery.protocol_version == MCP_PROTOCOL_VERSION
        assert discovery.server_name == "fixture-mcp"
        assert [tool.name for tool in discovery.tools] == ["echo"]
        assert [resource.name for resource in discovery.resources] == ["guide"]
        assert [prompt.name for prompt in discovery.prompts] == ["research"]

        called = await client.call_tool("echo", {"message": "hello"})
        assert called.isError is False
        assert called.structuredContent == {"echo": "hello"}
        invalid = await client.call_tool("echo", {})
        assert invalid.isError is True
        assert invalid.content
        resource = await client.read_resource("memory://guide")
        assert resource.contents[0].text == "Fixture MCP guide"
        prompt = await client.get_prompt("research", {"topic": "protocols"})
        assert prompt.messages[0].content.text == "Research protocols"
        assert await client.ping() == ToolHealthStatus.HEALTHY

        remote_registry = ToolRegistry()
        definitions = await register_discovered_mcp_tools(client, remote_registry, namespace="remote")
        assert definitions[0].name == "remote.echo"
        assert definitions[0].metadata["mcp_protocol_version"] == MCP_PROTOCOL_VERSION
        with SQLiteToolStateStore(tmp_path / "remote-state.sqlite3") as remote_state:
            gateway = ProtocolToolGateway(registry=remote_registry, state_store=remote_state, retry_base_seconds=0)
            command = Command(
                command_id="command_remote_mcp",
                run_id="run_remote_mcp",
                task_id="task_remote_mcp",
                actor_id="agent_remote_mcp",
                kind=CommandKind.TOOL,
                name="remote.echo",
                arguments={"message": "through adapter"},
                idempotency_key="idempotency-remote-mcp",
                metadata={"tool_version": "1.0.0"},
            )
            result = await gateway.execute_tool(command, ToolInvocationContext())
            assert result.success
            assert result.data == {"echo": "through adapter"}
            assert result.protocol == ToolProtocol.MCP
    assert client.connected is False
    assert client.health_status == ToolHealthStatus.CLOSED


@pytest.mark.asyncio
async def test_real_mcp_streamable_http_lifecycle_and_session_termination(tmp_path):
    host, adapter, store = _build_host(tmp_path)
    app = host.streamable_http_app(path="/mcp")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started
        async with MCPProtocolClient(
            MCPClientConfig(
                transport="streamable_http",
                url=f"http://127.0.0.1:{port}/mcp",
                timeout_seconds=5,
            )
        ) as client:
            discovery = await client.discover()
            assert discovery.protocol_version == MCP_PROTOCOL_VERSION
            assert client.session_id
            result = await client.call_tool("echo", {"message": "http"})
            assert result.structuredContent == {"echo": "http"}
            assert await client.ping() == ToolHealthStatus.HEALTHY
        assert adapter.calls == 1
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        store.close()


@pytest.mark.asyncio
async def test_mcp_client_requires_connection_and_strict_pinned_configuration():
    with pytest.raises(ValueError, match="pinned"):
        MCPClientConfig(transport="stdio", command=sys.executable, protocol_version="2025-06-18")
    with pytest.raises(ValueError, match="requires a command"):
        MCPClientConfig(transport="stdio")
    with pytest.raises(ValueError, match="requires a URL"):
        MCPClientConfig(transport="streamable_http")
    client = MCPProtocolClient(MCPClientConfig(transport="stdio", command=sys.executable))
    with pytest.raises(MCPProtocolError, match="not connected"):
        await client.call_tool("missing", {})


def test_mcp_host_rejects_duplicate_resources_prompts_and_second_http_app(tmp_path):
    host, _, store = _build_host(tmp_path)

    async def resource() -> str:
        return "duplicate"

    async def prompt(arguments: dict[str, str]) -> mcp_types.GetPromptResult:
        return mcp_types.GetPromptResult(messages=[])

    with pytest.raises(ValueError, match="duplicate MCP resource"):
        host.register_resource(MCPResourceRegistration("memory://guide", "other", "Other", "text/plain", resource))
    with pytest.raises(ValueError, match="duplicate MCP prompt"):
        host.register_prompt(MCPPromptRegistration("research", "Other", (), prompt))
    host.streamable_http_app()
    with pytest.raises(RuntimeError, match="only be created once"):
        host.streamable_http_app()
    store.close()


@pytest.mark.asyncio
async def test_streamable_http_mcp_timeout_and_active_cancellation_interrupt_remote_calls(tmp_path):
    host, adapter, store = _build_host(tmp_path, delay=0.7)
    app = host.streamable_http_app(path="/mcp")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    server_task = asyncio.create_task(server.serve())
    try:
        for _ in range(200):
            if server.started:
                break
            await asyncio.sleep(0.01)
        assert server.started

        async with MCPProtocolClient(
            MCPClientConfig(
                transport="streamable_http",
                url=f"http://127.0.0.1:{port}/mcp",
                timeout_seconds=0.2,
            )
        ) as timeout_client:
            with pytest.raises(MCPRequestTimeout):
                await timeout_client.call_tool("echo", {"message": "timeout"})
            assert timeout_client.health_status == ToolHealthStatus.DEGRADED

        token = CancellationToken()
        async with MCPProtocolClient(
            MCPClientConfig(
                transport="streamable_http",
                url=f"http://127.0.0.1:{port}/mcp",
                timeout_seconds=2,
            )
        ) as cancellation_client:
            pending = asyncio.create_task(
                cancellation_client.call_tool("echo", {"message": "cancel"}, cancellation=token)
            )
            await asyncio.sleep(0.03)
            token.cancel()
            with pytest.raises(MCPRequestCancelled):
                await pending
        assert adapter.calls >= 1
    finally:
        server.should_exit = True
        await asyncio.wait_for(server_task, timeout=5)
        store.close()
