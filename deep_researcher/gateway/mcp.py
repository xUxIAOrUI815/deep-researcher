from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal

import httpx
from mcp import ClientSession, StdioServerParameters, types as mcp_types
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client
from mcp.server import Server
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import McpError
from pydantic import AnyUrl
from starlette.applications import Starlette
from starlette.routing import Route

from deep_researcher.contracts import Command, CommandKind, ErrorCategory, ErrorRecord

from .gateway import ProtocolToolGateway, redact_gateway_value
from .models import (
    MCP_PROTOCOL_VERSION,
    CancellationSignal,
    ToolAdapterResult,
    ToolDefinition,
    ToolErrorKind,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
)
from .registry import ToolRegistry


class MCPProtocolError(RuntimeError):
    pass


class MCPRequestCancelled(MCPProtocolError):
    pass


class MCPRequestTimeout(MCPProtocolError):
    pass


@dataclass(frozen=True)
class MCPClientConfig:
    transport: Literal["stdio", "streamable_http"]
    client_name: str = "deep-researcher"
    client_version: str = "1.0.0"
    protocol_version: str = MCP_PROTOCOL_VERSION
    timeout_seconds: float = 30.0
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    cwd: str | Path | None = None
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    terminate_on_close: bool = True

    def __post_init__(self) -> None:
        if self.timeout_seconds <= 0:
            raise ValueError("MCP timeout must be positive")
        if self.protocol_version != MCP_PROTOCOL_VERSION:
            raise ValueError(f"MCP protocol must be pinned to {MCP_PROTOCOL_VERSION}")
        if self.transport == "stdio" and not self.command:
            raise ValueError("stdio MCP transport requires a command")
        if self.transport == "streamable_http" and not self.url:
            raise ValueError("streamable HTTP MCP transport requires a URL")


@dataclass(frozen=True)
class MCPDiscovery:
    protocol_version: str
    server_name: str
    server_version: str
    instructions: str
    capabilities: dict[str, Any]
    tools: tuple[mcp_types.Tool, ...]
    resources: tuple[mcp_types.Resource, ...]
    prompts: tuple[mcp_types.Prompt, ...]


class MCPProtocolClient:
    """Official-SDK MCP client with strict lifecycle and version negotiation."""

    def __init__(self, config: MCPClientConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config
        self._provided_http_client = http_client
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._initialize_result: mcp_types.InitializeResult | None = None
        self._session_id_getter: Callable[[], str | None] | None = None
        self.health_status = ToolHealthStatus.CLOSED
        self.last_error: str | None = None

    @property
    def connected(self) -> bool:
        return self._session is not None

    @property
    def session_id(self) -> str | None:
        return self._session_id_getter() if self._session_id_getter else None

    async def connect(self) -> mcp_types.InitializeResult:
        if self._session is not None and self._initialize_result is not None:
            return self._initialize_result
        stack = AsyncExitStack()
        try:
            if self.config.transport == "stdio":
                parameters = StdioServerParameters(
                    command=str(self.config.command),
                    args=list(self.config.args),
                    env=self.config.env,
                    cwd=str(self.config.cwd) if self.config.cwd is not None else None,
                )
                read_stream, write_stream = await stack.enter_async_context(stdio_client(parameters))
            else:
                client = self._provided_http_client
                if client is None:
                    client = await stack.enter_async_context(
                        httpx.AsyncClient(headers=self.config.headers, timeout=self.config.timeout_seconds)
                    )
                read_stream, write_stream, self._session_id_getter = await stack.enter_async_context(
                    streamable_http_client(
                        str(self.config.url),
                        http_client=client,
                        terminate_on_close=self.config.terminate_on_close,
                    )
                )
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
                    client_info=mcp_types.Implementation(name=self.config.client_name, version=self.config.client_version),
                )
            )
            initialized = await self._timeout(session.initialize())
            if initialized.protocolVersion != self.config.protocol_version:
                raise MCPProtocolError(
                    f"MCP version negotiation returned {initialized.protocolVersion}; required {self.config.protocol_version}"
                )
        except Exception as exc:
            self.health_status = ToolHealthStatus.UNHEALTHY
            self.last_error = f"{type(exc).__name__}: {exc}"
            await stack.aclose()
            raise
        self._stack = stack
        self._session = session
        self._initialize_result = initialized
        self.health_status = ToolHealthStatus.HEALTHY
        self.last_error = None
        return initialized

    async def close(self) -> None:
        stack, self._stack = self._stack, None
        self._session = None
        self._initialize_result = None
        self._session_id_getter = None
        if stack is not None:
            await stack.aclose()
        self.health_status = ToolHealthStatus.CLOSED

    async def __aenter__(self) -> "MCPProtocolClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()

    async def discover(self) -> MCPDiscovery:
        initialized = await self.connect()
        tools: list[mcp_types.Tool] = []
        resources: list[mcp_types.Resource] = []
        prompts: list[mcp_types.Prompt] = []
        cursor: str | None = None
        while True:
            page = await self._timeout(self._require_session().list_tools(cursor=cursor))
            tools.extend(page.tools)
            cursor = page.nextCursor
            if not cursor:
                break
        if initialized.capabilities.resources is not None:
            cursor = None
            while True:
                page = await self._timeout(self._require_session().list_resources(cursor=cursor))
                resources.extend(page.resources)
                cursor = page.nextCursor
                if not cursor:
                    break
        if initialized.capabilities.prompts is not None:
            cursor = None
            while True:
                page = await self._timeout(self._require_session().list_prompts(cursor=cursor))
                prompts.extend(page.prompts)
                cursor = page.nextCursor
                if not cursor:
                    break
        return MCPDiscovery(
            protocol_version=initialized.protocolVersion,
            server_name=initialized.serverInfo.name,
            server_version=initialized.serverInfo.version,
            instructions=initialized.instructions or "",
            capabilities=initialized.capabilities.model_dump(mode="json", by_alias=True),
            tools=tuple(tools),
            resources=tuple(resources),
            prompts=tuple(prompts),
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        cancellation: CancellationSignal | None = None,
    ) -> mcp_types.CallToolResult:
        return await self._controlled(
            self._require_session().call_tool(
                name,
                arguments,
                read_timeout_seconds=timedelta(seconds=self.config.timeout_seconds),
            ),
            cancellation=cancellation,
        )

    async def read_resource(
        self,
        uri: str,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> mcp_types.ReadResourceResult:
        return await self._controlled(self._require_session().read_resource(AnyUrl(uri)), cancellation=cancellation)

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
        *,
        cancellation: CancellationSignal | None = None,
    ) -> mcp_types.GetPromptResult:
        return await self._controlled(self._require_session().get_prompt(name, arguments), cancellation=cancellation)

    async def ping(self) -> ToolHealthStatus:
        try:
            await self._timeout(self._require_session().send_ping())
        except Exception as exc:
            self.health_status = ToolHealthStatus.UNHEALTHY
            self.last_error = f"{type(exc).__name__}: {exc}"
        else:
            self.health_status = ToolHealthStatus.HEALTHY
            self.last_error = None
        return self.health_status

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise MCPProtocolError("MCP client is not connected")
        return self._session

    async def _timeout(self, awaitable: Awaitable[Any]) -> Any:
        try:
            return await __import__("asyncio").wait_for(awaitable, timeout=self.config.timeout_seconds)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            if isinstance(exc, (TimeoutError, httpx.TimeoutException)):
                self.health_status = ToolHealthStatus.DEGRADED
            raise

    async def _controlled(self, awaitable: Awaitable[Any], *, cancellation: CancellationSignal | None) -> Any:
        if cancellation is not None and cancellation.cancelled:
            if hasattr(awaitable, "close"):
                awaitable.close()
            raise MCPRequestCancelled("MCP request was cancelled")
        operation = asyncio.ensure_future(awaitable)
        cancellation_wait = asyncio.create_task(cancellation.wait()) if cancellation is not None else None
        waiters = {operation, *({cancellation_wait} if cancellation_wait else set())}
        try:
            done, _ = await asyncio.wait(
                waiters,
                timeout=self.config.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancellation_wait is not None and cancellation_wait in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                raise MCPRequestCancelled("MCP request was cancelled")
            if operation not in done:
                operation.cancel()
                with suppress(asyncio.CancelledError):
                    await operation
                self.health_status = ToolHealthStatus.DEGRADED
                raise MCPRequestTimeout(f"MCP request timed out after {self.config.timeout_seconds:.3f}s")
            return await operation
        except MCPRequestCancelled:
            raise
        except MCPRequestTimeout as exc:
            self.last_error = str(exc)
            raise
        except Exception as exc:
            if isinstance(exc, McpError) and exc.error.code == httpx.codes.REQUEST_TIMEOUT:
                timeout = MCPRequestTimeout(str(exc))
                self.health_status = ToolHealthStatus.DEGRADED
                self.last_error = str(timeout)
                raise timeout from exc
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if cancellation_wait is not None:
                cancellation_wait.cancel()
                with suppress(asyncio.CancelledError):
                    await cancellation_wait


class MCPRemoteToolAdapter:
    def __init__(self, client: MCPProtocolClient, remote_name: str) -> None:
        self.client = client
        self.remote_name = remote_name

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        try:
            result = await self.client.call_tool(self.remote_name, arguments, cancellation=context.cancellation)
        except Exception as exc:
            cancelled = isinstance(exc, MCPRequestCancelled)
            retryable = isinstance(exc, (MCPRequestTimeout, TimeoutError, httpx.TimeoutException, httpx.ConnectError))
            return ToolAdapterResult(
                success=False,
                error=ErrorRecord(
                    category=ErrorCategory.CANCELLED if cancelled else (ErrorCategory.TRANSIENT_PROVIDER if retryable else ErrorCategory.PROTOCOL),
                    code="mcp_cancelled" if cancelled else ("mcp_timeout" if retryable else "mcp_call_failed"),
                    message=f"{type(exc).__name__}: {exc}"[:2000],
                    retryable=retryable,
                    fatal=cancelled or not retryable,
                ),
                retryable=retryable,
            )
        data: Any = result.structuredContent
        if data is None:
            data = {
                "content": [item.model_dump(mode="json", by_alias=True) for item in result.content]
            }
        if result.isError:
            message = "MCP tool returned an error result"
            if result.content and isinstance(result.content[0], mcp_types.TextContent):
                message = result.content[0].text
            return ToolAdapterResult(
                success=False,
                data=data,
                error=ErrorRecord(
                    category=ErrorCategory.PROTOCOL,
                    code="mcp_tool_error",
                    message=message[:2000],
                    retryable=False,
                    fatal=True,
                ),
            )
        return ToolAdapterResult(success=True, data=data, metadata={"mcp_session_id": self.client.session_id})

    async def health(self) -> ToolHealthStatus:
        return await self.client.ping()


async def register_discovered_mcp_tools(
    client: MCPProtocolClient,
    registry: ToolRegistry,
    *,
    namespace: str = "",
    version: str | None = None,
) -> tuple[ToolDefinition, ...]:
    discovery = await client.discover()
    selected_version = version or discovery.server_version
    if not _is_semver(selected_version):
        raise MCPProtocolError("MCP server version must be semantic or an explicit semantic tool version must be provided")
    definitions: list[ToolDefinition] = []
    for tool in discovery.tools:
        name = f"{namespace}.{tool.name}" if namespace else tool.name
        definition = ToolDefinition(
            name=name,
            version=selected_version,
            description=tool.description or f"MCP tool {tool.name}",
            input_schema=tool.inputSchema,
            output_schema=tool.outputSchema,
            operations=(name,),
            protocol=ToolProtocol.MCP,
            provider=discovery.server_name,
            metadata={
                "remote_name": tool.name,
                "mcp_protocol_version": discovery.protocol_version,
                "server_version": discovery.server_version,
            },
        )
        registry.register(definition, MCPRemoteToolAdapter(client, tool.name))
        definitions.append(definition)
    return tuple(definitions)


@dataclass(frozen=True)
class MCPResourceRegistration:
    uri: str
    name: str
    description: str
    mime_type: str
    reader: Callable[[], Awaitable[str | bytes]]


@dataclass(frozen=True)
class MCPPromptRegistration:
    name: str
    description: str
    arguments: tuple[mcp_types.PromptArgument, ...]
    renderer: Callable[[dict[str, str]], Awaitable[mcp_types.GetPromptResult]]


class GatewayMCPHost:
    """Low-level MCP server exposing the governed ToolRegistry and catalogs."""

    def __init__(
        self,
        *,
        gateway: ProtocolToolGateway,
        registry: ToolRegistry,
        name: str = "deep-researcher-gateway",
        version: str = "1.0.0",
        context_factory: Callable[[str, dict[str, Any]], ToolInvocationContext] | None = None,
    ) -> None:
        self.gateway = gateway
        self.registry = registry
        self.name = name
        self.version = version
        self.context_factory = context_factory or (lambda name, arguments: ToolInvocationContext())
        self.server = Server(name, version=version, instructions="Governed tools, resources, and prompts.")
        self._resources: dict[str, MCPResourceRegistration] = {}
        self._prompts: dict[str, MCPPromptRegistration] = {}
        self._session_manager: StreamableHTTPSessionManager | None = None
        self._install_handlers()

    def register_resource(self, registration: MCPResourceRegistration) -> None:
        if registration.uri in self._resources:
            raise ValueError(f"duplicate MCP resource URI: {registration.uri}")
        self._resources[registration.uri] = registration

    def register_prompt(self, registration: MCPPromptRegistration) -> None:
        if registration.name in self._prompts:
            raise ValueError(f"duplicate MCP prompt: {registration.name}")
        self._prompts[registration.name] = registration

    def _install_handlers(self) -> None:
        @self.server.list_tools()
        async def list_tools() -> list[mcp_types.Tool]:
            return [
                mcp_types.Tool(
                    name=item.name,
                    description=item.description,
                    inputSchema=item.input_schema,
                    outputSchema=item.output_schema,
                    _meta={
                        "toolVersion": item.version,
                        "riskLevel": item.risk_level.value,
                        "requiresApproval": item.requires_approval,
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                    },
                )
                for item in self.registry.definitions()
            ]

        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
            definition, _ = self.registry.resolve(name)
            semantic = json.dumps([name, arguments], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            digest = hashlib.sha256(semantic.encode()).hexdigest()
            context = self.context_factory(name, arguments)
            command = Command(
                command_id=f"command_mcp_{digest[:24]}",
                run_id=str(context.metadata.get("run_id") or f"run_mcp_{digest[:16]}"),
                task_id=str(context.metadata.get("task_id") or f"task_mcp_{digest[:16]}"),
                actor_id=str(context.metadata.get("actor_id") or "agent_mcp_client"),
                kind=CommandKind.TOOL,
                name=name,
                arguments=arguments,
                expected_output_schema="ToolExecutionResult@1",
                idempotency_key=f"mcp-{digest}",
                requires_approval=definition.requires_approval,
                risk_level=definition.risk_level.value,
                metadata={"tool_version": definition.version, "protocol": "mcp"},
            )
            result = await self.gateway.execute_tool(command, context)
            normalized = result.data if isinstance(result.data, dict) else {"value": result.data}
            structured = redact_gateway_value(normalized) if result.success else {
                "error": result.error.model_dump(mode="json") if result.error else {"message": "tool failed"}
            }
            text = json.dumps(structured, ensure_ascii=False, sort_keys=True)
            return mcp_types.CallToolResult(
                content=[mcp_types.TextContent(type="text", text=text)],
                structuredContent=structured,
                isError=not result.success,
                _meta={
                    "gateway": {
                        "outputArtifactIds": list(result.output_artifact_ids),
                        "usage": result.usage.model_dump(mode="json"),
                        "toolVersion": result.tool_version,
                    }
                },
            )

        @self.server.list_resources()
        async def list_resources() -> list[mcp_types.Resource]:
            return [
                mcp_types.Resource(
                    uri=registration.uri,
                    name=registration.name,
                    description=registration.description,
                    mimeType=registration.mime_type,
                )
                for registration in self._resources.values()
            ]

        @self.server.read_resource()
        async def read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
            registration = self._resources.get(str(uri))
            if registration is None:
                raise ValueError(f"unknown MCP resource: {uri}")
            return [ReadResourceContents(await registration.reader(), registration.mime_type)]

        @self.server.list_prompts()
        async def list_prompts() -> list[mcp_types.Prompt]:
            return [
                mcp_types.Prompt(
                    name=registration.name,
                    description=registration.description,
                    arguments=list(registration.arguments),
                )
                for registration in self._prompts.values()
            ]

        @self.server.get_prompt()
        async def get_prompt(name: str, arguments: dict[str, str] | None) -> mcp_types.GetPromptResult:
            registration = self._prompts.get(name)
            if registration is None:
                raise ValueError(f"unknown MCP prompt: {name}")
            values = arguments or {}
            missing = [item.name for item in registration.arguments if item.required and item.name not in values]
            if missing:
                raise ValueError(f"missing prompt arguments: {missing}")
            return await registration.renderer(values)

    async def run_stdio(self) -> None:
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                self.server.create_initialization_options(),
            )

    def streamable_http_app(self, *, path: str = "/mcp", stateless: bool = False) -> Starlette:
        if self._session_manager is not None:
            raise RuntimeError("streamable HTTP app can only be created once per MCP host")
        manager = StreamableHTTPSessionManager(
            app=self.server,
            json_response=False,
            stateless=stateless,
        )
        self._session_manager = manager
        endpoint = StreamableHTTPASGIApp(manager)
        return Starlette(routes=[Route(path, endpoint=endpoint)], lifespan=lambda app: manager.run())


def _is_semver(value: str) -> bool:
    import re

    return re.fullmatch(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-[0-9A-Za-z.-]+)?", value) is not None
