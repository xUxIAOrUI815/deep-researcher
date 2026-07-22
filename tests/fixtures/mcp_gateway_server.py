from __future__ import annotations

import asyncio
import os

from mcp import types as mcp_types

from deep_researcher.gateway import (
    GatewayMCPHost,
    MCPPromptRegistration,
    MCPResourceRegistration,
    ProtocolToolGateway,
    RateLimit,
    SQLiteToolStateStore,
    ToolAdapterResult,
    ToolDefinition,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
    ToolRegistry,
)


class EchoAdapter:
    async def execute(self, arguments: dict, context: ToolInvocationContext) -> ToolAdapterResult:
        return ToolAdapterResult(success=True, data={"echo": arguments["message"]})

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


async def main() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            version="1.0.0",
            description="Echo a message through the governed MCP host.",
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
            provider="fixture",
        ),
        EchoAdapter(),
    )
    store = SQLiteToolStateStore(os.environ["MCP_FIXTURE_STATE"])
    gateway = ProtocolToolGateway(registry=registry, state_store=store, retry_base_seconds=0)
    host = GatewayMCPHost(gateway=gateway, registry=registry, name="fixture-mcp", version="1.0.0")

    async def read_guide() -> str:
        return "Fixture MCP guide"

    async def render_prompt(arguments: dict[str, str]) -> mcp_types.GetPromptResult:
        return mcp_types.GetPromptResult(
            description="Fixture prompt",
            messages=[
                mcp_types.PromptMessage(
                    role="user",
                    content=mcp_types.TextContent(type="text", text=f"Research {arguments['topic']}"),
                )
            ],
        )

    host.register_resource(
        MCPResourceRegistration(
            uri="memory://guide",
            name="guide",
            description="Fixture resource",
            mime_type="text/plain",
            reader=read_guide,
        )
    )
    host.register_prompt(
        MCPPromptRegistration(
            name="research",
            description="Fixture research prompt",
            arguments=(mcp_types.PromptArgument(name="topic", required=True),),
            renderer=render_prompt,
        )
    )
    try:
        await host.run_stdio()
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
