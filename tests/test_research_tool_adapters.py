from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from deep_researcher.contracts import Command, CommandKind
from deep_researcher.gateway import (
    ToolAdapterError,
    ToolHealthStatus,
    ToolInvocationContext,
)
from deep_researcher.providers import (
    ExaSearchToolAdapter,
    MockScraper,
    SEARCH_TOOL,
    TavilySearchToolAdapter,
    build_governed_research_tools,
)
from deep_researcher.providers.research_tools import (
    READ_TOOL,
    SEARCH_FALLBACK_TOOL,
)


def _context(permission: str = "web:search") -> ToolInvocationContext:
    return ToolInvocationContext(permissions=frozenset({permission}))


def _command(
    suffix: str,
    *,
    name: str,
    kind: CommandKind,
    arguments: dict,
) -> Command:
    return Command(
        command_id=f"command_{suffix}",
        run_id="run_provider",
        task_id="task_provider",
        actor_id="agent_research_worker",
        kind=kind,
        name=name,
        arguments=arguments,
        expected_output_schema="Observation@1",
        idempotency_key=f"provider-{suffix}",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "variable"),
    [
        (TavilySearchToolAdapter(api_key=""), "TAVILY_API_KEY"),
        (ExaSearchToolAdapter(api_key=""), "EXA_API_KEY"),
    ],
)
async def test_live_search_adapters_require_explicit_credentials(
    adapter,
    variable,
):
    with pytest.raises(ToolAdapterError, match=variable) as captured:
        await adapter.execute(
            {"query": "transformer", "max_results": 3},
            _context(),
        )
    assert captured.value.retryable is False
    assert await adapter.health() == ToolHealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_tavily_adapter_uses_bearer_auth_and_normalizes_response():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "url": "https://example.org/transformer",
                        "title": "Transformer overview",
                        "content": "Transformer overview content.",
                        "raw_content": "Complete content.",
                        "score": 0.9,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        result = await TavilySearchToolAdapter(
            api_key="tvly-test-key",
            http_client=client,
        ).execute(
            {"query": "transformer", "max_results": 3},
            _context(),
        )

    assert result.success is True
    assert result.data["items"][0]["url"] == (
        "https://example.org/transformer"
    )
    assert calls[0].headers["Authorization"] == "Bearer tvly-test-key"
    assert b'"api_key"' not in calls[0].content
    assert b'"max_results":3' in calls[0].content


@pytest.mark.asyncio
async def test_exa_adapter_uses_key_header_and_normalizes_response():
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "url": "https://example.org/exa",
                        "title": "Exa result",
                        "text": "Full Exa content.",
                        "score": 0.8,
                    }
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        result = await ExaSearchToolAdapter(
            api_key="exa-test-key",
            http_client=client,
        ).execute(
            {"query": "evidence", "max_results": 2},
            _context(),
        )

    assert result.data["items"][0]["snippet"] == "Full Exa content."
    assert calls[0].headers["x-api-key"] == "exa-test-key"
    assert b'"num_results":2' in calls[0].content


@pytest.mark.asyncio
async def test_production_registry_governs_search_fallback_and_scraping(
    tmp_path: Path,
):
    def exa_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "url": "https://example.org/fallback",
                        "title": "Fallback",
                        "text": "Verified fallback content.",
                        "score": 0.7,
                    }
                ]
            },
        )

    fixtures = {
        "https://example.org/page": {
            "title": "Fixture page",
            "markdown": "# Fixture\n\nA complete fixture body.",
        }
    }
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(exa_handler)
    ) as client:
        tools = build_governed_research_tools(
            tmp_path,
            tavily=TavilySearchToolAdapter(api_key=""),
            exa=ExaSearchToolAdapter(
                api_key="exa-test",
                http_client=client,
            ),
            scraper=MockScraper(fixtures),
            retry_base_seconds=0,
        )
        try:
            search = await tools.gateway.execute(
                _command(
                    "search",
                    name=SEARCH_TOOL,
                    kind=CommandKind.SEARCH,
                    arguments={
                        "operation": "search",
                        "query": "fallback query",
                        "max_results": 3,
                    },
                )
            )
            read = await tools.gateway.execute(
                _command(
                    "read",
                    name=READ_TOOL,
                    kind=CommandKind.READ,
                    arguments={
                        "operation": "read",
                        "urls": ["https://example.org/page"],
                        "source_context": {},
                        "force_playwright": False,
                    },
                )
            )
            definitions = {
                item.name: item for item in tools.registry.definitions()
            }
        finally:
            tools.close()

    assert search.status.value == "succeeded"
    assert search.normalized_data["sources"][0]["url"] == (
        "https://example.org/fallback"
    )
    assert search.normalized_data["_tool"]["fallback_chain"] == [
        f"{SEARCH_FALLBACK_TOOL}@1.0.0"
    ]
    assert read.status.value == "succeeded"
    assert read.normalized_data["scraped_data_cache"][0]["title"] == (
        "Fixture page"
    )
    assert definitions[SEARCH_TOOL].fallback_tools == (
        SEARCH_FALLBACK_TOOL,
    )
    assert definitions[READ_TOOL].permission_scopes == ("web:read",)
    assert all(
        item.protocol.value == "native" for item in definitions.values()
    )


@pytest.mark.asyncio
async def test_provider_protocol_error_is_structured_and_not_retryable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            content=b"not-json",
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        adapter = TavilySearchToolAdapter(
            api_key="tvly-test",
            http_client=client,
        )
        with pytest.raises(ToolAdapterError) as captured:
            await adapter.execute(
                {"query": "broken", "max_results": 2},
                _context(),
            )
    assert captured.value.kind.value == "protocol"
    assert captured.value.retryable is False
