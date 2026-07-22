from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from deep_researcher.gateway import ToolAdapterError, ToolHealthStatus, ToolInvocationContext
from providers import (
    EXA_TOOL,
    SCRAPER_TOOL,
    TAVILY_TOOL,
    ExaSearchToolAdapter,
    MockScraper,
    ResearchToolGateway,
    ResearchToolGatewayError,
    ScraperToolAdapter,
    TavilySearchToolAdapter,
)
from schemas.state import ScrapedData


def _context(permission: str = "web:search") -> ToolInvocationContext:
    return ToolInvocationContext(permissions=frozenset({permission}))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("adapter", "variable"),
    [
        (TavilySearchToolAdapter(api_key=""), "TAVILY_API_KEY"),
        (ExaSearchToolAdapter(api_key=""), "EXA_API_KEY"),
    ],
)
async def test_live_search_adapters_require_real_credentials_and_never_return_demo_data(adapter, variable):
    with pytest.raises(ToolAdapterError, match=variable) as captured:
        await adapter.execute({"query": "transformer", "max_results": 3}, _context())
    assert captured.value.retryable is False
    assert await adapter.health() == ToolHealthStatus.UNHEALTHY


@pytest.mark.asyncio
async def test_tavily_adapter_uses_bearer_auth_and_normalizes_provider_response():
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await TavilySearchToolAdapter(api_key="tvly-test-key", http_client=client).execute(
            {"query": "transformer", "max_results": 3},
            _context(),
        )

    assert result.success is True
    assert result.data["items"][0]["url"] == "https://example.org/transformer"
    assert calls[0].headers["Authorization"] == "Bearer tvly-test-key"
    assert b'"api_key"' not in calls[0].content
    assert b'"max_results":3' in calls[0].content


@pytest.mark.asyncio
async def test_exa_adapter_uses_api_key_header_and_normalizes_provider_response():
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

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        result = await ExaSearchToolAdapter(api_key="exa-test-key", http_client=client).execute(
            {"query": "evidence", "max_results": 2},
            _context(),
        )

    assert result.data["items"][0]["snippet"] == "Full Exa content."
    assert calls[0].headers["x-api-key"] == "exa-test-key"
    assert b'"num_results":2' in calls[0].content


@pytest.mark.asyncio
async def test_research_gateway_owns_retry_and_emits_retry_callback(tmp_path: Path):
    calls = 0
    retries: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ReadTimeout("temporary timeout", request=request)
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"url": "https://example.org/recovered", "title": "Recovered", "content": "Recovered content", "score": 0.7}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ResearchToolGateway(
            state_path=tmp_path / "state.sqlite3",
            tavily=TavilySearchToolAdapter(api_key="tvly-test", http_client=client),
            exa=ExaSearchToolAdapter(api_key=""),
            scraper=MockScraper(),
            retry_base_seconds=0,
        )
        results = await gateway.search("transformer", max_results=3, on_retry=retries.append)
        gateway.close()

    assert calls == 2
    assert len(retries) == 1
    assert retries[0]["attempt"] == 2
    assert results[0].url == "https://example.org/recovered"


@pytest.mark.asyncio
async def test_research_gateway_falls_back_from_unconfigured_tavily_to_exa(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={"results": [{"url": "https://example.org/fallback", "title": "Fallback", "text": "Exa fallback", "score": 0.6}]},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ResearchToolGateway(
            state_path=tmp_path / "state.sqlite3",
            tavily=TavilySearchToolAdapter(api_key=""),
            exa=ExaSearchToolAdapter(api_key="exa-test", http_client=client),
            scraper=MockScraper(),
            retry_base_seconds=0,
        )
        results = await gateway.search("fallback query")
        fallback_chain = gateway.last_result.fallback_chain if gateway.last_result else ()
        gateway.close()

    assert results[0].url == "https://example.org/fallback"
    assert fallback_chain == (f"{EXA_TOOL}@1.0.0",)


@pytest.mark.asyncio
async def test_scraper_is_a_governed_tool_adapter_and_round_trips_schema(tmp_path: Path):
    fixtures = {
        "https://example.org/page": {
            "title": "Fixture page",
            "markdown": "# Fixture\n\nA sufficiently complete fixture body.",
        }
    }
    gateway = ResearchToolGateway(
        state_path=tmp_path / "state.sqlite3",
        tavily=TavilySearchToolAdapter(api_key=""),
        exa=ExaSearchToolAdapter(api_key=""),
        scraper=MockScraper(fixtures),
        retry_base_seconds=0,
    )
    values = await gateway.scrape_batch(
        ["https://example.org/page"],
        source_context={"https://example.org/page": {"query": "fixture"}},
    )
    definitions = {item.name: item for item in gateway.definitions()}
    gateway.close()

    assert values[0].title == "Fixture page"
    assert values[0].fetch_method == "mock"
    assert definitions[SCRAPER_TOOL].permission_scopes == ("web:read",)
    assert definitions[TAVILY_TOOL].fallback_tools == (EXA_TOOL,)
    assert all(item.protocol.value == "native" for item in definitions.values())


@pytest.mark.asyncio
async def test_scraper_tool_adapter_delegates_one_batch_without_internal_retry():
    class CountingScraper:
        mode = "test"

        def __init__(self) -> None:
            self.calls = 0

        async def scrape_batch(self, urls, *, source_context=None, force_playwright=False):
            self.calls += 1
            return [ScrapedData(url=urls[0], markdown="body", title="title", fetch_method="unknown")]

    backend = CountingScraper()
    result = await ScraperToolAdapter(backend).execute(
        {"urls": ["https://example.org"], "source_context": {}, "force_playwright": False},
        _context("web:read"),
    )
    assert backend.calls == 1
    assert result.data["items"][0]["url"] == "https://example.org"


@pytest.mark.asyncio
async def test_gateway_exposes_structured_permanent_provider_failure(tmp_path: Path):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, request=request, text="unauthorized")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        gateway = ResearchToolGateway(
            state_path=tmp_path / "state.sqlite3",
            tavily=TavilySearchToolAdapter(api_key=""),
            exa=ExaSearchToolAdapter(api_key="exa-invalid", http_client=client),
            scraper=MockScraper(),
            retry_base_seconds=0,
        )
        with pytest.raises(ResearchToolGatewayError) as captured:
            await gateway.search("transformer", provider="exa")
        gateway.close()

    assert captured.value.retryable is False
    assert captured.value.status_code == 401
    assert captured.value.attempts == 1
    assert "HTTP error: 401" in str(captured.value)
