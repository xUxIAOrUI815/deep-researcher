from __future__ import annotations

import pytest

import providers.mcp_gateway as mcp_gateway
from providers.mcp_gateway import MCPGateway, MCPToolResult, TavilySearchProvider


@pytest.mark.asyncio
async def test_tavily_live_search_requires_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = await TavilySearchProvider().execute("tavily_search", {"query": "transformer"})

    assert result.success is False
    assert "TAVILY_API_KEY is required" in str(result.error)
    assert "RESEARCHER_SEARCH_MODE=mock" in str(result.error)


@pytest.mark.asyncio
async def test_tavily_search_uses_bearer_header(monkeypatch):
    calls: list[dict] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "url": "https://example.org/transformer",
                        "title": "Transformer overview",
                        "content": "Transformer overview content.",
                        "score": 0.9,
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, headers: dict, json: dict):
            calls.append({"url": url, "headers": headers, "json": json})
            return FakeResponse()

    monkeypatch.setattr(mcp_gateway.httpx, "AsyncClient", FakeClient)

    result = await TavilySearchProvider(api_key="tvly-test-key").execute(
        "tavily_search",
        {"query": "transformer", "max_results": 3},
    )

    assert result.success is True
    assert result.data[0].url == "https://example.org/transformer"
    assert calls[0]["headers"]["Authorization"] == "Bearer tvly-test-key"
    assert "api_key" not in calls[0]["json"]
    assert calls[0]["json"]["query"] == "transformer"
    assert calls[0]["json"]["max_results"] == 3


@pytest.mark.asyncio
async def test_tavily_retries_retryable_timeout(monkeypatch):
    calls: list[int] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {
                "results": [
                    {
                        "url": "https://example.org/retry-success",
                        "title": "Retry success",
                        "content": "Recovered after timeout.",
                        "score": 0.8,
                    }
                ]
            }

    class FakeClient:
        def __init__(self, *, timeout: float):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, url: str, *, headers: dict, json: dict):
            calls.append(1)
            if len(calls) == 1:
                raise mcp_gateway.httpx.ReadTimeout("temporary timeout")
            return FakeResponse()

    async def fake_sleep(delay: float):
        return None

    monkeypatch.setattr(mcp_gateway.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(mcp_gateway.asyncio, "sleep", fake_sleep)

    result = await TavilySearchProvider(api_key="tvly-test-key").execute(
        "tavily_search",
        {"query": "transformer", "max_results": 3},
    )

    assert result.success is True
    assert result.attempts == 2
    assert len(calls) == 2
    assert result.data[0].url == "https://example.org/retry-success"


@pytest.mark.asyncio
async def test_mcp_gateway_search_raises_on_provider_failure():
    class FailingHandler:
        async def execute(self, tool_name: str, parameters: dict) -> MCPToolResult:
            return MCPToolResult(success=False, data=None, error="HTTP error: 401", tool_name=tool_name)

    gateway = MCPGateway()
    gateway._tools["tavily_search"] = FailingHandler()

    with pytest.raises(RuntimeError, match="tavily_search failed: HTTP error: 401"):
        await gateway.search("transformer", provider="tavily")
