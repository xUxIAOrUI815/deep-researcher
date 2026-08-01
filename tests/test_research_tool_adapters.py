from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

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
    SmartScraper,
    build_governed_research_tools,
)
from deep_researcher.providers.research_tools import (
    EXTRACT_TOOL,
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
async def test_live_scraper_failure_uses_explicit_capture_metadata(
    monkeypatch,
):
    scraper = SmartScraper()

    async def fail_scrape(url: str, force_playwright: bool = False):
        del url, force_playwright
        raise RuntimeError("bounded scrape failure")

    monkeypatch.setattr(scraper, "scrape", fail_scrape)
    documents = await scraper.scrape_batch(["https://example.org/paper"])

    assert len(documents) == 1
    assert documents[0].error == "bounded scrape failure"
    captured_at = datetime.fromisoformat(
        str(documents[0].metadata["captured_at"])
    )
    assert captured_at.tzinfo is not None


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
async def test_large_tavily_results_are_bounded_before_gateway_safety_scan(
    tmp_path: Path,
    monkeypatch,
):
    monkeypatch.delenv("EXA_API_KEY", raising=False)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            request=request,
            json={
                "results": [
                    {
                        "url": f"https://example.org/paper-{index}",
                        "title": f"DRAGIN:\f Paper {index}",
                        "content": "RAG-TP: Relevant paper; doi:10.1234/example.",
                        "raw_content": "\f" + "x" * 120_000,
                        "score": 0.9,
                    }
                    for index in range(10)
                ]
            },
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    ) as client:
        tools = build_governed_research_tools(
            tmp_path,
            tavily=TavilySearchToolAdapter(
                api_key="tvly-test-key",
                http_client=client,
            ),
            retry_base_seconds=0,
        )
        try:
            result = await tools.gateway.execute(
                _command(
                    "large-search",
                    name=SEARCH_TOOL,
                    kind=CommandKind.SEARCH,
                    arguments={
                        "operation": "search",
                        "query": "large papers",
                        "max_results": 10,
                    },
                )
            )
            definition = tools.registry.resolve(SEARCH_TOOL)[0]
        finally:
            tools.close()

    assert result.status.value == "succeeded"
    assert len(json.dumps(result.normalized_data).encode()) < 1_000_000
    assert all(
        "raw_content" not in item
        for item in result.normalized_data["items"]
    )
    assert all(
        len(item["text"]) <= 8_000
        for item in result.normalized_data["passages"]
    )
    assert sum(
        len(item["text"])
        for item in result.normalized_data["passages"]
    ) <= 40_000
    assert all(
        not item["markdown"]
        for item in result.normalized_data["scraped_data_cache"]
    )
    assert "\f" not in repr(result.normalized_data)
    assert definition.fallback_tools == ()


@pytest.mark.asyncio
async def test_extract_schema_rejects_empty_facts_before_domain_ingestion(
    tmp_path: Path,
):
    tools = build_governed_research_tools(
        tmp_path,
        scraper=MockScraper(),
        retry_base_seconds=0,
    )
    base = {
        "operation": "extract",
        "section_id": "section_findings",
        "evidence": [
            {
                "id": "evidence_one",
                "source_id": "source_result_0",
                "quote": "Exact persisted quote.",
            }
        ],
        "atomic_facts": [
            {
                "id": "fact_one",
                "source_id": "source_result_0",
                "statement": "A grounded fact.",
            }
        ],
        "claims": [
            {
                "id": "claim_one",
                "statement": "A grounded claim.",
                "fact_ids": ["fact_one"],
            }
        ],
        "conflicts": [],
    }
    try:
        invalid = await tools.gateway.execute(
            _command(
                "invalid-extract",
                name=EXTRACT_TOOL,
                kind=CommandKind.EXTRACT,
                arguments={
                    **base,
                    "atomic_facts": [
                        {
                            "id": "fact_one",
                            "source_id": "source_result_0",
                            "statement": "",
                        }
                    ],
                },
            )
        )
        valid = await tools.gateway.execute(
            _command(
                "valid-extract",
                name=EXTRACT_TOOL,
                kind=CommandKind.EXTRACT,
                arguments=base,
            )
        )
    finally:
        tools.close()

    assert invalid.status.value == "failed"
    assert invalid.error is not None
    assert invalid.error.category.value == "schema_validation"
    assert valid.status.value == "succeeded"


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
            "markdown": "# Fixture\n\nA complete\f fixture body.",
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
    assert "\f" not in read.normalized_data["passages"][0]["text"]
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
