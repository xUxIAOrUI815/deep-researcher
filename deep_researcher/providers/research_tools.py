from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deep_researcher.contracts import (
    BudgetUsage,
    Command,
)
from deep_researcher.gateway import (
    CircuitBreakerPolicy,
    ProtocolToolGateway,
    RateLimit,
    SQLiteToolStateStore,
    ToolAdapterResult,
    ToolDefinition,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
    ToolRegistry,
    ToolRiskLevel,
)

from .scraper_backend import ScraperInterface, build_scraper
from .tool_gateway import ExaSearchToolAdapter, TavilySearchToolAdapter


SEARCH_TOOL = "research.search"
SEARCH_FALLBACK_TOOL = "research.search_fallback"
READ_TOOL = "research.read"
EXTRACT_TOOL = "research.extract"
COMPARE_TOOL = "research.compare"
VERIFY_SOURCE_TOOL = "research.verify_source"
WORKER_TOOL_NAMES = (
    SEARCH_TOOL,
    READ_TOOL,
    EXTRACT_TOOL,
    COMPARE_TOOL,
    VERIFY_SOURCE_TOOL,
)


class _SearchAdapter:
    def __init__(self, provider: TavilySearchToolAdapter | ExaSearchToolAdapter):
        self.provider = provider

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        result = await self.provider.execute(arguments, context)
        items = list((result.data or {}).get("items", []))
        sources = [
            {
                "source_id": f"source_result_{index}",
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "score": item.get("score", 0.0),
                "query": arguments["query"],
                "source_type": "other",
            }
            for index, item in enumerate(items)
            if item.get("url")
        ]
        passages = [
            {
                "source_id": f"source_result_{index}",
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "text": item.get("raw_content") or item.get("snippet") or "",
                "query": arguments["query"],
                "extraction_method": "search_provider",
            }
            for index, item in enumerate(items)
            if item.get("url")
            and (item.get("raw_content") or item.get("snippet"))
        ]
        scraped = [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "markdown": item.get("raw_content") or item.get("snippet") or "",
                "fetch_method": "search_provider",
                "http_status": 200,
            }
            for item in items
            if item.get("url")
            and (item.get("raw_content") or item.get("snippet"))
        ]
        return ToolAdapterResult(
            success=True,
            data={
                "items": items,
                "sources": sources,
                "passages": passages,
                "scraped_data_cache": scraped,
                "semantic_complete": False,
            },
            usage=result.usage,
            metadata=result.metadata,
        )

    async def health(self) -> ToolHealthStatus:
        return await self.provider.health()


class _ReadAdapter:
    def __init__(self, scraper: ScraperInterface) -> None:
        self.scraper = scraper

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        urls = list(arguments.get("urls") or ())
        if not urls and arguments.get("url"):
            urls = [str(arguments["url"])]
        documents = await self.scraper.scrape_batch(
            urls,
            source_context=arguments.get("source_context") or {},
            force_playwright=bool(arguments.get("force_playwright", False)),
        )
        sources: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            payload = document.model_dump(mode="json")
            snapshots.append(payload)
            if document.error or not document.markdown.strip():
                continue
            source_id = f"source_read_{index}"
            sources.append(
                {
                    "source_id": source_id,
                    "url": document.url,
                    "title": document.title,
                    "source_type": "other",
                    "extraction_method": document.fetch_method,
                }
            )
            passages.append(
                {
                    "source_id": source_id,
                    "url": document.url,
                    "title": document.title,
                    "text": document.markdown,
                    "extraction_method": document.fetch_method,
                }
            )
        return ToolAdapterResult(
            success=True,
            data={
                "sources": sources,
                "passages": passages,
                "scraped_data_cache": snapshots,
                "semantic_complete": False,
            },
            usage=BudgetUsage(tool_calls=1),
        )

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


class _StructuredResearchAdapter:
    """Validates a model-proposed, non-side-effecting research transformation."""

    def __init__(self, operation: str) -> None:
        self.operation = operation

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        data = {key: value for key, value in arguments.items() if key != "operation"}
        data["operation"] = self.operation
        data.setdefault("semantic_complete", self.operation == "extract")
        return ToolAdapterResult(
            success=True,
            data=data,
            usage=BudgetUsage(tool_calls=1),
        )

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


def _search_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "operation": {"const": "search"},
            "query": {"type": "string", "minLength": 1, "maxLength": 4000},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["operation", "query"],
        "additionalProperties": False,
    }


def _read_input_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "operation": {"const": "read"},
            "url": {"type": "string", "format": "uri"},
            "urls": {
                "type": "array",
                "minItems": 1,
                "maxItems": 20,
                "items": {"type": "string", "format": "uri"},
            },
            "source_context": {"type": "object"},
            "force_playwright": {"type": "boolean"},
        },
        "required": ["operation"],
        "anyOf": [{"required": ["url"]}, {"required": ["urls"]}],
        "additionalProperties": False,
    }


def _structured_input_schema(operation: str) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "operation": {"const": operation},
        "section_id": {"type": "string"},
        "semantic_complete": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "object"}},
        "passages": {"type": "array", "items": {"type": "object"}},
        "scraped_data_cache": {"type": "array", "items": {"type": "object"}},
        "evidence": {"type": "array", "items": {"type": "object"}},
        "atomic_facts": {"type": "array", "items": {"type": "object"}},
        "claims": {"type": "array", "items": {"type": "object"}},
        "conflicts": {"type": "array", "items": {"type": "object"}},
        "section_evidence_packs": {"type": "array", "items": {"type": "object"}},
        "items": {"type": "array"},
        "result": {},
        "summary": {"type": "string"},
    }
    required = ["operation"]
    if operation == "extract":
        required.append("claims")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _definition(
    *,
    name: str,
    operation: str,
    description: str,
    input_schema: dict[str, Any],
    permission: str,
    provider: str,
    fallback_tools: tuple[str, ...] = (),
    cache_ttl_seconds: float | None = None,
    timeout_seconds: float = 60.0,
    max_attempts: int = 2,
) -> ToolDefinition:
    return ToolDefinition(
        name=name,
        version="1.0.0",
        description=description,
        input_schema=input_schema,
        output_schema={"type": "object"},
        operations=(operation,),
        permission_scopes=(permission,),
        risk_level=ToolRiskLevel.LOW,
        timeout_seconds=timeout_seconds,
        max_attempts=max_attempts,
        rate_limit=RateLimit(calls=120, window_seconds=60.0),
        cache_ttl_seconds=cache_ttl_seconds,
        fallback_tools=fallback_tools,
        circuit_breaker=CircuitBreakerPolicy(
            failure_threshold=3,
            recovery_seconds=30.0,
        ),
        protocol=ToolProtocol.NATIVE,
        provider=provider,
    )


def build_worker_tool_registry(
    *,
    scraper: ScraperInterface | None = None,
    tavily: TavilySearchToolAdapter | None = None,
    exa: ExaSearchToolAdapter | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        _definition(
            name=SEARCH_TOOL,
            operation="search",
            description="Search the public web through governed Tavily access.",
            input_schema=_search_input_schema(),
            permission="web:search",
            provider="tavily",
            fallback_tools=(SEARCH_FALLBACK_TOOL,),
            cache_ttl_seconds=300.0,
        ),
        _SearchAdapter(tavily or TavilySearchToolAdapter()),
        activate=True,
    )
    registry.register(
        _definition(
            name=SEARCH_FALLBACK_TOOL,
            operation="search",
            description="Search the public web through governed Exa fallback.",
            input_schema=_search_input_schema(),
            permission="web:search",
            provider="exa",
            cache_ttl_seconds=300.0,
        ),
        _SearchAdapter(exa or ExaSearchToolAdapter()),
        activate=True,
    )
    registry.register(
        _definition(
            name=READ_TOOL,
            operation="read",
            description="Fetch and normalize public pages with Jina/Playwright fallback.",
            input_schema=_read_input_schema(),
            permission="web:read",
            provider="smart_scraper",
            timeout_seconds=120.0,
        ),
        _ReadAdapter(scraper or build_scraper()),
        activate=True,
    )
    for name, operation, permission, description in (
        (
            EXTRACT_TOOL,
            "extract",
            "knowledge:extract",
            "Validate structured candidate evidence, facts, claims, and conflicts.",
        ),
        (
            COMPARE_TOOL,
            "compare",
            "knowledge:compare",
            "Validate a structured comparison over prior observations.",
        ),
        (
            VERIFY_SOURCE_TOOL,
            "verify_source",
            "source:verify",
            "Validate a structured source-authority and provenance assessment.",
        ),
    ):
        registry.register(
            _definition(
                name=name,
                operation=operation,
                description=description,
                input_schema=_structured_input_schema(operation),
                permission=permission,
                provider="local_governed_transform",
                cache_ttl_seconds=300.0,
                timeout_seconds=30.0,
                max_attempts=1,
            ),
            _StructuredResearchAdapter(operation),
            activate=True,
        )
    return registry


@dataclass
class GovernedResearchToolRuntime:
    state_store: SQLiteToolStateStore
    registry: ToolRegistry
    gateway: ProtocolToolGateway

    def integrity_check(self) -> None:
        self.state_store.integrity_check()

    def close(self) -> None:
        self.state_store.close()


def build_governed_research_tools(
    root: str | Path,
    *,
    scraper: ScraperInterface | None = None,
    tavily: TavilySearchToolAdapter | None = None,
    exa: ExaSearchToolAdapter | None = None,
    retry_base_seconds: float = 0.1,
) -> GovernedResearchToolRuntime:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    state = SQLiteToolStateStore(root_path / "tool_gateway.sqlite3")
    registry = build_worker_tool_registry(
        scraper=scraper,
        tavily=tavily,
        exa=exa,
    )

    def context(command: Command) -> ToolInvocationContext:
        return ToolInvocationContext(
            principal_id=command.actor_id,
            permissions=frozenset(
                {
                    "web:search",
                    "web:read",
                    "knowledge:extract",
                    "knowledge:compare",
                    "source:verify",
                }
            ),
            correlation_id=str(
                command.metadata.get("correlation_id")
                or f"correlation_{command.run_id}"
            ),
            trace_id=str(
                command.metadata.get("trace_id")
                or f"trace_{command.run_id}"
            ),
        )

    runtime = GovernedResearchToolRuntime(
        state_store=state,
        registry=registry,
        gateway=ProtocolToolGateway(
            registry=registry,
            state_store=state,
            context_factory=context,
            retry_base_seconds=retry_base_seconds,
        ),
    )
    runtime.integrity_check()
    return runtime
