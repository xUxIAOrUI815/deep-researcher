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
# Provider payloads are persisted before they reach the model context.  These
# limits therefore bound only the evidence excerpt carried through each model
# turn; they do not discard the durable source/snapshot records.  A standard
# four-worker cycle must leave enough context budget for the mandatory extract
# turn instead of spending the entire run budget re-sending search prose.
MAX_SEARCH_CONTENT_CHARS = 2_500
MAX_SEARCH_TOTAL_CONTENT_CHARS = 12_000
MAX_READ_CONTENT_CHARS = 10_000
MAX_READ_TOTAL_CONTENT_CHARS = 30_000


def _sanitize_text(value: str) -> str:
    return "".join(
        character
        for character in value
        if (
            ord(character) >= 32
            and ord(character) != 127
        )
        or character in "\r\n\t"
    )


def _sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _sanitize_text(str(key)): _sanitize_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_sanitize_value(item) for item in value]
    if isinstance(value, str):
        return _sanitize_text(value)
    return value


def _bounded_contents(
    values: list[str],
    *,
    per_item_limit: int,
    total_limit: int,
) -> tuple[list[str], int, int]:
    populated = sum(bool(item) for item in values)
    effective_limit = (
        min(per_item_limit, max(1, total_limit // populated))
        if populated
        else per_item_limit
    )
    bounded = [item[:effective_limit] for item in values]
    truncated = sum(
        len(item) > len(bounded_item)
        for item, bounded_item in zip(values, bounded, strict=True)
    )
    return bounded, truncated, effective_limit


class _SearchAdapter:
    def __init__(self, provider: TavilySearchToolAdapter | ExaSearchToolAdapter):
        self.provider = provider

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        result = await self.provider.execute(arguments, context)
        provider_items = [
            _sanitize_value(dict(item))
            for item in (result.data or {}).get("items", [])
            if isinstance(item, dict)
        ]
        items = [
            {
                key: value
                for key, value in item.items()
                if key != "raw_content"
            }
            for item in provider_items
        ]
        raw_content = [
            str(item.get("raw_content") or item.get("snippet") or "")
            for item in provider_items
        ]
        bounded_content, truncated_count, effective_limit = (
            _bounded_contents(
                raw_content,
                per_item_limit=MAX_SEARCH_CONTENT_CHARS,
                total_limit=MAX_SEARCH_TOTAL_CONTENT_CHARS,
            )
        )
        sources = [
            {
                "source_id": f"source_result_{index}",
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "score": item.get("score", 0.0),
                "query": arguments["query"],
                "source_type": "other",
            }
            for index, item in enumerate(provider_items)
            if item.get("url")
        ]
        passages = [
            {
                "source_id": f"source_result_{index}",
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                "text": bounded_content[index],
                "query": arguments["query"],
                "extraction_method": "search_provider",
            }
            for index, item in enumerate(provider_items)
            if item.get("url")
            and (item.get("raw_content") or item.get("snippet"))
        ]
        scraped = [
            {
                "url": item.get("url", ""),
                "title": item.get("title", ""),
                # The content is represented once in passages so the model
                # context and gateway payload do not carry a duplicate copy.
                # Knowledge ingestion falls back from snapshot markdown to
                # the matching passage while retaining this fetch metadata.
                "markdown": "",
                "fetch_method": "search_provider",
                "http_status": 200,
                "content_available_in_passages": True,
            }
            for index, item in enumerate(provider_items)
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
            metadata={
                **result.metadata,
                "content_limit_chars": effective_limit,
                "total_content_limit_chars": (
                    MAX_SEARCH_TOTAL_CONTENT_CHARS
                ),
                "truncated_result_count": truncated_count,
            },
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
        raw_content = [
            _sanitize_text(document.markdown)
            for document in documents
        ]
        bounded_content, truncated_count, effective_limit = (
            _bounded_contents(
                raw_content,
                per_item_limit=MAX_READ_CONTENT_CHARS,
                total_limit=MAX_READ_TOTAL_CONTENT_CHARS,
            )
        )
        sources: list[dict[str, Any]] = []
        passages: list[dict[str, Any]] = []
        snapshots: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            payload = _sanitize_value(document.model_dump(mode="json"))
            payload["markdown"] = ""
            payload["content_available_in_passages"] = bool(
                bounded_content[index]
            )
            snapshots.append(payload)
            if document.error or not bounded_content[index].strip():
                continue
            source_id = f"source_read_{index}"
            sources.append(
                {
                    "source_id": source_id,
                    "url": _sanitize_text(document.url),
                    "title": _sanitize_text(document.title),
                    "source_type": "other",
                    "extraction_method": document.fetch_method,
                }
            )
            passages.append(
                {
                    "source_id": source_id,
                    "url": _sanitize_text(document.url),
                    "title": _sanitize_text(document.title),
                    "text": bounded_content[index],
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
            metadata={
                "content_limit_chars": effective_limit,
                "total_content_limit_chars": MAX_READ_TOTAL_CONTENT_CHARS,
                "truncated_result_count": truncated_count,
            },
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
    identifier = {"type": "string", "minLength": 1, "maxLength": 500}
    bounded_text = {"type": "string", "minLength": 1, "maxLength": 8000}
    score = {"type": "number", "minimum": 0.0, "maximum": 1.0}
    evidence_item = {
        "type": "object",
        "properties": {
            "id": identifier,
            "evidence_id": identifier,
            "source_id": identifier,
            "source_url": {"type": "string", "format": "uri"},
            "quote": {
                "type": "string",
                "minLength": 1,
                "maxLength": 3000,
            },
            "summary": {
                "type": "string",
                "minLength": 1,
                "maxLength": 4000,
            },
            "confidence": score,
            "quality_score": score,
        },
        "required": ["quote"],
        "anyOf": [
            {"required": ["source_id"]},
            {"required": ["source_url"]},
        ],
        "additionalProperties": False,
    }
    fact_item = {
        "type": "object",
        "properties": {
            "id": identifier,
            "fact_id": identifier,
            "source_id": identifier,
            "source_url": {"type": "string", "format": "uri"},
            "text": bounded_text,
            "statement": bounded_text,
            "section_id": identifier,
            "confidence": score,
        },
        "allOf": [
            {"anyOf": [{"required": ["text"]}, {"required": ["statement"]}]},
            {
                "anyOf": [
                    {"required": ["source_id"]},
                    {"required": ["source_url"]},
                ]
            },
        ],
        "additionalProperties": False,
    }
    claim_item = {
        "type": "object",
        "properties": {
            "id": identifier,
            "claim_id": identifier,
            "text": bounded_text,
            "statement": bounded_text,
            "fact_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": identifier,
            },
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": identifier,
            },
            "confidence": score,
        },
        "allOf": [
            {"anyOf": [{"required": ["text"]}, {"required": ["statement"]}]},
            {
                "anyOf": [
                    {"required": ["fact_ids"]},
                    {"required": ["evidence_ids"]},
                ]
            },
        ],
        "additionalProperties": False,
    }
    conflict_item = {
        "type": "object",
        "properties": {
            "id": identifier,
            "conflict_id": identifier,
            "claim_ids": {
                "type": "array",
                "minItems": 2,
                "uniqueItems": True,
                "items": identifier,
            },
            "fact_ids": {
                "type": "array",
                "uniqueItems": True,
                "items": identifier,
            },
            "summary": bounded_text,
            "description": bounded_text,
        },
        "required": ["claim_ids"],
        "additionalProperties": False,
    }
    properties: dict[str, Any] = {
        "operation": {"const": operation},
        "section_id": {"type": "string"},
        "semantic_complete": {"type": "boolean"},
        "sources": {"type": "array", "items": {"type": "object"}},
        "passages": {"type": "array", "items": {"type": "object"}},
        "scraped_data_cache": {"type": "array", "items": {"type": "object"}},
        "evidence": {"type": "array", "items": evidence_item},
        "atomic_facts": {"type": "array", "items": fact_item},
        "claims": {"type": "array", "items": claim_item},
        "conflicts": {"type": "array", "items": conflict_item},
        "section_evidence_packs": {"type": "array", "items": {"type": "object"}},
        "items": {"type": "array"},
        "result": {},
        "summary": {"type": "string"},
        "url": {"type": "string", "format": "uri"},
        "source_url": {"type": "string", "format": "uri"},
        "source_id": {"type": "string", "minLength": 1},
    }
    required = ["operation"]
    if operation == "extract":
        required.extend(
            ["section_id", "evidence", "atomic_facts", "claims"]
        )
        properties["section_id"] = identifier
        for name in ("evidence", "atomic_facts", "claims"):
            properties[name]["minItems"] = 1
            properties[name]["maxItems"] = 3
    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    if operation == "compare":
        schema["properties"]["items"] = {
            "type": "array",
            "minItems": 2,
        }
        schema["required"] = ["operation", "items"]
    elif operation == "verify_source":
        schema["anyOf"] = [
            {"required": ["url"]},
            {"required": ["source_url"]},
            {"required": ["source_id"]},
        ]
    return schema


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
    tavily_adapter = tavily or TavilySearchToolAdapter()
    exa_adapter = exa or ExaSearchToolAdapter()
    exa_configured = exa is not None or bool(
        getattr(exa_adapter, "api_key", "")
    )
    registry = ToolRegistry()
    registry.register(
        _definition(
            name=SEARCH_TOOL,
            operation="search",
            description="Search the public web through governed Tavily access.",
            input_schema=_search_input_schema(),
            permission="web:search",
            provider="tavily",
            fallback_tools=(
                (SEARCH_FALLBACK_TOOL,) if exa_configured else ()
            ),
            cache_ttl_seconds=300.0,
        ),
        _SearchAdapter(tavily_adapter),
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
        _SearchAdapter(exa_adapter),
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
