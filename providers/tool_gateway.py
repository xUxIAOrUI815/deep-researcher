from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

import httpx

from deep_researcher.contracts import BudgetUsage, Command, CommandKind
from deep_researcher.gateway import (
    CircuitBreakerPolicy,
    ProtocolToolGateway,
    RateLimit,
    SQLiteToolStateStore,
    ToolAdapterError,
    ToolAdapterResult,
    ToolDefinition,
    ToolErrorKind,
    ToolExecutionResult,
    ToolHealthStatus,
    ToolInvocationContext,
    ToolProtocol,
    ToolRegistry,
    ToolRiskLevel,
)
from schemas.state import ScrapedData, SearchResult

from .scraper_backend import ScraperInterface, build_scraper


TAVILY_TOOL = "web.search.tavily"
EXA_TOOL = "web.search.exa"
SCRAPER_TOOL = "web.scrape"
TOOL_VERSION = "1.0.0"


class ResearchToolGatewayError(RuntimeError):
    def __init__(
        self,
        *,
        tool_name: str,
        error: str,
        error_kind: str = "permanent",
        retryable: bool = False,
        status_code: int | None = None,
        attempts: int = 1,
    ) -> None:
        super().__init__(f"{tool_name} failed: {error}")
        self.tool_name = tool_name
        self.error_kind = error_kind
        self.retryable = retryable
        self.status_code = status_code
        self.attempts = attempts


class _HTTPAdapter:
    def __init__(self, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._http_client = http_client

    async def _post(self, url: str, **kwargs: Any) -> httpx.Response:
        if self._http_client is not None:
            return await self._http_client.post(url, **kwargs)
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await client.post(url, **kwargs)

    @staticmethod
    def _provider_error(exc: httpx.HTTPStatusError) -> ToolAdapterError:
        status = exc.response.status_code
        retryable = status in {408, 409, 425, 429, 500, 502, 503, 504}
        body = exc.response.text[:500].strip()
        suffix = f"; response={body}" if body else ""
        return ToolAdapterError(
            f"HTTP error: {status}{suffix}",
            kind=ToolErrorKind.TRANSIENT if retryable else ToolErrorKind.PERMANENT,
            retryable=retryable,
            status_code=status,
        )


class TavilySearchToolAdapter(_HTTPAdapter):
    base_url = "https://api.tavily.com/search"

    def __init__(self, api_key: str | None = None, *, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(http_client=http_client)
        raw = api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        self.api_key = self._normalize_key(raw, "TAVILY_API_KEY")

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        if not self.api_key or self.api_key == "demo":
            raise ToolAdapterError(
                "TAVILY_API_KEY is required for live Tavily search; use explicit mock mode for offline execution",
                kind=ToolErrorKind.PERMANENT,
            )
        try:
            response = await self._post(
                self.base_url,
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={
                    "query": arguments["query"],
                    "max_results": arguments.get("max_results", 5),
                    "include_answer": True,
                    "include_raw_content": True,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._provider_error(exc) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ToolAdapterError(f"invalid Tavily response: {exc}", kind=ToolErrorKind.PROTOCOL) from exc
        items = [
            {
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("content", ""))[:200],
                "score": float(item.get("score", 0.0) or 0.0),
                "raw_content": str(item["raw_content"]) if item.get("raw_content") is not None else None,
            }
            for item in payload.get("results", [])
        ]
        return ToolAdapterResult(success=True, data={"items": items}, usage=BudgetUsage(search_calls=1))

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY if self.api_key and self.api_key != "demo" else ToolHealthStatus.UNHEALTHY

    @staticmethod
    def _normalize_key(raw: str, variable: str) -> str:
        value = str(raw or "").strip().strip('"').strip("'")
        if value.startswith(f"{variable}="):
            value = value.split("=", 1)[1].strip().strip('"').strip("'")
        return value


class ExaSearchToolAdapter(_HTTPAdapter):
    base_url = "https://api.exa.ai/search"

    def __init__(self, api_key: str | None = None, *, http_client: httpx.AsyncClient | None = None) -> None:
        super().__init__(http_client=http_client)
        raw = api_key if api_key is not None else os.getenv("EXA_API_KEY", "")
        self.api_key = TavilySearchToolAdapter._normalize_key(raw, "EXA_API_KEY")

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        if not self.api_key or self.api_key == "demo":
            raise ToolAdapterError(
                "EXA_API_KEY is required for live Exa search; use explicit mock mode for offline execution",
                kind=ToolErrorKind.PERMANENT,
            )
        try:
            response = await self._post(
                self.base_url,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                json={
                    "query": arguments["query"],
                    "num_results": arguments.get("max_results", 5),
                    "text": {"max_characters": 500},
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise self._provider_error(exc) from exc
        except (ValueError, TypeError, KeyError) as exc:
            raise ToolAdapterError(f"invalid Exa response: {exc}", kind=ToolErrorKind.PROTOCOL) from exc
        items = []
        for item in payload.get("results", []):
            text = item.get("text", "")
            if isinstance(text, dict):
                text = text.get("text", "")
            items.append(
                {
                    "url": str(item.get("url", "")),
                    "title": str(item.get("title", "")),
                    "snippet": str(text or "")[:200],
                    "score": float(item.get("score", 0.0) or 0.0),
                    "raw_content": str(text) if text else None,
                }
            )
        return ToolAdapterResult(success=True, data={"items": items}, usage=BudgetUsage(search_calls=1))

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY if self.api_key and self.api_key != "demo" else ToolHealthStatus.UNHEALTHY


class ScraperToolAdapter:
    def __init__(self, scraper: ScraperInterface) -> None:
        self.scraper = scraper

    async def execute(self, arguments: dict[str, Any], context: ToolInvocationContext) -> ToolAdapterResult:
        values = await self.scraper.scrape_batch(
            list(arguments["urls"]),
            source_context=arguments.get("source_context") or {},
            force_playwright=bool(arguments.get("force_playwright", False)),
        )
        return ToolAdapterResult(
            success=True,
            data={"items": [item.model_dump(mode="json") for item in values]},
        )

    async def health(self) -> ToolHealthStatus:
        return ToolHealthStatus.HEALTHY


def _search_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "operation": {"const": "search"},
            "query": {"type": "string", "minLength": 1, "maxLength": 4000},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 20},
        },
        "required": ["operation", "query", "max_results"],
        "additionalProperties": False,
    }


def _search_output_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "title": {"type": "string"},
                        "snippet": {"type": "string"},
                        "score": {"type": "number"},
                        "raw_content": {"type": ["string", "null"]},
                    },
                    "required": ["url", "title", "snippet", "score", "raw_content"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def build_research_tool_registry(
    *,
    scraper: ScraperInterface | None = None,
    tavily: TavilySearchToolAdapter | None = None,
    exa: ExaSearchToolAdapter | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    search_schema = _search_schema()
    search_output = _search_output_schema()
    registry.register(
        ToolDefinition(
            name=TAVILY_TOOL,
            version=TOOL_VERSION,
            description="Search the public web through Tavily.",
            input_schema=search_schema,
            output_schema=search_output,
            operations=("search",),
            permission_scopes=("web:search",),
            risk_level=ToolRiskLevel.LOW,
            timeout_seconds=30.0,
            max_attempts=3,
            rate_limit=RateLimit(calls=60, window_seconds=60.0),
            cache_ttl_seconds=300.0,
            estimated_cost_usd=0.008,
            fallback_tools=(EXA_TOOL,),
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=3, recovery_seconds=30.0),
            protocol=ToolProtocol.NATIVE,
            provider="tavily",
        ),
        tavily or TavilySearchToolAdapter(),
        activate=True,
    )
    registry.register(
        ToolDefinition(
            name=EXA_TOOL,
            version=TOOL_VERSION,
            description="Search the public web through Exa.",
            input_schema=search_schema,
            output_schema=search_output,
            operations=("search",),
            permission_scopes=("web:search",),
            risk_level=ToolRiskLevel.LOW,
            timeout_seconds=30.0,
            max_attempts=3,
            rate_limit=RateLimit(calls=60, window_seconds=60.0),
            cache_ttl_seconds=300.0,
            estimated_cost_usd=0.008,
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=3, recovery_seconds=30.0),
            protocol=ToolProtocol.NATIVE,
            provider="exa",
        ),
        exa or ExaSearchToolAdapter(),
        activate=True,
    )
    scraper_backend = scraper or build_scraper()
    registry.register(
        ToolDefinition(
            name=SCRAPER_TOOL,
            version=TOOL_VERSION,
            description="Fetch and normalize public web pages through the configured scraper backend.",
            input_schema={
                "type": "object",
                "properties": {
                    "operation": {"const": "read"},
                    "urls": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "uniqueItems": True,
                        "items": {"type": "string", "format": "uri"},
                    },
                    "source_context": {"type": "object"},
                    "force_playwright": {"type": "boolean"},
                },
                "required": ["operation", "urls", "source_context", "force_playwright"],
                "additionalProperties": False,
            },
            output_schema={
                "type": "object",
                "properties": {"items": {"type": "array", "items": {"type": "object"}}},
                "required": ["items"],
                "additionalProperties": False,
            },
            operations=("read",),
            permission_scopes=("web:read",),
            risk_level=ToolRiskLevel.MEDIUM,
            timeout_seconds=90.0,
            max_attempts=2,
            rate_limit=RateLimit(calls=30, window_seconds=60.0),
            cache_ttl_seconds=60.0,
            circuit_breaker=CircuitBreakerPolicy(failure_threshold=3, recovery_seconds=30.0),
            protocol=ToolProtocol.NATIVE,
            provider=getattr(scraper_backend, "mode", "scraper"),
        ),
        ScraperToolAdapter(scraper_backend),
        activate=True,
    )
    return registry


@dataclass(frozen=True)
class GatewayCallIdentity:
    run_id: str = "run_tool_gateway"
    task_id: str = "task_tool_gateway"
    actor_id: str = "agent_research_worker"
    principal_id: str = "principal_research_worker"
    correlation_id: str = "correlation_tool_gateway"
    trace_id: str = "trace_tool_gateway"


class ResearchToolGateway:
    """Typed compatibility facade over the governed ProtocolToolGateway."""

    def __init__(
        self,
        *,
        registry: ToolRegistry | None = None,
        state_store: SQLiteToolStateStore | None = None,
        state_path: str | Path | None = None,
        scraper: ScraperInterface | None = None,
        tavily: TavilySearchToolAdapter | None = None,
        exa: ExaSearchToolAdapter | None = None,
        identity: GatewayCallIdentity | None = None,
        retry_base_seconds: float = 0.1,
    ) -> None:
        self.registry = registry or build_research_tool_registry(scraper=scraper, tavily=tavily, exa=exa)
        if state_store is None:
            resolved_path = Path(state_path or os.getenv("TOOL_GATEWAY_DB_PATH", "knowledge_data/tool_gateway.sqlite3"))
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            self.state_store = SQLiteToolStateStore(resolved_path)
        else:
            self.state_store = state_store
        self._owns_state_store = state_store is None
        self.identity = identity or GatewayCallIdentity()
        self.gateway = ProtocolToolGateway(
            registry=self.registry,
            state_store=self.state_store,
            retry_base_seconds=retry_base_seconds,
        )
        self.last_result: ToolExecutionResult | None = None

    async def search(
        self,
        query: str,
        max_results: int = 5,
        provider: str = "tavily",
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[SearchResult]:
        tool_name = {"tavily": TAVILY_TOOL, "exa": EXA_TOOL}.get(provider)
        if tool_name is None:
            raise ValueError(f"unsupported search provider: {provider}")
        result = await self._invoke(
            tool_name,
            CommandKind.SEARCH,
            {"operation": "search", "query": query, "max_results": max_results},
            permissions=frozenset({"web:search"}),
            on_retry=on_retry,
        )
        return [SearchResult.model_validate(item) for item in result.get("items", [])]

    async def scrape_batch(
        self,
        urls: list[str],
        *,
        source_context: Mapping[str, Mapping[str, Any]] | None = None,
        force_playwright: bool = False,
        on_retry: Callable[[dict[str, Any]], None] | None = None,
    ) -> list[ScrapedData]:
        result = await self._invoke(
            SCRAPER_TOOL,
            CommandKind.READ,
            {
                "operation": "read",
                "urls": urls,
                "source_context": {key: dict(value) for key, value in (source_context or {}).items()},
                "force_playwright": force_playwright,
            },
            permissions=frozenset({"web:read"}),
            on_retry=on_retry,
        )
        return [ScrapedData.model_validate(item, strict=False) for item in result.get("items", [])]

    async def _invoke(
        self,
        tool_name: str,
        kind: CommandKind,
        arguments: dict[str, Any],
        *,
        permissions: frozenset[str],
        on_retry: Callable[[dict[str, Any]], None] | None,
    ) -> dict[str, Any]:
        payload = f"{self.identity.run_id}\0{self.identity.task_id}\0{tool_name}\0{arguments}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        command = Command(
            command_id=f"command_{digest[:32]}",
            run_id=self._namespaced_id(self.identity.run_id, "run"),
            task_id=self._namespaced_id(self.identity.task_id, "task"),
            actor_id=self._namespaced_id(self.identity.actor_id, "agent"),
            kind=kind,
            name=tool_name,
            arguments=arguments,
            expected_output_schema="ToolExecutionResult@1",
            idempotency_key=f"tool-{digest}",
            metadata={"tool_version": TOOL_VERSION},
        )
        context = ToolInvocationContext(
            principal_id=self.identity.principal_id,
            permissions=permissions,
            correlation_id=self.identity.correlation_id,
            trace_id=self.identity.trace_id,
            on_retry=on_retry,
        )
        result = await self.gateway.execute_tool(command, context)
        self.last_result = result
        if result.success:
            return dict(result.data or {})
        error = result.error
        raise ResearchToolGatewayError(
            tool_name=result.tool_name,
            error=error.message if error else "unknown tool error",
            error_kind=error.code if error else "permanent",
            retryable=bool(error.retryable) if error else False,
            status_code=result.metadata.get("status_code"),
            attempts=result.attempts,
        )

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return self.registry.definitions()

    @staticmethod
    def _namespaced_id(value: str, prefix: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value).strip()).strip("_.:-")
        if not normalized:
            normalized = hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]
        if not normalized.startswith(f"{prefix}_"):
            normalized = f"{prefix}_{normalized}"
        return normalized

    def close(self) -> None:
        if self._owns_state_store:
            self.state_store.close()
            self._owns_state_store = False

    def __enter__(self) -> "ResearchToolGateway":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            # Destructors cannot safely surface cleanup failures. Explicit close
            # remains the observable lifecycle path.
            pass
