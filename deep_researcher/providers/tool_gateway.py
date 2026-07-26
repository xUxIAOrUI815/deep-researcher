from __future__ import annotations

import os
from typing import Any

import httpx

from deep_researcher.contracts import BudgetUsage
from deep_researcher.gateway import (
    ToolAdapterError,
    ToolAdapterResult,
    ToolErrorKind,
    ToolHealthStatus,
    ToolInvocationContext,
)


class _HTTPAdapter:
    def __init__(
        self,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
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
            kind=(
                ToolErrorKind.TRANSIENT
                if retryable
                else ToolErrorKind.PERMANENT
            ),
            retryable=retryable,
            status_code=status,
        )


class TavilySearchToolAdapter(_HTTPAdapter):
    """Native Tavily provider adapter used by the governed worker registry."""

    base_url = "https://api.tavily.com/search"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(http_client=http_client)
        raw = api_key if api_key is not None else os.getenv(
            "TAVILY_API_KEY",
            "",
        )
        self.api_key = self._normalize_key(raw, "TAVILY_API_KEY")

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        del context
        if not self.api_key or self.api_key == "demo":
            raise ToolAdapterError(
                "TAVILY_API_KEY is required for live Tavily search.",
                kind=ToolErrorKind.PERMANENT,
            )
        try:
            response = await self._post(
                self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
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
            raise ToolAdapterError(
                f"invalid Tavily response: {exc}",
                kind=ToolErrorKind.PROTOCOL,
            ) from exc
        return ToolAdapterResult(
            success=True,
            data={"items": self._normalize_results(payload)},
            usage=BudgetUsage(search_calls=1),
        )

    async def health(self) -> ToolHealthStatus:
        return (
            ToolHealthStatus.HEALTHY
            if self.api_key and self.api_key != "demo"
            else ToolHealthStatus.UNHEALTHY
        )

    @staticmethod
    def _normalize_key(raw: str, variable: str) -> str:
        value = str(raw or "").strip().strip('"').strip("'")
        if value.startswith(f"{variable}="):
            value = value.split("=", 1)[1].strip().strip('"').strip("'")
        return value

    @staticmethod
    def _normalize_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "url": str(item.get("url", "")),
                "title": str(item.get("title", "")),
                "snippet": str(item.get("content", ""))[:200],
                "score": float(item.get("score", 0.0) or 0.0),
                "raw_content": (
                    str(item["raw_content"])
                    if item.get("raw_content") is not None
                    else None
                ),
            }
            for item in payload.get("results", [])
        ]


class ExaSearchToolAdapter(_HTTPAdapter):
    """Native Exa fallback adapter used by the governed worker registry."""

    base_url = "https://api.exa.ai/search"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(http_client=http_client)
        raw = api_key if api_key is not None else os.getenv("EXA_API_KEY", "")
        self.api_key = TavilySearchToolAdapter._normalize_key(
            raw,
            "EXA_API_KEY",
        )

    async def execute(
        self,
        arguments: dict[str, Any],
        context: ToolInvocationContext,
    ) -> ToolAdapterResult:
        del context
        if not self.api_key or self.api_key == "demo":
            raise ToolAdapterError(
                "EXA_API_KEY is required for live Exa search.",
                kind=ToolErrorKind.PERMANENT,
            )
        try:
            response = await self._post(
                self.base_url,
                headers={
                    "x-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
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
            raise ToolAdapterError(
                f"invalid Exa response: {exc}",
                kind=ToolErrorKind.PROTOCOL,
            ) from exc
        return ToolAdapterResult(
            success=True,
            data={"items": self._normalize_results(payload)},
            usage=BudgetUsage(search_calls=1),
        )

    async def health(self) -> ToolHealthStatus:
        return (
            ToolHealthStatus.HEALTHY
            if self.api_key and self.api_key != "demo"
            else ToolHealthStatus.UNHEALTHY
        )

    @staticmethod
    def _normalize_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for item in payload.get("results", []):
            content = item.get("text", "")
            if isinstance(content, dict):
                content = content.get("text", "")
            items.append(
                {
                    "url": str(item.get("url", "")),
                    "title": str(item.get("title", "")),
                    "snippet": str(content or "")[:200],
                    "score": float(item.get("score", 0.0) or 0.0),
                    "raw_content": str(content) if content else None,
                }
            )
        return items
