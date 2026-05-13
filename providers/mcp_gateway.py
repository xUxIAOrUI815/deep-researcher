import asyncio
import httpx
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from schemas.state import SearchResult


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"


@dataclass
class MCPToolDefinition:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Any


@dataclass
class MCPToolResult:
    success: bool
    data: Any
    error: Optional[str] = None
    tool_name: str = ""
    error_type: str = ""
    retryable: bool = False
    status_code: Optional[int] = None
    attempts: int = 1


class MCPGatewayError(RuntimeError):
    def __init__(
        self,
        *,
        tool_name: str,
        error: str,
        retryable: bool = False,
        status_code: Optional[int] = None,
        attempts: int = 1,
    ):
        super().__init__(f"{tool_name} failed: {error}")
        self.tool_name = tool_name
        self.retryable = retryable
        self.status_code = status_code
        self.attempts = attempts


class BaseMCPHandler:
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        raise NotImplementedError


class TavilySearchProvider(BaseMCPHandler):
    retryable_status_codes = {429, 500, 502, 503, 504}
    retryable_exceptions = (
        httpx.ReadTimeout,
        httpx.ConnectTimeout,
        httpx.ConnectError,
        httpx.RemoteProtocolError,
    )

    def __init__(self, api_key: Optional[str] = None):
        raw_api_key = api_key if api_key is not None else os.getenv("TAVILY_API_KEY", "")
        self.api_key = raw_api_key.strip().strip('"').strip("'")
        if self.api_key.startswith("TAVILY_API_KEY="):
            self.api_key = self.api_key.split("=", 1)[1].strip().strip('"').strip("'")
        self.base_url = "https://api.tavily.com/search"

    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        if tool_name == "tavily_search":
            return await self._search(parameters)
        return MCPToolResult(success=False, data=None, error=f"Unknown tool: {tool_name}")

    async def _search(self, params: Dict[str, Any]) -> MCPToolResult:
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not self.api_key or self.api_key == "demo":
            return MCPToolResult(
                success=False,
                data=None,
                error=(
                    "TAVILY_API_KEY is required for live Tavily search. "
                    "Set TAVILY_API_KEY or set RESEARCHER_SEARCH_MODE=mock for offline testing."
                ),
                tool_name="tavily_search",
                error_type="MissingAPIKey",
                retryable=False,
            )

        max_attempts = max(1, int(params.get("max_attempts", 3) or 3))
        async with httpx.AsyncClient(timeout=30.0) as client:
            for attempt in range(1, max_attempts + 1):
                try:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "query": query,
                            "max_results": max_results,
                            "include_answer": True,
                            "include_raw_content": True,
                        },
                    )
                    response.raise_for_status()
                    data = response.json()

                    results = [
                        SearchResult(
                            url=item.get("url", ""),
                            title=item.get("title", ""),
                            snippet=item.get("content", "")[:200],
                            score=item.get("score", 0.0),
                        )
                        for item in data.get("results", [])
                    ]
                    return MCPToolResult(success=True, data=results, tool_name="tavily_search", attempts=attempt)

                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code
                    response_text = e.response.text[:500] if e.response is not None else ""
                    retryable = status_code in self.retryable_status_codes
                    if retryable and attempt < max_attempts:
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    detail = f"HTTP error: {status_code}"
                    if response_text:
                        detail = f"{detail}; response={response_text}"
                    return MCPToolResult(
                        success=False,
                        data=None,
                        error=detail,
                        tool_name="tavily_search",
                        error_type="HTTPStatusError",
                        retryable=retryable,
                        status_code=status_code,
                        attempts=attempt,
                    )
                except Exception as e:
                    retryable = isinstance(e, self.retryable_exceptions)
                    if retryable and attempt < max_attempts:
                        await asyncio.sleep(0.5 * attempt)
                        continue
                    return MCPToolResult(
                        success=False,
                        data=None,
                        error=_format_exception(e),
                        tool_name="tavily_search",
                        error_type=type(e).__name__,
                        retryable=retryable,
                        attempts=attempt,
                    )


class ExaSearchProvider(BaseMCPHandler):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or "demo"
        self.base_url = "https://api.exa.ai/search"

    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        if tool_name == "exa_search":
            return await self._search(parameters)
        return MCPToolResult(success=False, data=None, error=f"Unknown tool: {tool_name}")

    async def _search(self, params: Dict[str, Any]) -> MCPToolResult:
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not self.api_key or self.api_key == "demo":
            await asyncio.sleep(0.1)
            mock_results = [
                SearchResult(
                    url=f"https://exa-mock.com/article{i}",
                    title=f"Exa Mock Article {i} - {query}",
                    snippet=f"Mock Exa search result for {query}...",
                    score=0.85 - i * 0.15
                )
                for i in range(max_results)
            ]
            return MCPToolResult(success=True, data=mock_results, tool_name="exa_search")

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    headers={"x-api-key": self.api_key},
                    json={
                        "query": query,
                        "num_results": max_results,
                        "text": {"max_characters": 500}
                    }
                )
                response.raise_for_status()
                data = response.json()

                results = [
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("text", "")[:200],
                        score=item.get("score", 0.0)
                    )
                    for item in data.get("results", [])
                ]
                return MCPToolResult(success=True, data=results, tool_name="exa_search")

        except Exception as e:
            return MCPToolResult(success=False, data=None, error=_format_exception(e), tool_name="exa_search")


class MCPGateway:
    def __init__(self):
        self._tools: Dict[str, BaseMCPHandler] = {}
        self._tool_definitions: Dict[str, MCPToolDefinition] = {}
        self._register_default_tools()

    def _register_default_tools(self):
        tavily = TavilySearchProvider()
        exa = ExaSearchProvider()

        self._tools["tavily_search"] = tavily
        self._tool_definitions["tavily_search"] = MCPToolDefinition(
            name="tavily_search",
            description="Search the web using Tavily API for relevant articles and information",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results"}
                },
                "required": ["query"]
            },
            handler=tavily
        )

        self._tools["exa_search"] = exa
        self._tool_definitions["exa_search"] = MCPToolDefinition(
            name="exa_search",
            description="Search the web using Exa.ai for relevant content",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5, "description": "Maximum number of results"}
                },
                "required": ["query"]
            },
            handler=exa
        )

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": defn.name,
                "description": defn.description,
                "input_schema": defn.input_schema
            }
            for defn in self._tool_definitions.values()
        ]

    async def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        if tool_name not in self._tools:
            return MCPToolResult(
                success=False,
                data=None,
                error=f"Tool '{tool_name}' not found. Available tools: {list(self._tools.keys())}"
            )

        handler = self._tools[tool_name]
        try:
            result = await handler.execute(tool_name, parameters)
            return result
        except Exception as e:
            return MCPToolResult(success=False, data=None, error=_format_exception(e), tool_name=tool_name)

    async def search(self, query: str, max_results: int = 5, provider: str = "tavily") -> List[SearchResult]:
        tool_name = f"{provider}_search"
        result = await self.call_tool(tool_name, {"query": query, "max_results": max_results})

        if result.success:
            return result.data
        # else:
        #     print(f"[MCPGateway] Search error: {result.error}")
        #     return []
        raise MCPGatewayError(
            tool_name=tool_name,
            error=str(result.error or ""),
            retryable=bool(result.retryable),
            status_code=result.status_code,
            attempts=result.attempts,
        )
