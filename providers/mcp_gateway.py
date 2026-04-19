import asyncio
import httpx
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json

from schemas.state import SearchResult

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


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


class BaseMCPHandler:
    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        raise NotImplementedError


class TavilySearchProvider(BaseMCPHandler):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or TAVILY_API_KEY
        self.base_url = "https://api.tavily.com/search"

    async def execute(self, tool_name: str, parameters: Dict[str, Any]) -> MCPToolResult:
        if tool_name == "tavily_search":
            return await self._search(parameters)
        return MCPToolResult(success=False, data=None, error=f"Unknown tool: {tool_name}")

    async def _search(self, params: Dict[str, Any]) -> MCPToolResult:
        query = params.get("query", "")
        max_results = params.get("max_results", 5)

        if not self.api_key or self.api_key == "demo":
            # await asyncio.sleep(0.1)
            # mock_results = [
            #     SearchResult(
            #         url=f"https://example.com/article{i}",
            #         title=f"Mock Article {i} about {query}",
            #         snippet=f"This is a mock search result snippet for {query}...",
            #         score=0.9 - i * 0.1
            #     )
            #     for i in range(max_results)
            # ]
            # return MCPToolResult(success=True, data=mock_results, tool_name="tavily_search")

            return MCPToolResult(
                success=False,
                data=None,
                error=(
                    "TAVILY_API_KEY is required for live Tavily search. "
                    "Set TAVILY_API_KEY or set RESEARCHER_SEARCH_MODE=mock for offline testing."
                ),
                tool_name="tavily_search",
            )
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.base_url,
                    json={
                        "api_key": self.api_key,
                        "query": query,
                        "max_results": max_results,
                        "include_answer": True,
                        "include_raw_content": True
                    }
                )
                response.raise_for_status()
                data = response.json()

                results = [
                    SearchResult(
                        url=item.get("url", ""),
                        title=item.get("title", ""),
                        snippet=item.get("content", "")[:200],
                        score=item.get("score", 0.0)
                    )
                    for item in data.get("results", [])
                ]
                return MCPToolResult(success=True, data=results, tool_name="tavily_search")

        except httpx.HTTPStatusError as e:
            return MCPToolResult(success=False, data=None, error=f"HTTP error: {e.response.status_code}")
        except Exception as e:
            return MCPToolResult(success=False, data=None, error=str(e))


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
            return MCPToolResult(success=False, data=None, error=str(e))


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
            return MCPToolResult(success=False, data=None, error=str(e))

    async def search(self, query: str, max_results: int = 5, provider: str = "tavily") -> List[SearchResult]:
        tool_name = f"{provider}_search"
        result = await self.call_tool(tool_name, {"query": query, "max_results": max_results})

        if result.success:
            return result.data
        # else:
        #     print(f"[MCPGateway] Search error: {result.error}")
        #     return []
        raise RuntimeError(f"{tool_name} failed: {result.error}")
