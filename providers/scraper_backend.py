from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional, Protocol

from schemas.state import ScrapedData

from .scraper import SmartScraper


class ScraperInterface(Protocol):
    mode: str

    async def scrape_batch(
        self,
        urls: list[str],
        *,
        source_context: Optional[Dict[str, Dict[str, Any]]] = None,
        force_playwright: bool = False,
    ) -> list[ScrapedData]:
        ...


class SmartScraperAdapter:
    mode = "live"

    def __init__(self, scraper: Optional[SmartScraper] = None):
        self.scraper = scraper or SmartScraper()

    async def scrape_batch(
        self,
        urls: list[str],
        *,
        source_context: Optional[Dict[str, Dict[str, Any]]] = None,
        force_playwright: bool = False,
    ) -> list[ScrapedData]:
        return await self.scraper.scrape_batch(urls, force_playwright=force_playwright)


class MockScraper:
    mode = "mock"

    def __init__(self, fixtures: Optional[Dict[str, Dict[str, Any]]] = None):
        self.fixtures = fixtures or {}

    async def scrape_batch(
        self,
        urls: list[str],
        *,
        source_context: Optional[Dict[str, Dict[str, Any]]] = None,
        force_playwright: bool = False,
    ) -> list[ScrapedData]:
        source_context = source_context or {}
        return [self._build_scraped_data(url, source_context.get(url, {})) for url in urls]

    def _build_scraped_data(self, url: str, context: Dict[str, Any]) -> ScrapedData:
        fixture = self.fixtures.get(url)
        if fixture:
            return ScrapedData(
                url=url,
                markdown=str(fixture.get("markdown", "")),
                title=str(fixture.get("title", context.get("title", url))),
                fetch_method="mock",
                error=fixture.get("error"),
            )

        title = str(context.get("title", "") or f"Mock page for {url}")
        query = str(context.get("query", "") or "the requested research topic")
        snippet = str(context.get("snippet", "") or "")
        numeric_seed = 50 + (int(hashlib.sha1(url.encode("utf-8")).hexdigest()[:4], 16) % 75)
        markdown = (
            f"# {title}\n\n"
            f"{snippet}\n\n"
            f"This is deterministic mock content for offline DeepResearch testing. "
            f"It summarizes evidence related to {query}. "
            f"Reported benchmark value: {numeric_seed}. "
            f"Reported comparison value: {numeric_seed + 7}. "
            f"The source is intentionally stable across runs so Researcher, Distiller, and Writer can be exercised without live network access. "
            f"Source URL: {url}."
        )
        return ScrapedData(
            url=url,
            markdown=markdown,
            title=title,
            fetch_method="mock",
        )


def resolve_scraper_mode(mode: Optional[str] = None) -> str:
    candidate = (mode or os.getenv("RESEARCHER_SCRAPER_MODE", "live")).strip().lower()
    if candidate not in {"live", "mock"}:
        return "live"
    return candidate


def build_scraper(
    mode: Optional[str] = None,
    *,
    fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ScraperInterface:
    resolved = resolve_scraper_mode(mode)
    if resolved == "mock":
        return MockScraper(fixtures=fixtures)
    return SmartScraperAdapter()
