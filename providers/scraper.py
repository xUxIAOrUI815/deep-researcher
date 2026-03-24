import asyncio
import httpx
import os
import re
from typing import Optional, Literal, Dict, Any, List
from datetime import datetime
from dataclasses import dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from httpx import HTTPStatusError, TimeoutException

from schemas.state import ScrapedData

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")


NAVIGATION_PATTERNS = [
    r"(?i)(navbar|nav-menu|header|sidebar|footer|cookie|banner|popup|modal)",
    r"(?i)(login|signin|register|subscribe|newsletter)",
    r"(?i)(copyright|©|terms|privacy|policy)",
    r"(?i)(social media|facebook|twitter|linkedin)",
]

AD_PLACEHOLDERS = [
    r"(?i)(advertisement|ad-|ads-|sponsor)",
    r"(?i)sidebar",
    r"(?i)related posts",
]

CHINESE_SIDEBAR_PATTERNS = [
    r"(?i)(相关阅读|相关文章，相关推荐|猜你喜欢|为你推荐|热门文章|热点新闻)",
    r"(?i)(合作伙伴|广告合作|关于我们|联系 我们)",
    r"(?i)(最新资讯|今日热点|实时新闻)",
]

RISK_LEGAL_PATTERNS = [
    r"(?i)(风险提示|风险揭示|风险声明|投资有风险)",
    r"(?i)(免责声明|法律声明|用户协议|隐私政策|服务条款)",
    r"(?i)(copyright\s*©)",
]

PLACEHOLDER_PATTERNS = [
    r'!\[\]\(javascript:;\)',
    r'!\[\]\([^)]+\)',
    r'\[](javascript:;|#)',
    r'^\s*\[?\s*\]\s*\(?\s*(javascript:|#)\s*\)?\s*$',
]


@dataclass
class DenoiseStats:
    original_length: int = 0
    cleaned_length: int = 0
    link_density_removed: int = 0
    risk_block_removed: int = 0
    semantic_score: float = 0.0
    was_discarded: bool = False

    @property
    def denoise_rate(self) -> float:
        if self.original_length == 0:
            return 0.0
        return (self.original_length - self.cleaned_length) / self.original_length


class SemanticFilter:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or ZHIPU_API_KEY
        self.base_url = "https://open.bigmodel.cn/api/paas/v4"

    async def score_relevance(self, content: str, query: str) -> float:
        if not self.api_key or self.api_key == "":
            return 0.8

        prompt = f"""判断以下内容是否包含与调研任务「{query}」相关的核心事实或数据？

内容片段：
{content[:2000]}

请仅返回一个 0.0 到 1.0 之间的数字分数，其中：
- 1.0 = 内容高度相关，包含核心事实和数据
- 0.5 = 内容部分相关，但有较多无关信息
- 0.0 = 内容完全不相关，是广告或噪音

直接返回数字，不要其他文字："""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "glm-4-flash",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0.1
                    }
                )
                response.raise_for_status()
                data = response.json()

                content_text = data.get("choices", [{}])[0].get("message", {}).get("content", "0.5")
                score = float(content_text.strip())

                return max(0.0, min(1.0, score))

        except Exception as e:
            print(f"[SemanticFilter] Error scoring relevance: {e}")
            return 0.5


class SmartScraper:
    def __init__(
        self,
        jina_api_base: str = "https://r.jina.ai/",
        timeout: float = 20.0,
        maxConcurrency: int = 3
    ):
        self.jina_api_base = jina_api_base
        self.jina_api_key = JINA_API_KEY
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(maxConcurrency)
        self._playwright_available = False
        self.semantic_filter = SemanticFilter()
        self._token_savings = 0

    async def _fetch_with_playwright(self, url: str) -> ScrapedData:
        try:
            from playwright.async_api import async_playwright

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                page = await browser.new_page()

                await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)

                title = await page.title()

                markdown = await page.evaluate("""
                    () => {
                        const main = document.querySelector('article') || document.querySelector('main') || document.body;
                        return main.innerText;
                    }
                """)

                await browser.close()

                return ScrapedData(
                    url=url,
                    markdown=markdown,
                    title=title,
                    fetch_method="playwright",
                    timestamp=datetime.now()
                )

        except ImportError:
            return ScrapedData(
                url=url,
                markdown="",
                title="",
                fetch_method="playwright",
                timestamp=datetime.now(),
                error="Playwright not installed"
            )
        except Exception as e:
            return ScrapedData(
                url=url,
                markdown="",
                title="",
                fetch_method="playwright",
                timestamp=datetime.now(),
                error=str(e)
            )

    def _level1_heuristic_clean(self, markdown: str) -> tuple[str, DenoiseStats]:
        stats = DenoiseStats()
        stats.original_length = len(markdown)

        lines = markdown.split("\n")
        cleaned_lines = []
        skip_mode = False
        risk_block_buffer = []
        in_risk_block = False

        for line in lines:
            line_lower = line.lower().strip()

            if re.match(r'^#{1,3}\s*(navbar|nav|menu|header|footer|sidebar)', line_lower):
                continue

            if any(re.search(pattern, line_lower) for pattern in NAVIGATION_PATTERNS):
                if "content" not in line_lower and len(line.strip()) < 50:
                    continue

            skip_line = False
            for pattern in AD_PLACEHOLDERS:
                try:
                    if re.search(pattern, line_lower):
                        skip_line = True
                        break
                except re.error:
                    continue
            if skip_line:
                continue

            for pattern in CHINESE_SIDEBAR_PATTERNS:
                if re.search(pattern, line_lower):
                    skip_line = True
                    break
            if skip_line:
                continue

            if line.strip().startswith("![](") and any(kw in line_lower for kw in ["ad", "banner", "promo"]):
                continue

            for pattern in PLACEHOLDER_PATTERNS:
                if re.match(pattern, line.strip()):
                    continue

            link_count = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', line))
            link_chars = sum(len(m) for m in re.findall(r'\[([^\]]+)\]\([^)]+\)', line))
            total_content_chars = len(re.sub(r'\[([^\]]+)\]\([^)]+\)', '', line))
            if total_content_chars > 0 and link_count > 0:
                link_ratio = link_chars / (link_chars + total_content_chars)
                if link_ratio > 0.4:
                    stats.link_density_removed += len(line)
                    continue

            risk_match = False
            for pattern in RISK_LEGAL_PATTERNS:
                if re.search(pattern, line_lower):
                    risk_match = True
                    break

            if risk_match:
                if not in_risk_block:
                    in_risk_block = True
                    risk_block_buffer = []
                risk_block_buffer.append(line)
                if len(line) > 100 or len(risk_block_buffer) > 5:
                    stats.risk_block_removed += sum(len(l) for l in risk_block_buffer)
                    risk_block_buffer = []
                continue
            else:
                if in_risk_block and risk_block_buffer:
                    combined_block = "\n".join(risk_block_buffer)
                    if len(combined_block) > 500 and ('$' in combined_block or '基金' in combined_block):
                        stats.risk_block_removed += sum(len(l) for l in risk_block_buffer)
                    else:
                        cleaned_lines.extend(risk_block_buffer)
                    risk_block_buffer = []
                in_risk_block = False

            if re.match(r'^\s*[-*]\s*(login|sign in|register|subscribe)\s*$', line_lower):
                continue

            if len(line.strip()) < 2 and len(cleaned_lines) > 0 and cleaned_lines[-1].strip() == "":
                continue

            cleaned_lines.append(line)

        if risk_block_buffer and not in_risk_block:
            cleaned_lines.extend(risk_block_buffer)

        result = "\n".join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)

        lines_before = len(result.split("\n"))
        lines_after = len([l for l in result.split("\n") if l.strip()])
        if lines_after > 0 and lines_before > lines_after * 2:
            result = "\n".join([l for l in result.split("\n") if l.strip()])

        result = result.strip()
        stats.cleaned_length = len(result)

        return result, stats

    def _level3_token_optimize(self, markdown: str) -> str:
        lines = markdown.split("\n")
        optimized_lines = []
        prev_heading = None

        for line in lines:
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if heading_match:
                heading_level, heading_text = heading_match.groups()
                if heading_text == prev_heading:
                    continue
                prev_heading = heading_text
            else:
                if not line.strip().startswith("#"):
                    prev_heading = None

            optimized_lines.append(line)

        result = "\n".join(optimized_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)

        return result.strip()

    async def _level2_semantic_filter(
        self,
        content: str,
        query: str,
        url: str
    ) -> tuple[bool, float]:
        score = await self.semantic_filter.score_relevance(content, query)

        print(f"[SemanticFilter] URL: {url}")
        print(f"[SemanticFilter] Relevance Score: {score:.2f}")

        if score < 0.5:
            print(f"[SemanticFilter] Score < 0.5, URL Discarded")
            return False, score

        return True, score

    async def scrape_with_denoise(
        self,
        url: str,
        query: str,
        force_playwright: bool = False
    ) -> tuple[ScrapedData, DenoiseStats]:
        stats = DenoiseStats()

        scraped = await self.scrape(url, force_playwright)

        if scraped.error or not scraped.markdown:
            stats.was_discarded = True
            return scraped, stats

        original_md = scraped.markdown
        stats.original_length = len(original_md)

        level1_md, stats = self._level1_heuristic_clean(original_md)

        keep, stats.semantic_score = await self._level2_semantic_filter(
            level1_md, query, url
        )

        if not keep:
            stats.was_discarded = True
            return ScrapedData(
                url=url,
                markdown="",
                title=scraped.title,
                fetch_method=scraped.fetch_method,
                timestamp=datetime.now(),
                error="Semantic filter discarded"
            ), stats

        level3_md = self._level3_token_optimize(level1_md)
        stats.cleaned_length = len(level3_md)

        token_saved = stats.original_length - stats.cleaned_length
        self._token_savings += token_saved

        scraped.markdown = level3_md

        return scraped, stats

    async def scrape_batch_with_denoise(
        self,
        urls: List[str],
        query: str,
        force_playwright: bool = False
    ) -> tuple[List[ScrapedData], List[DenoiseStats]]:
        tasks = [
            self.scrape_with_denoise(url, query, force_playwright)
            for url in urls
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped_data = []
        all_stats = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                stats = DenoiseStats(was_discarded=True)
                scraped_data.append(ScrapedData(
                    url=urls[i],
                    markdown="",
                    title="",
                    fetch_method="unknown",
                    timestamp=datetime.now(),
                    error=str(result)
                ))
                all_stats.append(stats)
            else:
                scraped, stats = result
                scraped_data.append(scraped)
                all_stats.append(stats)

        return scraped_data, all_stats

    def get_token_savings(self) -> int:
        return self._token_savings

    async def _fetch_with_jina(self, url: str) -> ScrapedData:
        jina_url = f"{self.jina_api_base}{url}"

        headers = {}
        if self.jina_api_key:
            headers["Authorization"] = f"Bearer {self.jina_api_key}"

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(jina_url, headers=headers)
            response.raise_for_status()

            text = response.text

            title_match = re.search(r'#\s*(.+?)\n', text)
            title = title_match.group(1) if title_match else url

            return ScrapedData(
                url=url,
                markdown=text,
                title=title,
                fetch_method="jina",
                timestamp=datetime.now()
            )

    async def scrape(self, url: str, force_playwright: bool = False) -> ScrapedData:
        async with self.semaphore:
            if force_playwright:
                return await self._fetch_with_playwright(url)

            try:
                result = await self._fetch_with_jina(url)
                if result.markdown and len(result.markdown) > 100:
                    return result
                else:
                    print(f"[SmartScraper] Jina returned insufficient content for {url}, falling back to Playwright")
                    return await self._fetch_with_playwright(url)

            except HTTPStatusError as e:
                if e.response.status_code == 403 or e.response.status_code == 429:
                    print(f"[SmartScraper] Jina got {e.response.status_code}, falling back to Playwright")
                    return await self._fetch_with_playwright(url)
                raise
            except TimeoutException:
                print(f"[SmartScraper] Jina timeout for {url}, falling back to Playwright")
                return await self._fetch_with_playwright(url)
            except Exception as e:
                print(f"[SmartScraper] Jina error: {e}, falling back to Playwright")
                return await self._fetch_with_playwright(url)

    async def scrape_batch(self, urls: list[str], force_playwright: bool = False) -> list[ScrapedData]:
        tasks = [self.scrape(url, force_playwright) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        scraped = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                scraped.append(ScrapedData(
                    url=urls[i],
                    markdown="",
                    title="",
                    fetch_method="unknown",
                    timestamp=datetime.now(),
                    error=str(result)
                ))
            else:
                scraped.append(result)

        return scraped

    async def close(self):
        pass
