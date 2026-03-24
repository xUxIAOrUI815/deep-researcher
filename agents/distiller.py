import asyncio
import os
import json
import re
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from schemas.state import AtomicFact


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com"


class DistillationError(Exception):
    pass


@dataclass
class DistillationResult:
    facts: List[AtomicFact]
    summary: str
    raw_response: str


class DistillerAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(2)
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_request_interval = 1.5

    async def distill(self, markdown_text: str, source_url: str, task_id: Optional[str] = None) -> DistillationResult:
        async with self._semaphore:
            async with self._request_lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self._last_request_time
                if elapsed < self._min_request_interval:
                    await asyncio.sleep(self._min_request_interval - elapsed)
                self._last_request_time = asyncio.get_event_loop().time()

            if not self.api_key:
                raise DistillationError("DEEPSEEK_API_KEY not set in environment")

            prompt = self._build_prompt(markdown_text)

            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=90.0) as client:
                        response = await client.post(
                            f"{DEEPSEEK_API_BASE}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "messages": [
                                    {"role": "system", "content": self._get_system_prompt()},
                                    {"role": "user", "content": prompt}
                                ],
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature
                            }
                        )

                        if response.status_code == 429:
                            wait_time = 2 ** attempt
                            print(f"[DistillerAgent] Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        if not raw_content:
                            raise DistillationError(f"Empty response from DeepSeek API for {source_url}")

                        facts = self._parse_facts_from_response(raw_content, source_url, task_id)

                        if not facts:
                            raise DistillationError(f"No valid facts parsed from response for {source_url}")

                        summary = self._extract_summary(raw_content)

                        print(f"[DistillerAgent] Successfully distilled {len(facts)} facts from {source_url}")

                        return DistillationResult(
                            facts=facts,
                            summary=summary,
                            raw_response=raw_content
                        )

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait_time = 2 ** attempt
                        print(f"[DistillerAgent] Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise DistillationError(f"HTTP error calling DeepSeek API: {e}")

                except DistillationError:
                    raise

                except Exception as e:
                    raise DistillationError(f"Unexpected error calling DeepSeek API: {e}")

            raise DistillationError(f"Max retries ({3}) exceeded for {source_url}")

    def _get_system_prompt(self) -> str:
        return """你是一个专业的原子事实提炼专家。你的任务是将网页内容拆解为语义完整、不可分割的事实陈述。

核心原则：
1. 每个事实必须语义完整，单独拎出来阅读时无需额外上下文
2. 实体必须明确，不使用代词（将"该公司"替换为具体公司名如"华为"）
3. 每条事实必须直接关联数值、时间或具体事件
4. 事实来源必须可信，基于文本内容推断原始来源权威性

输出格式要求：
- 使用JSON数组格式输出
- 每个fact对象包含：text(事实文本), confidence(可信度0.0-1.0)
- 事实按重要性和可信度降序排列
- 提取5-15条核心事实

注意：只输出JSON数组，不要有其他文字。"""

    def _build_prompt(self, markdown_text: str) -> str:
        truncated = markdown_text[:8000]

        return f"""请分析以下文本，提取原子事实：

{'-'*60}
{truncated}
{'-'*60}

要求：
1. 提取所有包含具体数据、时间、公司/人物名称的事实
2. 每条事实必须语义完整
3. 将所有代词替换为具体名称
4. 根据来源权威性给出置信度评分
5. 仅返回JSON格式的事实数组"""

    def _parse_facts_from_response(
        self,
        raw_response: str,
        source_url: str,
        task_id: Optional[str] = None
    ) -> List[AtomicFact]:
        facts = []

        try:
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                fact_data = json.loads(json_match.group())

                for item in fact_data:
                    if isinstance(item, dict) and "text" in item:
                        fact = AtomicFact(
                            text=item["text"],
                            source_url=source_url,
                            confidence=float(item.get("confidence", 0.7)),
                            task_id=task_id
                        )
                        facts.append(fact)
        except json.JSONDecodeError as e:
            raise DistillationError(f"Failed to parse JSON from DeepSeek response: {e}")

        if not facts:
            raise DistillationError(f"No facts found in JSON response for {source_url}")

        return facts

    def _extract_summary(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        summary_lines = []

        for line in lines[:3]:
            if line.strip() and len(line.strip()) > 20:
                clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
                summary_lines.append(clean.strip())

        return " | ".join(summary_lines[:2])

    async def distill_batch(
        self,
        items: List[Dict[str, str]]
    ) -> List[DistillationResult]:
        async def distill_with_url(item: Dict[str, str]) -> DistillationResult:
            text = item.get("markdown", item.get("text", ""))
            url = item.get("source_url", item.get("url", ""))
            tid = item.get("task_id")
            return await self.distill(text, url, tid)

        tasks = [distill_with_url(item) for item in items]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[DistillerAgent] Batch item {i} failed: {result}")
                raise result
            processed.append(result)

        return processed
