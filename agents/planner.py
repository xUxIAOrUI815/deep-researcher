import asyncio
import os
import json
import re
import httpx
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from schemas.state import TaskNode, AtomicFact, SourceLevel
from core.config import ModelConfig


class PlannerError(Exception):
    pass


@dataclass
class PlannerDecision:
    action: str
    new_tasks: List[Dict[str, Any]]
    reasoning: str


class BasePlannerAgent:
    SYSTEM_PROMPT = """你是 AIRE 首席情报分析官。你的职责是评估当前调研进度并决定是否需要"深挖"。

1. **全量事实分析**：基于 all_fact_ids 对应的全部历史事实（而非仅本轮）判断信息饱和度。若已有事实足够回答初始问题，输出 finish。

2. **线索识别**：扫描全部事实，寻找：(a) 关键技术瓶颈 (b) 未定义的专业实体 (c) 数据冲突点 (d) 来源不可信的孤证。

3. **增量拆解**：若发现上述线索且当前深度 < {max_depth}，生成 1-2 个具体的子任务。

4. **任务要求**：Query 必须是原子化的（如"ASML High-NA 光刻机 2025 年交付计划"），严禁泛化。

5. **收敛判断**：若已有事实足够回答初始调研目标，或新信息无法提供显著增益，必须输出 finish。

6. **任务去重**：若计划生成的子任务与已存在任务的 Query 语义相似度 > 90%，直接丢弃该任务。

输出格式（必须严格遵循）：
{{"action": "continue" 或 "finish", "new_tasks": [{{"query": "...", "priority": 0.0-1.0, "reason": "...", "parent_id": "...", "depth": 0}}], "reasoning": "决策理由"}}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ):
        self.api_key = api_key
        self.model_name = model_name
        self.api_base = api_base
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(2)
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_request_interval = 1.5

    async def plan(
        self,
        task_tree: Dict[str, Any],
        root_task_id: Optional[str],
        completed_facts: List[Dict[str, Any]],
        current_depth: int,
        max_depth: int = 3,
        is_user_triggered: bool = False
    ) -> PlannerDecision:
        async with self._semaphore:
            async with self._request_lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self._last_request_time
                if elapsed < self._min_request_interval:
                    await asyncio.sleep(self._min_request_interval - elapsed)
                self._last_request_time = asyncio.get_event_loop().time()

            if not self.api_key:
                raise PlannerError(f"API key not set for {self.__class__.__name__}")

            context = self._build_context(task_tree, root_task_id, completed_facts, current_depth, max_depth)
            prompt = self._build_prompt(context, max_depth, is_user_triggered)

            for attempt in range(3):
                try:
                    response = await self._call_api(prompt, max_depth)
                    decision = self._parse_decision(response, root_task_id, current_depth)
                    return decision
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait_time = 2 ** attempt
                        print(f"[PLANNER:{self.__class__.__name__}] Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise PlannerError(f"HTTP error calling API: {e}")
                except Exception as e:
                    if attempt == 2:
                        raise PlannerError(f"Max retries exceeded: {e}")
                    print(f"[PLANNER:{self.__class__.__name__}] Attempt {attempt + 1} failed: {e}")
                    await asyncio.sleep(2 ** attempt)

            raise PlannerError(f"Max retries exceeded for {self.__class__.__name__} planning")

    async def _call_api(self, prompt: str, max_depth: int) -> str:
        raise NotImplementedError("Subclasses must implement _call_api method")

    def _build_context(
        self,
        task_tree: Dict[str, Any],
        root_task_id: Optional[str],
        completed_facts: List[Dict[str, Any]],
        current_depth: int,
        max_depth: int
    ) -> str:
        context_parts = []

        context_parts.append(f"=== 任务树状态 (深度 {current_depth}/{max_depth}) ===")
        if root_task_id and root_task_id in task_tree:
            root = task_tree[root_task_id]
            context_parts.append(f"根任务: {root.get('query', '')} (ID: {root_task_id})")
            context_parts.append(f"深度: {root.get('depth', 0)}, 优先级: {root.get('priority', 0)}")
            context_parts.append(f"状态: {root.get('status', 'unknown')}")

            children = root.get('children_ids', [])
            if children:
                context_parts.append(f"子任务数: {len(children)}")
                for child_id in children[:5]:
                    if child_id in task_tree:
                        child = task_tree[child_id]
                        context_parts.append(f"  - {child.get('query', '')[:50]} (depth={child.get('depth', 0)})")
        else:
            context_parts.append("根任务尚未创建")

        context_parts.append(f"\n=== 已完成事实 ({len(completed_facts)} 条) ===")
        for i, fact in enumerate(completed_facts[:10]):
            conflict_mark = " ⚠️ 冲突" if fact.get('is_conflict') else ""
            level_mark = f"[{fact.get('source_level', 'C')}]"
            context_parts.append(f"{i+1}. {level_mark} {fact.get('text', '')[:80]}... (置信度: {fact.get('confidence', 0):.2f}){conflict_mark}")

        if len(completed_facts) > 10:
            context_parts.append(f"... 还有 {len(completed_facts) - 10} 条事实")

        conflicts = [f for f in completed_facts if f.get('is_conflict')]
        if conflicts:
            context_parts.append(f"\n=== 冲突点 ({len(conflicts)} 处) ===")
            for conflict in conflicts[:3]:
                context_parts.append(f"⚠️ {conflict.get('text', '')[:60]}...")
                conflict_with = conflict.get('conflict_with', [])
                if conflict_with:
                    context_parts.append(f"   冲突ID: {conflict_with[:2]}")

        return "\n".join(context_parts)

    def _build_prompt(
        self,
        context: str,
        max_depth: int,
        is_user_triggered: bool
    ) -> str:
        user_trigger_note = "\n\n[重要] 用户明确要求继续深挖此问题(is_user_triggered=True)，请务必生成具体的子任务。" if is_user_triggered else ""

        return f"""基于以下调研上下文，分析是否需要继续深挖，并决定下一步行动。

{context}

{user_trigger_note}

请严格按以下格式输出（只输出JSON，不要有其他文字）：
{{"action": "continue" 或 "finish", "new_tasks": [{{"query": "具体问题", "priority": 0.0-1.0, "reason": "发现什么线索", "parent_id": "父任务ID", "depth": 当前深度+1}}], "reasoning": "决策理由"}}"""

    def _parse_decision(
        self,
        raw_response: str,
        root_task_id: Optional[str],
        current_depth: int
    ) -> PlannerDecision:
        try:
            json_match = re.search(r'\{.*\}', raw_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                action = data.get("action", "finish")
                new_tasks = data.get("new_tasks", [])
                reasoning = data.get("reasoning", "")

                for task in new_tasks:
                    if "parent_id" not in task or not task["parent_id"]:
                        task["parent_id"] = root_task_id
                    task["depth"] = current_depth + 1

                return PlannerDecision(
                    action=action,
                    new_tasks=new_tasks,
                    reasoning=reasoning
                )
        except json.JSONDecodeError as e:
            print(f"[PlannerAgent] JSON parse error: {e}, raw: {raw_response[:200]}")

        return PlannerDecision(
            action="finish",
            new_tasks=[],
            reasoning=f"解析失败，默认结束: {raw_response[:100]}"
        )


class GLMPlannerAgent(BasePlannerAgent):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ):
        super().__init__(
            api_key=api_key or ModelConfig.GLM.API_KEY,
            model_name=model_name or ModelConfig.GLM.MODEL_NAME,
            api_base=api_base or ModelConfig.GLM.API_BASE,
            max_tokens=max_tokens,
            temperature=temperature
        )

    async def _call_api(self, prompt: str, max_depth: int) -> str:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT.format(max_depth=max_depth)},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )

            response.raise_for_status()
            data = response.json()

            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not raw_content:
                raise PlannerError("Empty response from GLM API")

            decision = self._parse_decision(raw_content, None, 0)
            print(f"[PLANNER:GLM] Decision: {decision.action}, new_tasks: {len(decision.new_tasks)}")
            print(f"[PLANNER:GLM] Reasoning: {decision.reasoning[:100]}...")

            return raw_content


class DeepSeekPlannerAgent(BasePlannerAgent):
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        api_base: Optional[str] = None,
        max_tokens: int = 2000,
        temperature: float = 0.1
    ):
        super().__init__(
            api_key=api_key or ModelConfig.DeepSeek.API_KEY,
            model_name=model_name or ModelConfig.DeepSeek.MODEL_NAME,
            api_base=api_base or ModelConfig.DeepSeek.API_BASE,
            max_tokens=max_tokens,
            temperature=temperature
        )

    async def _call_api(self, prompt: str, max_depth: int) -> str:
        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(
                f"{self.api_base}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": self.SYSTEM_PROMPT.format(max_depth=max_depth)},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            )

            response.raise_for_status()
            data = response.json()

            raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

            if not raw_content:
                raise PlannerError("Empty response from DeepSeek API")

            decision = self._parse_decision(raw_content, None, 0)
            print(f"[PLANNER:DeepSeek] Decision: {decision.action}, new_tasks: {len(decision.new_tasks)}")
            if decision.reasoning:
                print(f"[PLANNER:DeepSeek] Reasoning: {decision.reasoning[:100]}...")

            return raw_content


class PlannerAgentFactory:
    @staticmethod
    def create_agent(model_type: Optional[str] = None) -> BasePlannerAgent:
        model_type = model_type or ModelConfig.DEFAULT_MODEL
        
        if model_type.lower() == "glm":
            return GLMPlannerAgent()
        elif model_type.lower() == "deepseek":
            return DeepSeekPlannerAgent()
        else:
            raise PlannerError(f"Unsupported model type: {model_type}")


# 为了向后兼容，保留原有的 PlannerAgent 名称
PlannerAgent = DeepSeekPlannerAgent
