import asyncio
from typing import List, Any
from core.knowledge import KnowledgeManager
from core.config import ResearchConfig


class SaturationResult:
    def __init__(
        self,
        is_saturated: bool,
        repetition_rate: float,
        new_facts_count: int,
        similar_count: int,
        forced_finish: bool = False
    ):
        self.is_saturated = is_saturated
        self.repetition_rate = repetition_rate
        self.new_facts_count = new_facts_count
        self.similar_count = similar_count
        self.forced_finish = forced_finish


class SaturationChecker:

    def __init__(self, knowledge_manager: KnowledgeManager):
        self.knowledge = knowledge_manager

    async def check_saturation(
        self,
        new_facts: List[Any],
        collection_name: str,
        is_user_triggered: bool = False
    ) -> SaturationResult:
        if not new_facts:
            return SaturationResult(
                is_saturated=False,
                repetition_rate=0.0,
                new_facts_count=0,
                similar_count=0
            )

        if is_user_triggered:
            print(f"[SaturationChecker] User-triggered task, skipping saturation check")
            return SaturationResult(
                is_saturated=False,
                repetition_rate=0.0,
                new_facts_count=len(new_facts),
                similar_count=0,
                forced_finish=False
            )

        similar_count = 0
        total_checked = 0

        for fact in new_facts:
            fact_text = getattr(fact, 'text', None)
            if not fact_text:
                continue

            total_checked += 1

            similar = await self.knowledge._find_similar(
                collection_name,
                fact_text,
                threshold=ResearchConfig.SATURATION_SIMILARITY_THRESHOLD
            )

            if len(similar) > 0:
                similar_count += 1

        if total_checked == 0:
            return SaturationResult(
                is_saturated=False,
                repetition_rate=0.0,
                new_facts_count=len(new_facts),
                similar_count=0
            )

        repetition_rate = similar_count / total_checked if total_checked > 0 else 0.0

        forced_finish = repetition_rate > ResearchConfig.SATURATION_SIMILARITY_THRESHOLD

        if forced_finish:
            print(f"[SaturationChecker] ⚠️ 信息饱和检测: R={repetition_rate:.2%} > {ResearchConfig.SATURATION_SIMILARITY_THRESHOLD:.0%}, 强制结束")
        else:
            print(f"[SaturationChecker] 饱和度检查: {similar_count}/{total_checked} 相似 (R={repetition_rate:.2%})")

        return SaturationResult(
            is_saturated=repetition_rate > ResearchConfig.SATURATION_SIMILARITY_THRESHOLD,
            repetition_rate=repetition_rate,
            new_facts_count=len(new_facts),
            similar_count=similar_count,
            forced_finish=forced_finish
        )


async def check_information_saturation(
    new_facts: List[Any],
    collection_name: str,
    knowledge_manager: KnowledgeManager,
    is_user_triggered: bool = False
) -> SaturationResult:
    checker = SaturationChecker(knowledge_manager)
    return await checker.check_saturation(new_facts, collection_name, is_user_triggered)
