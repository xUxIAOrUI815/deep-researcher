from __future__ import annotations

from typing import Any, Dict

from agents.distiller import run_distiller
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_RESEARCHER_OUTPUTS,
    MOCK_SECTION_GOALS,
)


class StubKnowledgeManager:
    def __init__(self, snapshot: Dict[str, Any]):
        self.snapshot = snapshot
        self.calls: list[tuple[str, str]] = []

    def get_session_snapshot(self, research_id: str, session_id: str) -> Dict[str, Any]:
        self.calls.append((research_id, session_id))
        return self.snapshot


async def build_distiller_outputs(
    *,
    task_id: str = "task-1",
    knowledge_refs: Dict[str, Any] | None = None,
) -> Any:
    return await run_distiller(
        task_id=task_id,
        task={"id": task_id, "query": "AI chip market 2026"},
        researcher_outputs=MOCK_RESEARCHER_OUTPUTS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        knowledge_refs=knowledge_refs or MOCK_KNOWLEDGE_REFS,
    )
