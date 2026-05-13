from __future__ import annotations

import pytest

from agents.researcher import run_researcher
from tests.fixtures.offline_research_inputs import MOCK_KNOWLEDGE_REFS, MOCK_TASK, ROOT_QUERY
from tests.fixtures.session_knowledge_fixtures import StubKnowledgeManager


@pytest.mark.asyncio
async def test_researcher_deduplicates_against_session_source_registry():
    baseline = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )
    duplicate_source_ids = [source["source_id"] for source in baseline.sources]

    knowledge_manager = StubKnowledgeManager(
        {
            "source_registry": {
                source_id: {"source_id": source_id}
                for source_id in duplicate_source_ids
            },
            "knowledge_refs": {"source_ids": duplicate_source_ids},
            "facts": [{"id": "fact-1"}],
            "evidence": [{"id": "evidence-1"}],
        }
    )

    filtered = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        knowledge_manager=knowledge_manager,
        research_id="research-1",
        session_id="session-a",
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )

    assert knowledge_manager.calls
    assert set(knowledge_manager.calls) == {("research-1", "session-a")}
    assert filtered.metadata["session_knowledge"]["source_count"] == len(duplicate_source_ids)
    assert filtered.sources == []
    assert filtered.passages == []
    assert filtered.metadata["stop_reason"] == "marginal_gain_stop"
    assert all(item["reason"] == "duplicate_or_already_used" for item in filtered.metadata["rejected_sources"])
