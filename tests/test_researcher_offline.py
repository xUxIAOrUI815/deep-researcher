import pytest

from agents.researcher import run_researcher
from tests.fixtures.offline_research_inputs import MOCK_KNOWLEDGE_REFS, MOCK_TASK, ROOT_QUERY


@pytest.mark.asyncio
async def test_researcher_runs_in_mock_mode_and_returns_stable_outputs():
    outputs = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        scraper_mode="mock",
        search_mode="mock",
        enable_scraping=True,
    )

    assert outputs.task_id == "task-1"
    assert len(outputs.queries) > 0
    assert len(outputs.sources) > 0
    assert len(outputs.passages) > 0
    assert outputs.metadata["scraper_mode"] == "mock"
    assert outputs.metadata["search_mode"] == "mock"
    assert outputs.metadata["stop_reason"] in {
        "max_search_iterations_reached",
        "query_budget_exhausted",
        "marginal_gain_stop",
    }
    assert all(source.get("scraper_mode") == "mock" for source in outputs.sources)
    assert all(passage.get("extraction_method") == "mock" for passage in outputs.passages)
    assert outputs.summary


@pytest.mark.asyncio
async def test_researcher_records_recoverable_search_errors_without_failing_run():
    class FailingSearchGateway:
        async def search(self, query: str, max_results: int = 5, provider: str = "tavily"):
            raise RuntimeError("tavily_search failed: ReadTimeout: temporary timeout")

    outputs = await run_researcher(
        task_id="task-1",
        task=MOCK_TASK,
        root_user_query=ROOT_QUERY,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
        scraper_mode="mock",
        search_mode="live",
        search_gateway=FailingSearchGateway(),
        enable_scraping=True,
    )

    assert outputs.task_id == "task-1"
    assert outputs.sources == []
    assert outputs.passages == []
    assert outputs.metadata["stop_reason"] == "search_errors_exhausted"
    assert outputs.metadata["recoverable_failure"] is True
    assert outputs.metadata["search_errors"]
    assert all(item["recoverable"] for item in outputs.metadata["search_errors"])
