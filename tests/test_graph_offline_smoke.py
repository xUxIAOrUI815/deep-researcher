import pytest

from core.graph import create_research_graph
from tests.fixtures.offline_research_inputs import build_initial_graph_state


@pytest.mark.asyncio
async def test_graph_offline_smoke_runs_with_mock_research_modes(monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    graph = create_research_graph(None)
    result = await graph.ainvoke(
        build_initial_graph_state(),
        {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}},
    )

    assert result["final_report"] is not None
    assert "markdown" in result["final_report"]
    assert len(result["final_report"]["markdown"]) > 200
    assert len(result.get("section_evidence_packs", [])) > 0
