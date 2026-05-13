import pytest

from agents.distiller import run_distiller
from agents.writer import run_writer
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_RESEARCHER_OUTPUTS,
    MOCK_SECTION_GOALS,
)


@pytest.mark.asyncio
async def test_writer_generates_markdown_report_from_stable_evidence_inputs():
    distiller_outputs = await run_distiller(
        task_id="task-1",
        task={"id": "task-1", "query": "AI chip market 2026"},
        researcher_outputs=MOCK_RESEARCHER_OUTPUTS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
    )

    report = await run_writer(
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        section_evidence_packs=distiller_outputs.section_evidence_packs,
    )

    assert report.markdown.startswith("# ")
    assert len(report.markdown) > 500
    assert "## Executive Summary" in report.markdown
    assert "## Open Questions / Research Gaps" in report.markdown
    assert len(report.section_ids) == len(MOCK_REPORT_OUTLINE["sections"])
    assert len(report.evidence_pack_ids) > 0
    assert isinstance(report.citation_map, dict)
