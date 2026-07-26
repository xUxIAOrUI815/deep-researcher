from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from console_app.service import ResearchConsoleService
from deep_researcher.application import ResearchCreateRequest
from tests.fixtures.background001_application import (
    STATEMENT,
    build_deterministic_application,
)


def test_console_shell_routes_render_html(tmp_path: Path):
    runtime, tools, _ = build_deterministic_application(tmp_path / "runtime")
    app = create_app(runtime_dir=str(tmp_path / "runtime"), runtime=runtime)
    try:
        with TestClient(app) as client:
            for route in ["/", "/console/demo-run", "/report/demo-run"]:
                response = client.get(route)
                assert response.status_code == 200
                assert "DeepResearcher 研究控制台" in response.text
                assert "/static/app.js" in response.text
    finally:
        asyncio.run(runtime.aclose())
        tools.close()


@pytest.mark.asyncio
async def test_console_service_projects_the_production_application_runtime(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(tmp_path / "runtime")
    service = ResearchConsoleService(
        runtime_dir=str(tmp_path / "runtime"),
        runtime=runtime,
    )
    try:
        created = await service.create_run(
            ResearchCreateRequest(
                query="Compare verified benchmark claims across vendors",
                instructions="Preserve conflicts and cite verified sources.",
                depth="standard",
            )
        )

        summary = None
        for _ in range(200):
            summary = await service.get_console_summary(created.research_id)
            if summary.status in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.01)

        assert summary is not None
        assert summary.status == "completed"
        assert summary.current_stage == "completed"
        assert summary.knowledge_summary.fact_count > 0
        assert summary.knowledge_summary.claim_count > 0
        assert summary.knowledge_summary.evidence_count > 0
        assert summary.context_summary.planner["required_sections"]
        assert summary.context_summary.researcher["source_count"] > 0
        assert summary.timeline
        assert summary.run_metadata["scheduler_status"] == "completed"

        report = await service.get_report_view(created.research_id)
        debug = await service.get_debug_view(created.research_id)

        assert STATEMENT in report.markdown
        assert report.knowledge_summary.section_pack_count > 0
        assert debug.trace
        assert debug.state_summary["terminal_event_id"]
        assert debug.raw_state["projection_schema"] == "StudioProjection@1"
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()
