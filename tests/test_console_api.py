from __future__ import annotations

import asyncio
import shutil
import tempfile
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from console_app.service import ResearchConsoleService
from schemas.console import ResearchCreateRequest


@pytest.fixture
def console_tmp_dir():
    path = Path(tempfile.gettempdir()) / "mini-deep-research-console-tests" / uuid.uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def test_console_shell_routes_render_html(console_tmp_dir):
    app = create_app(runtime_dir=str(console_tmp_dir))
    with TestClient(app) as client:
        for route in ["/", "/console/demo-run", "/report/demo-run"]:
            response = client.get(route)
            assert response.status_code == 200
            assert "DeepResearcher 研究控制台" in response.text
            assert "/static/app.js" in response.text


@pytest.mark.asyncio
async def test_console_service_exposes_console_and_report_views_for_offline_run(console_tmp_dir, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")

    service = ResearchConsoleService(runtime_dir=str(console_tmp_dir))
    try:
        created = await service.create_run(
            ResearchCreateRequest(
                query="Compare HBM4 timeline claims across vendors",
                instructions="Focus on conflicts and primary sources",
                depth="standard",
            )
        )

        summary = None
        for _ in range(80):
            summary = await service.get_console_summary(created.research_id)
            if summary.status in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)

        assert summary is not None
        assert summary.status == "completed"
        assert summary.current_stage == "completed"
        assert summary.knowledge_summary.fact_count > 0
        assert summary.context_summary.planner["coverage"] >= 0.0
        assert summary.timeline

        report = await service.get_report_view(created.research_id)
        debug = await service.get_debug_view(created.research_id)

        assert report.markdown
        assert report.knowledge_summary.section_pack_count > 0
        assert debug.trace
        assert "planner_action" in debug.state_summary
    finally:
        await service.aclose()
