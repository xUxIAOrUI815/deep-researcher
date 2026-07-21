from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from console_app.service import ResearchConsoleService
from schemas.console import ResearchCreateRequest


@pytest.mark.asyncio
async def test_studio_service_survives_restart_and_never_uses_graph_state_for_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")
    service = ResearchConsoleService(str(tmp_path))
    try:
        created = await service.create_run(ResearchCreateRequest(query="persistent Studio trace"))
        for _ in range(80):
            runs = await service.list_runs()
            if runs and runs[0]["status"] in {"completed", "failed"}:
                break
            await asyncio.sleep(0.25)
        run_id = service._resolve_run_id(created.research_id)
        assert run_id

        async def forbidden(*args, **kwargs):
            raise AssertionError("Studio projection API must not read LangGraph state")

        monkeypatch.setattr(service, "_load_graph_state", forbidden)
        run = service.get_studio_run(run_id)
        spans = service.list_studio_spans(run_id, limit=1000)
        timeline = service.get_timeline_page(__import__("deep_researcher.studio", fromlist=["TimelineQuery"]).TimelineQuery(run_id, limit=1000))
        assert run["status"] == "succeeded"
        assert spans["items"] and timeline["items"]
    finally:
        await service.aclose()

    restarted = ResearchConsoleService(str(tmp_path))
    try:
        run = restarted.get_studio_run(run_id)
        assert run["status"] == "succeeded"
        assert restarted.export_studio_trace(run_id)["events"]
    finally:
        await restarted.aclose()


def test_studio_http_filters_streams_and_exports_trace(tmp_path, monkeypatch):
    monkeypatch.setenv("RESEARCHER_SCRAPER_MODE", "mock")
    monkeypatch.setenv("RESEARCHER_SEARCH_MODE", "mock")
    app = create_app(str(tmp_path))
    with TestClient(app) as client:
        created = client.post("/api/runs", json={"query": "Studio API test", "instructions": "", "depth": "quick"}).json()
        summary = None
        for _ in range(80):
            summary = client.get(f"/api/runs/{created['research_id']}/console").json()
            if summary["status"] in {"completed", "failed"}:
                break
            import time
            time.sleep(0.1)
        assert summary["status"] == "completed"
        run_id = summary["timeline"][0]["run_id"]
        threads = client.get("/api/studio/threads").json()
        runs = client.get("/api/studio/runs", params={"limit": 1}).json()
        assert threads["items"] and runs["items"]
        page = client.get(
            f"/api/studio/runs/{run_id}/timeline",
            params={"event_types": "run_started,run_completed", "limit": 1},
        ).json()
        assert page["items"][0]["event_type"] == "run_started"
        assert page["next_after_sequence"] is not None
        second = client.get(
            f"/api/studio/runs/{run_id}/timeline",
            params={"event_types": "run_started,run_completed", "after_sequence": page["next_after_sequence"], "limit": 1},
        ).json()
        assert second["items"][0]["event_type"] == "run_completed"
        stream = client.get(f"/api/studio/runs/{run_id}/timeline/stream", params={"follow": "false"})
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert "event: timeline" in stream.text
        exported = client.get(f"/api/studio/runs/{run_id}/export").json()
        assert exported["schema"] == "StudioTraceExport@1"
        ndjson = client.get(f"/api/studio/runs/{run_id}/export", params={"format": "ndjson"})
        assert ndjson.headers["content-type"].startswith("application/x-ndjson")
        assert json.loads(ndjson.text.splitlines()[0])["record_type"] == "thread"
