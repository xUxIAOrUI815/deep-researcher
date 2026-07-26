from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from console_app.app import create_app
from console_app.service import ResearchConsoleService
from deep_researcher.application import ResearchCreateRequest
from deep_researcher.studio import TimelineQuery
from tests.fixtures.background001_application import (
    build_deterministic_application,
)


@pytest.mark.asyncio
async def test_studio_service_survives_restart_without_graph_state(
    tmp_path: Path,
):
    runtime, tools, _ = build_deterministic_application(tmp_path)
    service = ResearchConsoleService(str(tmp_path), runtime=runtime)
    try:
        created = await service.create_run(
            ResearchCreateRequest(query="persistent Studio trace")
        )
        for _ in range(200):
            record = runtime.application_store.get(created.research_id)
            assert record is not None
            if record.status.value in {"completed", "failed", "cancelled"}:
                break
            await asyncio.sleep(0.01)
        assert record.status.value == "completed"
        run_id = record.run_id

        assert not hasattr(service, "_load_graph_state")
        assert not hasattr(service, "_resolve_run_id")
        run = service.get_studio_run(run_id)
        spans = service.list_studio_spans(run_id, limit=1000)
        timeline = service.get_timeline_page(
            TimelineQuery(run_id=run_id, limit=1000)
        )
        assert run["status"] == "succeeded"
        assert spans["items"] and timeline["items"]
    finally:
        await service.aclose()
        await runtime.aclose()
        tools.close()

    restarted_runtime, restarted_tools, _ = (
        build_deterministic_application(tmp_path)
    )
    restarted = ResearchConsoleService(
        str(tmp_path),
        runtime=restarted_runtime,
    )
    try:
        run = restarted.get_studio_run(run_id)
        assert run["status"] == "succeeded"
        assert restarted.export_studio_trace(run_id)["events"]
    finally:
        await restarted.aclose()
        await restarted_runtime.aclose()
        restarted_tools.close()


def test_studio_http_filters_streams_and_exports_trace(tmp_path: Path):
    runtime, tools, _ = build_deterministic_application(tmp_path)
    app = create_app(str(tmp_path), runtime=runtime)
    try:
        with TestClient(app) as client:
            created = client.post(
                "/api/runs",
                json={
                    "query": "Studio API test",
                    "instructions": "",
                    "depth": "quick",
                },
            ).json()
            summary = None
            for _ in range(200):
                summary = client.get(
                    f"/api/runs/{created['research_id']}/console"
                ).json()
                if summary["status"] in {
                    "completed",
                    "failed",
                    "cancelled",
                }:
                    break
                time.sleep(0.01)
            assert summary is not None
            assert summary["status"] == "completed"
            run_id = summary["run_metadata"]["run_id"]
            threads = client.get("/api/studio/threads").json()
            runs = client.get(
                "/api/studio/runs",
                params={"limit": 1},
            ).json()
            assert threads["items"] and runs["items"]
            page = client.get(
                f"/api/studio/runs/{run_id}/timeline",
                params={
                    "event_types": "run_started,run_completed",
                    "limit": 1,
                },
            ).json()
            assert page["items"][0]["event_type"] == "run_started"
            assert page["next_after_sequence"] is not None
            second = client.get(
                f"/api/studio/runs/{run_id}/timeline",
                params={
                    "event_types": "run_started,run_completed",
                    "after_sequence": page["next_after_sequence"],
                    "limit": 1,
                },
            ).json()
            assert second["items"][0]["event_type"] == "run_completed"
            stream = client.get(
                f"/api/studio/runs/{run_id}/timeline/stream",
                params={"follow": "false"},
            )
            assert stream.headers["content-type"].startswith(
                "text/event-stream"
            )
            assert "event: timeline" in stream.text
            exported = client.get(
                f"/api/studio/runs/{run_id}/export"
            ).json()
            assert exported["schema"] == "StudioTraceExport@1"
            ndjson = client.get(
                f"/api/studio/runs/{run_id}/export",
                params={"format": "ndjson"},
            )
            assert ndjson.headers["content-type"].startswith(
                "application/x-ndjson"
            )
            assert json.loads(ndjson.text.splitlines()[0])[
                "record_type"
            ] == "thread"
    finally:
        asyncio.run(runtime.aclose())
        tools.close()
