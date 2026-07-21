from __future__ import annotations

from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
import json
from pathlib import Path
import tempfile
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from schemas.console import ResearchCreateRequest
from deep_researcher.studio import TimelineQuery
from .service import ResearchConsoleService

load_dotenv()


def create_app(runtime_dir: str = ".console_runtime") -> FastAPI:
    service = ResearchConsoleService(runtime_dir=runtime_dir)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            await service.aclose()

    app = FastAPI(title="DeepResearcher Console", version="0.1.0", lifespan=lifespan)
    app.state.console_service = service

    base_dir = Path(__file__).resolve().parent
    templates = Jinja2Templates(directory=str(base_dir / "templates"))
    app.mount("/static", StaticFiles(directory=str(base_dir / "static")), name="static")

    @app.get("/api/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/runs")
    async def list_runs() -> list[dict]:
        return await service.list_runs()

    @app.post("/api/runs")
    async def create_run(payload: ResearchCreateRequest):
        return await service.create_run(payload)

    @app.get("/api/studio/threads")
    async def list_studio_threads(
        after_created_at: str | None = None,
        after_thread_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.list_studio_threads(
                after_created_at=after_created_at, after_thread_id=after_thread_id, limit=limit
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/runs")
    async def list_studio_runs(
        thread_id: str | None = None,
        after_started_at: str | None = None,
        after_run_id: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.list_studio_runs(
                thread_id=thread_id, after_started_at=after_started_at,
                after_run_id=after_run_id, limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/runs/{run_id}")
    async def get_studio_run(run_id: str):
        try:
            return service.get_studio_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/studio/runs/{run_id}/spans")
    async def list_studio_spans(
        run_id: str,
        after_started_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.list_studio_spans(
                run_id, after_started_sequence=after_started_sequence, limit=limit
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    def timeline_query(
        run_id: str,
        *,
        after_sequence: int,
        limit: int,
        event_types: str | None = None,
        span_kinds: str | None = None,
        statuses: str | None = None,
        actor_id: str | None = None,
        task_id: str | None = None,
        text: str | None = None,
        error_only: bool = False,
        has_artifacts: bool | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
    ) -> TimelineQuery:
        split = lambda value: tuple(part.strip() for part in (value or "").split(",") if part.strip())
        return TimelineQuery(
            run_id=run_id, after_sequence=after_sequence, limit=limit,
            event_types=split(event_types), span_kinds=split(span_kinds), statuses=split(statuses),
            actor_id=actor_id, task_id=task_id, text=text, error_only=error_only,
            has_artifacts=has_artifacts, occurred_from=occurred_from, occurred_to=occurred_to,
        )

    @app.get("/api/studio/runs/{run_id}/timeline")
    async def get_studio_timeline(
        run_id: str,
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
        event_types: str | None = None,
        span_kinds: str | None = None,
        statuses: str | None = None,
        actor_id: str | None = None,
        task_id: str | None = None,
        text: str | None = None,
        error_only: bool = False,
        has_artifacts: bool | None = None,
        occurred_from: datetime | None = None,
        occurred_to: datetime | None = None,
    ):
        try:
            return service.get_timeline_page(timeline_query(
                run_id, after_sequence=after_sequence, limit=limit, event_types=event_types,
                span_kinds=span_kinds, statuses=statuses, actor_id=actor_id, task_id=task_id,
                text=text, error_only=error_only, has_artifacts=has_artifacts,
                occurred_from=occurred_from, occurred_to=occurred_to,
            ))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/runs/{run_id}/timeline/stream")
    async def stream_studio_timeline(
        request: Request,
        run_id: str,
        after_sequence: int = Query(default=0, ge=0),
        follow: bool = True,
    ):
        async def stream():
            cursor = after_sequence
            while True:
                try:
                    page = service.get_timeline_page(TimelineQuery(run_id, after_sequence=cursor, limit=200))
                except KeyError:
                    yield "event: error\ndata: {\"detail\":\"Run not found\"}\n\n"
                    return
                for event in page["items"]:
                    cursor = int(event["sequence_no"])
                    yield f"id: {cursor}\nevent: timeline\ndata: {json.dumps(event, ensure_ascii=False, separators=(',', ':'))}\n\n"
                if not follow:
                    return
                if await request.is_disconnected():
                    return
                if not page["items"]:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(
            stream(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/studio/runs/{run_id}/export")
    async def export_studio_trace(run_id: str, format: Literal["json", "ndjson"] = "json"):
        try:
            trace = service.export_studio_trace(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        if format == "json":
            return trace
        records = [
            {"record_type": "thread", "value": trace["thread"]},
            {"record_type": "run", "value": trace["run"]},
            *({"record_type": "span", "value": value} for value in trace["spans"]),
            *({"record_type": "event", "value": value} for value in trace["events"]),
        ]
        body = "".join(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n" for record in records)
        return Response(
            body, media_type="application/x-ndjson",
            headers={"Content-Disposition": f'attachment; filename="{run_id}.ndjson"'},
        )

    @app.get("/api/runs/{research_id}/console")
    async def get_console(research_id: str):
        try:
            return await service.get_console_summary(research_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

    @app.get("/api/runs/{research_id}/report")
    async def get_report(research_id: str):
        try:
            return await service.get_report_view(research_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

    @app.get("/api/runs/{research_id}/debug")
    async def get_debug(research_id: str):
        try:
            return await service.get_debug_view(research_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Run not found")

    @app.get("/", response_class=HTMLResponse)
    @app.get("/console/{research_id}", response_class=HTMLResponse)
    @app.get("/report/{research_id}", response_class=HTMLResponse)
    async def console_shell(request: Request, research_id: str | None = None):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "request": request,
                "page_research_id": research_id or "",
            },
        )

    return app

try:
    app = create_app()
except Exception:
    app = create_app(runtime_dir=str(Path(tempfile.gettempdir()) / "mini-deep-research-console"))
