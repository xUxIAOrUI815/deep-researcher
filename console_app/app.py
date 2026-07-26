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

from schemas.console import (
    ResearchCreateRequest,
    StudioABComparisonRequest,
    StudioBadcaseCreateRequest,
    StudioReplayApprovalRequest,
    StudioReplayCreateRequest,
)
from deep_researcher.studio import (
    StudioAdvancedConflict,
    TimelineQuery,
)
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

    def csv_values(value: str | None) -> tuple[str, ...]:
        return tuple(
            part.strip()
            for part in (value or "").split(",")
            if part.strip()
        )

    @app.get("/api/studio/v2/runs/{run_id}/task-graph")
    async def get_studio_v2_task_graph(
        run_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        statuses: str | None = None,
    ):
        try:
            return service.get_studio_v2_task_graph(
                run_id,
                cursor=cursor,
                limit=limit,
                statuses=csv_values(statuses),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/runs/{run_id}/evidence-graph")
    async def get_studio_v2_evidence_graph(
        run_id: str,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
        entity_types: str | None = None,
        statuses: str | None = None,
    ):
        try:
            return service.get_studio_v2_evidence_graph(
                run_id,
                cursor=cursor,
                limit=limit,
                entity_types=csv_values(entity_types),
                statuses=csv_values(statuses),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/runs/{run_id}/state-diff")
    async def get_studio_v2_state_diff(
        run_id: str,
        domain: Literal["scheduler", "evidence"],
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.get_studio_v2_state_diff(
                run_id,
                domain=domain,
                after_sequence=after_sequence,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/runs/{run_id}/error-retry-chain")
    async def get_studio_v2_error_retry_chain(
        run_id: str,
        domain: Literal["runtime", "scheduler"] = "runtime",
        after_sequence: int = Query(default=0, ge=0),
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.get_studio_v2_errors(
                run_id,
                domain=domain,
                after_sequence=after_sequence,
                limit=limit,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/runs/{run_id}/conflicts")
    async def get_studio_v2_conflicts(
        run_id: str,
        cursor: str | None = None,
        limit: int = Query(default=50, ge=1, le=1000),
        statuses: str | None = None,
    ):
        try:
            return service.get_studio_v2_conflicts(
                run_id,
                cursor=cursor,
                limit=limit,
                statuses=csv_values(statuses),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/runs/{run_id}/components")
    async def get_studio_v2_components(run_id: str):
        try:
            return {"items": service.get_studio_v2_components(run_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get(
        "/api/studio/advanced/runs/{run_id}/component-selection"
    )
    async def get_studio_component_selection(run_id: str):
        try:
            return service.get_studio_component_selection(run_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Run not found",
            ) from exc

    @app.get("/api/studio/v2/runs/{run_id}/metrics")
    async def get_studio_v2_metrics(run_id: str):
        try:
            return service.get_studio_v2_metrics(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Run not found") from exc

    @app.get("/api/studio/v2/events/{event_id}")
    async def get_studio_v2_event(event_id: str):
        try:
            return service.studio_v2.run_event(event_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Event not found") from exc

    @app.get(
        "/api/studio/v2/runs/{run_id}/scheduler-events/{sequence_no}"
    )
    async def get_studio_v2_scheduler_event(
        run_id: str,
        sequence_no: int,
    ):
        try:
            return service.studio_v2.scheduler_event(run_id, sequence_no)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Scheduler event not found",
            ) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/v2/snapshots/{snapshot_id}")
    async def get_studio_v2_snapshot(snapshot_id: str):
        try:
            return service.studio_v2.snapshot_navigation(snapshot_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Source snapshot not found",
            ) from exc

    @app.get("/api/studio/v2/artifacts/{artifact_id}/content")
    async def get_studio_v2_artifact_content(artifact_id: str):
        try:
            content, media_type, content_hash = (
                service.studio_v2.artifact_content(artifact_id)
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Artifact not found",
            ) from exc
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        return Response(
            content,
            media_type=media_type,
            headers={
                "ETag": f'"sha256-{content_hash}"',
                "Cache-Control": "private, immutable, max-age=31536000",
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/studio/v2/artifacts/{artifact_id}")
    async def get_studio_v2_artifact(artifact_id: str):
        try:
            return service.studio_v2.artifact_metadata(artifact_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Artifact not found",
            ) from exc

    @app.get(
        "/api/studio/advanced/runs/{run_id}/spans/{span_id}/"
        "replay-eligibility"
    )
    async def get_studio_replay_eligibility(
        run_id: str,
        span_id: str,
    ):
        try:
            return service.get_studio_replay_eligibility(
                run_id,
                span_id,
            )
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail=str(exc),
            ) from exc

    @app.post(
        "/api/studio/advanced/runs/{run_id}/spans/{span_id}/replays",
        status_code=201,
    )
    async def prepare_studio_replay(
        run_id: str,
        span_id: str,
        payload: StudioReplayCreateRequest,
    ):
        try:
            return service.prepare_studio_replay(
                run_id=run_id,
                span_id=span_id,
                mode=payload.mode,
                selected_component_versions=(
                    payload.selected_component_versions
                ),
                requested_by=payload.requested_by,
                reason=payload.reason,
                restart_failed_span=payload.restart_failed_span,
                environment_label=payload.environment_label,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/advanced/replays")
    async def list_studio_replays(
        statuses: str | None = None,
        cursor: str | None = None,
        limit: int = Query(default=100, ge=1, le=1000),
    ):
        try:
            return service.list_studio_replays(
                statuses=csv_values(statuses),
                cursor=cursor,
                limit=limit,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/advanced/replays/{request_id}")
    async def get_studio_replay(request_id: str):
        try:
            return service.get_studio_replay(request_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Replay request not found",
            ) from exc

    @app.post(
        "/api/studio/advanced/replays/{request_id}/approvals",
        status_code=201,
    )
    async def approve_studio_replay(
        request_id: str,
        payload: StudioReplayApprovalRequest,
    ):
        try:
            return service.approve_studio_replay(
                request_id=request_id,
                command_fingerprint=payload.command_fingerprint,
                approved_by=payload.approved_by,
                reason=payload.reason,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StudioAdvancedConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/studio/advanced/replays/{request_id}/execute"
    )
    async def execute_studio_replay(request_id: str):
        try:
            return await service.execute_studio_replay(request_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except StudioAdvancedConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post(
        "/api/studio/advanced/comparisons",
        status_code=201,
    )
    async def compare_studio_runs(
        payload: StudioABComparisonRequest,
    ):
        try:
            return service.compare_studio_runs(
                left_run_id=payload.left_run_id,
                right_run_id=payload.right_run_id,
                dataset_sample_artifact_id=(
                    payload.dataset_sample_artifact_id
                ),
                left_span_id=payload.left_span_id,
                right_span_id=payload.right_span_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get(
        "/api/studio/advanced/comparisons/{comparison_id}"
    )
    async def get_studio_comparison(comparison_id: str):
        try:
            return service.get_studio_comparison(comparison_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Comparison not found",
            ) from exc

    @app.get(
        "/api/studio/advanced/component-diff/"
        "{left_version_id}/{right_version_id}"
    )
    async def get_studio_component_diff(
        left_version_id: str,
        right_version_id: str,
    ):
        try:
            return service.get_studio_component_diff(
                left_version_id,
                right_version_id,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post(
        "/api/studio/advanced/badcases",
        status_code=201,
    )
    async def create_studio_badcase(
        payload: StudioBadcaseCreateRequest,
    ):
        try:
            return service.create_studio_badcase(
                source_run_id=payload.source_run_id,
                source_span_id=payload.source_span_id,
                dataset_sample_artifact_id=(
                    payload.dataset_sample_artifact_id
                ),
                evaluation_ids=tuple(payload.evaluation_ids),
                evaluation_artifact_ids=tuple(
                    payload.evaluation_artifact_ids
                ),
                human_note=payload.human_note,
                created_by=payload.created_by,
                additional_input_artifact_ids=tuple(
                    payload.additional_input_artifact_ids
                ),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/api/studio/advanced/badcases/{badcase_id}")
    async def get_studio_badcase(badcase_id: str):
        try:
            return service.get_studio_badcase(badcase_id)
        except KeyError as exc:
            raise HTTPException(
                status_code=404,
                detail="Badcase not found",
            ) from exc

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

    @app.get("/studio/{run_id}", response_class=HTMLResponse)
    async def studio_v2_shell(request: Request, run_id: str):
        return templates.TemplateResponse(
            request,
            "studio_v2.html",
            {
                "request": request,
                "studio_run_id": run_id,
            },
        )

    return app

try:
    app = create_app()
except Exception:
    app = create_app(runtime_dir=str(Path(tempfile.gettempdir()) / "mini-deep-research-console"))
