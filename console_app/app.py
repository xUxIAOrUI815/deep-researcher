from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
import tempfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

from schemas.console import ResearchCreateRequest
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
