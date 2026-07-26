"""Deterministic local Console server for browser-level development checks.

This helper is intentionally test-only. It composes the same FastAPI Console
with the complete deterministic Background001 application fixture, so UI tests
can exercise successful research and reporting without provider credentials.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
import tempfile

import uvicorn

from console_app.app import create_app
from deep_researcher.contracts import BudgetUsage
from deep_researcher.kernel import ModelResponse
from tests.fixtures.background001_application import (
    DeterministicSupervisorModel,
    build_deterministic_application,
)


class ApprovalThenResearchModel:
    def __init__(self) -> None:
        self.calls = 0
        self.delegate = DeterministicSupervisorModel()

    async def complete(self, request):
        self.calls += 1
        if self.calls == 1:
            return ModelResponse(
                structured={
                    "action": "request_approval",
                    "tasks": [],
                    "decision_summary": (
                        "Explicit approval is required."
                    ),
                    "approval_reason": (
                        "Approve the bounded source and evidence scope."
                    ),
                },
                usage=BudgetUsage(model_calls=1),
            )
        return await self.delegate.complete(request)

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--runtime-dir", type=Path)
    parser.add_argument(
        "--scenario",
        choices=("complete", "approval"),
        default="complete",
    )
    args = parser.parse_args()

    temporary = None
    if args.runtime_dir is None:
        temporary = tempfile.TemporaryDirectory(
            prefix="deep-researcher-console-ui-"
        )
        root = Path(temporary.name)
    else:
        root = args.runtime_dir
        root.mkdir(parents=True, exist_ok=True)

    runtime, tools, _ = build_deterministic_application(
        root,
        supervisor_model=(
            ApprovalThenResearchModel()
            if args.scenario == "approval"
            else None
        ),
    )
    app = create_app(str(root), runtime=runtime)
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="warning",
        )
    finally:
        asyncio.run(runtime.aclose())
        tools.close()
        if temporary is not None:
            temporary.cleanup()


if __name__ == "__main__":
    main()
