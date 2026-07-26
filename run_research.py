from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from deep_researcher.application import (
    ResearchCreateRequest,
    build_live_application_runtime,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the durable Background001 research application."
    )
    parser.add_argument(
        "query",
        nargs="?",
        default="Test research query",
        help="Research question.",
    )
    parser.add_argument(
        "--instructions",
        default="",
        help="Additional report and research constraints.",
    )
    parser.add_argument(
        "--depth",
        choices=("quick", "standard", "deep"),
        default="standard",
    )
    parser.add_argument(
        "--runtime-dir",
        default=os.getenv("DEEP_RESEARCH_RUNTIME_DIR", ".research_runtime"),
        help="Durable runtime directory.",
    )
    return parser.parse_args()


async def main() -> int:
    load_dotenv()
    args = _arguments()
    runtime = build_live_application_runtime(Path(args.runtime_dir))
    try:
        record = runtime.new_run(
            ResearchCreateRequest(
                query=args.query,
                instructions=args.instructions,
                depth=args.depth,
            )
        )
        print(f"run_id={record.run_id}")
        print(f"research_id={record.research_id}")
        result = await runtime.execute(record.research_id)
        print(f"status={result.status.value}")
        print(f"stage={result.current_stage}")
        if result.error_message:
            print(f"error={result.error_message}")
        if result.report_artifact_id:
            markdown = runtime.artifact_store.read_bytes(
                result.report_artifact_id
            ).decode("utf-8")
            print(markdown)
        return 0 if result.status.value == "completed" else 1
    finally:
        await runtime.aclose()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
