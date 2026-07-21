from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deep_researcher.baseline import capture_current_mock_pipeline


async def _main() -> int:
    parser = argparse.ArgumentParser(description="Capture or verify the Background001 draft-pipeline baseline.")
    parser.add_argument("--check", type=Path, help="Compare the captured baseline with an existing JSON file.")
    args = parser.parse_args()
    captured = await capture_current_mock_pipeline()
    if args.check is not None:
        expected = json.loads(args.check.read_text(encoding="utf-8"))
        if captured != expected:
            print(json.dumps({"expected": expected, "actual": captured}, ensure_ascii=False, indent=2, sort_keys=True))
            return 1
    print(json.dumps(captured, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
