from __future__ import annotations

import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "console_app" / "static"


def _run_node(source: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["node", "--input-type=module", "--eval", source],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_console_frontend_modules_are_valid_and_old_contract_is_removed():
    for filename in ("console_models.js", "app.js"):
        result = subprocess.run(
            ["node", "--check", str(STATIC / filename)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stderr

    application = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "ConsoleWorkspace@2" in (
        STATIC / "console_models.js"
    ).read_text(encoding="utf-8")
    assert "Planner 上下文" not in application
    assert "Researcher 上下文" not in application
    assert "Distiller" not in application
    assert "knowledge_updating" not in application
    assert "get_console_summary" not in application
    assert "document.body.dataset.researchId" not in application
    assert '"approve"' in application
    assert '"cancel"' in application
    assert "report_reviewer" in application
    assert "evidence_verifier" in application


def test_console_frontend_contract_polling_filtering_and_safe_markdown():
    module_url = (STATIC / "console_models.js").resolve().as_uri()
    workspace = {
        "schema_version": "ConsoleWorkspace@2",
        "identity": {},
        "runtime": {
            "status": "waiting_approval",
            "progress": [],
            "roles": [],
            "decision_reasons": [],
        },
        "actions": {"approvals": []},
        "scheduler": {"tasks": [], "task_counts": {}},
        "evidence": {
            "sections": [],
            "gaps": [],
            "conflicts": [],
            "packets": [],
            "sources": [],
        },
        "reporting": {},
        "timeline": [],
        "navigation": {},
    }
    source = f"""
      import assert from "node:assert/strict";
      import {{
        filterRuns,
        isNamespacedIdentifier,
        normalizeWorkspace,
        pollingDelay,
        renderMarkdownSafe,
        safeExternalUrl,
        safeInternalPath,
        shouldPoll,
      }} from {json.dumps(module_url)};

      const waiting = normalizeWorkspace({json.dumps(workspace)});
      assert.equal(shouldPoll(waiting), true);
      assert.equal(pollingDelay(waiting), 5000);
      waiting.runtime.status = "completed";
      assert.equal(shouldPoll(waiting), false);

      const runs = [
        {{ query: "Alpha evidence", status: "completed" }},
        {{ query: "Beta evidence", status: "failed" }},
      ];
      assert.equal(filterRuns(runs, "alpha", "all").length, 1);
      assert.equal(filterRuns(runs, "", "failed").length, 1);
      assert.equal(isNamespacedIdentifier("user_research_owner"), true);
      assert.equal(isNamespacedIdentifier("local-operator"), false);

      assert.equal(safeInternalPath("//evil.test"), "/");
      assert.equal(safeExternalUrl("javascript:alert(1)"), null);
      const html = renderMarkdownSafe(
        "# Safe\\n<script>alert(1)</script>\\n"
        + "[bad](javascript:alert(1)) [good](https://example.com)"
      );
      assert.equal(html.includes("<script>"), false);
      assert.equal(html.includes("&lt;script&gt;"), true);
      assert.equal(html.includes('href="javascript:'), false);
      assert.equal(html.includes('href="https://example.com/"'), true);
      process.stdout.write("ok");
    """
    result = _run_node(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout == "ok"


def test_console_styles_cover_accessibility_and_responsive_boundaries():
    styles = (STATIC / "styles.css").read_text(encoding="utf-8")
    assert ".skip-link" in styles
    assert ":focus-visible" in styles
    assert "@media (prefers-reduced-motion: reduce)" in styles
    assert "@media (max-width: 720px)" in styles
    assert "@media (max-width: 420px)" in styles
    assert "@media print" in styles
