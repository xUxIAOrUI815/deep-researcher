from __future__ import annotations

import json
from pathlib import Path

import pytest

from deep_researcher.baseline import capture_current_mock_pipeline, fixture_fingerprint, load_frozen_replay_fixture, normalize_run_result


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_frozen_replay_fixture_is_fingerprinted_and_offline():
    fixture = load_frozen_replay_fixture()
    assert fixture["fixture_fingerprint"] == fixture_fingerprint(fixture)
    assert fixture["mock_researcher_outputs"]["metadata"]["search_mode"] == "mock"
    assert fixture["mock_researcher_outputs"]["metadata"]["scraper_mode"] == "mock"
    assert all(source["url"].startswith("https://mock.local/") for source in fixture["mock_researcher_outputs"]["sources"])


def test_baseline_normalization_ignores_identity_and_timestamps():
    result = {
        "researcher_outputs": {"queries": [{"query": "q"}], "sources": [], "passages": []},
        "distiller_outputs": {},
        "section_evidence_packs": [],
        "final_report": {
            "report_id": "random-one",
            "created_at": "2026-01-01T00:00:00Z",
            "markdown": "# Report",
            "section_ids": ["section_one"],
            "citation_map": {"section_one": []},
        },
        "state_events": [],
        "token_usage": {},
        "failed_tasks": [],
        "error_state": None,
    }
    changed_identity = json.loads(json.dumps(result))
    changed_identity["final_report"]["report_id"] = "random-two"
    changed_identity["final_report"]["created_at"] = "2030-01-01T00:00:00Z"
    assert normalize_run_result(result) == normalize_run_result(changed_identity)


@pytest.mark.asyncio
async def test_current_mock_pipeline_matches_committed_baseline():
    expected_path = REPOSITORY_ROOT / "docs" / "baselines" / "background001_current_draft.json"
    expected = json.loads(expected_path.read_text(encoding="utf-8"))
    assert await capture_current_mock_pipeline() == expected
