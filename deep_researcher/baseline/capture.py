from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
from statistics import fmean
from typing import Any


BASELINE_SCHEMA_VERSION = "1.0.0"
_FIXTURE_PATH = Path(__file__).resolve().parents[2] / "datasets" / "background001" / "frozen_replay_v1.json"
_CITATION_PATTERN = re.compile(r"\[([^\[\]]+)\]")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fixture_fingerprint(payload: Mapping[str, Any]) -> str:
    content = dict(payload)
    content.pop("fixture_fingerprint", None)
    return hashlib.sha256(_canonical_json(content).encode("utf-8")).hexdigest()


def load_frozen_replay_fixture(path: str | Path | None = None) -> dict[str, Any]:
    fixture_path = Path(path) if path is not None else _FIXTURE_PATH
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    expected = str(payload.get("fixture_fingerprint", ""))
    actual = fixture_fingerprint(payload)
    if expected != actual:
        raise ValueError(f"frozen replay fixture fingerprint mismatch: expected {expected}, got {actual}")
    if payload.get("schema_version") != BASELINE_SCHEMA_VERSION:
        raise ValueError("unsupported frozen replay schema version")
    return payload


def _mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def normalize_run_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a draft-pipeline run to deterministic, behavior-level measurements."""

    researcher = _mapping(result.get("researcher_outputs"))
    distiller = _mapping(result.get("distiller_outputs"))
    report = _mapping(result.get("final_report"))
    markdown = str(report.get("markdown", ""))
    citation_map = _mapping(report.get("citation_map"))
    citation_refs = {
        str(reference)
        for references in citation_map.values()
        for reference in _sequence(references)
        if str(reference)
    }
    citation_markers = [
        marker
        for marker in _CITATION_PATTERN.findall(markdown)
        if marker and not marker.isspace()
    ]
    packs = _sequence(result.get("section_evidence_packs"))
    coverages = [float(_mapping(pack).get("coverage_score", 0.0) or 0.0) for pack in packs]
    state_events = [_mapping(event) for event in _sequence(result.get("state_events"))]
    event_types = sorted(
        {
            str(event.get("event_type") or event.get("type") or event.get("event") or "unknown")
            for event in state_events
        }
    )
    token_usage = _mapping(result.get("token_usage"))
    failed_tasks = _sequence(result.get("failed_tasks"))
    error_state = result.get("error_state")
    section_ids = _sequence(report.get("section_ids"))
    sections_with_citations = sum(1 for section_id in section_ids if _sequence(citation_map.get(str(section_id))))
    retrieval_sources = _sequence(researcher.get("sources"))
    retrieval_passages = _sequence(researcher.get("passages"))
    claims = _sequence(distiller.get("claims"))
    facts = _sequence(distiller.get("atomic_facts"))
    evidence = _sequence(distiller.get("evidence"))
    conflicts = _sequence(distiller.get("conflicts"))

    return {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "retrieval": {
            "query_count": len(_sequence(researcher.get("queries"))),
            "source_count": len(retrieval_sources),
            "passage_count": len(retrieval_passages),
            "accepted_source_count": sum(1 for item in retrieval_sources if _mapping(item).get("status") == "accepted"),
            "unique_source_url_count": len({str(_mapping(item).get("url", "")) for item in retrieval_sources if _mapping(item).get("url")}),
        },
        "knowledge": {
            "atomic_fact_count": len(facts),
            "claim_count": len(claims),
            "evidence_count": len(evidence),
            "conflict_count": len(conflicts),
            "section_pack_count": len(packs),
            "mean_section_coverage": round(fmean(coverages), 6) if coverages else 0.0,
        },
        "citations": {
            "citation_map_reference_count": len(citation_refs),
            "markdown_reference_count": len(citation_markers),
            "sections_with_citations": sections_with_citations,
        },
        "report": {
            "section_count": len(section_ids),
            "character_count": len(markdown),
            "has_executive_summary": "## Executive Summary" in markdown,
            "has_open_questions": "## Open Questions / Research Gaps" in markdown,
            "nonempty": bool(markdown.strip()),
        },
        "trace": {
            "state_event_count": len(state_events),
            "event_types": event_types,
            "has_run_events": bool(state_events),
            "has_model_usage_events": any("model" in event_type for event_type in event_types),
            "has_tool_usage_events": any("tool" in event_type for event_type in event_types),
        },
        "resources": {
            "token_usage": {
                "planning_tokens": int(token_usage.get("planning_tokens", 0) or 0),
                "research_tokens": int(token_usage.get("research_tokens", 0) or 0),
                "distillation_tokens": int(token_usage.get("distillation_tokens", 0) or 0),
                "writing_tokens": int(token_usage.get("writing_tokens", 0) or 0),
                "total_tokens": int(token_usage.get("total_tokens", 0) or 0),
            },
            "cost_usd": 0.0,
            "latency_ms": 0.0,
            "cost_instrumented": False,
            "latency_instrumented": False,
        },
        "failures": {
            "failed_task_count": len(failed_tasks),
            "has_error_state": error_state is not None,
            "error_code": str(_mapping(error_state).get("code", "")) if error_state is not None else "",
        },
    }


async def capture_current_mock_pipeline() -> dict[str, Any]:
    """Run the current draft graph against its frozen offline scenario."""

    os.environ["RESEARCHER_SCRAPER_MODE"] = "mock"
    os.environ["RESEARCHER_SEARCH_MODE"] = "mock"
    fixture = load_frozen_replay_fixture()

    import core.graph as graph_module
    from core.context_builders import PlannerContextBuilder, ResearcherContextBuilder, WriterContextBuilder
    from core.session_knowledge import KnowledgeManager
    from core.session_retrieval import SessionRetrievalService
    from tests.fixtures.offline_research_inputs import build_initial_graph_state

    original = (
        graph_module.SESSION_KNOWLEDGE_MANAGER,
        graph_module.SESSION_RETRIEVAL_SERVICE,
        graph_module.PLANNER_CONTEXT_BUILDER,
        graph_module.RESEARCHER_CONTEXT_BUILDER,
        graph_module.WRITER_CONTEXT_BUILDER,
    )
    manager = KnowledgeManager(base_storage_path=".", sqlite_filename=":memory:")
    retrieval = SessionRetrievalService(manager)
    graph_module.SESSION_KNOWLEDGE_MANAGER = manager
    graph_module.SESSION_RETRIEVAL_SERVICE = retrieval
    graph_module.PLANNER_CONTEXT_BUILDER = PlannerContextBuilder(retrieval)
    graph_module.RESEARCHER_CONTEXT_BUILDER = ResearcherContextBuilder(retrieval)
    graph_module.WRITER_CONTEXT_BUILDER = WriterContextBuilder(retrieval)
    try:
        graph = graph_module.create_research_graph(None)
        result = await graph.ainvoke(build_initial_graph_state(), fixture["run_config"])
        return normalize_run_result(result)
    finally:
        (
            graph_module.SESSION_KNOWLEDGE_MANAGER,
            graph_module.SESSION_RETRIEVAL_SERVICE,
            graph_module.PLANNER_CONTEXT_BUILDER,
            graph_module.RESEARCHER_CONTEXT_BUILDER,
            graph_module.WRITER_CONTEXT_BUILDER,
        ) = original
        manager.close()
