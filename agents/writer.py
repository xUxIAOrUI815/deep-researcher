from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from core.observability import EventType, get_observer
from schemas.state import FinalReport


async def run_writer(
    *,
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    section_evidence_packs: List[Dict[str, Any]],
    run_context: Any = None,
) -> FinalReport:
    """Graph-facing writer entry.

    Writer is intentionally downstream of stable writing inputs only:
    `report_outline`, `section_goals`, and `section_evidence_packs`.
    It should not depend on raw passages, planner hints, or distiller debug
    metadata once the full writer is implemented.
    """
    observer = get_observer()
    if run_context is not None:
        observer.record_run_event(
            run_context,
            EventType.WRITER_STARTED,
            message="Writer started evidence-based report generation.",
        )

    sections = list((report_outline or {}).get("sections", []) or [])
    if not sections and section_goals:
        sections = [
            {
                "section_id": goal.get("section_id", f"section_{index}"),
                "title": goal.get("goal", f"Section {index + 1}"),
                "goal": goal.get("goal", ""),
                "order": index + 1,
            }
            for index, goal in enumerate(section_goals)
        ]

    title = str((report_outline or {}).get("title", "") or "Research Report").strip() or "Research Report"
    goals_by_section = {str(goal.get("section_id", "")): goal for goal in section_goals}
    packs_by_section = {str(pack.get("section_id", "")): pack for pack in section_evidence_packs}

    section_markdown_blocks: list[str] = []
    section_ids: list[str] = []
    evidence_pack_ids: list[str] = []
    citation_map: dict[str, list[str]] = {}

    executive_summary = _build_executive_summary(sections, packs_by_section, goals_by_section)
    if executive_summary:
        section_markdown_blocks.append("## Executive Summary\n\n" + executive_summary)

    ordered_sections = sorted(
        sections,
        key=lambda item: int(item.get("order", 9999) or 9999),
    )
    for index, section in enumerate(ordered_sections, start=1):
        section_id = str(section.get("section_id", f"section_{index}"))
        goal = goals_by_section.get(section_id, {})
        pack = packs_by_section.get(section_id, {})
        section_markdown, used_refs = _render_section(
            title=str(section.get("title", f"Section {index}")),
            section_id=section_id,
            goal_text=str(goal.get("goal", "") or section.get("goal", "")),
            pack=pack,
        )
        section_markdown_blocks.append(section_markdown)
        section_ids.append(section_id)
        if pack.get("pack_id"):
            evidence_pack_ids.append(str(pack["pack_id"]))
        citation_map[section_id] = used_refs

        if run_context is not None:
            observer.record_evidence_event(
                run_context,
                EventType.SECTION_GENERATED,
                section_id=section_id,
                message="Writer generated report section from evidence pack.",
                payload={
                    "title": section.get("title", ""),
                    "evidence_pack_id": pack.get("pack_id"),
                    "reference_count": len(used_refs),
                },
            )

    open_questions = _build_open_questions(sections, packs_by_section)
    if open_questions:
        section_markdown_blocks.append("## Open Questions / Research Gaps\n\n" + open_questions)

    markdown = f"# {title}\n\n" + "\n\n".join(block for block in section_markdown_blocks if block.strip())

    report = FinalReport(
        markdown=markdown,
        section_ids=section_ids,
        evidence_pack_ids=evidence_pack_ids,
        citation_map=citation_map,
    )

    if run_context is not None:
        observer.record_run_event(
            run_context,
            EventType.WRITER_COMPLETED,
            message="Writer completed report generation.",
            payload={
                "section_count": len(section_ids),
                "report_length": len(markdown),
            },
        )

    return report


def _as_strings(items: Iterable[Any]) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("summary")
                or item.get("quote")
                or item.get("id")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            values.append(text)
    return values


def _ref_ids(prefix: str, items: Iterable[Any]) -> list[str]:
    refs: list[str] = []
    for item in items:
        if isinstance(item, str):
            value = item.strip()
        elif isinstance(item, dict):
            value = str(item.get("id") or item.get(f"{prefix}_id") or "").strip()
        else:
            value = ""
        if value:
            refs.append(value)
    return refs


def _format_refs(refs: Iterable[str]) -> str:
    deduped: list[str] = []
    seen: set[str] = set()
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    if not deduped:
        return ""
    return " " + " ".join(f"[{ref}]" for ref in deduped[:6])


def _build_executive_summary(
    sections: List[Dict[str, Any]],
    packs_by_section: Dict[str, Dict[str, Any]],
    goals_by_section: Dict[str, Dict[str, Any]],
) -> str:
    lines: list[str] = []
    for section in sections[:4]:
        section_id = str(section.get("section_id", ""))
        pack = packs_by_section.get(section_id, {})
        coverage = float(pack.get("coverage_score", 0.0) or 0.0)
        conflict_count = len(pack.get("conflict_ids", []) or [])
        evidence_count = len(pack.get("evidence_ids", []) or pack.get("evidence", []) or [])
        title = str(section.get("title", ""))
        if coverage >= 0.5:
            lines.append(
                f"- {title}: available evidence is reasonably strong, with {evidence_count} evidence item(s) supporting this section."
            )
        elif evidence_count > 0:
            lines.append(
                f"- {title}: current material supports only a partial assessment and should be interpreted cautiously."
            )
        else:
            lines.append(
                f"- {title}: current coverage is weak and conclusions remain preliminary."
            )
        if conflict_count > 0:
            lines.append(f"  Conflicting evidence remains in this area and is reflected in the section discussion.")
    return "\n".join(lines)


def _render_section(
    *,
    title: str,
    section_id: str,
    goal_text: str,
    pack: Dict[str, Any],
) -> tuple[str, list[str]]:
    claims = pack.get("claims", []) or []
    facts = pack.get("facts", []) or pack.get("atomic_facts", []) or []
    evidence = pack.get("evidence", []) or []
    claim_texts = _as_strings(claims)
    fact_texts = _as_strings(facts)
    evidence_texts = _as_strings(evidence)
    claim_ids = list(pack.get("claim_ids", []) or []) + _ref_ids("claim", claims)
    fact_ids = list(pack.get("fact_ids", []) or []) + _ref_ids("fact", facts)
    evidence_ids = list(pack.get("evidence_ids", []) or []) + _ref_ids("evidence", evidence)
    conflict_ids = list(pack.get("conflict_ids", []) or [])
    coverage = float(pack.get("coverage_score", 0.0) or 0.0)

    paragraphs: list[str] = [f"## {title}"]
    if goal_text:
        paragraphs.append(f"{goal_text.strip()}.")

    if claim_texts:
        primary_claims = claim_texts[:3]
        paragraphs.append(
            "Available evidence suggests "
            + "; ".join(_sentence_case(text.rstrip(". ")) for text in primary_claims)
            + "."
            + _format_refs(claim_ids[:3] + evidence_ids[:2])
        )
    elif fact_texts:
        paragraphs.append(
            "The available material supports the following factual points: "
            + "; ".join(_sentence_case(text.rstrip(". ")) for text in fact_texts[:3])
            + "."
            + _format_refs(fact_ids[:3] + evidence_ids[:2])
        )
    else:
        paragraphs.append(
            "Evidence for this section remains limited, so only a cautious summary can be provided."
            + _format_refs(evidence_ids[:2])
        )

    if evidence_texts:
        paragraphs.append(
            "Supporting evidence includes "
            + "; ".join(_sentence_case(text.rstrip(". ")) for text in evidence_texts[:2])
            + "."
            + _format_refs(evidence_ids[:3])
        )

    if conflict_ids:
        paragraphs.append(
            "Current sources disagree on important details in this section, so competing interpretations are preserved rather than collapsed into a single conclusion."
            + _format_refs(conflict_ids)
        )
    elif coverage < 0.35:
        paragraphs.append(
            "Further investigation is required to determine whether the current material is sufficient for a confident conclusion."
        )
    elif coverage < 0.6:
        paragraphs.append(
            "The section is supported by some evidence, but the result should still be treated as provisional rather than definitive."
        )

    notes = str(pack.get("notes", "") or "").strip()
    if notes:
        paragraphs.append(notes)

    used_refs = _dedupe_refs(claim_ids + fact_ids + evidence_ids + conflict_ids)
    return "\n\n".join(paragraphs), used_refs


def _build_open_questions(
    sections: List[Dict[str, Any]],
    packs_by_section: Dict[str, Dict[str, Any]],
) -> str:
    bullets: list[str] = []
    for section in sections:
        section_id = str(section.get("section_id", ""))
        title = str(section.get("title", ""))
        pack = packs_by_section.get(section_id, {})
        coverage = float(pack.get("coverage_score", 0.0) or 0.0)
        conflict_count = len(pack.get("conflict_ids", []) or [])
        if conflict_count > 0:
            bullets.append(f"- {title}: sources remain in conflict and require explicit verification before a firmer conclusion is possible.")
        elif coverage < 0.35:
            bullets.append(f"- {title}: evidence coverage is currently too thin to support a strong conclusion.")
    return "\n".join(bullets)


def _dedupe_refs(refs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped


def _sentence_case(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    return text[0].upper() + text[1:]
