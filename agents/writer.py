from __future__ import annotations

from typing import Any, Dict, Iterable, List

from core.context_builders import WriterContextBuilder
from core.observability import EventType, get_observer
from core.session_retrieval import SessionRetrievalService
from schemas.state import FinalReport


def _writer_context_from_snapshot(
    *,
    knowledge_manager: Any,
    research_id: str,
    session_id: str,
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    section_evidence_packs: List[Dict[str, Any]],
) -> Dict[str, Any] | None:
    """从持久化会话数据构建 writer 上下文，失败时返回 None。"""
    try:
        builder = WriterContextBuilder(SessionRetrievalService(knowledge_manager))
        return builder.build(
            research_id=research_id,
            session_id=session_id,
            report_outline=report_outline,
            section_goals=section_goals,
            fallback_section_packs=section_evidence_packs,
        ).model_dump()
    except Exception:
        return None


async def run_writer(
    *,
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    section_evidence_packs: List[Dict[str, Any]],
    knowledge_manager: Any = None,
    research_id: str = "",
    session_id: str = "",
    writer_context: Dict[str, Any] | None = None,
    run_context: Any = None,
) -> FinalReport:
    """根据大纲和证据包生成最终报告 Markdown。"""
    observer = get_observer()
    if run_context is not None:
        observer.record_run_event(
            run_context,
            EventType.WRITER_STARTED,
            message="Writer started evidence-based report generation.",
        )

    session_snapshot: dict[str, Any] | None = None
    if writer_context is None and knowledge_manager is not None and research_id and session_id:
        writer_context = _writer_context_from_snapshot(
            knowledge_manager=knowledge_manager,
            research_id=research_id,
            session_id=session_id,
            report_outline=report_outline,
            section_goals=section_goals,
            section_evidence_packs=section_evidence_packs,
        )
    if writer_context is None and knowledge_manager is not None and research_id and session_id:
        try:
            session_snapshot = knowledge_manager.get_session_snapshot(research_id, session_id)
        except Exception:
            session_snapshot = None

    effective_packs = list(section_evidence_packs or [])
    context_source = "state_fallback"
    if writer_context:
        effective_packs = list(writer_context.get("section_evidence_packs", []) or effective_packs)
        context_source = str(writer_context.get("context_source", "session_context"))
    elif session_snapshot:
        effective_packs = list(session_snapshot.get("section_evidence_packs", []))
        context_source = "session_snapshot"

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
    section_markdown_blocks: list[str] = []
    section_ids: list[str] = []
    evidence_pack_ids: list[str] = []
    citation_map: dict[str, list[str]] = {}

    ordered_sections = sorted(
        sections,
        key=lambda item: int(item.get("order", 9999) or 9999),
    )
    packs_by_section = {str(pack.get("section_id", "")): pack for pack in effective_packs}
    contexts_by_section = {
        str(item.get("section_id", "")): item
        for item in (writer_context or {}).get("section_contexts", [])
        if str(item.get("section_id", ""))
    }
    for section in ordered_sections:
        section_id = str(section.get("section_id", ""))
        if not _pack_has_refs(packs_by_section.get(section_id, {})):
            fallback_pack = _pack_from_section_context(contexts_by_section.get(section_id, {}))
            if _pack_has_refs(fallback_pack):
                packs_by_section[section_id] = fallback_pack

    executive_summary = _build_executive_summary(ordered_sections, packs_by_section)
    if executive_summary:
        section_markdown_blocks.append("## Executive Summary\n\n" + executive_summary)

    for index, section in enumerate(ordered_sections, start=1):
        section_id = str(section.get("section_id", f"section_{index}"))
        goal = goals_by_section.get(section_id, {})
        pack = packs_by_section.get(section_id, {})
        section_markdown, used_refs = _render_section(
            title=str(section.get("title", f"Section {index}")),
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
                "context_source": context_source,
            },
        )

    return report


def _pack_has_refs(pack: Dict[str, Any]) -> bool:
    return any(pack.get(key) for key in ("claim_ids", "fact_ids", "evidence_ids", "conflict_ids"))


def _ids_from(items: Iterable[Any], id_key: str) -> list[str]:
    refs: list[str] = []
    for item in items:
        if isinstance(item, dict):
            value = str(item.get("id") or item.get(id_key) or "").strip()
        else:
            value = str(getattr(item, "id", "") or getattr(item, id_key, "") or "").strip()
        if value:
            refs.append(value)
    return _dedupe_refs(refs)


def _pack_from_section_context(context: Dict[str, Any]) -> Dict[str, Any]:
    if not context:
        return {}
    claims = list(context.get("claims", []) or [])
    facts = list(context.get("facts", []) or [])
    evidence = list(context.get("evidence", []) or [])
    conflicts = list(context.get("conflicts", []) or [])
    claim_ids = _ids_from(claims, "claim_id")
    fact_ids = _ids_from(facts, "fact_id")
    evidence_ids = _ids_from(evidence, "evidence_id")
    conflict_ids = _ids_from(conflicts, "conflict_id")
    coverage_score = min(1.0, 0.2 * len(claim_ids) + 0.1 * len(evidence_ids) + 0.05 * len(fact_ids))
    return {
        "pack_id": str((context.get("pack") or {}).get("pack_id") or ""),
        "section_id": str(context.get("section_id", "")),
        "claim_ids": claim_ids,
        "fact_ids": fact_ids,
        "evidence_ids": evidence_ids,
        "conflict_ids": conflict_ids,
        "claims": claims,
        "facts": facts,
        "evidence": evidence,
        "conflicts": conflicts,
        "coverage_score": coverage_score,
        "notes": "Recovered section support from session retrieval context.",
    }


def _as_strings(items: Iterable[Any]) -> list[str]:
    values: list[str] = []
    for item in items:
        if isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("canonical_text")
                or item.get("summary")
                or item.get("summary_text")
                or item.get("quote")
                or item.get("quote_text")
                or ""
            ).strip()
        else:
            text = str(item).strip()
        if text:
            values.append(text)
    return values


def _format_refs(refs: Iterable[str]) -> str:
    """将引用 ID 列表格式化为文内引用标记。"""
    deduped = _dedupe_refs(refs)
    if not deduped:
        return ""
    return " " + " ".join(f"[{ref}]" for ref in deduped[:6])


def _build_executive_summary(
    sections: List[Dict[str, Any]],
    packs_by_section: Dict[str, Dict[str, Any]],
) -> str:
    """根据各小节证据覆盖情况生成执行摘要文本。"""
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
    goal_text: str,
    pack: Dict[str, Any],
) -> tuple[str, list[str]]:
    """渲染单个报告小节，并返回所使用的引用 ID。"""
    claim_ids = list(pack.get("claim_ids", []) or [])
    fact_ids = list(pack.get("fact_ids", []) or [])
    evidence_ids = list(pack.get("evidence_ids", []) or [])
    conflict_ids = list(pack.get("conflict_ids", []) or [])
    claim_texts = _as_strings(pack.get("claims", []) or [])
    fact_texts = _as_strings(pack.get("facts", []) or [])
    coverage = float(pack.get("coverage_score", 0.0) or 0.0)
    claim_count = len(claim_ids)
    fact_count = len(fact_ids)
    evidence_count = len(evidence_ids)

    paragraphs: list[str] = [f"## {title}"]
    if goal_text:
        paragraphs.append(f"{goal_text.strip()}.")

    if claim_count > 0:
        summary = (
            f"This section is supported by {claim_count} distilled claim(s), "
            f"{fact_count} atomic fact(s), and {evidence_count} evidence item(s)."
        )
        if coverage >= 0.6:
            summary += " The available material is strong enough to support a reasonably confident synthesis."
        elif coverage >= 0.35:
            summary += " The current material supports a provisional synthesis, but some conclusions remain tentative."
        else:
            summary += " Coverage remains thin, so conclusions should be treated cautiously."
        paragraphs.append(
            summary + _format_refs(claim_ids[:3] + fact_ids[:2] + evidence_ids[:2])
        )
        if claim_texts:
            paragraphs.append("Representative claims: " + "; ".join(text.rstrip(". ") for text in claim_texts[:3]) + ".")
    elif fact_count > 0 or evidence_count > 0:
        if fact_texts:
            paragraphs.append(
                "The available material supports these factual points: "
                + "; ".join(text.rstrip(". ") for text in fact_texts[:3])
                + "."
                + _format_refs(fact_ids[:3] + evidence_ids[:2])
            )
        else:
            paragraphs.append(
                "This section has limited structured support, so the draft relies on a sparse evidence base rather than a dense factual record."
                + _format_refs(fact_ids[:3] + evidence_ids[:2])
            )
    else:
        paragraphs.append(
            "Evidence for this section remains limited, so only a cautious summary can be provided."
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
    """根据覆盖不足或冲突情况生成开放问题列表。"""
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
    """按原始顺序去重引用 ID。"""
    seen: set[str] = set()
    deduped: list[str] = []
    for ref in refs:
        if ref and ref not in seen:
            seen.add(ref)
            deduped.append(ref)
    return deduped
