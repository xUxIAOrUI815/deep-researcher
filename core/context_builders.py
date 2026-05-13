from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.session_retrieval import SessionRetrievalService
from schemas.retrieval import (
    PlannerContext,
    ResearcherContext,
    SessionRetrievalQuery,
    WriterContext,
)


class PlannerContextBuilder:
    """为规划器组装决策所需的会话上下文。"""

    def __init__(self, retrieval_service: SessionRetrievalService):
        self.retrieval_service = retrieval_service

    def build(
        self,
        *,
        research_id: str,
        session_id: str,
        user_query: str,
        task_tree: Dict[str, Any],
        active_task_id: Optional[str] = None,
    ) -> PlannerContext:
        # 优先使用当前任务的标题/查询作为检索主题；没有活动任务时回退到用户原始问题。
        active_task = dict(task_tree.get(active_task_id or "", {}) or {})
        topic = str(active_task.get("title") or active_task.get("query") or user_query)
        result = self.retrieval_service.retrieve(
            research_id=research_id,
            session_id=session_id,
            query=SessionRetrievalQuery(
                task_id=active_task_id,
                topic=topic,
                semantic_query=topic,
                semantic_weight=0.2,
                limit_per_type=8,
            ),
        )
        section_readiness = self._section_readiness(result.section_packs, result.unresolved_gaps)
        latest_coverage = dict(result.latest_coverage_snapshot or {})
        # 没有历史覆盖率快照时，基于当前章节成熟度构造一个保守的默认摘要。
        if not latest_coverage:
            latest_coverage = {
                "avg_section_coverage": 0.0,
                "sufficiency_level": "insufficient",
                "completed_section_count": 0,
                "partial_section_count": 0,
                "uncovered_section_count": len([row for row in section_readiness if row["maturity"] == "weak"]),
            }
        coverage_summary = {
            "avg_section_coverage": latest_coverage.get("avg_section_coverage", 0.0),
            "sufficiency_level": latest_coverage.get("sufficiency_level", "insufficient"),
            "covered_sections": [row["section_id"] for row in section_readiness if row["maturity"] == "ready"],
            "uncovered_sections": [row["section_id"] for row in section_readiness if row["maturity"] == "weak"],
                "section_status": section_readiness,
            }
        # ready 和 developing 都可以进入写作候选，weak 章节应继续补证据。
        writing_ready_sections = [row["section_id"] for row in section_readiness if row["maturity"] in {"ready", "developing"}]
        return PlannerContext(
            research_id=research_id,
            session_id=session_id,
            coverage_summary=coverage_summary,
            latest_coverage_snapshot=result.latest_coverage_snapshot,
            latest_novelty_snapshot=result.latest_novelty_snapshot,
            section_readiness=section_readiness,
            section_packs=result.section_packs,
            unresolved_gaps=result.unresolved_gaps,
            conflict_hotspots=result.conflicts,
            relevant_claims=result.claims,
            relevant_evidence=result.evidence,
            writing_ready_sections=writing_ready_sections,
            retrieval_meta=result.retrieval_meta,
        )

    def _section_readiness(
        self,
        section_packs: List[Dict[str, Any]],
        unresolved_gaps: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """根据覆盖率、未解决缺口和冲突数量评估每个章节的成熟度。"""

        gap_counts: Dict[str, int] = {}
        for gap in unresolved_gaps:
            section_id = str(gap.get("section_id", "")).strip()
            if section_id:
                gap_counts[section_id] = gap_counts.get(section_id, 0) + 1

        readiness: list[Dict[str, Any]] = []
        for pack in section_packs:
            section_id = str(pack.get("section_id", "")).strip()
            coverage = float(pack.get("coverage_score", 0.0) or 0.0)
            conflict_count = len(pack.get("conflict_ids", []) or [])
            gap_count = gap_counts.get(section_id, 0)
            # 阈值用于给规划器一个粗粒度信号：可写、发展中或仍薄弱。
            if coverage >= 0.65 and gap_count == 0:
                maturity = "ready"
            elif coverage >= 0.35:
                maturity = "developing"
            else:
                maturity = "weak"
            readiness.append(
                {
                    "section_id": section_id,
                    "section_title": pack.get("section_title", ""),
                    "coverage_score": coverage,
                    "gap_count": gap_count,
                    "conflict_count": conflict_count,
                    "maturity": maturity,
                    "pack_id": pack.get("pack_id"),
                }
            )
        # 覆盖率高、缺口少的章节排在前面，便于规划器优先推进。
        readiness.sort(key=lambda item: (item["coverage_score"], -item["gap_count"]), reverse=True)
        return readiness


class ResearcherContextBuilder:
    """为研究员组装检索去重、缺口和事实线索上下文。"""

    def __init__(self, retrieval_service: SessionRetrievalService):
        self.retrieval_service = retrieval_service

    def build(
        self,
        *,
        research_id: str,
        session_id: str,
        root_user_query: str,
        task_id: Optional[str],
        task: Optional[Dict[str, Any]],
    ) -> ResearcherContext:
        # 研究任务可能是树上的子任务；没有子任务标题时回退到根问题。
        topic = str((task or {}).get("title") or (task or {}).get("query") or root_user_query)
        result = self.retrieval_service.retrieve(
            research_id=research_id,
            session_id=session_id,
            query=SessionRetrievalQuery(
                task_id=task_id,
                topic=topic,
                semantic_query=topic,
                semantic_weight=0.2,
                limit_per_type=200,
            ),
        )
        # 已见来源用于后续搜索和入库去重，减少重复抓取同一批资料。
        already_seen_source_ids = [str(row.get("source_id", "")) for row in result.sources if row.get("source_id")]
        already_seen_source_urls = [str(row.get("url", "")) for row in result.sources if row.get("url")]
        authority_gaps = []
        # 如果当前材料缺少高权威来源，显式提醒研究员优先补一手/权威资料。
        if not any(float(row.get("authority_score", 0.0) or 0.0) >= 0.8 for row in result.sources):
            authority_gaps.append("high_authority_primary_source")
        focus_sections = [
            str(row.get("section_id", ""))
            for row in result.unresolved_gaps
            if str(row.get("section_id", "")).strip()
        ]
        return ResearcherContext(
            research_id=research_id,
            session_id=session_id,
            already_seen_source_ids=already_seen_source_ids,
            already_seen_source_urls=already_seen_source_urls,
            source_registry=result.source_registry,
            unresolved_gaps=result.unresolved_gaps,
            relevant_claims=result.claims,
            relevant_facts=result.facts,
            focus_sections=focus_sections,
            authority_gaps=authority_gaps,
            search_dedup_hints={
                "source_ids": already_seen_source_ids,
                "source_urls": already_seen_source_urls,
            },
            retrieval_meta=result.retrieval_meta,
        )


class WriterContextBuilder:
    """为写作者按报告章节组装证据包和相关事实。"""

    def __init__(self, retrieval_service: SessionRetrievalService):
        self.retrieval_service = retrieval_service

    def build(
        self,
        *,
        research_id: str,
        session_id: str,
        report_outline: Dict[str, Any],
        section_goals: List[Dict[str, Any]],
        fallback_section_packs: List[Dict[str, Any]],
    ) -> WriterContext:
        sections = list((report_outline or {}).get("sections", []) or [])
        # 如果尚未生成正式大纲，就用 section_goals 临时构造写作章节。
        if not sections and section_goals:
            sections = [
                {
                    "section_id": goal.get("section_id", ""),
                    "title": goal.get("goal", ""),
                    "goal": goal.get("goal", ""),
                    "order": index + 1,
                }
                for index, goal in enumerate(section_goals)
            ]

        used_packs: list[Dict[str, Any]] = []
        section_contexts: list[Dict[str, Any]] = []
        retrieval_count = 0
        used_session_pack = False
        used_fallback_pack = False
        for section in sections:
            section_id = str(section.get("section_id", "")).strip()
            # 写作阶段按章节检索，保证 claims/facts/evidence 与当前章节目标相关。
            retrieval = self.retrieval_service.retrieve(
                research_id=research_id,
                session_id=session_id,
                query=SessionRetrievalQuery(
                    section_id=section_id,
                    limit_per_type=6,
                    semantic_query=str(section.get("goal") or section.get("title") or section_id),
                    semantic_weight=0.15,
                ),
            )
            retrieval_count += 1
            session_pack = retrieval.section_packs[0] if retrieval.section_packs else {}
            fallback_pack = next(
                (item for item in fallback_section_packs if str(item.get("section_id", "")).strip() == section_id),
                {},
            )
            # 会话内最新证据包优先；没有命中时使用调用方提供的 fallback。
            pack = session_pack or fallback_pack
            if session_pack:
                used_session_pack = True
            elif fallback_pack:
                used_fallback_pack = True
            if pack:
                used_packs.append(pack)
            section_contexts.append(
                {
                    "section_id": section_id,
                    "title": section.get("title", ""),
                    "goal": section.get("goal", ""),
                    "pack": pack,
                    "claims": retrieval.claims,
                    "facts": retrieval.facts,
                    "evidence": retrieval.evidence,
                    "conflicts": retrieval.conflicts,
                }
            )

        # 标记上下文来源，便于下游判断证据是来自会话检索、fallback 还是二者混合。
        context_source = "fallback"
        if used_session_pack and used_fallback_pack:
            context_source = "mixed"
        elif used_session_pack:
            context_source = "session"
        elif not used_packs and fallback_section_packs:
            used_packs = list(fallback_section_packs)
            context_source = "fallback"

        return WriterContext(
            research_id=research_id,
            session_id=session_id,
            context_source=context_source,
            section_evidence_packs=used_packs,
            section_contexts=section_contexts,
            retrieval_meta={"section_count": len(sections), "retrieval_calls": retrieval_count},
        )
