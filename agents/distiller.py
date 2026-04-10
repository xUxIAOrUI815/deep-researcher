import asyncio
import os
import json
import re
import httpx
from collections import defaultdict
from typing import List, Optional, Dict, Any, Iterable
from dataclasses import dataclass

from core.observability import EventType, get_observer
from schemas.state import AtomicFact, Claim, ConflictRecord, DistillerOutputs, Evidence, SectionEvidencePack


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com"


class DistillationError(Exception):
    pass


@dataclass
class DistillationResult:
    facts: List[AtomicFact]
    summary: str
    raw_response: str


class DistillerAgent:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 4000,
        temperature: float = 0.1
    ):
        self.api_key = api_key or DEEPSEEK_API_KEY
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._semaphore = asyncio.Semaphore(2)
        self._request_lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_request_interval = 1.5

    async def distill(self, markdown_text: str, source_url: str, task_id: Optional[str] = None) -> DistillationResult:
        async with self._semaphore:
            async with self._request_lock:
                now = asyncio.get_event_loop().time()
                elapsed = now - self._last_request_time
                if elapsed < self._min_request_interval:
                    await asyncio.sleep(self._min_request_interval - elapsed)
                self._last_request_time = asyncio.get_event_loop().time()

            if not self.api_key:
                raise DistillationError("DEEPSEEK_API_KEY not set in environment")

            prompt = self._build_prompt(markdown_text)

            for attempt in range(3):
                try:
                    async with httpx.AsyncClient(timeout=90.0) as client:
                        response = await client.post(
                            f"{DEEPSEEK_API_BASE}/chat/completions",
                            headers={
                                "Authorization": f"Bearer {self.api_key}",
                                "Content-Type": "application/json"
                            },
                            json={
                                "model": self.model,
                                "messages": [
                                    {"role": "system", "content": self._get_system_prompt()},
                                    {"role": "user", "content": prompt}
                                ],
                                "max_tokens": self.max_tokens,
                                "temperature": self.temperature
                            }
                        )

                        if response.status_code == 429:
                            wait_time = 2 ** attempt
                            print(f"[DistillerAgent] Rate limited, waiting {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue

                        response.raise_for_status()
                        data = response.json()

                        raw_content = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                        if not raw_content:
                            raise DistillationError(f"Empty response from DeepSeek API for {source_url}")

                        facts = self._parse_facts_from_response(raw_content, source_url, task_id)

                        if not facts:
                            raise DistillationError(f"No valid facts parsed from response for {source_url}")

                        summary = self._extract_summary(raw_content)

                        print(f"[DistillerAgent] Successfully distilled {len(facts)} facts from {source_url}")

                        return DistillationResult(
                            facts=facts,
                            summary=summary,
                            raw_response=raw_content
                        )

                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        wait_time = 2 ** attempt
                        print(f"[DistillerAgent] Rate limited, waiting {wait_time}s...")
                        await asyncio.sleep(wait_time)
                        continue
                    raise DistillationError(f"HTTP error calling DeepSeek API: {e}")

                except DistillationError:
                    raise

                except Exception as e:
                    raise DistillationError(f"Unexpected error calling DeepSeek API: {e}")

            raise DistillationError(f"Max retries ({3}) exceeded for {source_url}")

    def _get_system_prompt(self) -> str:
        return """你是一个专业的原子事实提炼专家。你的任务是将网页内容拆解为语义完整、不可分割的事实陈述。

核心原则：
1. 每个事实必须语义完整，单独拎出来阅读时无需额外上下文
2. 实体必须明确，不使用代词（将"该公司"替换为具体公司名如"华为"）
3. 每条事实必须直接关联数值、时间或具体事件
4. 事实来源必须可信，基于文本内容推断原始来源权威性

输出格式要求：
- 使用JSON数组格式输出
- 每个fact对象包含：text(事实文本), confidence(可信度0.0-1.0)
- 事实按重要性和可信度降序排列
- 提取5-15条核心事实

注意：只输出JSON数组，不要有其他文字。"""

    def _build_prompt(self, markdown_text: str) -> str:
        truncated = markdown_text[:8000]

        return f"""请分析以下文本，提取原子事实：

{'-'*60}
{truncated}
{'-'*60}

要求：
1. 提取所有包含具体数据、时间、公司/人物名称的事实
2. 每条事实必须语义完整
3. 将所有代词替换为具体名称
4. 根据来源权威性给出置信度评分
5. 仅返回JSON格式的事实数组"""

    def _parse_facts_from_response(
        self,
        raw_response: str,
        source_url: str,
        task_id: Optional[str] = None
    ) -> List[AtomicFact]:
        facts = []

        try:
            json_match = re.search(r'\[.*\]', raw_response, re.DOTALL)
            if json_match:
                fact_data = json.loads(json_match.group())

                for item in fact_data:
                    if isinstance(item, dict) and "text" in item:
                        fact = AtomicFact(
                            text=item["text"],
                            source_url=source_url,
                            confidence=float(item.get("confidence", 0.7)),
                            task_id=task_id
                        )
                        facts.append(fact)
        except json.JSONDecodeError as e:
            raise DistillationError(f"Failed to parse JSON from DeepSeek response: {e}")

        if not facts:
            raise DistillationError(f"No facts found in JSON response for {source_url}")

        return facts

    def _extract_summary(self, raw_response: str) -> str:
        lines = raw_response.split("\n")
        summary_lines = []

        for line in lines[:3]:
            if line.strip() and len(line.strip()) > 20:
                clean = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', line)
                summary_lines.append(clean.strip())

        return " | ".join(summary_lines[:2])

    async def distill_batch(
        self,
        items: List[Dict[str, str]]
    ) -> List[DistillationResult]:
        async def distill_with_url(item: Dict[str, str]) -> DistillationResult:
            text = item.get("markdown", item.get("text", ""))
            url = item.get("source_url", item.get("url", ""))
            tid = item.get("task_id")
            return await self.distill(text, url, tid)

        tasks = [distill_with_url(item) for item in items]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        processed = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[DistillerAgent] Batch item {i} failed: {result}")
                raise result
            processed.append(result)

        return processed


async def run_distiller(
    *,
    task_id: Optional[str],
    task: Optional[Dict[str, Any]] = None,
    researcher_outputs: Dict[str, Any],
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    knowledge_refs: Dict[str, Any],
    run_context: Any = None,
) -> "DistillerOutputs":
    """Graph-facing distiller entry.

    The existing DistillerAgent remains available above; the new graph path
    calls this function so graph.py does not host distillation behavior.
    """
    observer = get_observer()
    if run_context is not None:
        observer.record_task_event(
            run_context,
            EventType.DISTILL_STARTED,
            task_id or "unknown-task",
            message="Distiller started processing researcher outputs.",
            payload={"task_id": task_id},
        )

    passages = list((researcher_outputs or {}).get("passages", []) or [])
    sources = list((researcher_outputs or {}).get("sources", []) or [])
    metadata = dict((researcher_outputs or {}).get("metadata", {}) or {})
    follow_up_hints = list(metadata.get("follow_up_hints", []) or [])

    clean_passages = _clean_passages(passages, run_context=run_context, task_id=task_id)
    claims = _extract_claims(clean_passages, run_context=run_context, task_id=task_id)
    evidence = _map_evidence(claims, clean_passages)
    facts = _extract_atomic_facts(claims, clean_passages, run_context=run_context, task_id=task_id)
    compressed_facts, compression_summary = _compress_atomic_facts(facts, run_context=run_context, task_id=task_id)
    _remap_claim_fact_ids(claims, compressed_facts)
    _link_claims_and_evidence(claims, evidence)
    conflicts = _detect_conflicts(claims, evidence, run_context=run_context, task_id=task_id)
    section_evidence_packs = _build_section_evidence_packs(
        report_outline=report_outline,
        section_goals=section_goals,
        claims=claims,
        evidence=evidence,
        facts=compressed_facts,
        conflicts=conflicts,
        run_context=run_context,
    )
    unresolved_gaps = _derive_unresolved_gaps(section_goals, section_evidence_packs, follow_up_hints)
    coverage_summary = _build_coverage_summary(
        section_goals=section_goals,
        section_evidence_packs=section_evidence_packs,
        claims=claims,
        evidence=evidence,
        conflicts=conflicts,
        unresolved_gaps=unresolved_gaps,
    )

    source_ids = [str(source.get("source_id", "")) for source in sources if source.get("source_id")]
    updated_knowledge_refs = dict(knowledge_refs or {})
    updated_knowledge_refs.setdefault("collection_name", "")
    updated_knowledge_refs["fact_ids"] = _unique_ids(
        list(updated_knowledge_refs.get("fact_ids", [])) + [fact.id for fact in compressed_facts]
    )
    updated_knowledge_refs["claim_ids"] = _unique_ids(
        list(updated_knowledge_refs.get("claim_ids", [])) + [claim.id for claim in claims]
    )
    updated_knowledge_refs["evidence_ids"] = _unique_ids(
        list(updated_knowledge_refs.get("evidence_ids", [])) + [item.id for item in evidence]
    )
    updated_knowledge_refs["conflict_ids"] = _unique_ids(
        list(updated_knowledge_refs.get("conflict_ids", [])) + [item.id for item in conflicts]
    )
    updated_knowledge_refs["source_ids"] = _unique_ids(
        list(updated_knowledge_refs.get("source_ids", [])) + source_ids
    )

    return DistillerOutputs(
        task_id=task_id,
        clean_passages=clean_passages,
        atomic_facts=compressed_facts,
        claims=claims,
        evidence=evidence,
        conflicts=conflicts,
        fact_ids=[fact.id for fact in compressed_facts],
        claim_ids=[claim.id for claim in claims],
        evidence_ids=[item.id for item in evidence],
        conflict_ids=[item.id for item in conflicts],
        knowledge_refs=updated_knowledge_refs,
        section_evidence_packs=[pack.model_dump() for pack in section_evidence_packs],
        compression_summary=compression_summary,
        coverage_summary=coverage_summary,
        unresolved_gaps=unresolved_gaps,
        summary=(
            f"Distilled {len(clean_passages)} clean passages into "
            f"{len(compressed_facts)} atomic facts, {len(claims)} claims, "
            f"{len(evidence)} evidence items, {len(conflicts)} conflicts, and "
            f"{len(section_evidence_packs)} section evidence packs."
        ),
    )


BOILERPLATE_PATTERNS = [
    r"(?i)cookie",
    r"(?i)privacy policy",
    r"(?i)terms of service",
    r"(?i)subscribe",
    r"(?i)newsletter",
    r"(?i)advertisement",
    r"(?i)related articles",
]

CONTRADICTION_TERMS = {
    "increase": "decrease",
    "decrease": "increase",
    "rise": "fall",
    "fall": "rise",
    "up": "down",
    "down": "up",
    "enabled": "disabled",
    "available": "unavailable",
}


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_key(text: str) -> str:
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())


def _jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _unique_ids(values: List[str]) -> List[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _clean_passages(
    passages: List[Dict[str, Any]],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    observer = get_observer()
    cleaned: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for passage in passages:
        raw_text = _normalize_text(str(passage.get("text", "")))
        if not raw_text:
            continue
        lines = []
        for line in raw_text.splitlines():
            line = _normalize_text(line)
            if not line:
                continue
            if any(re.search(pattern, line) for pattern in BOILERPLATE_PATTERNS):
                continue
            lines.append(line)
        text = _normalize_text(" ".join(lines))
        if len(text) < 60:
            continue
        key = _normalize_key(text[:500])
        if key in seen_keys:
            continue
        seen_keys.add(key)
        cleaned_passage = {
            "passage_id": passage.get("passage_id"),
            "source_id": passage.get("source_id"),
            "url": passage.get("url", ""),
            "title": passage.get("title", ""),
            "query": passage.get("query", ""),
            "text": text,
        }
        cleaned.append(cleaned_passage)
        if run_context is not None:
            observer.record_task_event(
                run_context,
                EventType.PASSAGE_CLEANED,
                task_id or "unknown-task",
                message="Distiller cleaned passage.",
                payload={
                    "passage_id": cleaned_passage.get("passage_id"),
                    "source_id": cleaned_passage.get("source_id"),
                    "text_length": len(text),
                },
            )
    return cleaned


def _split_sentences(text: str) -> List[str]:
    chunks = re.split(r"(?<=[.!?。！？；;])\s+", text)
    return [_normalize_text(chunk) for chunk in chunks if _normalize_text(chunk)]


def _sentence_confidence(sentence: str, source_id: str) -> float:
    score = 0.5
    if re.search(r"\d", sentence):
        score += 0.15
    if re.search(r"\b(according|reported|announced|stated|showed|grew|declined|launched)\b", sentence.lower()):
        score += 0.1
    if len(sentence) > 120:
        score += 0.05
    if source_id:
        score += 0.05
    return max(0.3, min(0.95, score))


def _extract_claims(
    clean_passages: List[Dict[str, Any]],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> List[Claim]:
    observer = get_observer()
    claims: list[Claim] = []
    seen_claims: set[str] = set()
    for passage in clean_passages:
        sentences = _split_sentences(passage.get("text", ""))
        selected = 0
        for sentence in sentences:
            if len(sentence) < 50:
                continue
            if not re.search(r"\d|\b(is|are|was|were|has|have|will|reported|announced|according)\b", sentence.lower()):
                continue
            normalized = _normalize_key(sentence)
            if normalized in seen_claims:
                continue
            seen_claims.add(normalized)
            claim = Claim(
                text=sentence,
                confidence=_sentence_confidence(sentence, passage.get("source_id", "")),
                task_id=task_id,
            )
            claim.evidence_ids = []
            claims.append(claim)
            selected += 1
            if run_context is not None:
                observer.record_task_event(
                    run_context,
                    EventType.CLAIM_EXTRACTED,
                    task_id or "unknown-task",
                    message="Distiller extracted claim.",
                    payload={"claim_id": claim.id, "source_id": passage.get("source_id"), "passage_id": passage.get("passage_id")},
                )
            if selected >= 3:
                break
    return claims


def _extract_atomic_facts(
    claims: List[Claim],
    clean_passages: List[Dict[str, Any]],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> List[AtomicFact]:
    observer = get_observer()
    facts: list[AtomicFact] = []
    for claim in claims:
        source_id = ""
        source_url = ""
        snippet = claim.text[:280]
        for passage in clean_passages:
            if claim.text in passage.get("text", ""):
                source_id = passage.get("source_id", "")
                source_url = passage.get("url", "")
                snippet = claim.text[:280]
                break
        fact = AtomicFact(
            text=claim.text,
            source_url=source_url,
            confidence=claim.confidence,
            task_id=task_id,
            snippet=snippet,
            verified_count=1,
        )
        facts.append(fact)
        if run_context is not None:
            observer.record_task_event(
                run_context,
                EventType.FACT_EXTRACTED,
                task_id or "unknown-task",
                message="Distiller extracted atomic fact.",
                payload={"fact_id": fact.id, "source_id": source_id},
            )
    return facts


def _map_evidence(claims: List[Claim], clean_passages: List[Dict[str, Any]]) -> List[Evidence]:
    evidence_items: list[Evidence] = []
    for claim in claims:
        matching_passage = next((passage for passage in clean_passages if claim.text in passage.get("text", "")), None)
        if matching_passage is None:
            continue
        evidence = Evidence(
            source_id=str(matching_passage.get("source_id", "")),
            source_url=str(matching_passage.get("url", "")),
            quote=claim.text[:500],
            summary=(matching_passage.get("title", "") or claim.text[:160])[:240],
            quality_score=max(0.3, min(0.95, claim.confidence + 0.05)),
            task_id=claim.task_id,
        )
        evidence_items.append(evidence)
    return evidence_items


def _link_claims_and_evidence(claims: List[Claim], evidence: List[Evidence]) -> None:
    evidence_iter = iter(evidence)
    for claim in claims:
        try:
            evidence_item = next(evidence_iter)
        except StopIteration:
            break
        claim.evidence_ids = [evidence_item.id]
        evidence_item.claim_ids = [claim.id]


def _compress_atomic_facts(
    facts: List[AtomicFact],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> tuple[List[AtomicFact], str]:
    observer = get_observer()
    compressed: list[AtomicFact] = []
    for fact in facts:
        matched = None
        for kept in compressed:
            similarity = _jaccard_similarity(_tokenize(fact.text), _tokenize(kept.text))
            if similarity >= 0.82:
                matched = kept
                break
        if matched:
            matched.verified_count += 1
            matched.confidence = max(matched.confidence, fact.confidence)
            continue
        compressed.append(fact)
    summary = f"Compressed {len(facts)} extracted facts into {len(compressed)} unique atomic facts."
    if run_context is not None:
        observer.record_task_event(
            run_context,
            EventType.COMPRESSION_COMPLETED,
            task_id or "unknown-task",
            message="Distiller completed knowledge compression.",
            payload={"input_facts": len(facts), "compressed_facts": len(compressed)},
        )
    return compressed, summary


def _remap_claim_fact_ids(claims: List[Claim], compressed_facts: List[AtomicFact]) -> None:
    for claim in claims:
        matched_ids = [
            fact.id for fact in compressed_facts
            if _jaccard_similarity(_tokenize(claim.text), _tokenize(fact.text)) >= 0.7
        ]
        claim.fact_ids = matched_ids[:3]


def _claim_overlap(left: Claim, right: Claim) -> float:
    return _jaccard_similarity(_tokenize(left.text), _tokenize(right.text))


def _extract_numbers(text: str) -> List[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def _extract_years(text: str) -> List[str]:
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def _detect_conflicts(
    claims: List[Claim],
    evidence: List[Evidence],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> List[ConflictRecord]:
    observer = get_observer()
    conflicts: list[ConflictRecord] = []
    seen_pairs: set[tuple[str, str]] = set()
    for index, left in enumerate(claims):
        for right in claims[index + 1:]:
            if _claim_overlap(left, right) < 0.45:
                continue
            pair = tuple(sorted((left.id, right.id)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            numbers_left = set(_extract_numbers(left.text))
            numbers_right = set(_extract_numbers(right.text))
            years_left = set(_extract_years(left.text))
            years_right = set(_extract_years(right.text))
            conflict_type = None
            severity = "medium"

            if numbers_left and numbers_right and numbers_left != numbers_right:
                conflict_type = "numerical_conflict"
                severity = "high"
            elif years_left and years_right and years_left != years_right:
                conflict_type = "temporal_conflict"
            else:
                lower_left = left.text.lower()
                lower_right = right.text.lower()
                for term, opposite in CONTRADICTION_TERMS.items():
                    if term in lower_left and opposite in lower_right:
                        conflict_type = "contradiction"
                        severity = "high"
                        break
                if conflict_type is None and lower_left != lower_right and _claim_overlap(left, right) > 0.7:
                    conflict_type = "definition_mismatch"

            if conflict_type is None:
                continue

            related_evidence_ids = _unique_ids(list(left.evidence_ids) + list(right.evidence_ids))
            conflict = ConflictRecord(
                claim_ids=[left.id, right.id],
                evidence_ids=related_evidence_ids,
                description=f"{conflict_type} between: {left.text[:120]} <> {right.text[:120]}",
                severity=severity,
            )
            conflicts.append(conflict)
            if run_context is not None:
                observer.record_task_event(
                    run_context,
                    EventType.CONFLICT_DETECTED,
                    task_id or "unknown-task",
                    message="Distiller detected conflicting claims.",
                    payload={"conflict_id": conflict.id, "type": conflict_type, "severity": severity},
                )
    return conflicts


def _build_section_evidence_packs(
    *,
    report_outline: Dict[str, Any],
    section_goals: List[Dict[str, Any]],
    claims: List[Claim],
    evidence: List[Evidence],
    facts: List[AtomicFact],
    conflicts: List[ConflictRecord],
    run_context: Any = None,
) -> List[SectionEvidencePack]:
    observer = get_observer()
    sections = list((report_outline or {}).get("sections", []) or [])
    packs: list[SectionEvidencePack] = []

    if not sections and section_goals:
        sections = [
            {"section_id": goal.get("section_id", ""), "title": goal.get("goal", ""), "goal": goal.get("goal", "")}
            for goal in section_goals
        ]

    if not sections and not section_goals:
        return packs

    goals_by_section = {str(goal.get("section_id")): goal for goal in section_goals}

    for section in sections:
        section_id = str(section.get("section_id", ""))
        title = str(section.get("title", ""))
        goal = goals_by_section.get(section_id, {})
        goal_text = str(goal.get("goal", "") or section.get("goal", ""))
        section_tokens = _tokenize(f"{title} {goal_text}")

        selected_claims = [
            claim for claim in claims
            if _jaccard_similarity(_tokenize(claim.text), section_tokens) >= 0.12
        ]
        selected_evidence_ids = _unique_ids(
            [evidence_id for claim in selected_claims for evidence_id in claim.evidence_ids]
        )
        selected_fact_ids = _unique_ids(
            [fact_id for claim in selected_claims for fact_id in claim.fact_ids]
        )
        selected_conflict_ids = [
            conflict.id for conflict in conflicts
            if set(conflict.claim_ids) & {claim.id for claim in selected_claims}
        ]
        coverage_score = min(1.0, 0.2 * len(selected_claims) + 0.1 * len(selected_evidence_ids))
        pack = SectionEvidencePack(
            section_id=section_id,
            goal=goal_text,
            claim_ids=[claim.id for claim in selected_claims],
            evidence_ids=selected_evidence_ids,
            fact_ids=selected_fact_ids,
            conflict_ids=selected_conflict_ids,
            coverage_score=coverage_score,
            notes=f"Selected {len(selected_claims)} claims for section '{title}'.",
        )
        packs.append(pack)
        if run_context is not None:
            observer.record_evidence_event(
                run_context,
                EventType.EVIDENCE_PACK_CREATED,
                section_id=section_id,
                message="Distiller created section evidence pack.",
                payload={
                    "pack_id": pack.pack_id,
                    "claim_count": len(pack.claim_ids),
                    "evidence_count": len(pack.evidence_ids),
                    "fact_count": len(pack.fact_ids),
                },
            )
    return packs


def _derive_unresolved_gaps(
    section_goals: List[Dict[str, Any]],
    packs: List[SectionEvidencePack],
    follow_up_hints: List[str],
) -> List[str]:
    pack_by_section = {pack.section_id: pack for pack in packs}
    gaps: list[str] = []
    for goal in section_goals:
        section_id = str(goal.get("section_id", ""))
        pack = pack_by_section.get(section_id)
        if pack is None or pack.coverage_score < 0.35:
            gaps.append(f"Low evidence coverage for section {section_id}: {goal.get('goal', '')}".strip())
    for hint in follow_up_hints[:3]:
        gaps.append(f"Coverage gap around hinted topic: {hint}")
    return gaps[:6]


def _build_coverage_summary(
    *,
    section_goals: List[Dict[str, Any]],
    section_evidence_packs: List[SectionEvidencePack],
    claims: List[Claim],
    evidence: List[Evidence],
    conflicts: List[ConflictRecord],
    unresolved_gaps: List[str],
) -> Dict[str, Any]:
    pack_by_section = {pack.section_id: pack for pack in section_evidence_packs}
    covered_sections: list[str] = []
    uncovered_sections: list[str] = []
    section_status: list[dict[str, Any]] = []

    for goal in section_goals:
        section_id = str(goal.get("section_id", ""))
        goal_text = str(goal.get("goal", ""))
        pack = pack_by_section.get(section_id)
        coverage_score = float(pack.coverage_score) if pack else 0.0
        status = "covered" if coverage_score >= 0.5 else "partial" if coverage_score >= 0.25 else "uncovered"
        section_status.append(
            {
                "section_id": section_id,
                "goal": goal_text,
                "coverage_score": coverage_score,
                "status": status,
                "claim_count": len(pack.claim_ids) if pack else 0,
                "evidence_count": len(pack.evidence_ids) if pack else 0,
                "conflict_count": len(pack.conflict_ids) if pack else 0,
            }
        )
        if status == "covered":
            covered_sections.append(section_id)
        else:
            uncovered_sections.append(section_id)

    avg_section_coverage = (
        sum(item["coverage_score"] for item in section_status) / len(section_status)
        if section_status else 0.0
    )
    evidence_density = (len(evidence) / max(1, len(claims))) if claims else 0.0
    conflict_pressure = min(1.0, len(conflicts) / max(1, len(claims) or 1))

    if avg_section_coverage >= 0.65 and evidence_density >= 0.8 and conflict_pressure <= 0.2:
        sufficiency_level = "sufficient_for_writing"
    elif avg_section_coverage >= 0.35 or len(claims) > 0:
        sufficiency_level = "partial"
    else:
        sufficiency_level = "insufficient"

    return {
        "covered_sections": covered_sections,
        "uncovered_sections": uncovered_sections,
        "section_status": section_status,
        "avg_section_coverage": round(avg_section_coverage, 3),
        "evidence_density": round(evidence_density, 3),
        "conflict_pressure": round(conflict_pressure, 3),
        "sufficiency_level": sufficiency_level,
        "ready_for_writing_sections": [item["section_id"] for item in section_status if item["coverage_score"] >= 0.5],
        "unresolved_gap_count": len(unresolved_gaps),
    }
