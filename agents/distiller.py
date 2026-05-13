import json
import os
import re
from typing import List, Optional, Dict, Any, Iterable

import httpx
from dotenv import load_dotenv

from core.observability import EventType, get_observer
from schemas.state import AtomicFact, Claim, ConflictRecord, DistillerOutputs, Evidence, SectionEvidencePack


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
    """执行蒸馏流程，将 researcher 输出整理为结构化知识。"""
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
    distillation_mode = "llm" if _should_use_llm_distiller(metadata) else "local"
    if distillation_mode == "llm":
        facts, claims, evidence, distiller_note = await _distill_with_deepseek(
            clean_passages,
            task=task or {},
            task_id=task_id,
            run_context=run_context,
        )
        if not facts and not claims and not evidence:
            facts, claims, evidence = _distill_locally(clean_passages, run_context=run_context, task_id=task_id)
            distillation_mode = "local_fallback"
            distiller_note = "DeepSeek returned no usable structured outputs; used local fallback."
    else:
        facts, claims, evidence = _distill_locally(clean_passages, run_context=run_context, task_id=task_id)
        distiller_note = "Used local deterministic distiller."
    compressed_facts, compression_summary = _compress_atomic_facts(
        facts,
        run_context=run_context,
        task_id=task_id,
    )
    _remap_claim_fact_ids(claims, compressed_facts)
    _link_claims_and_evidence(claims, evidence)
    _link_facts_and_evidence(compressed_facts, evidence)
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

    passthrough_knowledge_refs = dict(knowledge_refs or {})

    return DistillerOutputs(
        task_id=task_id,
        sources=sources,
        clean_passages=clean_passages,
        atomic_facts=compressed_facts,
        claims=claims,
        evidence=evidence,
        conflicts=conflicts,
        fact_ids=[fact.id for fact in compressed_facts],
        claim_ids=[claim.id for claim in claims],
        evidence_ids=[item.id for item in evidence],
        conflict_ids=[item.id for item in conflicts],
        knowledge_refs=passthrough_knowledge_refs,
        section_evidence_packs=[pack.model_dump() for pack in section_evidence_packs],
        compression_summary=f"{compression_summary} Distiller mode: {distillation_mode}. {distiller_note}",
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

DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_DEFAULT_MODEL = "deepseek-chat"


def _load_deepseek_api_key() -> str:
    """Load the DeepSeek API key from .env and normalize common shell quoting mistakes."""
    load_dotenv(override=False)
    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip().strip('"').strip("'")
    if api_key.startswith("DEEPSEEK_API_KEY="):
        api_key = api_key.split("=", 1)[1].strip().strip('"').strip("'")
    return api_key


def _env_flag(name: str, default: str = "auto") -> str:
    """Return a normalized environment flag value."""
    return os.getenv(name, default).strip().lower()


def _should_use_llm_distiller(metadata: Dict[str, Any]) -> bool:
    """Decide whether to use the online DeepSeek distiller for this run."""
    api_key = _load_deepseek_api_key()
    mode = _env_flag("DISTILLER_USE_LLM", "auto")
    if mode in {"0", "false", "no", "off", "local"}:
        return False
    if mode in {"1", "true", "yes", "on", "llm"}:
        return bool(api_key)

    # Auto mode keeps offline/mock tests deterministic while using DeepSeek for live research.
    if not api_key:
        return False
    search_mode = str(metadata.get("search_mode", "")).lower()
    scraper_mode = str(metadata.get("scraper_mode", "")).lower()
    return search_mode != "mock" and scraper_mode != "mock"


def _deepseek_chat_url() -> str:
    """Build an OpenAI-compatible DeepSeek chat completions URL from environment config."""
    base_url = os.getenv("DEEPSEEK_API_BASE", DEEPSEEK_DEFAULT_BASE_URL).strip().rstrip("/")
    if base_url.endswith("/v1"):
        return f"{base_url}/chat/completions"
    return f"{base_url}/v1/chat/completions"


async def _distill_with_deepseek(
    clean_passages: List[Dict[str, Any]],
    *,
    task: Dict[str, Any],
    task_id: Optional[str],
    run_context: Any = None,
) -> tuple[List[AtomicFact], List[Claim], List[Evidence], str]:
    """Use DeepSeek as the LLM Distiller Agent and materialize grounded outputs."""
    api_key = _load_deepseek_api_key()
    if not api_key:
        return [], [], [], "DEEPSEEK_API_KEY is empty."

    max_passages = max(1, int(os.getenv("DISTILLER_MAX_PASSAGES", "5") or 5))
    max_chars = max(1200, int(os.getenv("DISTILLER_MAX_CHARS_PER_PASSAGE", "4500") or 4500))
    model = os.getenv("DEEPSEEK_MODEL_NAME") or os.getenv("DEEPSEEK_MODEL") or DEEPSEEK_DEFAULT_MODEL
    facts: list[AtomicFact] = []
    claims: list[Claim] = []
    evidence: list[Evidence] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=float(os.getenv("DISTILLER_LLM_TIMEOUT", "45"))) as client:
        for passage in clean_passages[:max_passages]:
            prompt = _build_distiller_prompt(task=task, passage=passage, max_chars=max_chars)
            try:
                response = await client.post(
                    _deepseek_chat_url(),
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {
                                "role": "system",
                                "content": (
                                    "You are a strict evidence distillation agent. "
                                    "Return only valid JSON grounded in the supplied passage."
                                ),
                            },
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": 0.0,
                        "max_tokens": int(os.getenv("DISTILLER_LLM_MAX_TOKENS", "1800") or 1800),
                        "response_format": {"type": "json_object"},
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                payload = _parse_llm_json(content)
                new_facts, new_claims, new_evidence = _materialize_distillation_payload(
                    payload,
                    passage=passage,
                    task_id=task_id,
                )
                facts.extend(new_facts)
                claims.extend(new_claims)
                evidence.extend(new_evidence)
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {str(exc)[:240]}")

    note = "DeepSeek distiller completed."
    if errors:
        note += f" Passage errors: {len(errors)}; first={errors[0]}"
    return facts, claims, evidence, note


def _build_distiller_prompt(*, task: Dict[str, Any], passage: Dict[str, Any], max_chars: int) -> str:
    """Build the strict JSON prompt for one cleaned passage."""
    task_text = _normalize_text(str(task.get("query") or task.get("title") or ""))
    passage_text = _normalize_text(str(passage.get("text", "")))[:max_chars]
    title = _normalize_text(str(passage.get("title", "")))
    url = _normalize_text(str(passage.get("url", "")))
    return f"""
你是证据蒸馏 agent。你只能基于输入 passage 输出结构化知识，不得引入外部事实。

任务主题：
{task_text}

来源标题：
{title}

来源 URL：
{url}

passage：
{passage_text}

请输出严格 JSON，格式如下：
{{
  "evidence_units": [
    {{
      "quote": "必须从 passage 原文中截取的最小支持片段",
      "summary": "用中文说明该证据支持什么",
      "evidence_type": "definition|mechanism|timeline|metric|comparison|causal|background",
      "quality_score": 0.0,
      "facts": [
        {{
          "text": "中文客观事实。必须原子化、可验证，不要照抄整句。",
          "confidence": 0.0,
          "fact_type": "definition|mechanism|timeline|metric|composition"
        }}
      ],
      "claims": [
        {{
          "text": "中文来源主张。用于判断、解释、归纳、预测或争议。",
          "confidence": 0.0,
          "claim_type": "definition|mechanism|comparison|prediction|evaluation",
          "modality": "asserted|reported|estimated|projected|disputed"
        }}
      ]
    }}
  ]
}}

规则：
1. fact 必须是客观事实，不能是营销、评价、预测或不确定说法。
2. fact 只表达一个关系，例如“Transformer 是一种神经网络架构。”。
3. claim 表达来源认为、报告称、作者认为等主张或解释。
4. 每个 fact/claim 都必须能由同一个 quote 直接支持。
5. 如果 passage 没有高质量信息，返回 {{"evidence_units": []}}。
6. 不要输出 markdown，不要输出 JSON 外文本。
""".strip()


def _parse_llm_json(content: str) -> Dict[str, Any]:
    """Parse JSON returned by the model, tolerating fenced or lightly wrapped content."""
    text = _normalize_text(content)
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(text[start:end + 1])
    return payload if isinstance(payload, dict) else {"evidence_units": []}


def _materialize_distillation_payload(
    payload: Dict[str, Any],
    *,
    passage: Dict[str, Any],
    task_id: Optional[str],
) -> tuple[List[AtomicFact], List[Claim], List[Evidence]]:
    """Convert LLM JSON into Pydantic records while enforcing grounding and quality gates."""
    facts: list[AtomicFact] = []
    claims: list[Claim] = []
    evidence_items: list[Evidence] = []
    source_id = str(passage.get("source_id", ""))
    source_url = str(passage.get("url", ""))
    passage_text = str(passage.get("text", ""))

    for unit in _iter_evidence_units(payload):
        quote = _resolve_quote(str(unit.get("quote", "")), passage_text, unit)
        if len(quote) < 30:
            continue
        evidence = Evidence(
            source_id=source_id,
            source_url=source_url,
            quote=quote[:500],
            summary=_normalize_text(str(unit.get("summary", "")))[:240],
            quality_score=_bounded_score(unit.get("quality_score"), default=0.75),
            task_id=task_id,
        )
        unit_fact_ids: list[str] = []
        for fact_payload in _as_list(unit.get("facts"))[:6]:
            fact_text = _clean_fact_text(str(fact_payload.get("text", "")))
            if not _is_quality_fact(fact_text, quote):
                continue
            fact = AtomicFact(
                text=fact_text,
                source_id=source_id,
                source_url=source_url,
                confidence=_bounded_score(fact_payload.get("confidence"), default=0.78),
                task_id=task_id,
                snippet=quote[:280],
                verified_count=1,
                confidence_reason=str(fact_payload.get("fact_type") or "llm_distilled_fact"),
            )
            facts.append(fact)
            unit_fact_ids.append(fact.id)
            evidence.fact_ids.append(fact.id)

        for claim_payload in _as_list(unit.get("claims"))[:6]:
            claim_text = _clean_claim_text(str(claim_payload.get("text", "")))
            if not _is_quality_claim(claim_text):
                continue
            claim = Claim(
                text=claim_text,
                confidence=_bounded_score(claim_payload.get("confidence"), default=0.72),
                task_id=task_id,
                source_ids=[source_id] if source_id else [],
            )
            claim.evidence_ids = [evidence.id]
            claim.fact_ids = list(unit_fact_ids)
            claims.append(claim)
            evidence.claim_ids.append(claim.id)

        if evidence.fact_ids or evidence.claim_ids:
            if not evidence.summary:
                evidence.summary = _summarize_evidence_unit(facts, claims, evidence)
            evidence_items.append(evidence)

    return facts, claims, evidence_items


def _iter_evidence_units(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return evidence units from the expected schema or build units from top-level facts/claims."""
    units = _as_list(payload.get("evidence_units") or payload.get("items") or payload.get("evidence"))
    if units:
        return [unit for unit in units if isinstance(unit, dict)]

    grouped: Dict[str, Dict[str, Any]] = {}
    for key in ("facts", "claims"):
        for item in _as_list(payload.get(key)):
            if not isinstance(item, dict):
                continue
            quote = str(item.get("quote") or item.get("evidence_quote") or item.get("source_quote") or "")
            bucket = grouped.setdefault(quote, {"quote": quote, "facts": [], "claims": []})
            bucket[key].append(item)
    return list(grouped.values())


def _as_list(value: Any) -> list[Any]:
    """Normalize model fields that may be omitted, null, a dict, or a list."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return [value]
    return []


def _resolve_quote(quote: str, passage_text: str, unit: Dict[str, Any]) -> str:
    """Ensure an evidence quote is grounded in the original passage."""
    quote = _normalize_text(quote)
    passage_text = _normalize_text(passage_text)
    if quote and quote in passage_text:
        return quote
    target_text = " ".join(
        [quote, str(unit.get("summary", ""))]
        + [str(item.get("text", "")) for item in _as_list(unit.get("facts")) if isinstance(item, dict)]
        + [str(item.get("text", "")) for item in _as_list(unit.get("claims")) if isinstance(item, dict)]
    )
    return _best_sentence(target_text, passage_text)


def _best_sentence(target_text: str, passage_text: str) -> str:
    """Pick the passage sentence with the strongest token overlap with the target text."""
    target_tokens = set(_tokenize(target_text))
    candidates = _split_sentences(passage_text)
    if not candidates:
        return passage_text[:500]
    scored = [
        (_jaccard_similarity(target_tokens, _tokenize(sentence)), sentence)
        for sentence in candidates
        if len(sentence) >= 30
    ]
    if not scored:
        return candidates[0][:500]
    return max(scored, key=lambda item: item[0])[1][:500]


def _bounded_score(value: Any, *, default: float) -> float:
    """Clamp confidence and quality scores to the schema range."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = default
    return max(0.0, min(1.0, score))


def _clean_fact_text(text: str) -> str:
    """Normalize a fact string into one concise sentence."""
    text = _normalize_text(text).strip("-• ")
    if text and text[-1] not in ".。":
        text += "。"
    return text


def _clean_claim_text(text: str) -> str:
    """Normalize a claim string into one concise sentence."""
    text = _normalize_text(text).strip("-• ")
    if text and text[-1] not in ".。":
        text += "。"
    return text


def _is_quality_fact(text: str, quote: str) -> bool:
    """Reject sentence copies, subjective statements, and non-atomic fact candidates."""
    if len(text) < 8 or len(text) > 180:
        return False
    if len(re.findall(r"[.。!?！？]", text)) > 1:
        return False
    lower = text.lower()
    subjective_markers = (
        "important", "significant", "best", "leading", "revolutionary", "likely", "may", "could",
        "重要", "显著", "最好", "领先", "可能", "预计", "有望",
    )
    if any(marker in lower for marker in subjective_markers):
        return False
    quote_key = _normalize_key(quote)
    text_key = _normalize_key(text)
    if len(quote) > 220 and text_key and text_key == quote_key:
        return False
    return True


def _is_quality_claim(text: str) -> bool:
    """Reject empty or overlong claim candidates before storing them."""
    return 8 <= len(text) <= 240 and len(re.findall(r"[.。!?！？]", text)) <= 2


def _summarize_evidence_unit(facts: List[AtomicFact], claims: List[Claim], evidence: Evidence) -> str:
    """Create a short evidence summary when the model omitted one."""
    linked_fact_ids = set(evidence.fact_ids)
    linked_claim_ids = set(evidence.claim_ids)
    fact_text = next((fact.text for fact in facts if fact.id in linked_fact_ids), "")
    claim_text = next((claim.text for claim in claims if claim.id in linked_claim_ids), "")
    return (fact_text or claim_text or evidence.quote[:160])[:240]


def _distill_locally(
    clean_passages: List[Dict[str, Any]],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> tuple[List[AtomicFact], List[Claim], List[Evidence]]:
    """Deterministic evidence-first fallback used for mock tests and LLM failures."""
    observer = get_observer()
    facts: list[AtomicFact] = []
    claims: list[Claim] = []
    evidence_items: list[Evidence] = []
    seen_units: set[str] = set()

    for passage in clean_passages:
        for sentence in _split_sentences(passage.get("text", "")):
            if not _is_informative_sentence(sentence):
                continue
            unit_key = _normalize_key(sentence)
            if unit_key in seen_units:
                continue
            seen_units.add(unit_key)
            local_fact_texts = _local_facts_from_sentence(sentence, passage)
            local_claim_texts = _local_claims_from_sentence(sentence, passage)
            if local_fact_texts and not local_claim_texts:
                local_claim_texts = [f"来源表明：{local_fact_texts[0].rstrip('。.')}。"]
            if not local_fact_texts and not local_claim_texts:
                continue

            evidence = Evidence(
                source_id=str(passage.get("source_id", "")),
                source_url=str(passage.get("url", "")),
                quote=sentence[:500],
                summary=(local_fact_texts[0] if local_fact_texts else local_claim_texts[0])[:240],
                quality_score=_sentence_confidence(sentence, str(passage.get("source_id", ""))),
                task_id=task_id,
            )
            unit_fact_ids: list[str] = []
            for fact_text in local_fact_texts[:5]:
                if not _is_quality_fact(fact_text, sentence):
                    continue
                fact = AtomicFact(
                    text=fact_text,
                    source_id=str(passage.get("source_id", "")),
                    source_url=str(passage.get("url", "")),
                    confidence=_sentence_confidence(sentence, str(passage.get("source_id", ""))),
                    task_id=task_id,
                    snippet=sentence[:280],
                    verified_count=1,
                    confidence_reason="local_evidence_first_fact",
                )
                facts.append(fact)
                unit_fact_ids.append(fact.id)
                evidence.fact_ids.append(fact.id)
                if run_context is not None:
                    observer.record_task_event(
                        run_context,
                        EventType.FACT_EXTRACTED,
                        task_id or "unknown-task",
                        message="Distiller extracted atomic fact.",
                        payload={"fact_id": fact.id, "source_id": passage.get("source_id")},
                    )

            for claim_text in local_claim_texts[:4]:
                if not _is_quality_claim(claim_text):
                    continue
                claim = Claim(
                    text=claim_text,
                    confidence=_sentence_confidence(sentence, str(passage.get("source_id", ""))),
                    task_id=task_id,
                    source_ids=[str(passage.get("source_id", ""))] if passage.get("source_id") else [],
                )
                claim.evidence_ids = [evidence.id]
                claim.fact_ids = list(unit_fact_ids)
                claims.append(claim)
                evidence.claim_ids.append(claim.id)
                if run_context is not None:
                    observer.record_task_event(
                        run_context,
                        EventType.CLAIM_EXTRACTED,
                        task_id or "unknown-task",
                        message="Distiller extracted claim.",
                        payload={"claim_id": claim.id, "source_id": passage.get("source_id"), "passage_id": passage.get("passage_id")},
                    )

            if evidence.fact_ids or evidence.claim_ids:
                evidence_items.append(evidence)
            if len(evidence_items) >= 16:
                break
        if len(evidence_items) >= 16:
            break
    return facts, claims, evidence_items


def _is_informative_sentence(sentence: str) -> bool:
    """Score whether a sentence is likely to contain extractable knowledge."""
    if len(sentence) < 35:
        return False
    lower = sentence.lower()
    return bool(
        re.search(r"\d", sentence)
        or re.search(r"\b(is|are|was|were|has|have|uses|relies|defined|introduced|proposed|reported|stated|argued|said|because|enables)\b", lower)
        or any(term in lower for term in ("transformer", "attention", "architecture", "llm", "large language model"))
    )


def _local_facts_from_sentence(sentence: str, passage: Dict[str, Any]) -> List[str]:
    """Extract conservative objective facts from one sentence without copying whole passages."""
    lower = sentence.lower()
    facts: list[str] = []

    if "transformer" in lower:
        if "architecture" in lower or "model architecture" in lower or "neural network" in lower:
            facts.append("Transformer 是一种神经网络架构。")
        if "attention" in lower:
            facts.append("Transformer 架构以注意力机制为核心。")
        if "recurrence" in lower or "recurrent" in lower or "rnn" in lower:
            facts.append("Transformer 不依赖循环网络作为主要序列建模机制。")
        if "attention is all you need" in lower or "introduced" in lower or "proposed" in lower:
            facts.append("Transformer 最初由论文 Attention Is All You Need 提出。")

    market_match = re.search(r"the ([a-z0-9 \-]+ market) (?:reached|was) ([^.]+?)(?:\.|$)", sentence, re.I)
    if market_match:
        facts.append(f"{market_match.group(1).strip()} was {market_match.group(2).strip()}.".capitalize())

    defined_match = re.search(r"(?:defined|defines) ([a-z0-9 \-]+) (?:mainly )?as ([^.]+)", sentence, re.I)
    if defined_match:
        facts.append(f"{defined_match.group(1).strip()} is defined as {defined_match.group(2).strip()}.")

    if not facts and re.search(r"\b(is|are|was|were|has|uses|relies on|consists of)\b", lower):
        facts.append(sentence[:180].rstrip(". ") + ".")
    return _unique_ids([_clean_fact_text(fact) for fact in facts])


def _local_claims_from_sentence(sentence: str, passage: Dict[str, Any]) -> List[str]:
    """Extract source-attributed claims from explanatory or evaluative sentences."""
    lower = sentence.lower()
    claims: list[str] = []
    metric_claim_added = False
    if re.search(r"\bmarket\b", lower) and re.search(r"\b(reached|was)\b", lower) and re.search(r"\d", sentence):
        claims.append(f"来源称：{sentence[:190].rstrip('. ')}。")
        metric_claim_added = True
    if "transformer" in lower and ("parallel" in lower or "parallelization" in lower):
        claims.append("来源认为 Transformer 的注意力机制使模型更容易并行训练。")
    if "transformer" in lower and ("large language model" in lower or "llm" in lower or "foundation" in lower):
        claims.append("来源认为 Transformer 架构是现代大语言模型的基础之一。")
    if not metric_claim_added and re.search(r"\b(reported|stated|argued|said|according|because|enables|allows)\b", lower):
        claims.append(f"来源认为：{sentence[:190].rstrip('. ')}。")
    return _unique_ids([_clean_claim_text(claim) for claim in claims])


def _normalize_text(text: str) -> str:
    """规范化文本中的空白字符并去除首尾空格。"""
    return re.sub(r"\s+", " ", text or "").strip()


def _normalize_key(text: str) -> str:
    """将文本归一化为适合去重和比对的键。"""
    return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", (text or "").lower()).strip()


def _tokenize(text: str) -> list[str]:
    """将文本切分为用于相似度计算的词项列表。"""
    return re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())


def _jaccard_similarity(left: Iterable[str], right: Iterable[str]) -> float:
    """计算两个词项集合的 Jaccard 相似度。"""
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _unique_ids(values: List[str]) -> List[str]:
    """按出现顺序去重 ID 列表。"""
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
    """清洗 passage 文本，去除样板内容、短文本和重复项。"""
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
    """将段落按句子边界拆分为句子列表。"""
    chunks = re.split(r"(?<=[.!?銆傦紒锛燂紱;])\s+", text)
    return [_normalize_text(chunk) for chunk in chunks if _normalize_text(chunk)]


def _sentence_confidence(sentence: str, source_id: str) -> float:
    """根据句子特征和来源信息估算单句置信度。"""
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
    """从清洗后的段落中提取可陈述的 claim。"""
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
                source_ids=[str(passage.get("source_id", ""))] if passage.get("source_id") else [],
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
    """将 claim 转换为更原子的事实记录。"""
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
            source_id=source_id,
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
    """为每条 claim 匹配对应证据，并生成 Evidence 记录。"""
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
    """Link claims and evidence without overwriting links produced by the LLM distiller."""
    evidence_iter = iter(evidence)
    evidence_by_id = {item.id: item for item in evidence}
    for claim in claims:
        existing_ids = [evidence_id for evidence_id in claim.evidence_ids if evidence_id in evidence_by_id]
        if existing_ids:
            claim.evidence_ids = _unique_ids(existing_ids)
            for evidence_id in claim.evidence_ids:
                if claim.id not in evidence_by_id[evidence_id].claim_ids:
                    evidence_by_id[evidence_id].claim_ids.append(claim.id)
            continue
        try:
            evidence_item = next(evidence_iter)
        except StopIteration:
            break
        claim.evidence_ids = [evidence_item.id]
        if claim.id not in evidence_item.claim_ids:
            evidence_item.claim_ids.append(claim.id)


def _compress_atomic_facts(
    facts: List[AtomicFact],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> tuple[List[AtomicFact], str]:
    """合并高相似度事实，压缩重复的原子事实集合。"""
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
    """Refresh claim-to-fact links after fact compression while preserving exact LLM links."""
    compressed_ids = {fact.id for fact in compressed_facts}
    for claim in claims:
        preserved_ids = [fact_id for fact_id in claim.fact_ids if fact_id in compressed_ids]
        matched_ids = [
            fact.id for fact in compressed_facts
            if _jaccard_similarity(_tokenize(claim.text), _tokenize(fact.text)) >= 0.7
        ]
        claim.fact_ids = _unique_ids(preserved_ids + matched_ids)[:5]


def _link_facts_and_evidence(facts: List[AtomicFact], evidence: List[Evidence]) -> None:
    """Ensure evidence records point at compressed fact IDs after deduplication."""
    fact_by_id = {fact.id: fact for fact in facts}
    for item in evidence:
        preserved_ids = [fact_id for fact_id in item.fact_ids if fact_id in fact_by_id]
        matched_ids = [
            fact.id for fact in facts
            if fact.snippet and item.quote and _jaccard_similarity(_tokenize(fact.snippet), _tokenize(item.quote)) >= 0.5
        ]
        item.fact_ids = _unique_ids(preserved_ids + matched_ids)[:6]


def _claim_overlap(left: Claim, right: Claim) -> float:
    """计算两条 claim 在文本层面的重叠程度。"""
    return _jaccard_similarity(_tokenize(left.text), _tokenize(right.text))


def _extract_numbers(text: str) -> List[str]:
    """提取文本中的数字片段。"""
    return re.findall(r"\d+(?:\.\d+)?", text)


def _extract_years(text: str) -> List[str]:
    """提取文本中的年份信息。"""
    return re.findall(r"\b(?:19|20)\d{2}\b", text)


def _detect_conflicts(
    claims: List[Claim],
    evidence: List[Evidence],
    *,
    run_context: Any = None,
    task_id: Optional[str] = None,
) -> List[ConflictRecord]:
    """识别 claim 之间的数值、时间或语义冲突。"""
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
    """按报告小节聚合相关 claim、evidence、fact 和 conflict。"""
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
        if not selected_claims and claims:
            section_hint = f"{title} {goal_text}".lower()
            conflict_claim_ids = {claim_id for conflict in conflicts for claim_id in conflict.claim_ids}
            if any(term in section_hint for term in ("conflict", "gap", "uncertain", "risk")) and conflict_claim_ids:
                selected_claims = [claim for claim in claims if claim.id in conflict_claim_ids][:6]
            elif any(term in section_hint for term in ("key", "finding", "synthesize", "conclusion", "supports")):
                selected_claims = sorted(
                    claims,
                    key=lambda claim: (
                        float(claim.confidence or 0.0),
                        len(claim.evidence_ids or []),
                        len(claim.fact_ids or []),
                    ),
                    reverse=True,
                )[:6]
            elif len(claims) <= 6:
                selected_claims = list(claims)
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
        selected_claim_ids = {claim.id for claim in selected_claims}
        for claim in selected_claims:
            if not claim.section_id:
                claim.section_id = section_id
        for item in evidence:
            if item.id in selected_evidence_ids and not item.section_id:
                item.section_id = section_id
        for fact in facts:
            if fact.id in selected_fact_ids and not fact.section_id:
                fact.section_id = section_id
        for conflict in conflicts:
            if conflict.id in selected_conflict_ids and not conflict.section_id:
                conflict.section_id = section_id
            if set(conflict.claim_ids) & selected_claim_ids and not conflict.section_id:
                conflict.section_id = section_id
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
    """根据小节覆盖情况和后续提示生成未解决缺口列表。"""
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
    """汇总小节覆盖度、证据密度和可写作充分性指标。"""
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
        if section_status
        else 0.0
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
