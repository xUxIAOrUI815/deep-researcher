from __future__ import annotations

import hashlib
import math
import os
import re
from typing import Any, Dict, Iterable, List, Optional

import httpx

from core.context_builders import ResearcherContextBuilder
from core.observability import EventType, get_observer
from core.session_retrieval import SessionRetrievalService
from providers import MCPGateway, MCPGatewayError, build_scraper, resolve_scraper_mode
from schemas.state import ResearcherOutputs, ScrapedData, SearchResult


MAX_QUERY_PER_TASK = 5
MAX_SEARCH_ITERATIONS = 3
MAX_SOURCES_PER_QUERY = 5
MIN_RELEVANCE_SCORE = 6
DUPLICATE_QUERY_THRESHOLD = 0.85
MIN_PASSAGE_TEXT_LENGTH = 80
MIN_SCRAPED_TEXT_LENGTH = 200


def _normalize_text(text: str) -> str:
    """规范化文本中的空白字符并去除首尾空格。"""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def _tokenize(text: str) -> list[str]:
    """将文本切分为英文单词或中文词块，供匹配和去重使用。"""
    return re.findall(r"[\w\u4e00-\u9fff]+", (text or "").lower())


def _unique_preserve_order(values: Iterable[str]) -> list[str]:
    """在保留原始顺序的前提下对字符串序列去重。"""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = _normalize_text(value)
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        result.append(normalized)
    return result


def _source_id(url: str) -> str:
    """根据 URL 生成稳定的来源 ID。"""
    return f"src_{hashlib.sha1(url.encode('utf-8')).hexdigest()[:16]}"


def _passage_id(task_id: Optional[str], url: str, index: int) -> str:
    """根据任务、URL 和序号生成稳定的段落 ID。"""
    base = f"{task_id or 'task'}:{url}:{index}"
    return f"passage_{hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]}"


def _query_embedding(text: str, dimensions: int = 64) -> list[float]:
    """生成确定性的本地向量表示，用于查询准入阶段的去重。"""
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    """计算两个向量的余弦相似度。"""
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


def _heuristic_relevance_score(query: str, root_user_query: str) -> int:
    """使用词项重叠启发式估算候选查询与根问题的相关度。"""
    query_tokens = set(_tokenize(query))
    root_tokens = set(_tokenize(root_user_query))
    if not query_tokens or not root_tokens:
        return 6 if query_tokens else 1
    overlap = len(query_tokens & root_tokens) / max(1, len(root_tokens))
    containment = len(query_tokens & root_tokens) / max(1, len(query_tokens))
    score = 3 + int(round(4 * overlap + 3 * containment))
    return max(1, min(10, score))


async def _score_query_relevance(query: str, root_user_query: str) -> tuple[int, str]:
    """为候选查询打分；启用 LLM 时走模型评分，否则回退到启发式规则。"""
    prompt = f"""Score the relevance between the candidate retrieval query and the root research question.

Root research question:
{root_user_query}

Candidate retrieval query:
{query}

Return only one integer from 1 to 10:
- 10 = directly relevant and necessary
- 6 = acceptable for exploratory retrieval
- 1 = unrelated
"""
    use_llm = os.getenv("RESEARCHER_USE_LLM_SCORING", "").lower() in {"1", "true", "yes"}
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if use_llm and api_key:
        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                response = await client.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.0,
                        "max_tokens": 5,
                    },
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                match = re.search(r"\d+", content)
                if match:
                    return max(1, min(10, int(match.group(0)))), "llm"
        except Exception:
            pass
    return _heuristic_relevance_score(query, root_user_query), "heuristic"


def _generate_candidate_queries(task: Dict[str, Any]) -> list[str]:
    """根据任务内容和类型生成一组候选检索查询。"""
    query = _normalize_text(str(task.get("query", "")))
    title = _normalize_text(str(task.get("title", "")))
    rationale = _normalize_text(str(task.get("rationale", "")))
    node_type = str(task.get("node_type", "") or "").lower()

    rationale_terms = " ".join(_tokenize(rationale)[:10])
    candidates = [
        query,
        f"{title} {query}",
        f"{query} {rationale_terms}",
    ]
    if "conflict" in node_type:
        candidates.append(f"{query} conflicting evidence verification")
    elif "verification" in node_type:
        candidates.append(f"{query} verification authoritative source")
    elif "source" in node_type:
        candidates.append(f"{query} authoritative sources primary data")
    elif "gap" in node_type:
        candidates.append(f"{query} missing evidence research gap")
    else:
        candidates.append(f"{query} current evidence")
        candidates.append(f"{query} latest reliable sources")
    return _unique_preserve_order(candidates)


def _collect_existing_source_keys(
    task: Dict[str, Any],
    knowledge_refs: Optional[Dict[str, Any]],
    session_snapshot: Optional[Dict[str, Any]] = None,
    research_context: Optional[Dict[str, Any]] = None,
) -> set[str]:
    """汇总已使用来源的 ID 和 URL 键，避免重复抓取。"""
    keys: set[str] = set()
    for value in task.get("source_span_ids", []) or []:
        keys.add(str(value).lower())
    source_ids = list((knowledge_refs or {}).get("source_ids", []))
    if research_context:
        source_ids = list(research_context.get("already_seen_source_ids", [])) + source_ids
    if session_snapshot:
        source_ids = list(session_snapshot.get("knowledge_refs", {}).get("source_ids", [])) + source_ids
    for value in source_ids or []:
        keys.add(str(value).lower())
    for value in (research_context or {}).get("already_seen_source_urls", []) or []:
        keys.add(str(value).lower().rstrip("/"))
    return keys


def _research_context_from_snapshot(
    *,
    knowledge_manager: Any,
    research_id: str,
    session_id: str,
    root_user_query: str,
    task_id: Optional[str],
    task: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """从持久化会话数据构建 researcher 上下文，失败时返回 None。"""
    try:
        builder = ResearcherContextBuilder(SessionRetrievalService(knowledge_manager))
        return builder.build(
            research_id=research_id,
            session_id=session_id,
            root_user_query=root_user_query,
            task_id=task_id,
            task=task,
        ).model_dump()
    except Exception:
        return None


def _result_to_source(result: SearchResult, *, query: str, task_id: Optional[str]) -> dict[str, Any]:
    """将搜索结果转换为统一的来源记录结构。"""
    source_id = _source_id(result.url)
    return {
        "source_id": source_id,
        "task_id": task_id,
        "query": query,
        "url": result.url,
        "title": result.title,
        "snippet": result.snippet,
        "score": result.score,
        "status": "candidate",
    }


def _make_passage(
    *,
    task_id: Optional[str],
    source_id: str,
    url: str,
    title: str,
    text: str,
    query: str,
    index: int,
    extraction_method: str,
) -> dict[str, Any]:
    """把抓取或检索得到的文本整理为 passage 结构。"""
    return {
        "passage_id": _passage_id(task_id, url, index),
        "task_id": task_id,
        "source_id": source_id,
        "url": url,
        "title": title,
        "text": text[:4000],
        "query": query,
        "extraction_method": extraction_method,
    }


def _emit_task_event(run_context: Any, event_type: EventType, task_id: Optional[str], message: str, payload: dict[str, Any]) -> None:
    """在存在运行上下文时记录 researcher 相关事件。"""
    if run_context is None:
        return
    get_observer().record_task_event(
        run_context,
        event_type,
        task_id or "unknown-task",
        message=message,
        payload=payload,
    )


def _resolve_search_mode(mode: Optional[str], scraper_mode: str) -> str:
    """解析最终搜索模式，并与抓取器模式保持兼容。"""
    candidate = (mode or os.getenv("RESEARCHER_SEARCH_MODE", "")).strip().lower()
    if not candidate:
        return "mock" if scraper_mode == "mock" else "live"
    return candidate if candidate in {"live", "mock"} else "live"


def _format_exception(exc: Exception) -> str:
    message = str(exc).strip()
    if message:
        return f"{type(exc).__name__}: {message}"
    return f"{type(exc).__name__}: {exc!r}"


def _is_recoverable_search_error(exc: Exception) -> bool:
    if isinstance(exc, MCPGatewayError):
        return bool(exc.retryable)
    error_text = _format_exception(exc)
    fatal_markers = (
        "TAVILY_API_KEY",
        "HTTP error: 400",
        "HTTP error: 401",
        "HTTP error: 403",
        "Illegal header value",
    )
    if any(marker in error_text for marker in fatal_markers):
        return False
    recoverable_markers = (
        "ReadTimeout",
        "ConnectTimeout",
        "ConnectError",
        "RemoteProtocolError",
        "PoolTimeout",
        "HTTP error: 429",
        "HTTP error: 500",
        "HTTP error: 502",
        "HTTP error: 503",
        "HTTP error: 504",
    )
    return any(marker in error_text for marker in recoverable_markers)


def _mock_search(query: str, max_results: int = 5) -> list[SearchResult]:
    """生成稳定可复现的 mock 搜索结果，用于离线测试。"""
    base = hashlib.sha1(query.encode("utf-8")).hexdigest()[:8]
    results: list[SearchResult] = []
    for index in range(max_results):
        score = max(0.1, 0.95 - (index * 0.08))
        results.append(
            SearchResult(
                url=f"https://mock.local/{base}/doc-{index + 1}",
                title=f"Mock source {index + 1} for {query}",
                snippet=(
                    f"Deterministic offline source {index + 1} about {query}. "
                    f"It includes stable benchmark values and explanatory context for integration testing."
                ),
                score=score,
            )
        )
    return results


async def run_researcher(
    *,
    task_id: Optional[str],
    task: Optional[Dict[str, Any]],
    root_user_query: str = "",
    knowledge_refs: Optional[Dict[str, Any]] = None,
    knowledge_manager: Any = None,
    research_id: Optional[str] = None,
    session_id: Optional[str] = None,
    research_context: Optional[Dict[str, Any]] = None,
    run_context: Any = None,
    max_query_per_task: int = MAX_QUERY_PER_TASK,
    max_search_iterations: int = MAX_SEARCH_ITERATIONS,
    max_sources_per_query: int = MAX_SOURCES_PER_QUERY,
    enable_scraping: bool = True,
    scraper_mode: Optional[str] = None,
    search_mode: Optional[str] = None,
    scraper_backend: Any = None,
    search_gateway: Optional[MCPGateway] = None,
    scraper_fixtures: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ResearcherOutputs:
    """执行检索探索流程，并返回供 distiller 使用的原始材料。"""
    if not task_id or not task:
        return ResearcherOutputs(
            task_id=task_id,
            metadata={"stop_reason": "missing_active_task"},
            summary="No active task was provided to researcher.",
        )

    root_query = _normalize_text(root_user_query or getattr(run_context, "root_query", ""))
    task_query = _normalize_text(str(task.get("query", "")))
    if not root_query:
        root_query = task_query

    metadata: dict[str, Any] = {
        "constraint_domain": "retrieval_exploration",
        "limits": {
            "max_query_per_task": max_query_per_task,
            "max_search_iterations": max_search_iterations,
            "max_sources_per_query": max_sources_per_query,
        },
        "scraper_mode": "",
        "search_mode": "",
        "rejected_queries": [],
        "duplicate_queries": [],
        "rejected_sources": [],
        "search_errors": [],
        "follow_up_hints": [],
        "scoring_methods": [],
        "iterations": 0,
        "stop_reason": "",
    }

    generated_queries = _generate_candidate_queries(task)
    admitted_queries: list[dict[str, Any]] = []
    accepted_embeddings: list[tuple[str, list[float]]] = []

    for query in generated_queries:
        _emit_task_event(
            run_context,
            EventType.QUERY_GENERATED,
            task_id,
            "Researcher generated candidate query.",
            {"query": query},
        )
        score, scoring_method = await _score_query_relevance(query, root_query)
        metadata["scoring_methods"].append(scoring_method)
        if score < MIN_RELEVANCE_SCORE:
            rejection = {"query": query, "score": score, "reason": "low_relevance"}
            metadata["rejected_queries"].append(rejection)
            _emit_task_event(
                run_context,
                EventType.QUERY_REJECTED,
                task_id,
                "Researcher rejected query below relevance threshold.",
                rejection,
            )
            continue

        embedding = _query_embedding(query)
        duplicate_of: Optional[str] = None
        for accepted_query, accepted_embedding in accepted_embeddings:
            if _cosine_similarity(embedding, accepted_embedding) > DUPLICATE_QUERY_THRESHOLD:
                duplicate_of = accepted_query
                break
        if duplicate_of:
            duplicate = {"query": query, "duplicate_of": duplicate_of, "reason": "embedding_similarity"}
            metadata["duplicate_queries"].append(duplicate)
            _emit_task_event(
                run_context,
                EventType.QUERY_DROPPED,
                task_id,
                "Researcher dropped duplicate query.",
                duplicate,
            )
            continue

        admitted = {"query": query, "relevance_score": score, "scoring_method": scoring_method}
        admitted_queries.append(admitted)
        accepted_embeddings.append((query, embedding))
        if len(admitted_queries) >= max_query_per_task:
            break

    effective_scraper_mode = resolve_scraper_mode(scraper_mode)
    effective_search_mode = _resolve_search_mode(search_mode, effective_scraper_mode)
    scraper = scraper_backend or build_scraper(effective_scraper_mode, fixtures=scraper_fixtures)
    gateway = search_gateway or MCPGateway()
    metadata["scraper_mode"] = getattr(scraper, "mode", effective_scraper_mode)
    metadata["search_mode"] = effective_search_mode

    session_snapshot: Optional[Dict[str, Any]] = None
    if research_context is None and knowledge_manager is not None and research_id and session_id:
        research_context = _research_context_from_snapshot(
            knowledge_manager=knowledge_manager,
            research_id=research_id,
            session_id=session_id,
            root_user_query=root_query,
            task_id=task_id,
            task=task,
        )
    if research_context is None and knowledge_manager is not None and research_id and session_id:
        try:
            session_snapshot = knowledge_manager.get_session_snapshot(research_id, session_id)
        except Exception:
            session_snapshot = None

    used_source_keys = _collect_existing_source_keys(task, knowledge_refs, session_snapshot, research_context)
    if research_context:
        metadata["session_knowledge"] = {
            "fact_count": len(research_context.get("relevant_facts", [])),
            "evidence_count": len((session_snapshot or {}).get("evidence", [])),
            "source_count": len(research_context.get("already_seen_source_ids", [])),
            "gap_count": len(research_context.get("unresolved_gaps", [])),
            "context_source": "research_context",
        }
        metadata["follow_up_hints"] = [
            *(f"Investigate gap: {row.get('gap_text', '')}" for row in research_context.get("unresolved_gaps", [])[:3]),
            *(f"Seek {item}" for item in research_context.get("authority_gaps", [])[:2]),
        ]
    elif session_snapshot:
        metadata["session_knowledge"] = {
            "fact_count": len(session_snapshot.get("facts", [])),
            "evidence_count": len(session_snapshot.get("evidence", [])),
            "source_count": len(session_snapshot.get("knowledge_refs", {}).get("source_ids", [])),
        }
    seen_urls: set[str] = set()
    accepted_sources: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    search_results_cache: list[SearchResult] = []
    scraped_data_cache: list[ScrapedData] = []
    no_gain_iterations = 0

    for iteration, admitted in enumerate(admitted_queries[:max_search_iterations], start=1):
        query = admitted["query"]
        metadata["iterations"] = iteration
        before_sources = len(accepted_sources)
        before_passages = len(passages)

        if effective_search_mode == "mock":
            search_results = _mock_search(query, max_results=max_sources_per_query)
        else:
            try:
                search_results = await gateway.search(query, max_results=max_sources_per_query, provider="tavily")
            except Exception as exc:
                recoverable = _is_recoverable_search_error(exc)
                search_error = {
                    "query": query,
                    "provider": "tavily",
                    "error": _format_exception(exc),
                    "recoverable": recoverable,
                    "attempts": getattr(exc, "attempts", None),
                    "status_code": getattr(exc, "status_code", None),
                }
                metadata["search_errors"].append(search_error)
                _emit_task_event(
                    run_context,
                    EventType.QUERY_REJECTED,
                    task_id,
                    "Researcher skipped query after recoverable search provider error." if recoverable else "Researcher hit fatal search provider error.",
                    search_error,
                )
                if recoverable:
                    continue
                raise
        search_results_cache.extend(search_results)

        candidate_sources: list[dict[str, Any]] = []
        for result in search_results[:max_sources_per_query]:
            url_key = result.url.lower().rstrip("/")
            sid = _source_id(result.url)
            if not result.url or url_key in seen_urls or sid.lower() in used_source_keys or url_key in used_source_keys:
                rejection = {"url": result.url, "reason": "duplicate_or_already_used", "query": query}
                metadata["rejected_sources"].append(rejection)
                _emit_task_event(
                    run_context,
                    EventType.SOURCE_REJECTED,
                    task_id,
                    "Researcher rejected duplicate or already used source.",
                    rejection,
                )
                continue
            candidate_sources.append(_result_to_source(result, query=query, task_id=task_id))

        scraped_by_url: dict[str, ScrapedData] = {}
        if enable_scraping and candidate_sources:
            try:
                source_context = {source["url"]: dict(source) for source in candidate_sources}
                scraped_batch = await scraper.scrape_batch(
                    [source["url"] for source in candidate_sources],
                    source_context=source_context,
                )
                for scraped in scraped_batch:
                    scraped_data_cache.append(scraped)
                    scraped_by_url[scraped.url] = scraped
            except Exception as exc:
                metadata["scrape_error"] = str(exc)

        for source in candidate_sources:
            url = source["url"]
            scraped = scraped_by_url.get(url)
            text = ""
            extraction_method = "search_snippet"
            if scraped and not scraped.error:
                text = _normalize_text(scraped.markdown)
                extraction_method = scraped.fetch_method
            if not text:
                text = _normalize_text(source.get("snippet", ""))

            min_length = MIN_SCRAPED_TEXT_LENGTH if extraction_method != "search_snippet" else MIN_PASSAGE_TEXT_LENGTH
            if len(text) < min_length:
                rejection = {
                    "url": url,
                    "reason": "short_text",
                    "text_length": len(text),
                    "query": query,
                }
                metadata["rejected_sources"].append(rejection)
                _emit_task_event(
                    run_context,
                    EventType.SOURCE_REJECTED,
                    task_id,
                    "Researcher rejected source with too little text.",
                    rejection,
                )
                continue

            source["status"] = "accepted"
            source["text_length"] = len(text)
            source["extraction_method"] = extraction_method
            source["scraper_mode"] = getattr(scraper, "mode", effective_scraper_mode)
            accepted_sources.append(source)
            seen_urls.add(url.lower().rstrip("/"))
            used_source_keys.add(source["source_id"].lower())
            passages.append(
                _make_passage(
                    task_id=task_id,
                    source_id=source["source_id"],
                    url=url,
                    title=source.get("title", ""),
                    text=text,
                    query=query,
                    index=len(passages),
                    extraction_method=extraction_method,
                )
            )
            _emit_task_event(
                run_context,
                EventType.SOURCE_ACCEPTED,
                task_id,
                "Researcher accepted source.",
                {
                    "url": url,
                    "source_id": source["source_id"],
                    "query": query,
                    "text_length": len(text),
                    "scraper_mode": getattr(scraper, "mode", effective_scraper_mode),
                    "search_mode": effective_search_mode,
                },
            )

        if len(accepted_sources) == before_sources and len(passages) == before_passages:
            no_gain_iterations += 1
        else:
            no_gain_iterations = 0

        if no_gain_iterations >= 2:
            metadata["stop_reason"] = "marginal_gain_stop"
            _emit_task_event(
                run_context,
                EventType.EXPLORATION_STOPPED,
                task_id,
                "Researcher stopped exploration after two no-gain iterations.",
                {"iterations": iteration, "reason": metadata["stop_reason"]},
            )
            break

    if not metadata["stop_reason"]:
        if metadata["search_errors"] and not accepted_sources and not passages:
            metadata["stop_reason"] = "search_errors_exhausted"
            metadata["recoverable_failure"] = all(item.get("recoverable") for item in metadata["search_errors"])
        elif not admitted_queries:
            metadata["stop_reason"] = "no_admitted_queries"
        elif metadata["iterations"] >= max_search_iterations:
            metadata["stop_reason"] = "max_search_iterations_reached"
        else:
            metadata["stop_reason"] = "query_budget_exhausted"

    derived_hints = _derive_follow_up_hints(accepted_sources, task_query)
    metadata["follow_up_hints"] = list(metadata.get("follow_up_hints", [])) + derived_hints
    _emit_task_event(
        run_context,
        EventType.EXPLORATION_STOPPED,
        task_id,
        "Researcher completed exploration.",
        {
            "reason": metadata["stop_reason"],
            "queries": len(admitted_queries),
            "sources": len(accepted_sources),
            "passages": len(passages),
            "scraper_mode": getattr(scraper, "mode", effective_scraper_mode),
            "search_mode": effective_search_mode,
        },
    )

    return ResearcherOutputs(
        task_id=task_id,
        queries=admitted_queries,
        sources=accepted_sources,
        passages=passages,
        metadata=metadata,
        search_result_ids=[_source_id(result.url) for result in search_results_cache],
        source_ids=[source["source_id"] for source in accepted_sources],
        scraped_data_ids=[_source_id(scraped.url) for scraped in scraped_data_cache],
        search_results_cache=search_results_cache,
        scraped_data_cache=scraped_data_cache,
        summary=(
            f"Collected {len(accepted_sources)} sources and {len(passages)} passages "
            f"for task {task_id} using {len(admitted_queries)} admitted queries."
        ),
    )


def _derive_follow_up_hints(sources: list[dict[str, Any]], task_query: str) -> list[str]:
    """从已接受来源标题中提取可能的后续检索提示词。"""
    hints: list[str] = []
    task_tokens = set(_tokenize(task_query))
    for source in sources:
        title_tokens = set(_tokenize(source.get("title", "")))
        novel_tokens = [token for token in title_tokens - task_tokens if len(token) > 3]
        if novel_tokens:
            hints.append(" ".join(novel_tokens[:5]))
    return _unique_preserve_order(hints)[:5]
