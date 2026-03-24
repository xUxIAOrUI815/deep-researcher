import asyncio
import json
import os
import httpx
from typing import TypedDict, Annotated, Sequence
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import aiosqlite

from schemas.state import TaskNode, AtomicFact, ResearchState, TokenUsage, compute_priority, ScrapedData, SearchResult
from providers import MCPGateway, SmartScraper


DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_BASE = "https://api.deepseek.com"


class GraphState(TypedDict):
    task_tree: dict
    fact_pool: list
    atomic_facts: list
    token_usage: dict
    current_focus: str | None
    root_task_id: str | None
    completed_tasks: list
    failed_tasks: list
    messages: list
    raw_scraped_data: list
    search_results: list
    final_report: str | None


def planner(state: GraphState) -> GraphState:
    print(f"[PLANNER] Initializing research plan...")
    print(f"[PLANNER] Current task tree: {len(state['task_tree'])} tasks")

    if not state["root_task_id"]:
        root_task = TaskNode(
            query="Initial research query",
            status="completed",
            depth=0,
            priority=1.0
        )
        state["task_tree"][root_task.id] = root_task.model_dump()
        state["root_task_id"] = root_task.id
        print(f"[PLANNER] Created root task: {root_task.id}")

    state["token_usage"]["planning_tokens"] += 100
    print(f"[PLANNER] Planning complete. Token usage: {state['token_usage']['planning_tokens']}")
    return state


async def researcher_async(state: GraphState) -> GraphState:
    print(f"[RESEARCHER] Starting research phase...")
    print(f"[RESEARCHER] Current focus: {state.get('current_focus', 'None')}")

    pending_tasks = [
        task_id for task_id, task in state["task_tree"].items()
        if task.get("status") == "pending"
    ]

    if pending_tasks:
        task_id = pending_tasks[0]
        state["task_tree"][task_id]["status"] = "running"
        state["current_focus"] = task_id
        print(f"[RESEARCHER] Researching task: {task_id}")

        query = state["task_tree"][task_id].get("query", "unknown")

        gateway = MCPGateway()
        scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

        print(f"[RESEARCHER] Searching for: {query}")
        search_results = await gateway.search(query, max_results=5, provider="tavily")
        state["search_results"] = [r.model_dump() for r in search_results]
        print(f"[RESEARCHER] Found {len(search_results)} search results")

        if search_results:
            urls = [r.url for r in search_results[:3]]
            print(f"[RESEARCHER] Scraping {len(urls)} URLs...")

            scraped_data = await scraper.scrape_batch(urls, force_playwright=False)
            state["raw_scraped_data"] = [s.model_dump() for s in scraped_data]

            for scraped in scraped_data:
                if scraped.markdown and not scraped.error:
                    state["fact_pool"].append(scraped.markdown[:500])

            print(f"[RESEARCHER] Scraped {len(scraped_data)} pages successfully")

        await scraper.close()

        state["task_tree"][task_id]["status"] = "completed"
        state["completed_tasks"].append(task_id)
        print(f"[RESEARCHER] Completed task: {task_id}")
    else:
        print(f"[RESEARCHER] No pending tasks found")

    state["token_usage"]["research_tokens"] += 500
    print(f"[RESEARCHER] Research complete. Token usage: {state['token_usage']['research_tokens']}")
    return state


def researcher(state: GraphState) -> GraphState:
    return asyncio.get_event_loop().run_until_complete(researcher_async(state))


def distiller(state: GraphState) -> GraphState:
    print(f"[DISTILLER] Distilling facts to atomic units...")

    for fact_text in state["fact_pool"][:3]:
        atomic_fact = AtomicFact(
            text=fact_text,
            source_url="https://example.com/mock-source",
            confidence=0.85,
            task_id=state.get("current_focus")
        )
        state["atomic_facts"].append(atomic_fact.model_dump())

    state["fact_pool"] = state["fact_pool"][3:]
    state["token_usage"]["distillation_tokens"] += 200
    print(f"[DISTILLER] Distillation complete. Atomic facts: {len(state['atomic_facts'])}")
    return state


async def writer_async(state: GraphState) -> GraphState:
    print(f"[WRITER] Synthesizing report with citations...")

    if not state["atomic_facts"]:
        print(f"[WRITER] No atomic facts available, skipping report generation")
        state["final_report"] = "No facts available for report generation."
        return state

    facts = state["atomic_facts"]
    print(f"[WRITER] Processing {len(facts)} atomic facts...")

    facts_text = []
    for i, fact in enumerate(facts, 1):
        fact_id = fact.get("id", f"fact_{i}")
        text = fact.get("text", "")
        source = fact.get("source_url", "")
        confidence = fact.get("confidence", 0.0)
        facts_text.append(f"[{fact_id}] {text} (来源: {source}, 置信度: {confidence:.2f})")

    facts_context = "\n\n".join(facts_text)

    prompt = f"""你是一位专业的行业分析师。请根据以下原子事实撰写一份结构清晰的调研简报。

## 写作要求

1. **结构顺序**：按以下逻辑顺序组织报告：
   - 背景（Context）：行业或主题的背景信息
   - 现状（Current State）：当前的主要进展、数据或事实
   - 挑战（Challenges）：面临的主要问题或障碍
   - 结论（Conclusion）：基于证据的总结和展望

2. **引用规范**：每个引用必须标注原始来源，格式为 [Fact_ID](Source_URL)
   - 例如：据台积电公告显示，其2nm芯片将在2025年量产 [fact_1](https://example.com)

3. **语言**：使用专业、简洁的中文

## 原子事实列表

{facts_context}

## 输出格式

请直接输出Markdown格式的调研简报，不要有其他说明文字。"""

    if not DEEPSEEK_API_KEY:
        print(f"[WRITER] DEEPSEEK_API_KEY not set, using mock report")
        mock_report = "# 调研简报（Mock）\n\n"
        for fact in facts[:5]:
            mock_report += f"\n- {fact.get('text', '')} [Source]({fact.get('source_url', '')})"
        state["final_report"] = mock_report
        state["token_usage"]["writing_tokens"] += 300
        return state

    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{DEEPSEEK_API_BASE}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [
                            {"role": "system", "content": "你是一位专业的行业分析师，擅长从事实出发撰写结构清晰、引用准确的调研简报。"},
                            {"role": "user", "content": prompt}
                        ],
                        "max_tokens": 4000,
                        "temperature": 0.3
                    }
                )

                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    print(f"[WRITER] Rate limited, waiting {wait_time}s...")
                    await asyncio.sleep(wait_time)
                    continue

                response.raise_for_status()
                data = response.json()

                report = data.get("choices", [{}])[0].get("message", {}).get("content", "")

                if report:
                    state["final_report"] = report
                    usage = data.get("usage", {})
                    tokens_used = usage.get("total_tokens", 0)
                    state["token_usage"]["writing_tokens"] += tokens_used
                    print(f"[WRITER] Report generated ({tokens_used} tokens)")
                    print(f"[WRITER] Report preview: {report[:200]}...")
                    return state
                else:
                    raise Exception("Empty response from DeepSeek API")

        except Exception as e:
            if attempt == 2:
                print(f"[WRITER] Failed after {3} attempts: {e}")
                state["final_report"] = f"Report generation failed: {e}"
                return state
            print(f"[WRITER] Attempt {attempt + 1} failed: {e}, retrying...")
            await asyncio.sleep(2 ** attempt)

    return state


def writer(state: GraphState) -> GraphState:
    return asyncio.get_event_loop().run_until_complete(writer_async(state))


def should_continue(state: GraphState) -> str:
    pending = [t for t, data in state["task_tree"].items() if data.get("status") == "pending"]
    if pending:
        return "researcher"
    return "end"


def create_research_graph(checkpointer: AsyncSqliteSaver) -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_async)
    workflow.add_node("distiller", distiller)
    workflow.add_node("writer", writer_async)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "distiller")
    workflow.add_edge("distiller", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=checkpointer)


async def init_sqlite_saver(db_path: str = "research.db") -> AsyncSqliteSaver:
    conn = await aiosqlite.connect(db_path)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA busy_timeout=5000")
    await conn.commit()

    saver = AsyncSqliteSaver(conn)

    return saver


async def run_research_cycle(initial_query: str = "Mock research query") -> dict:
    saver = await init_sqlite_saver()
    graph = create_research_graph(saver)

    initial_state = {
        "task_tree": {},
        "fact_pool": [],
        "atomic_facts": [],
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0
        },
        "current_focus": None,
        "root_task_id": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": [],
        "raw_scraped_data": [],
        "search_results": [],
        "final_report": None
    }

    config = {"configurable": {"thread_id": "main-research-thread"}}

    result = await graph.ainvoke(initial_state, config)

    print("\n" + "="*60)
    print("RESEARCH CYCLE COMPLETE")
    print("="*60)
    print(f"Tasks completed: {len(result.get('completed_tasks', []))}")
    print(f"Atomic facts extracted: {len(result.get('atomic_facts', []))}")
    print(f"Total tokens used: {sum(result.get('token_usage', {}).values())}")
    print(f"Search results: {len(result.get('search_results', []))}")
    print(f"Scraped pages: {len(result.get('raw_scraped_data', []))}")
    print(f"Final report: {result.get('final_report', 'N/A')[:200] if result.get('final_report') else 'N/A'}...")

    await saver.conn.close()
    return result
