# 事实生成流程问题分析报告

> 分析日期: 2026-04-05  
> 分析范围: 事实生成全流程代码审查

---

## 目录

1. [问题总览](#1-问题总览)
2. [截断问题详解](#2-截断问题详解)
3. [DistillerAgent集成方案](#3-distilleragent集成方案)
4. [固定值问题报告](#4-固定值问题报告)
5. [修改方案汇总](#5-修改方案汇总)

---

## 1. 问题总览

### 1.1 发现的问题清单

| 编号 | 问题 | 位置 | 严重程度 |
|-----|------|------|---------|
| 1 | researcher截断markdown至500字符 | graph.py:90 | 🔴 严重 |
| 2 | distiller使用Mock实现，未使用DistillerAgent | graph.py:111-126 | 🔴 严重 |
| 3 | DistillerAgent截断markdown至8000字符 | distiller.py:141 | 🟡 中等 |
| 4 | source_url使用固定Mock值 | graph.py:117 | 🔴 严重 |
| 5 | confidence使用固定值0.85 | graph.py:118 | 🟡 中等 |
| 6 | distiller只处理前3条fact_pool | graph.py:114 | 🟡 中等 |

### 1.2 问题影响分析

```
┌─────────────────────────────────────────────────────────────────────┐
│                        问题影响链                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  原始网页内容 (完整)                                                 │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ researcher: scraped.markdown[:500]      │ ← 问题1: 截断至500字符  │
│  │ 丢失: 500字符之后的所有内容              │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ fact_pool: ["截断片段1", "截断片段2"...] │                        │
│  └─────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ distiller: 只取fact_pool[:3]            │ ← 问题6: 只处理前3条   │
│  └─────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ AtomicFact:                             │                        │
│  │   text = "截断片段" (语义不完整)          │ ← 结果: 事实质量差     │
│  │   source_url = "mock-source" (假URL)    │ ← 问题4: 固定Mock值    │
│  │   confidence = 0.85 (固定值)             │ ← 问题5: 固定值        │
│  └─────────────────────────────────────────┘                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────┐                        │
│  │ Writer: 使用低质量事实生成报告           │ ← 最终影响: 报告质量差 │
│  └─────────────────────────────────────────┘                        │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. 截断问题详解

### 2.1 截断代码位置

#### 位置1: researcher截断 - [graph.py:88-90](file:///e:/mini-deep-research/core/graph.py#L88-L90)

```python
for scraped in scraped_data:
    if scraped.markdown and not scraped.error:
        state["fact_pool"].append(scraped.markdown[:500])  # ← 截断至500字符
```

**问题**:
- 网页内容通常有数千甚至数万字符
- 截断至500字符会丢失大部分内容
- 可能在句子中间截断，导致语义不完整

**示例**:
```
原文: "台积电宣布其2纳米制程技术研发进展顺利，预计将在2025年下半年开始量产。
      这款芯片将采用全新的Gate-All-Around晶体管架构，与当前的FinFET相比，
      性能提升可达10-15%。台积电CEO魏哲家表示，公司已投资超过200亿美元..."

截断后: "台积电宣布其2纳米制程技术研发进展顺利，预计将在2025年下半年开始量产。
        这款芯片将采用全新的Gate-All-Around晶体管架构，与当前的FinFET相比，
        性能提升可达10-15%。台积电CEO魏哲家表示，公司已投资超过200亿美"
                                                                    ↑ 截断处
```

#### 位置2: DistillerAgent截断 - [distiller.py:140-146](file:///e:/mini-deep-research/agents/distiller.py#L140-L146)

```python
def _build_prompt(self, markdown_text: str) -> str:
    truncated = markdown_text[:8000]  # ← 截断至8000字符

    return f"""请分析以下文本，提取原子事实：

{'-'*60}
{truncated}
{'-'*60}
...
```

**问题**:
- 8000字符相对合理，但仍可能在语义边界处截断
- 长文档会丢失后半部分内容
- 没有按语义边界分割

### 2.2 截断问题对比

| 维度 | researcher截断 (500字符) | DistillerAgent截断 (8000字符) |
|-----|-------------------------|------------------------------|
| 截断长度 | 500字符 | 8000字符 |
| 信息保留率 | ~5-10% | ~80-90% |
| 语义完整性 | 极差 | 较好 |
| 问题严重程度 | 🔴 严重 | 🟡 中等 |

### 2.3 根本原因

**问题根源**: `researcher_async` 和 `distiller` 的职责划分不清

```
当前设计 (错误):
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  researcher  │────►│  fact_pool   │────►│   distiller  │
│  (截断500)   │     │ (存储截断片段)│     │  (Mock实现)  │
└──────────────┘     └──────────────┘     └──────────────┘

正确设计:
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  researcher  │────►│ raw_scraped  │────►│   distiller  │
│  (完整存储)  │     │   (完整数据)  │     │ (智能提取)   │
└──────────────┘     └──────────────┘     └──────────────┘
```

---

## 3. DistillerAgent集成方案

### 3.1 当前实现 vs 正确实现

#### 当前实现 (Mock) - [graph.py:111-126](file:///e:/mini-deep-research/core/graph.py#L111-L126)

```python
def distiller(state: GraphState) -> GraphState:
    print(f"[DISTILLER] Distilling facts to atomic units...")

    for fact_text in state["fact_pool"][:3]:  # 只取前3条
        atomic_fact = AtomicFact(
            text=fact_text,  # 直接使用截断片段
            source_url="https://example.com/mock-source",  # Mock URL
            confidence=0.85,  # 固定置信度
            task_id=state.get("current_focus")
        )
        state["atomic_facts"].append(atomic_fact.model_dump())

    state["fact_pool"] = state["fact_pool"][3:]
    state["token_usage"]["distillation_tokens"] += 200
    return state
```

#### 正确实现 (使用DistillerAgent)

```python
from agents import DistillerAgent

async def distiller_async(state: GraphState) -> GraphState:
    print(f"[DISTILLER] Distilling facts to atomic units...")

    distiller_agent = DistillerAgent()
    
    # 从 raw_scraped_data 获取完整数据
    scraped_data = state.get("raw_scraped_data", [])
    
    if not scraped_data:
        print(f"[DISTILLER] No scraped data available")
        return state
    
    # 准备批量提取的数据
    items = []
    for scraped in scraped_data:
        if scraped.get("markdown") and not scraped.get("error"):
            items.append({
                "markdown": scraped["markdown"],  # 完整markdown，不截断
                "url": scraped["url"],
                "task_id": state.get("current_focus")
            })
    
    if not items:
        print(f"[DISTILLER] No valid items to distill")
        return state
    
    # 批量提取事实
    try:
        results = await distiller_agent.distill_batch(items)
        
        for result in results:
            for fact in result.facts:
                state["atomic_facts"].append(fact.model_dump())
        
        print(f"[DISTILLER] Extracted {len(state['atomic_facts'])} atomic facts")
        
    except Exception as e:
        print(f"[DISTILLER] Error during distillation: {e}")
    
    state["token_usage"]["distillation_tokens"] += 500  # 更新token统计
    return state
```

### 3.2 需要修改的位置

| 文件 | 位置 | 修改内容 |
|-----|------|---------|
| `core/graph.py` | 第14行 | 添加 `from agents import DistillerAgent` |
| `core/graph.py` | 第88-90行 | 移除截断，保留完整markdown到 `raw_scraped_data` |
| `core/graph.py` | 第111-126行 | 替换为 `distiller_async` 实现 |
| `core/graph.py` | 第251行 | 将 `distiller` 改为 `distiller_async` |

### 3.3 修改前后对比

| 维度 | 修改前 | 修改后 |
|-----|-------|-------|
| 数据来源 | `fact_pool` (截断片段) | `raw_scraped_data` (完整数据) |
| 提取方式 | 直接创建AtomicFact | DistillerAgent智能提取 |
| source_url | Mock值 | 真实URL |
| confidence | 固定0.85 | LLM评估 |
| 事实数量 | 最多3条 | 全部提取 |
| 事实质量 | 低 (截断片段) | 高 (语义完整) |

---

## 4. 固定值问题报告

### 4.1 固定值问题清单

| 编号 | 固定值 | 位置 | 影响 | 修改方案 |
|-----|-------|------|------|---------|
| 1 | `source_url="https://example.com/mock-source"` | graph.py:117 | 报告引用失效 | 使用真实URL |
| 2 | `confidence=0.85` | graph.py:118 | 可信度评估失真 | 使用LLM评估值 |
| 3 | `state["fact_pool"][:3]` | graph.py:114 | 丢失大部分事实 | 处理全部数据 |
| 4 | `token_usage += 200` | graph.py:124 | Token统计不准确 | 使用实际消耗 |
| 5 | `scraped.markdown[:500]` | graph.py:90 | 丢失大部分内容 | 保留完整内容 |

### 4.2 详细分析

#### 问题1: Mock source_url

**位置**: [graph.py:117](file:///e:/mini-deep-research/core/graph.py#L117)

```python
source_url="https://example.com/mock-source",
```

**影响**:
- 报告中的引用链接全部失效
- 用户无法追溯信息来源
- 报告可信度降低

**修改方案**:
```python
# 从 raw_scraped_data 获取真实URL
source_url=scraped["url"],  # 使用真实URL
```

---

#### 问题2: 固定confidence

**位置**: [graph.py:118](file:///e:/mini-deep-research/core/graph.py#L118)

```python
confidence=0.85,
```

**影响**:
- 所有事实的可信度相同，无法区分质量
- 低质量事实和高质量事实无法区分
- 报告中无法标注可信度差异

**修改方案**:
```python
# 使用DistillerAgent的LLM评估值
confidence=fact.confidence,  # 使用LLM评估的可信度
```

---

#### 问题3: 只处理前3条

**位置**: [graph.py:114](file:///e:/mini-deep-research/core/graph.py#L114)

```python
for fact_text in state["fact_pool"][:3]:
```

**影响**:
- 如果有10条数据，只处理3条
- 丢失70%的信息

**修改方案**:
```python
# 处理所有数据
for item in items:  # 处理全部数据
```

---

#### 问题4: 固定Token统计

**位置**: [graph.py:124](file:///e:/mini-deep-research/core/graph.py#L124)

```python
state["token_usage"]["distillation_tokens"] += 200
```

**影响**:
- Token统计不准确
- 无法评估真实成本

**修改方案**:
```python
# 使用API返回的实际token消耗
usage = data.get("usage", {})
state["token_usage"]["distillation_tokens"] += usage.get("total_tokens", 0)
```

---

#### 问题5: 截断至500字符

**位置**: [graph.py:90](file:///e:/mini-deep-research/core/graph.py#L90)

```python
state["fact_pool"].append(scraped.markdown[:500])
```

**影响**:
- 丢失90%以上的内容
- 语义断裂

**修改方案**:
```python
# 不截断，保留完整内容
state["raw_scraped_data"].append(scraped.model_dump())
```

---

## 5. 修改方案汇总

### 5.1 修改清单

| 序号 | 文件 | 行号 | 修改类型 | 说明 |
|-----|------|------|---------|------|
| 1 | core/graph.py | 14 | 新增 | 导入DistillerAgent |
| 2 | core/graph.py | 88-90 | 修改 | 移除截断，保留完整数据 |
| 3 | core/graph.py | 111-126 | 重写 | 使用DistillerAgent |
| 4 | core/graph.py | 251 | 修改 | 使用async版本 |

### 5.2 修改优先级

| 优先级 | 修改项 | 原因 |
|--------|-------|------|
| P0 | 集成DistillerAgent | 解决大部分问题 |
| P0 | 移除500字符截断 | 保留完整信息 |
| P0 | 使用真实source_url | 报告引用有效 |
| P1 | 使用LLM评估confidence | 可信度准确 |
| P1 | 处理全部数据 | 不丢失信息 |

### 5.3 预期效果

| 维度 | 修改前 | 修改后 | 提升 |
|-----|-------|-------|------|
| 信息保留率 | ~5-10% | ~90% | +80% |
| 事实数量 | 最多3条 | 全部提取 | +200%+ |
| 来源可追溯 | ❌ Mock URL | ✅ 真实URL | 质的飞跃 |
| 可信度评估 | ❌ 固定值 | ✅ LLM评估 | 更准确 |
| 报告质量 | 低 | 高 | 显著提升 |

---

## 附录: 完整修改代码

### A. researcher_async 修改

```python
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
            
            # 修改: 保留完整数据，不再截断
            for scraped in scraped_data:
                if scraped.markdown and not scraped.error:
                    state["raw_scraped_data"].append(scraped.model_dump())

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
```

### B. distiller_async 新实现

```python
from agents import DistillerAgent

async def distiller_async(state: GraphState) -> GraphState:
    print(f"[DISTILLER] Distilling facts to atomic units...")

    distiller_agent = DistillerAgent()
    
    scraped_data = state.get("raw_scraped_data", [])
    
    if not scraped_data:
        print(f"[DISTILLER] No scraped data available")
        return state
    
    items = []
    for scraped in scraped_data:
        if scraped.get("markdown") and not scraped.get("error"):
            items.append({
                "markdown": scraped["markdown"],
                "url": scraped["url"],
                "task_id": state.get("current_focus")
            })
    
    if not items:
        print(f"[DISTILLER] No valid items to distill")
        return state
    
    total_tokens = 0
    
    try:
        results = await distiller_agent.distill_batch(items)
        
        for result in results:
            for fact in result.facts:
                state["atomic_facts"].append(fact.model_dump())
        
        print(f"[DISTILLER] Extracted {len(state['atomic_facts'])} atomic facts")
        
    except Exception as e:
        print(f"[DISTILLER] Error during distillation: {e}")
    
    state["token_usage"]["distillation_tokens"] += total_tokens
    return state
```

### C. create_research_graph 修改

```python
def create_research_graph(checkpointer: AsyncSqliteSaver) -> StateGraph:
    workflow = StateGraph(GraphState)

    workflow.add_node("planner", planner)
    workflow.add_node("researcher", researcher_async)
    workflow.add_node("distiller", distiller_async)  # 使用async版本
    workflow.add_node("writer", writer_async)

    workflow.set_entry_point("planner")

    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "distiller")
    workflow.add_edge("distiller", "writer")
    workflow.add_edge("writer", END)

    return workflow.compile(checkpointer=checkpointer)
```

---

> 报告结束  
> 下一步: 根据本报告进行代码修改
