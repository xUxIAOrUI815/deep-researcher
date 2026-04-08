# Deep Research Agent 改进方案

> 版本: 2.0  
> 更新日期: 2026-04-05  
> 重要说明: 本文档基于对项目本质的重新理解，修正了之前将项目误判为RAG系统的错误

---

## 目录

1. [项目定位澄清](#1-项目定位澄清)
2. [原问题重新评估](#2-原问题重新评估)
3. [真正需要的改进方案](#3-真正需要的改进方案)
4. [实施优先级](#4-实施优先级)
5. [技术方案详解](#5-技术方案详解)

---

## 1. 项目定位澄清

### 1.1 项目本质

**本项目是一个 Deep Research Agent（深度研究智能体），不是 RAG（检索增强生成）系统。**

| 维度 | RAG系统 | 本项目 |
|-----|---------|----------------------|
| **核心模式** | 用户提问 → 检索知识库 → 生成回答 | 主动探索 → 收集信息 → 生成报告 |
| **知识来源** | 预先构建的静态知识库 | **实时网络搜索** |
| **数据新鲜度** | 取决于知识库更新频率 | **实时获取最新信息** |
| **知识持久化** | 长期存储，持续积累 | **临时存储，按需收集** |
| **典型场景** | 企业知识库问答、文档检索 | **深度调研、行业分析报告** |
| **用户角色** | 被动回答用户问题 | **主动规划、执行、总结** |

### 1.2 核心工作流

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Deep Research Agent 工作流                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  用户输入: "调研台积电2nm芯片进展"                                    │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Phase 1: 规划                              │   │
│  │  Planner 分析主题，生成研究任务树                              │   │
│  │  - 台积电2nm量产时间                                           │   │
│  │  - 竞争对手进展对比                                            │   │
│  │  - 技术路线分析                                                │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Phase 2: 执行                              │   │
│  │  Researcher 循环执行每个任务:                                  │   │
│  │    1. MCPGateway 搜索相关信息                                  │   │
│  │    2. SmartScraper 抓取网页内容                               │   │
│  │    3. DistillerAgent 提取原子事实                             │   │
│  │    4. KnowledgeManager 存储（工作记忆）                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Phase 3: 收敛                              │   │
│  │  ConvergenceChecker 判断是否需要继续深挖                       │   │
│  │  - 信息是否饱和？                                              │   │
│  │  - 是否有未探索的线索？                                        │   │
│  │  - 深度是否达到上限？                                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Phase 4: 输出                              │   │
│  │  Writer 综合所有事实，生成结构化调研报告                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│         │                                                           │
│         ▼                                                           │
│  输出: 带引用的调研报告                                              │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 KnowledgeManager 的真实定位

**KnowledgeManager 是"工作记忆"，不是"知识库"**

| 功能 | 定位 | 说明 |
|-----|-----|-----|
| 临时存储 | 工作记忆 | 在单次研究过程中存储提取的事实 |
| 去重 | 数据处理 | 避免重复收集相同信息 |
| 冲突检测 | 质量控制 | 发现不同来源的矛盾信息 |
| 语义检索 | 辅助功能 | 支持去重和冲突检测，非核心检索 |
| 持久化 | 可选功能 | 支持跨会话知识积累（非必需） |

---

## 2. 原问题重新评估

### 2.1 原问题清单（基于RAG误解）

| 编号 | 原问题描述 | 原判断 |
|-----|-----------|--------|
| P0-1 | 向量数据库未集成 | 严重问题 |
| P0-2 | 缺乏真正的RAG检索 | 严重问题 |
| P1-1 | 缺乏文档分块策略 | 中等问题 |
| P1-2 | 缺乏重排序机制 | 中等问题 |
| P1-3 | 缺乏混合检索 | 中等问题 |
| P2-1 | 缺乏来源可信度评估 | 较低问题 |
| P2-2 | 缺乏增量更新与版本管理 | 较低问题 |

### 2.2 重新评估结论

| 编号 | 原问题描述 | 重新评估 | 评估理由 |
|-----|-----------|----------|---------|
| P0-1 | 向量数据库未集成 | ⚠️ **非核心问题** | 当前内存存储足够支撑单次研究的事实量；向量数据库是优化手段，非必需 |
| P0-2 | 缺乏真正的RAG检索 | ✅ **部分有效** | Writer确实需要筛选事实，但不是传统RAG检索；是"报告生成优化"问题 |
| P1-1 | 缺乏文档分块策略 | ⚠️ **需重新定义** | 问题本质是"事实提取质量"，分块只是手段之一 |
| P1-2 | 缺乏重排序机制 | ❌ **不适用** | 事实数量有限，不需要复杂的重排序；简单的相关性筛选即可 |
| P1-3 | 缺乏混合检索 | ❌ **不适用** | 知识是实时收集的，不需要预先建立BM25索引 |
| P2-1 | 缺乏来源可信度评估 | ✅ **有效** | 对提升报告质量有直接价值 |
| P2-2 | 缺乏增量更新与版本管理 | ⚠️ **可选功能** | 对于长期跟踪场景有价值，但非核心需求 |

### 2.3 问题分类总结

```
┌─────────────────────────────────────────────────────────────────────┐
│                     问题重新分类                                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ✅ 真实问题（需要解决）                                              │
│  ├── Writer直接使用全部事实，缺乏筛选                                 │
│  ├── 事实提取质量不稳定                                              │
│  ├── 缺乏来源可信度评估                                              │
│  └── 研究深度控制不够智能                                            │
│                                                                     │
│  ⚠️ 优化项（可选改进）                                                │
│  ├── 向量数据库（大规模场景）                                         │
│  ├── 跨会话知识积累                                                  │
│  └── 事实版本管理                                                    │
│                                                                     │
│  ❌ 不适用（误解产生）                                                 │
│  ├── 混合检索 (BM25+向量)                                            │
│  ├── 复杂的重排序机制                                                │
│  └── 传统RAG架构改进                                                 │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. 真正需要的改进方案

### 3.1 改进方向总览

| 优先级 | 改进方向 | 核心目标 | 对应原问题 |
|--------|---------|---------|-----------|
| **P0** | 报告生成优化 | Writer智能筛选事实，生成高质量报告 | P0-2 (重新定义) |
| **P0** | 研究深度控制 | 智能判断何时停止深挖 | 新发现问题 |
| **P1** | 事实提取质量 | 提升DistillerAgent提取准确性 | P1-1 (重新定义) |
| **P1** | 来源可信度评估 | 客观评估信息来源质量 | P2-1 |
| **P2** | 跨会话知识积累 | 支持长期研究场景 | P2-2 |
| **P2** | 向量数据库集成 | 支持大规模事实存储 | P0-1 (降级) |

### 3.2 各改进方向详解

#### P0-1: 报告生成优化

**问题描述**:
当前 `writer_async` 直接使用 `state["atomic_facts"]` 中的全部事实，导致：
- 不相关事实干扰报告生成
- Token浪费
- 报告质量下降

**改进目标**:
Writer 应根据报告主题智能筛选相关事实，而非使用全部事实。

**改进方案**:
```python
# 改进思路（非完整代码）

async def writer_async(state: GraphState) -> GraphState:
    # 1. 获取报告主题
    root_query = get_root_query(state)
    
    # 2. 筛选相关事实
    relevant_facts = filter_facts_by_relevance(
        facts=state["atomic_facts"],
        query=root_query,
        min_relevance=0.3
    )
    
    # 3. 按重要性排序
    ranked_facts = rank_facts_by_importance(relevant_facts)
    
    # 4. 限制数量
    selected_facts = ranked_facts[:30]
    
    # 5. 生成报告
    report = generate_report(selected_facts, root_query)
```

**验收标准**:
- [ ] Writer只使用与主题相关的事实
- [ ] 报告中事实引用准确
- [ ] Token使用量减少30%以上

---

#### P0-2: 研究深度控制

**问题描述**:
当前 `ConvergenceChecker` 的判断逻辑较为简单，可能导致：
- 过早停止，信息收集不充分
- 过度深挖，浪费资源

**改进目标**:
更智能地判断何时停止研究，平衡信息完整性和资源消耗。

**改进方案**:
```python
# 改进思路

class IntelligentConvergenceChecker:
    """智能收敛检查器"""
    
    def check(self, state: dict) -> ConvergenceDecision:
        # 1. 硬性约束检查
        if self._check_hard_limits(state):
            return ConvergenceDecision(action="finish", reason="达到硬性限制")
        
        # 2. 信息饱和度检查
        saturation = self._check_saturation(state)
        if saturation > 0.8:
            return ConvergenceDecision(action="finish", reason="信息饱和")
        
        # 3. 线索分析
        clues = self._analyze_clues(state)
        if not clues:
            return ConvergenceDecision(action="finish", reason="无线索可追踪")
        
        # 4. 价值评估
        value = self._estimate_value(state, clues)
        if value < self.threshold:
            return ConvergenceDecision(action="finish", reason="边际价值过低")
        
        return ConvergenceDecision(action="continue", reason=f"发现{len(clues)}个线索")
    
    def _check_saturation(self, state: dict) -> float:
        """计算信息饱和度"""
        # 基于新事实与已有事实的相似度
        # 如果连续N轮新事实都与已有事实高度相似，则饱和
        pass
    
    def _analyze_clues(self, state: dict) -> List[Clue]:
        """分析未追踪的线索"""
        # 从已有事实中提取：
        # - 未定义的专业术语
        # - 数据冲突点
        # - 来源不可信的孤证
        pass
    
    def _estimate_value(self, state: dict, clues: List[Clue]) -> float:
        """估计继续研究的边际价值"""
        # 考虑因素：
        # - 线索数量和质量
        # - 当前深度
        # - 已收集信息的完整性
        pass
```

**验收标准**:
- [ ] 饱和度检测准确率 > 80%
- [ ] 线索分析能发现有价值的研究方向
- [ ] 避免无限循环

---

#### P1-1: 事实提取质量

**问题描述**:
当前 `DistillerAgent` 直接截断文档至8000字符，可能导致：
- 信息丢失
- 语义断裂
- 提取的事实不完整

**改进目标**:
提升事实提取的完整性和准确性。

**改进方案**:
```python
# 改进思路

class ImprovedDistillerAgent:
    """改进的事实提取器"""
    
    async def distill(self, markdown: str, source_url: str) -> List[AtomicFact]:
        # 1. 文档预处理
        cleaned = self._preprocess(markdown)
        
        # 2. 分段处理（而非简单截断）
        sections = self._split_into_sections(cleaned)
        
        # 3. 提取事实
        all_facts = []
        for section in sections:
            facts = await self._extract_facts(section, source_url)
            all_facts.extend(facts)
        
        # 4. 去重和合并
        merged = self._merge_facts(all_facts)
        
        # 5. 质量过滤
        filtered = self._filter_low_quality(merged)
        
        return filtered
    
    def _split_into_sections(self, markdown: str) -> List[str]:
        """按语义边界分割，而非简单截断"""
        # 按标题分割
        # 每个章节独立处理
        pass
    
    def _filter_low_quality(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        """过滤低质量事实"""
        # 过滤条件：
        # - 文本过短（< 20字符）
        # - 缺乏实质内容
        # - 置信度过低（< 0.5）
        pass
```

**验收标准**:
- [ ] 不丢失重要信息
- [ ] 提取的事实语义完整
- [ ] 事实质量评分 > 0.7

---

#### P1-2: 来源可信度评估

**问题描述**:
当前事实的 `confidence` 仅由 LLM 主观判断，缺乏客观评估标准。

**改进目标**:
建立客观的来源可信度评估体系。

**改进方案**:
```python
# 改进思路

class SourceCredibilityScorer:
    """来源可信度评估器"""
    
    TRUSTED_DOMAINS = {
        "gov.cn": 0.95,
        "edu.cn": 0.90,
        "reuters.com": 0.85,
        # ...
    }
    
    def score(self, source_url: str, content: str) -> float:
        # 1. 域名可信度 (40%)
        domain_score = self._score_domain(source_url)
        
        # 2. 内容质量 (30%)
        content_score = self._score_content(content)
        
        # 3. 引用质量 (20%)
        citation_score = self._score_citations(content)
        
        # 4. 时效性 (10%)
        freshness_score = self._score_freshness(content)
        
        return (
            0.4 * domain_score +
            0.3 * content_score +
            0.2 * citation_score +
            0.1 * freshness_score
        )
```

**验收标准**:
- [ ] 可信度评分有明确依据
- [ ] 低质量来源能被识别
- [ ] 报告中标注来源可信度

---

#### P2-1: 跨会话知识积累

**问题描述**:
每次研究都是独立的，无法利用历史研究成果。

**改进目标**:
支持跨会话的知识积累，提升长期研究效率。

**改进方案**:
```python
# 改进思路

class KnowledgeAccumulator:
    """知识积累器"""
    
    def __init__(self, storage_path: str):
        self.storage = PersistentStorage(storage_path)
    
    async def save_research(self, topic: str, facts: List[AtomicFact], report: str):
        """保存研究成果"""
        pass
    
    async def load_related_facts(self, topic: str, limit: int = 10) -> List[AtomicFact]:
        """加载相关历史事实"""
        pass
    
    async def merge_with_history(self, new_facts: List[AtomicFact], topic: str) -> List[AtomicFact]:
        """合并历史事实，去重"""
        pass
```

**验收标准**:
- [ ] 研究成果持久化存储
- [ ] 能检索相关历史事实
- [ ] 支持事实去重和更新

---

#### P2-2: 向量数据库集成

**问题描述**:
当前内存存储无法支撑大规模事实。

**改进目标**:
支持大规模事实存储和检索。

**改进方案**:
- 已有 `QdrantVectorStore` 实现
- 需要集成到 `KnowledgeManager`
- 作为可选功能，通过配置开关

**验收标准**:
- [ ] 配置开关控制存储后端
- [ ] 大规模数据测试通过
- [ ] 性能指标达标

---

## 4. 实施优先级

### 4.1 优先级矩阵

```
                    业务价值
           低                      高
        ┌─────────────────┬─────────────────┐
   高   │                 │   P0-1 报告生成   │
        │   P2-2 向量DB   │   P0-2 深度控制   │
 实      │                 │                 │
 施      ├─────────────────┼─────────────────┤
 复      │                 │   P1-1 事实提取   │
 杂      │   P2-1 知识积累 │   P1-2 可信度     │
 度      │                 │                 │
   低   └─────────────────┴─────────────────┘
```

### 4.2 实施路线

| 阶段 | 任务 | 预估时间 | 依赖 |
|-----|-----|---------|-----|
| **阶段1** | P0-1 报告生成优化 | 4h | 无 |
| | P0-2 研究深度控制 | 6h | 无 |
| **阶段2** | P1-1 事实提取质量 | 5h | 无 |
| | P1-2 来源可信度评估 | 4h | 无 |
| **阶段3** | P2-1 跨会话知识积累 | 6h | P2-2 |
| | P2-2 向量数据库集成 | 4h | 无 |
| **总计** | | **29h** | |

### 4.3 里程碑

| 里程碑 | 完成标准 | 预计完成 |
|-------|---------|---------|
| M1 | 报告生成优化完成，报告质量提升 | 阶段1开始后4h |
| M2 | 深度控制智能化，研究效率提升 | 阶段1开始后10h |
| M3 | 事实提取质量提升，信息更完整 | 阶段2开始后5h |
| M4 | 可信度评估上线，报告更可靠 | 阶段2开始后9h |
| M5 | 全部改进完成 | 阶段3开始后10h |

---

## 5. 技术方案详解

### 5.1 P0-1: 报告生成优化

#### 5.1.1 当前实现分析

```python
# 当前实现 (core/graph.py:129-148)
async def writer_async(state: GraphState) -> GraphState:
    # 直接使用全部事实
    facts = state["atomic_facts"]  # 问题：未筛选
    
    facts_text = []
    for i, fact in enumerate(facts, 1):
        facts_text.append(f"[{fact_id}] {text}...")
    
    # 生成报告
    prompt = f"...原子事实列表...\n{facts_context}"
```

#### 5.1.2 改进实现

```python
# 改进方案
async def writer_async(state: GraphState) -> GraphState:
    from core.fact_selector import FactSelector
    
    # 1. 获取报告主题
    root_query = _get_root_query(state)
    
    # 2. 智能筛选事实
    selector = FactSelector()
    selected_facts = await selector.select_facts(
        all_facts=state["atomic_facts"],
        query=root_query,
        max_facts=30,
        min_relevance=0.3
    )
    
    # 3. 按主题分组
    grouped_facts = _group_facts_by_topic(selected_facts, root_query)
    
    # 4. 生成报告
    report = await _generate_report(grouped_facts, root_query)
    
    state["final_report"] = report
    return state


class FactSelector:
    """事实选择器"""
    
    def __init__(self, knowledge_manager: KnowledgeManager = None):
        self.km = knowledge_manager or KnowledgeManager()
    
    async def select_facts(
        self,
        all_facts: List[dict],
        query: str,
        max_facts: int = 30,
        min_relevance: float = 0.3
    ) -> List[dict]:
        """
        智能选择相关事实
        
        策略:
        1. 计算每个事实与查询的相关性
        2. 过滤低相关性事实
        3. 按重要性排序
        4. 确保多样性（避免过于相似的事实）
        """
        if not all_facts:
            return []
        
        # 计算相关性
        scored_facts = []
        for fact in all_facts:
            relevance = await self._compute_relevance(fact["text"], query)
            if relevance >= min_relevance:
                scored_facts.append({
                    **fact,
                    "relevance": relevance
                })
        
        # 按相关性排序
        scored_facts.sort(key=lambda x: x["relevance"], reverse=True)
        
        # 多样性选择
        selected = []
        for fact in scored_facts:
            if len(selected) >= max_facts:
                break
            
            # 检查与已选事实的相似度
            is_duplicate = False
            for s in selected:
                sim = await self._compute_similarity(fact["text"], s["text"])
                if sim > 0.85:  # 高度相似则跳过
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                selected.append(fact)
        
        return selected
    
    async def _compute_relevance(self, fact_text: str, query: str) -> float:
        """计算事实与查询的相关性"""
        # 使用embedding相似度
        return await self.km._find_similar_score(query, fact_text)
    
    async def _compute_similarity(self, text1: str, text2: str) -> float:
        """计算两段文本的相似度"""
        # 使用embedding相似度
        pass


def _get_root_query(state: GraphState) -> str:
    """获取根查询"""
    root_id = state.get("root_task_id")
    if root_id and root_id in state["task_tree"]:
        return state["task_tree"][root_id].get("query", "")
    return ""


def _group_facts_by_topic(facts: List[dict], query: str) -> Dict[str, List[dict]]:
    """按主题分组事实"""
    # 简单实现：按关键词分组
    # 复杂实现：使用LLM进行主题聚类
    pass
```

#### 5.1.3 文件变更

| 文件 | 操作 | 说明 |
|-----|-----|-----|
| `core/fact_selector.py` | 新增 | 事实选择器 |
| `core/graph.py` | 修改 | 集成事实选择 |

---

### 5.2 P0-2: 研究深度控制

#### 5.2.1 当前实现分析

```python
# 当前实现 (core/convergence.py)
class ConvergenceChecker:
    @staticmethod
    def check(state: dict) -> ConvergenceDecision:
        # 硬性约束
        if current_depth >= MAX_DEPTH:
            return ConvergenceDecision(action="finish")
        
        if task_tree_size >= MAX_NODES:
            return ConvergenceDecision(action="finish")
        
        # 简单判断
        if pending_count > 0:
            return ConvergenceDecision(action="continue")
        
        return ConvergenceDecision(action="finish")
```

#### 5.2.2 改进实现

```python
# 改进方案
class IntelligentConvergenceChecker:
    """智能收敛检查器"""
    
    def __init__(self, config: ResearchConfig = None):
        self.config = config or ResearchConfig()
        self._saturation_history: List[float] = []
    
    def check(self, state: dict) -> ConvergenceDecision:
        """综合判断是否应该收敛"""
        
        # 1. 硬性约束检查
        hard_limit_result = self._check_hard_limits(state)
        if hard_limit_result:
            return hard_limit_result
        
        # 2. 信息饱和度检查
        saturation = self._compute_saturation(state)
        self._saturation_history.append(saturation)
        
        if self._is_saturated(saturation):
            return ConvergenceDecision(
                action="finish",
                reason=f"信息饱和度 {saturation:.2%} 超过阈值",
                skip_pending_tasks=True
            )
        
        # 3. 线索分析
        clues = self._analyze_clues(state)
        
        if not clues:
            return ConvergenceDecision(
                action="finish",
                reason="未发现新的研究线索",
                skip_pending_tasks=False
            )
        
        # 4. 边际价值评估
        value = self._estimate_marginal_value(state, clues)
        
        if value < self.config.VALUE_THRESHOLD:
            return ConvergenceDecision(
                action="finish",
                reason=f"边际研究价值 {value:.2f} 低于阈值",
                skip_pending_tasks=True
            )
        
        return ConvergenceDecision(
            action="continue",
            reason=f"发现 {len(clues)} 个研究线索，继续深挖",
            skip_pending_tasks=False
        )
    
    def _check_hard_limits(self, state: dict) -> Optional[ConvergenceDecision]:
        """检查硬性限制"""
        current_depth = state.get("current_depth", 0)
        if current_depth >= self.config.MAX_DEPTH:
            return ConvergenceDecision(
                action="finish",
                reason=f"深度 {current_depth} 达到上限 {self.config.MAX_DEPTH}",
                skip_pending_tasks=True
            )
        
        task_tree_size = len(state.get("task_tree", {}))
        if task_tree_size >= self.config.MAX_NODES:
            return ConvergenceDecision(
                action="finish",
                reason=f"节点数 {task_tree_size} 达到上限 {self.config.MAX_NODES}",
                skip_pending_tasks=True
            )
        
        total_facts = len(state.get("all_fact_ids", []))
        if total_facts > self.config.MAX_FACTS:
            return ConvergenceDecision(
                action="finish",
                reason=f"事实数 {total_facts} 超过上限 {self.config.MAX_FACTS}",
                skip_pending_tasks=True
            )
        
        return None
    
    def _compute_saturation(self, state: dict) -> float:
        """
        计算信息饱和度
        
        饱和度 = 最近N轮新事实与已有事实的平均相似度
        """
        recent_facts = state.get("recent_facts", [])
        all_facts = state.get("all_fact_ids", [])
        
        if not recent_facts or not all_facts:
            return 0.0
        
        # 计算新事实与已有事实的平均相似度
        # 如果高度相似，说明信息趋于饱和
        # 实际实现需要调用KnowledgeManager
        
        return 0.0  # 占位
    
    def _is_saturated(self, saturation: float) -> bool:
        """判断是否饱和"""
        # 条件1: 饱和度超过阈值
        if saturation > self.config.SATURATION_THRESHOLD:
            return True
        
        # 条件2: 连续N轮饱和度上升
        if len(self._saturation_history) >= 3:
            recent = self._saturation_history[-3:]
            if all(s > 0.7 for s in recent) and all(recent[i] <= recent[i+1] for i in range(2)):
                return True
        
        return False
    
    def _analyze_clues(self, state: dict) -> List[dict]:
        """
        分析未追踪的研究线索
        
        线索类型:
        1. 未定义的专业术语
        2. 数据冲突点
        3. 来源不可信的孤证
        4. 用户明确要求深挖的方向
        """
        clues = []
        
        # 从事实中提取线索
        facts = state.get("atomic_facts", [])
        for fact in facts:
            # 检查是否有未定义术语
            undefined_terms = self._find_undefined_terms(fact.get("text", ""))
            for term in undefined_terms:
                clues.append({
                    "type": "undefined_term",
                    "content": term,
                    "source_fact": fact.get("id")
                })
        
        # 检查冲突
        conflicts = state.get("conflicts", [])
        for conflict in conflicts:
            clues.append({
                "type": "data_conflict",
                "content": conflict.get("description"),
                "source_facts": [conflict.get("fact_id_1"), conflict.get("fact_id_2")]
            })
        
        return clues
    
    def _find_undefined_terms(self, text: str) -> List[str]:
        """查找未定义的专业术语"""
        # 简单实现：查找大写缩写、专业词汇
        # 复杂实现：使用NER或LLM
        import re
        acronyms = re.findall(r'\b[A-Z]{2,}\b', text)
        return acronyms
    
    def _estimate_marginal_value(self, state: dict, clues: List[dict]) -> float:
        """
        估计继续研究的边际价值
        
        考虑因素:
        1. 线索数量和质量
        2. 当前深度
        3. 已收集信息的完整性
        """
        if not clues:
            return 0.0
        
        # 线索价值
        clue_value = len(clues) * 0.2
        
        # 深度惩罚
        current_depth = state.get("current_depth", 0)
        depth_penalty = current_depth * 0.15
        
        # 信息完整性奖励
        total_facts = len(state.get("all_fact_ids", []))
        completeness_bonus = min(total_facts / 20, 1.0) * 0.1
        
        value = clue_value - depth_penalty + completeness_bonus
        return max(0.0, min(1.0, value))
```

#### 5.2.3 文件变更

| 文件 | 操作 | 说明 |
|-----|-----|-----|
| `core/convergence.py` | 修改 | 实现智能收敛检查 |
| `core/config.py` | 修改 | 添加收敛相关配置 |

---

## 附录

### A. 与原RAG方案的对比

| 维度 | 原RAG方案 | 新方案 |
|-----|----------|--------|
| 核心定位 | RAG检索增强生成 | Deep Research Agent |
| 知识来源 | 预构建知识库 | 实时网络搜索 |
| 检索需求 | 高（核心功能） | 低（辅助功能） |
| 向量数据库 | 必需 | 可选 |
| 混合检索 | 需要 | 不需要 |
| 重排序 | 需要 | 简单筛选即可 |
| 工作量 | 42h | 29h |

### B. 文件变更清单

| 文件 | 操作 | 优先级 |
|-----|-----|--------|
| `core/fact_selector.py` | 新增 | P0 |
| `core/graph.py` | 修改 | P0 |
| `core/convergence.py` | 修改 | P0 |
| `core/config.py` | 修改 | P0 |
| `agents/distiller.py` | 修改 | P1 |
| `core/credibility.py` | 新增 | P1 |
| `core/knowledge_accumulator.py` | 新增 | P2 |
| `core/vector_store_adapter.py` | 新增 | P2 |

### C. 配置项清单

```python
# core/config.py 新增配置

class ResearchConfig:
    # 现有配置
    MAX_DEPTH: int = 3
    MAX_NODES: int = 15
    MAX_FACTS: int = 30
    
    # 新增配置
    SATURATION_THRESHOLD: float = 0.80
    VALUE_THRESHOLD: float = 0.30
    MIN_FACT_RELEVANCE: float = 0.30
    MAX_FACTS_IN_REPORT: int = 30
    FACT_SIMILARITY_THRESHOLD: float = 0.85
```

---

> 文档结束  
> 下一步: 按照优先级顺序开始实施 P0-1 报告生成优化
