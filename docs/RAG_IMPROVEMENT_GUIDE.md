# RAG系统改进任务指导文档

> 版本: 1.0  
> 创建日期: 2026-04-05  
> 目标: 将当前的基础知识存储升级为完整的RAG检索增强生成系统

---

## 目录

1. [现状分析](#1-现状分析)
2. [改进任务清单](#2-改进任务清单)
3. [P0级任务详解](#3-p0级任务详解)
4. [P1级任务详解](#4-p1级任务详解)
5. [P2级任务详解](#5-p2级任务详解)
6. [实施路线图](#6-实施路线图)
7. [风险评估与应对](#7-风险评估与应对)

---

## 1. 现状分析

### 1.1 当前架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      当前数据流                                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MCPGateway ──► SmartScraper ──► DistillerAgent ──► KnowledgeManager
│  (搜索)         (抓取+降噪)       (提取原子事实)      (JSON本地存储)
│                                                                 │
│       ↓              ↓                ↓                  ↓       │
│  SearchResult   ScrapedData      AtomicFact         StoredFact  │
│                                                                 │
│                              ↓                                  │
│                        Writer (直接使用全部事实)                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 核心问题总结

| 问题编号 | 问题描述 | 影响程度 | 当前状态 |
|---------|---------|---------|---------|
| P0-1 | 向量数据库未集成 | 严重 | QdrantVectorStore已实现但未使用 |
| P0-2 | 缺乏真正的RAG检索 | 严重 | Writer直接使用全部事实 |
| P1-1 | 缺乏文档分块策略 | 中等 | 直接截断至8000字符 |
| P1-2 | 缺乏重排序机制 | 中等 | 仅按向量相似度排序 |
| P1-3 | 缺乏混合检索 | 中等 | 仅使用向量检索 |
| P2-1 | 缺乏来源可信度评估 | 较低 | confidence仅由LLM主观判断 |
| P2-2 | 缺乏增量更新与版本管理 | 较低 | 事实无版本历史 |

### 1.3 现有资源评估

| 组件 | 文件路径 | 可复用程度 | 备注 |
|-----|---------|-----------|-----|
| QdrantVectorStore | `core/vector_store_qdrant.py` | 高 | 完整实现，仅需集成 |
| KnowledgeManager | `core/knowledge.py` | 中 | 需要重构以支持向量数据库 |
| DistillerAgent | `agents/distiller.py` | 中 | 需要添加分块支持 |
| SmartScraper | `providers/scraper.py` | 高 | 已有降噪功能，可直接使用 |
| EmbeddingModel | `core/knowledge.py` | 高 | SiliconFlow API已集成 |

---

## 2. 改进任务清单

### 优先级定义

- **P0 (Critical)**: 必须完成，直接影响核心功能
- **P1 (High)**: 强烈建议完成，显著提升效果
- **P2 (Medium)**: 建议完成，提升系统健壮性

### 任务总览

```
┌────────────────────────────────────────────────────────────────────┐
│                         任务依赖关系                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  P0-1: 向量数据库集成                                               │
│    │                                                               │
│    ├──► P0-2: RAG检索实现                                          │
│    │         │                                                     │
│    │         └──► P1-2: 重排序机制                                 │
│    │                                                               │
│  P1-1: 文档分块策略                                                 │
│    │                                                               │
│    └──► P1-3: 混合检索 (依赖BM25索引)                               │
│                                                                    │
│  P2-1: 来源可信度评估                                               │
│                                                                    │
│  P2-2: 版本管理                                                     │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. P0级任务详解

### 3.1 P0-1: 向量数据库集成

#### 3.1.1 问题描述

当前 `KnowledgeManager` 使用本地JSON文件存储事实，并在内存中遍历所有向量计算相似度。这种方式存在以下问题：

1. **扩展性差**: 数据量增大后内存占用过高
2. **检索效率低**: O(n)复杂度，每次检索需遍历所有向量
3. **持久化风险**: JSON文件损坏会导致数据丢失
4. **无法支持高级查询**: 缺乏过滤、聚合等功能

#### 3.1.2 可行性分析

| 维度 | 评估 | 说明 |
|-----|-----|-----|
| 技术可行性 | ✅ 高 | QdrantVectorStore已完整实现 |
| 资源需求 | ✅ 低 | Qdrant支持Docker部署，资源占用小 |
| 兼容性 | ✅ 高 | 可通过配置开关切换存储后端 |
| 风险 | ⚠️ 中 | 需要数据迁移策略 |

#### 3.1.3 最优实现方案

**方案选择**: 适配器模式 + 配置开关

```python
# core/knowledge.py 重构方案

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.vector_store_qdrant import QdrantVectorStore
from core.config import QdrantConfig

class VectorStoreAdapter(ABC):
    """向量存储适配器接口"""
    
    @abstractmethod
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        pass
    
    @abstractmethod
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        pass
    
    @abstractmethod
    async def get(self, id: str) -> Optional[Dict]:
        pass


class InMemoryVectorStore(VectorStoreAdapter):
    """内存向量存储 (当前实现的封装)"""
    
    def __init__(self):
        self._vectors: Dict[str, tuple] = {}  # id -> (vector, payload)
    
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        self._vectors[id] = (vector, payload)
        return True
    
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        results = []
        query_vec = np.array(vector)
        
        for id, (vec, payload) in self._vectors.items():
            score = self._cosine_similarity(query_vec, np.array(vec))
            if score >= threshold:
                results.append({"id": id, "score": score, "payload": payload})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


class QdrantAdapter(VectorStoreAdapter):
    """Qdrant向量存储适配器"""
    
    def __init__(self, collection_name: str = "knowledge_facts"):
        self.client = QdrantVectorStore(
            host=QdrantConfig.HOST,
            port=QdrantConfig.PORT
        )
        self.collection_name = collection_name
    
    async def upsert(self, id: str, vector: List[float], payload: Dict) -> bool:
        return self.client.upsert_point(
            collection_name=self.collection_name,
            point_id=id,
            vector=vector,
            payload=payload
        )
    
    async def search(self, vector: List[float], limit: int, threshold: float) -> List[Dict]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=vector,
            limit=limit,
            score_threshold=threshold
        )
        return [{"id": r[0], "score": r[1], "payload": r[2]} for r in results]
    
    async def delete(self, id: str) -> bool:
        # Qdrant需要实现delete方法
        pass
    
    async def get(self, id: str) -> Optional[Dict]:
        return self.client.retrieve(self.collection_name, id)


class KnowledgeManager:
    """重构后的知识管理器"""
    
    def __init__(
        self,
        storage_path: str = "./knowledge_data",
        use_qdrant: Optional[bool] = None
    ):
        self.storage_path = storage_path
        self.embedding_model = EmbeddingModel()
        
        # 根据配置选择存储后端
        if use_qdrant is None:
            use_qdrant = QdrantConfig.USE_QDRANT
        
        if use_qdrant:
            self.vector_store = QdrantAdapter()
            self._backend_type = "qdrant"
        else:
            self.vector_store = InMemoryVectorStore()
            self._backend_type = "memory"
        
        self._stats = KnowledgeStats()
        self._conflicts: List[FactConflict] = []
        
        if self._backend_type == "memory":
            os.makedirs(storage_path, exist_ok=True)
            self._load()
```

#### 3.1.4 实施步骤

| 步骤 | 任务 | 预估时间 | 产出物 |
|-----|-----|---------|--------|
| 1 | 创建VectorStoreAdapter抽象类 | 30min | `core/vector_store_adapter.py` |
| 2 | 实现InMemoryVectorStore | 1h | 封装现有逻辑 |
| 3 | 实现QdrantAdapter | 1h | 集成QdrantVectorStore |
| 4 | 重构KnowledgeManager | 2h | 支持后端切换 |
| 5 | 编写单元测试 | 1h | `tests/test_vector_adapter.py` |
| 6 | 数据迁移脚本 | 1h | `scripts/migrate_to_qdrant.py` |

#### 3.1.5 验收标准

- [ ] 配置 `USE_QDRANT=true` 时使用Qdrant存储
- [ ] 配置 `USE_QDRANT=false` 时使用内存存储
- [ ] 现有测试全部通过
- [ ] 新增适配器测试覆盖率 > 80%
- [ ] 支持数据迁移命令

---

### 3.2 P0-2: RAG检索实现

#### 3.2.1 问题描述

当前 `writer_async` 函数直接使用 `state["atomic_facts"]` 中的全部事实生成报告，没有根据报告主题进行语义检索。这导致：

1. **上下文窗口浪费**: 不相关的事实占用token
2. **报告质量下降**: 噪音事实干扰生成
3. **无法处理大规模数据**: 事实数量超过上下文限制

#### 3.2.2 可行性分析

| 维度 | 评估 | 说明 |
|-----|-----|-----|
| 技术可行性 | ✅ 高 | 检索逻辑已存在于KnowledgeManager |
| 资源需求 | ✅ 低 | 复用现有embedding和检索 |
| 兼容性 | ✅ 高 | 对现有流程无破坏性变更 |
| 风险 | ✅ 低 | 可逐步迁移 |

#### 3.2.3 最优实现方案

**方案选择**: 查询扩展 + 多阶段检索

```python
# core/rag_retriever.py (新文件)

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from core.knowledge import KnowledgeManager, FactStatus

@dataclass
class RetrievedContext:
    """检索结果上下文"""
    facts: List[Dict[str, Any]]
    total_count: int
    query: str
    retrieval_strategy: str


class RAGRetriever:
    """RAG检索器"""
    
    def __init__(self, knowledge_manager: KnowledgeManager):
        self.km = knowledge_manager
    
    async def retrieve_for_report(
        self,
        root_query: str,
        task_tree: Dict[str, Any],
        max_facts: int = 30,
        min_relevance: float = 0.3
    ) -> RetrievedContext:
        """
        为报告生成检索相关事实
        
        策略:
        1. 使用根查询进行初始检索
        2. 使用子任务查询进行补充检索
        3. 合并去重后按相关性排序
        """
        all_results = []
        seen_ids = set()
        
        # 阶段1: 根查询检索
        root_results = await self.km.search_facts(
            query=root_query,
            limit=max_facts // 2,
            status_filter=FactStatus.VERIFIED
        )
        for r in root_results:
            if r["id"] not in seen_ids and r["score"] >= min_relevance:
                all_results.append(r)
                seen_ids.add(r["id"])
        
        # 阶段2: 子任务查询补充检索
        sub_queries = self._extract_sub_queries(task_tree)
        remaining_quota = max_facts - len(all_results)
        
        for query in sub_queries[:3]:  # 最多使用3个子查询
            if remaining_quota <= 0:
                break
            
            sub_results = await self.km.search_facts(
                query=query,
                limit=remaining_quota // len(sub_queries[:3]),
                status_filter=FactStatus.ACTIVE
            )
            
            for r in sub_results:
                if r["id"] not in seen_ids and r["score"] >= min_relevance:
                    all_results.append(r)
                    seen_ids.add(r["id"])
                    remaining_quota -= 1
        
        # 阶段3: 按相关性排序
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        return RetrievedContext(
            facts=all_results[:max_facts],
            total_count=len(all_results),
            query=root_query,
            retrieval_strategy="multi_stage"
        )
    
    def _extract_sub_queries(self, task_tree: Dict[str, Any]) -> List[str]:
        """从任务树提取子查询"""
        queries = []
        for task_id, task in task_tree.items():
            query = task.get("query", "")
            if query and task.get("status") == "completed":
                queries.append(query)
        return queries


# core/graph.py 修改

async def writer_async(state: GraphState) -> GraphState:
    """使用RAG检索的Writer"""
    
    from core.knowledge import KnowledgeManager
    from core.rag_retriever import RAGRetriever
    
    # 初始化检索器
    km = KnowledgeManager()
    retriever = RAGRetriever(km)
    
    # 获取根查询
    root_query = ""
    if state.get("root_task_id"):
        root_task = state["task_tree"].get(state["root_task_id"], {})
        root_query = root_task.get("query", "")
    
    # RAG检索
    context = await retriever.retrieve_for_report(
        root_query=root_query,
        task_tree=state["task_tree"],
        max_facts=30
    )
    
    # 格式化事实
    facts_text = []
    for i, fact in enumerate(context.facts, 1):
        facts_text.append(
            f"[{fact['id'][:8]}] {fact['text']} "
            f"(来源: {fact['source_url']}, 相关度: {fact['score']:.2f})"
        )
    
    facts_context = "\n\n".join(facts_text)
    
    # ... 后续报告生成逻辑
```

#### 3.2.4 实施步骤

| 步骤 | 任务 | 预估时间 | 产出物 |
|-----|-----|---------|--------|
| 1 | 创建RAGRetriever类 | 1h | `core/rag_retriever.py` |
| 2 | 实现多阶段检索策略 | 1.5h | retrieve_for_report方法 |
| 3 | 修改writer_async | 1h | 集成RAG检索 |
| 4 | 添加检索日志 | 30min | 便于调试和评估 |
| 5 | 编写集成测试 | 1h | `tests/test_rag_retriever.py` |

#### 3.2.5 验收标准

- [ ] Writer使用检索到的事实而非全部事实
- [ ] 检索结果按相关性排序
- [ ] 支持多阶段检索策略
- [ ] 检索过程有详细日志
- [ ] 测试覆盖率 > 80%

---

## 4. P1级任务详解

### 4.1 P1-1: 文档分块策略

#### 4.1.1 问题描述

当前 `DistillerAgent` 直接将文档截断至8000字符，导致：

1. **信息丢失**: 超出长度的内容被丢弃
2. **语义断裂**: 可能在句子中间截断
3. **提取质量下降**: LLM无法理解不完整的上下文

#### 4.1.2 可行性分析

| 维度 | 评估 | 说明 |
|-----|-----|-----|
| 技术可行性 | ✅ 高 | 分块算法成熟 |
| 资源需求 | ⚠️ 中 | 增加API调用次数 |
| 兼容性 | ✅ 高 | 对下游透明 |
| 风险 | ⚠️ 中 | 需要处理跨块事实 |

#### 4.1.3 最优实现方案

**方案选择**: 语义分块 + 滑动窗口

```python
# core/chunker.py (新文件)

import re
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    """文档块"""
    text: str
    start_index: int
    end_index: int
    token_count: int
    overlap_with_previous: int
    source_section: Optional[str] = None


class SemanticChunker:
    """语义分块器"""
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 50,
        min_chunk_tokens: int = 100
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens
    
    def chunk(self, markdown: str) -> List[Chunk]:
        """
        语义分块策略:
        1. 按标题分割文档
        2. 对每个章节进行段落分割
        3. 合并小段落，拆分大段落
        4. 添加重叠窗口
        """
        # 按标题分割
        sections = self._split_by_headers(markdown)
        
        chunks = []
        current_position = 0
        
        for section_title, section_text in sections:
            section_chunks = self._chunk_section(
                section_text,
                section_title,
                current_position
            )
            chunks.extend(section_chunks)
            current_position += len(section_text)
        
        # 添加重叠
        chunks = self._add_overlap(chunks)
        
        return chunks
    
    def _split_by_headers(self, markdown: str) -> List[tuple]:
        """按标题分割"""
        header_pattern = r'^(#{1,6})\s+(.+)$'
        lines = markdown.split('\n')
        
        sections = []
        current_title = "Introduction"
        current_text = []
        
        for line in lines:
            match = re.match(header_pattern, line)
            if match:
                if current_text:
                    sections.append((current_title, '\n'.join(current_text)))
                current_title = match.group(2)
                current_text = []
            else:
                current_text.append(line)
        
        if current_text:
            sections.append((current_title, '\n'.join(current_text)))
        
        return sections
    
    def _chunk_section(
        self,
        text: str,
        section_title: str,
        start_pos: int
    ) -> List[Chunk]:
        """对章节进行分块"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        chunk_start = start_pos
        
        for para in paragraphs:
            para_tokens = self._estimate_tokens(para)
            
            if current_tokens + para_tokens > self.max_tokens:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_index=chunk_start,
                        end_index=chunk_start + len(chunk_text),
                        token_count=current_tokens,
                        overlap_with_previous=0,
                        source_section=section_title
                    ))
                current_chunk = [para]
                current_tokens = para_tokens
                chunk_start = start_pos + text.find(para)
            else:
                current_chunk.append(para)
                current_tokens += para_tokens
        
        if current_chunk and current_tokens >= self.min_chunk_tokens:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_index=chunk_start,
                end_index=chunk_start + len(chunk_text),
                token_count=current_tokens,
                overlap_with_previous=0,
                source_section=section_title
            ))
        
        return chunks
    
    def _add_overlap(self, chunks: List[Chunk]) -> List[Chunk]:
        """添加重叠窗口"""
        if len(chunks) <= 1:
            return chunks
        
        overlapped = [chunks[0]]
        
        for i in range(1, len(chunks)):
            prev_chunk = chunks[i - 1]
            curr_chunk = chunks[i]
            
            # 从前一个块末尾提取重叠文本
            overlap_text = self._get_overlap_text(
                prev_chunk.text,
                self.overlap_tokens
            )
            
            # 创建带重叠的新块
            new_text = overlap_text + '\n\n' + curr_chunk.text
            overlapped.append(Chunk(
                text=new_text,
                start_index=curr_chunk.start_index,
                end_index=curr_chunk.end_index,
                token_count=self._estimate_tokens(new_text),
                overlap_with_previous=len(overlap_text),
                source_section=curr_chunk.source_section
            ))
        
        return overlapped
    
    def _get_overlap_text(self, text: str, max_tokens: int) -> str:
        """获取重叠文本"""
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        
        overlap_sentences = []
        token_count = 0
        
        for sentence in reversed(sentences):
            sent_tokens = self._estimate_tokens(sentence)
            if token_count + sent_tokens > max_tokens:
                break
            overlap_sentences.insert(0, sentence)
            token_count += sent_tokens
        
        return ' '.join(overlap_sentences)
    
    def _estimate_tokens(self, text: str) -> int:
        """估算token数量 (中文约1.5字符/token，英文约4字符/token)"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        other_chars = len(text) - chinese_chars
        return int(chinese_chars / 1.5 + other_chars / 4)


# agents/distiller.py 修改

class DistillerAgent:
    
    def __init__(self, ...):
        # ... 现有初始化
        self.chunker = SemanticChunker(
            max_tokens=512,
            overlap_tokens=50
        )
    
    async def distill(self, markdown_text: str, source_url: str, task_id: Optional[str] = None) -> DistillationResult:
        """使用分块策略提取事实"""
        
        # 分块
        chunks = self.chunker.chunk(markdown_text)
        print(f"[DistillerAgent] Split into {len(chunks)} chunks")
        
        all_facts = []
        seen_fact_texts = set()
        
        for i, chunk in enumerate(chunks):
            # 对每个块提取事实
            prompt = self._build_prompt(chunk.text)
            response = await self._call_api(prompt)
            chunk_facts = self._parse_facts_from_response(response, source_url, task_id)
            
            # 去重
            for fact in chunk_facts:
                normalized = self._normalize_fact_text(fact.text)
                if normalized not in seen_fact_texts:
                    all_facts.append(fact)
                    seen_fact_texts.add(normalized)
            
            print(f"[DistillerAgent] Chunk {i+1}/{len(chunks)}: {len(chunk_facts)} facts")
        
        # 合并相似事实
        merged_facts = self._merge_similar_facts(all_facts)
        
        return DistillationResult(
            facts=merged_facts,
            summary=self._extract_summary(markdown_text),
            raw_response=""
        )
    
    def _normalize_fact_text(self, text: str) -> str:
        """标准化事实文本用于去重"""
        return re.sub(r'\s+', ' ', text.lower().strip())
    
    def _merge_similar_facts(self, facts: List[AtomicFact]) -> List[AtomicFact]:
        """合并相似事实"""
        # 简单实现: 按文本长度排序，保留最详细的版本
        unique_facts = {}
        for fact in facts:
            key = fact.text[:50]  # 使用前50字符作为key
            if key not in unique_facts or len(fact.text) > len(unique_facts[key].text):
                unique_facts[key] = fact
        return list(unique_facts.values())
```

#### 4.1.4 实施步骤

| 步骤 | 任务 | 预估时间 | 产出物 |
|-----|-----|---------|--------|
| 1 | 创建SemanticChunker类 | 2h | `core/chunker.py` |
| 2 | 实现标题分割 | 1h | _split_by_headers方法 |
| 3 | 实现语义分块 | 1.5h | _chunk_section方法 |
| 4 | 实现重叠窗口 | 1h | _add_overlap方法 |
| 5 | 修改DistillerAgent | 1.5h | 集成分块策略 |
| 6 | 编写测试 | 1h | `tests/test_chunker.py` |

#### 4.1.5 验收标准

- [ ] 支持按标题分割文档
- [ ] 支持滑动窗口重叠
- [ ] 不在句子中间截断
- [ ] 跨块事实能正确合并
- [ ] 测试覆盖率 > 80%

---

### 4.2 P1-2: 重排序机制

#### 4.2.1 问题描述

当前检索结果仅按向量相似度排序，可能存在：

1. **语义漂移**: 向量相似但实际不相关
2. **遗漏重要结果**: 相关性高但向量距离远
3. **缺乏多样性**: 结果过于集中

#### 4.2.2 可行性分析

| 维度 | 评估 | 说明 |
|-----|-----|-----|
| 技术可行性 | ✅ 高 | 重排序算法成熟 |
| 资源需求 | ⚠️ 中 | 需要额外LLM调用 |
| 兼容性 | ✅ 高 | 可插拔实现 |
| 风险 | ✅ 低 | 不影响现有流程 |

#### 4.2.3 最优实现方案

**方案选择**: Cross-Encoder重排序 + 多样性重排

```python
# core/reranker.py (新文件)

from typing import List, Dict, Any
from dataclasses import dataclass
import httpx
import os

@dataclass
class RerankResult:
    """重排序结果"""
    id: str
    text: str
    original_score: float
    rerank_score: float
    final_score: float
    payload: Dict[str, Any]


class LLMReranker:
    """基于LLM的重排序器"""
    
    def __init__(self, api_key: str = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.model = model
        self.api_base = "https://api.deepseek.com"
    
    async def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> List[RerankResult]:
        """
        使用LLM对候选结果重排序
        
        策略: 让LLM对每个候选打分
        """
        if not candidates:
            return []
        
        # 批量打分
        scored_results = []
        
        for i, candidate in enumerate(candidates[:20]):  # 最多重排20个
            score = await self._score_relevance(
                query=query,
                text=candidate.get("text", ""),
                context=candidate.get("source_url", "")
            )
            
            scored_results.append(RerankResult(
                id=candidate.get("id", str(i)),
                text=candidate.get("text", ""),
                original_score=candidate.get("score", 0.0),
                rerank_score=score,
                final_score=0.0,  # 后续计算
                payload=candidate
            ))
        
        # 计算最终分数 (原始分数 + 重排序分数的加权)
        for result in scored_results:
            result.final_score = (
                0.3 * result.original_score + 
                0.7 * result.rerank_score
            )
        
        # 按最终分数排序
        scored_results.sort(key=lambda x: x.final_score, reverse=True)
        
        return scored_results[:top_k]
    
    async def _score_relevance(
        self,
        query: str,
        text: str,
        context: str = ""
    ) -> float:
        """使用LLM对单个文档打分"""
        
        prompt = f"""请判断以下内容与查询的相关性，并给出0.0到1.0的分数。

查询: {query}

内容: {text[:500]}

评分标准:
- 1.0: 内容直接回答查询，包含关键信息
- 0.7: 内容与查询相关，提供有用背景
- 0.4: 内容间接相关，可能有用
- 0.0: 内容与查询无关

请只返回一个数字分数，不要有其他文字:"""

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.api_base}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 10,
                        "temperature": 0.0
                    }
                )
                response.raise_for_status()
                data = response.json()
                
                content = data.get("choices", [{}])[0].get("message", {}).get("content", "0.5")
                score = float(content.strip())
                return max(0.0, min(1.0, score))
                
        except Exception as e:
            print(f"[LLMReranker] Error scoring: {e}")
            return 0.5


class DiversityReranker:
    """多样性重排序器"""
    
    def __init__(self, diversity_threshold: float = 0.8):
        self.diversity_threshold = diversity_threshold
    
    def rerank(
        self,
        candidates: List[RerankResult],
        embeddings: List[List[float]] = None
    ) -> List[RerankResult]:
        """
        多样性重排序 (MMR算法)
        
        确保结果多样性，避免过于相似的结果
        """
        if not candidates or len(candidates) <= 1:
            return candidates
        
        selected = [candidates[0]]  # 选择最高分的
        remaining = candidates[1:]
        
        while remaining and len(selected) < len(candidates):
            best_candidate = None
            best_score = -1
            
            for candidate in remaining:
                # 计算与已选结果的最大相似度
                max_sim = self._max_similarity_to_selected(
                    candidate, selected, embeddings
                )
                
                # MMR分数 = 相关性 - λ * 最大相似度
                mmr_score = 0.7 * candidate.final_score - 0.3 * max_sim
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                remaining.remove(best_candidate)
        
        return selected
    
    def _max_similarity_to_selected(
        self,
        candidate: RerankResult,
        selected: List[RerankResult],
        embeddings: List[List[float]] = None
    ) -> float:
        """计算候选与已选结果的最大相似度"""
        # 简化实现: 基于文本重叠
        max_sim = 0.0
        candidate_words = set(candidate.text.lower().split())
        
        for s in selected:
            selected_words = set(s.text.lower().split())
            overlap = len(candidate_words & selected_words)
            union = len(candidate_words | selected_words)
            sim = overlap / union if union > 0 else 0.0
            max_sim = max(max_sim, sim)
        
        return max_sim


# core/rag_retriever.py 修改

class RAGRetriever:
    
    def __init__(self, knowledge_manager: KnowledgeManager, use_reranker: bool = True):
        self.km = knowledge_manager
        self.llm_reranker = LLMReranker() if use_reranker else None
        self.diversity_reranker = DiversityReranker()
    
    async def retrieve_for_report(self, ...) -> RetrievedContext:
        # ... 现有检索逻辑
        
        # 重排序
        if self.llm_reranker:
            reranked = await self.llm_reranker.rerank(
                query=root_query,
                candidates=all_results,
                top_k=max_facts
            )
            
            # 多样性重排
            final_results = self.diversity_reranker.rerank(reranked)
            
            all_results = [
                {
                    "id": r.id,
                    "text": r.text,
                    "score": r.final_score,
                    "original_score": r.original_score,
                    **r.payload
                }
                for r in final_results
            ]
        
        # ... 返回结果
```

#### 4.2.4 实施步骤

| 步骤 | 任务 | 预估时间 | 产出物 |
|-----|-----|---------|--------|
| 1 | 创建LLMReranker类 | 1.5h | `core/reranker.py` |
| 2 | 实现相关性打分 | 1h | _score_relevance方法 |
| 3 | 创建DiversityReranker | 1h | MMR算法实现 |
| 4 | 集成到RAGRetriever | 1h | 修改retrieve_for_report |
| 5 | 编写测试 | 1h | `tests/test_reranker.py` |

#### 4.2.5 验收标准

- [ ] LLM重排序能提升检索相关性
- [ ] 多样性重排能减少重复结果
- [ ] 重排序过程有详细日志
- [ ] 支持配置开关控制是否启用
- [ ] 测试覆盖率 > 80%

---

### 4.3 P1-3: 混合检索

#### 4.3.1 问题描述

当前仅使用向量检索，存在以下问题：

1. **精确匹配失效**: 向量检索对关键词匹配不敏感
2. **专业术语检索差**: 嵌入模型可能无法理解专业术语
3. **召回率受限**: 单一检索方式覆盖面有限

#### 4.3.2 可行性分析

| 维度 | 评估 | 说明 |
|-----|-----|-----|
| 技术可行性 | ✅ 高 | BM25算法成熟，有现成库 |
| 资源需求 | ⚠️ 中 | 需要维护倒排索引 |
| 兼容性 | ✅ 高 | 可与向量检索并行 |
| 风险 | ✅ 低 | 不影响现有流程 |

#### 4.3.3 最优实现方案

**方案选择**: BM25 + 向量检索 + RRF融合

```python
# core/hybrid_retriever.py (新文件)

from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
import math
from collections import Counter

@dataclass
class HybridSearchResult:
    id: str
    text: str
    vector_score: float
    keyword_score: float
    combined_score: float
    payload: Dict[str, Any]


class BM25Index:
    """BM25倒排索引"""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_lengths: Dict[str, int] = {}
        self.avg_doc_length: float = 0.0
        self.inverted_index: Dict[str, Dict[str, int]] = {}  # term -> {doc_id: tf}
        self.doc_count: int = 0
        self.idf_cache: Dict[str, float] = {}
    
    def index(self, documents: List[Dict[str, Any]]):
        """构建索引"""
        self.doc_count = len(documents)
        total_length = 0
        
        for doc in documents:
            doc_id = doc.get("id")
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            
            self.doc_lengths[doc_id] = len(tokens)
            total_length += len(tokens)
            
            # 构建倒排索引
            term_freq = Counter(tokens)
            for term, freq in term_freq.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = {}
                self.inverted_index[term][doc_id] = freq
        
        self.avg_doc_length = total_length / self.doc_count if self.doc_count > 0 else 0
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """BM25搜索"""
        query_tokens = self._tokenize(query)
        scores: Dict[str, float] = {}
        
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            
            idf = self._compute_idf(term)
            
            for doc_id, tf in self.inverted_index[term].items():
                doc_length = self.doc_lengths.get(doc_id, 0)
                
                # BM25公式
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_length / self.avg_doc_length)
                score = idf * numerator / denominator
                
                if doc_id not in scores:
                    scores[doc_id] = 0.0
                scores[doc_id] += score
        
        # 排序
        sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def _compute_idf(self, term: str) -> float:
        """计算IDF"""
        if term in self.idf_cache:
            return self.idf_cache[term]
        
        df = len(self.inverted_index.get(term, {}))
        idf = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)
        
        self.idf_cache[term] = idf
        return idf
    
    def _tokenize(self, text: str) -> List[str]:
        """分词 (简单实现)"""
        import re
        # 中文按字符分割，英文按空格分割
        chinese = re.findall(r'[\u4e00-\u9fff]+', text)
        english = re.findall(r'[a-zA-Z]+', text)
        
        tokens = []
        for c in chinese:
            tokens.extend(list(c))
        for e in english:
            tokens.append(e.lower())
        
        return tokens


class HybridRetriever:
    """混合检索器"""
    
    def __init__(
        self,
        knowledge_manager: 'KnowledgeManager',
        vector_weight: float = 0.5,
        keyword_weight: float = 0.5
    ):
        self.km = knowledge_manager
        self.bm25_index = BM25Index()
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        self._indexed = False
    
    async def build_index(self):
        """构建混合索引"""
        # 获取所有文档
        all_facts = await self.km.search_facts("", limit=1000, threshold=0.0)
        
        if all_facts:
            self.bm25_index.index(all_facts)
            self._indexed = True
            print(f"[HybridRetriever] Indexed {len(all_facts)} documents")
    
    async def search(
        self,
        query: str,
        top_k: int = 10,
        threshold: float = 0.0
    ) -> List[HybridSearchResult]:
        """
        混合检索
        
        1. 向量检索
        2. BM25检索
        3. RRF融合
        """
        if not self._indexed:
            await self.build_index()
        
        # 向量检索
        vector_results = await self.km.search_facts(
            query=query,
            limit=top_k * 2,
            threshold=0.0
        )
        vector_ranking = {r["id"]: i for i, r in enumerate(vector_results)}
        
        # BM25检索
        bm25_results = self.bm25_index.search(query, top_k * 2)
        bm25_ranking = {r[0]: i for i, r in enumerate(bm25_results)}
        
        # RRF融合
        all_doc_ids = set(vector_ranking.keys()) | set(bm25_ranking.keys())
        rrf_scores = {}
        
        k = 60  # RRF参数
        for doc_id in all_doc_ids:
            vector_rank = vector_ranking.get(doc_id, len(vector_ranking))
            bm25_rank = bm25_ranking.get(doc_id, len(bm25_ranking))
            
            rrf_score = 1 / (k + vector_rank) + 1 / (k + bm25_rank)
            rrf_scores[doc_id] = rrf_score
        
        # 排序并构建结果
        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for doc_id, combined_score in sorted_ids[:top_k]:
            # 获取文档详情
            doc = await self.km.get_fact_by_id(doc_id)
            if doc:
                results.append(HybridSearchResult(
                    id=doc_id,
                    text=doc.get("text", ""),
                    vector_score=1 / (k + vector_ranking.get(doc_id, len(vector_ranking))),
                    keyword_score=1 / (k + bm25_ranking.get(doc_id, len(bm25_ranking))),
                    combined_score=combined_score,
                    payload=doc
                ))
        
        return results


# RRF (Reciprocal Rank Fusion) 算法说明:
# score(d) = Σ 1/(k + rank(d))
# 其中 k 是平滑参数，通常取 60
```

#### 4.3.4 实施步骤

| 步骤 | 任务 | 预估时间 | 产出物 |
|-----|-----|---------|--------|
| 1 | 创建BM25Index类 | 2h | `core/hybrid_retriever.py` |
| 2 | 实现倒排索引构建 | 1h | index方法 |
| 3 | 实现BM25搜索 | 1.5h | search方法 |
| 4 | 创建HybridRetriever | 1.5h | RRF融合 |
| 5 | 集成到RAGRetriever | 1h | 修改检索流程 |
| 6 | 编写测试 | 1h | `tests/test_hybrid_retriever.py` |

#### 4.3.5 验收标准

- [ ] BM25索引能正确构建
- [ ] 关键词检索能召回精确匹配结果
- [ ] RRF融合能综合两种检索结果
- [ ] 混合检索召回率高于单一检索
- [ ] 测试覆盖率 > 80%

---

## 5. P2级任务详解

### 5.1 P2-1: 来源可信度评估

#### 5.1.1 问题描述

当前事实的 `confidence` 仅由 LLM 主观判断，缺乏客观评估标准。

#### 5.1.2 最优实现方案

```python
# core/credibility.py (新文件)

from typing import Dict, List
from dataclasses import dataclass
import re

@dataclass
class CredibilityScore:
    overall: float
    domain_score: float
    content_score: float
    citation_score: float
    freshness_score: float
    details: Dict[str, str]


class SourceCredibilityScorer:
    """来源可信度评估器"""
    
    TRUSTED_DOMAINS = {
        # 政府机构
        "gov.cn": 0.95,
        "gov.uk": 0.95,
        "gov": 0.90,
        # 教育机构
        "edu.cn": 0.90,
        "edu": 0.85,
        # 新闻机构
        "reuters.com": 0.85,
        "bloomberg.com": 0.85,
        "ft.com": 0.85,
        "wsj.com": 0.80,
        # 学术
        "arxiv.org": 0.80,
        "nature.com": 0.90,
        "science.org": 0.90,
        # 科技公司
        "openai.com": 0.75,
        "deepmind.com": 0.75,
    }
    
    SUSPICIOUS_DOMAINS = {
        "blogspot.com": 0.40,
        "wordpress.com": 0.45,
        "medium.com": 0.50,
        "substack.com": 0.50,
    }
    
    def score(self, source_url: str, content: str, published_date: str = None) -> CredibilityScore:
        """评估来源可信度"""
        
        domain_score = self._score_domain(source_url)
        content_score = self._score_content(content)
        citation_score = self._score_citations(content)
        freshness_score = self._score_freshness(published_date)
        
        # 加权综合分数
        overall = (
            0.35 * domain_score +
            0.25 * content_score +
            0.25 * citation_score +
            0.15 * freshness_score
        )
        
        return CredibilityScore(
            overall=overall,
            domain_score=domain_score,
            content_score=content_score,
            citation_score=citation_score,
            freshness_score=freshness_score,
            details={
                "domain": self._extract_domain(source_url),
                "has_data": str(self._has_numerical_data(content)),
                "has_citations": str(citation_score > 0.5),
            }
        )
    
    def _score_domain(self, url: str) -> float:
        """评估域名可信度"""
        domain = self._extract_domain(url)
        
        for trusted, score in self.TRUSTED_DOMAINS.items():
            if trusted in domain:
                return score
        
        for suspicious, score in self.SUSPICIOUS_DOMAINS.items():
            if suspicious in domain:
                return score
        
        return 0.60  # 默认分数
    
    def _score_content(self, content: str) -> float:
        """评估内容质量"""
        score = 0.5
        
        # 有数值数据
        if self._has_numerical_data(content):
            score += 0.15
        
        # 有专业术语
        if self._has_technical_terms(content):
            score += 0.10
        
        # 内容长度适中
        if 200 < len(content) < 5000:
            score += 0.10
        
        # 有结构化内容
        if "##" in content or "1." in content:
            score += 0.10
        
        # 无明显广告
        if not self._has_ads(content):
            score += 0.05
        
        return min(1.0, score)
    
    def _score_citations(self, content: str) -> float:
        """评估引用质量"""
        # 检查是否有引用标记
        citation_patterns = [
            r'\[\d+\]',  # [1], [2]
            r'\(.*?\d{4}.*?\)',  # (Author, 2023)
            r'根据.*?报告',
            r'数据显示',
            r'来源:',
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, content))
        
        if citation_count >= 3:
            return 0.90
        elif citation_count >= 1:
            return 0.70
        else:
            return 0.40
    
    def _score_freshness(self, published_date: str) -> float:
        """评估时效性"""
        if not published_date:
            return 0.50
        
        from datetime import datetime, timedelta
        
        try:
            pub_date = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
            age = datetime.now(pub_date.tzinfo) - pub_date
            
            if age < timedelta(days=30):
                return 1.0
            elif age < timedelta(days=180):
                return 0.85
            elif age < timedelta(days=365):
                return 0.70
            else:
                return 0.50
        except:
            return 0.50
    
    def _extract_domain(self, url: str) -> str:
        """提取域名"""
        import re
        match = re.search(r'://([^/]+)', url)
        return match.group(1) if match else ""
    
    def _has_numerical_data(self, content: str) -> bool:
        """检查是否有数值数据"""
        numbers = re.findall(r'\d+\.?\d*[%亿万美元]?', content)
        return len(numbers) >= 3
    
    def _has_technical_terms(self, content: str) -> bool:
        """检查是否有专业术语"""
        terms = ['纳米', '制程', 'GAA', 'FinFET', 'EUV', 'AI', 'GPU', 'CPU']
        return any(term in content for term in terms)
    
    def _has_ads(self, content: str) -> bool:
        """检查是否有广告"""
        ad_patterns = ['广告', '赞助', '推广', 'advertisement']
        return any(ad in content.lower() for ad in ad_patterns)
```

#### 5.1.3 实施步骤

| 步骤 | 任务 | 预估时间 |
|-----|-----|---------|
| 1 | 创建SourceCredibilityScorer | 1.5h |
| 2 | 集成到DistillerAgent | 1h |
| 3 | 更新confidence计算逻辑 | 1h |
| 4 | 编写测试 | 1h |

---

### 5.2 P2-2: 版本管理

#### 5.2.1 问题描述

事实没有版本历史，无法追溯变更，无法回滚。

#### 5.2.2 最优实现方案

```python
# core/version_manager.py (新文件)

from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json

@dataclass
class FactVersion:
    version: int
    text: str
    confidence: float
    source_url: str
    changed_at: datetime
    change_reason: str
    previous_version: Optional[int] = None


class FactVersionManager:
    """事实版本管理器"""
    
    def __init__(self, storage_path: str = "./fact_versions"):
        self.storage_path = storage_path
        self._versions: Dict[str, List[FactVersion]] = {}
    
    def create_version(
        self,
        fact_id: str,
        text: str,
        confidence: float,
        source_url: str,
        reason: str = "initial"
    ) -> FactVersion:
        """创建新版本"""
        existing = self._versions.get(fact_id, [])
        
        new_version = FactVersion(
            version=len(existing) + 1,
            text=text,
            confidence=confidence,
            source_url=source_url,
            changed_at=datetime.now(),
            change_reason=reason,
            previous_version=existing[-1].version if existing else None
        )
        
        if fact_id not in self._versions:
            self._versions[fact_id] = []
        self._versions[fact_id].append(new_version)
        
        return new_version
    
    def get_version(self, fact_id: str, version: int = None) -> Optional[FactVersion]:
        """获取指定版本"""
        versions = self._versions.get(fact_id, [])
        if not versions:
            return None
        
        if version is None:
            return versions[-1]
        
        for v in versions:
            if v.version == version:
                return v
        return None
    
    def get_history(self, fact_id: str) -> List[FactVersion]:
        """获取版本历史"""
        return self._versions.get(fact_id, [])
    
    def rollback(self, fact_id: str, target_version: int) -> Optional[FactVersion]:
        """回滚到指定版本"""
        target = self.get_version(fact_id, target_version)
        if target:
            # 创建一个新版本，内容是目标版本的内容
            return self.create_version(
                fact_id=fact_id,
                text=target.text,
                confidence=target.confidence,
                source_url=target.source_url,
                reason=f"rollback to version {target_version}"
            )
        return None
```

#### 5.2.3 实施步骤

| 步骤 | 任务 | 预估时间 |
|-----|-----|---------|
| 1 | 创建FactVersionManager | 1.5h |
| 2 | 集成到KnowledgeManager | 1h |
| 3 | 添加版本历史API | 1h |
| 4 | 编写测试 | 1h |

---

## 6. 实施路线图

### 6.1 阶段划分

```
┌─────────────────────────────────────────────────────────────────────┐
│                         实施路线图                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  阶段1: 基础设施 (P0) ─────────────────────────────────────────────  │
│  │                                                                  │
│  ├── P0-1: 向量数据库集成                                           │
│  │   └── 产出: VectorStoreAdapter, KnowledgeManager重构             │
│  │                                                                  │
│  └── P0-2: RAG检索实现                                              │
│      └── 产出: RAGRetriever, Writer改造                             │
│                                                                     │
│  阶段2: 效果提升 (P1) ─────────────────────────────────────────────  │
│  │                                                                  │
│  ├── P1-1: 文档分块策略                                             │
│  │   └── 产出: SemanticChunker, DistillerAgent改造                  │
│  │                                                                  │
│  ├── P1-2: 重排序机制                                               │
│  │   └── 产出: LLMReranker, DiversityReranker                       │
│  │                                                                  │
│  └── P1-3: 混合检索                                                 │
│      └── 产出: BM25Index, HybridRetriever                           │
│                                                                     │
│  阶段3: 质量保障 (P2) ─────────────────────────────────────────────  │
│  │                                                                  │
│  ├── P2-1: 来源可信度评估                                           │
│  │   └── 产出: SourceCredibilityScorer                              │
│  │                                                                  │
│  └── P2-2: 版本管理                                                 │
│      └── 产出: FactVersionManager                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 时间规划

| 阶段 | 任务 | 预估时间 | 依赖 |
|-----|-----|---------|-----|
| **阶段1** | P0-1 向量数据库集成 | 6.5h | 无 |
| | P0-2 RAG检索实现 | 5h | P0-1 |
| **阶段2** | P1-1 文档分块策略 | 8h | 无 |
| | P1-2 重排序机制 | 5.5h | P0-2 |
| | P1-3 混合检索 | 8h | P0-1 |
| **阶段3** | P2-1 来源可信度评估 | 4.5h | 无 |
| | P2-2 版本管理 | 4.5h | 无 |
| **总计** | | **42h** | |

### 6.3 里程碑

| 里程碑 | 完成标准 | 预计完成 |
|-------|---------|---------|
| M1 | 向量数据库集成完成，测试通过 | 阶段1开始后8h |
| M2 | RAG检索可用，报告质量提升 | 阶段1开始后13h |
| M3 | 分块策略上线，事实提取质量提升 | 阶段2开始后8h |
| M4 | 重排序+混合检索完成，检索效果提升 | 阶段2开始后21.5h |
| M5 | 全部P0-P2任务完成 | 阶段3开始后9h |

---

## 7. 风险评估与应对

### 7.1 技术风险

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|-----|---------|
| Qdrant部署失败 | 低 | 高 | 提供内存存储作为fallback |
| LLM API限流 | 中 | 中 | 实现请求队列和重试机制 |
| 分块导致事实断裂 | 中 | 中 | 添加重叠窗口和跨块合并 |
| 重排序延迟过高 | 中 | 低 | 支持配置开关，可禁用 |

### 7.2 资源风险

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|-----|---------|
| API成本增加 | 高 | 中 | 实现缓存机制，优化调用次数 |
| 存储空间不足 | 低 | 中 | 实现数据清理策略 |
| 内存占用过高 | 中 | 中 | 使用Qdrant替代内存存储 |

### 7.3 兼容性风险

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|-----|---------|
| 现有测试失败 | 中 | 高 | 保持向后兼容，逐步迁移 |
| 配置迁移问题 | 低 | 中 | 提供迁移脚本和文档 |

---

## 附录

### A. 文件变更清单

| 文件 | 操作 | 说明 |
|-----|-----|-----|
| `core/vector_store_adapter.py` | 新增 | 向量存储适配器接口 |
| `core/knowledge.py` | 修改 | 重构以支持多种存储后端 |
| `core/rag_retriever.py` | 新增 | RAG检索器 |
| `core/chunker.py` | 新增 | 语义分块器 |
| `core/reranker.py` | 新增 | 重排序器 |
| `core/hybrid_retriever.py` | 新增 | 混合检索器 |
| `core/credibility.py` | 新增 | 可信度评估器 |
| `core/version_manager.py` | 新增 | 版本管理器 |
| `agents/distiller.py` | 修改 | 集成分块策略 |
| `core/graph.py` | 修改 | 集成RAG检索 |
| `core/config.py` | 修改 | 添加新配置项 |

### B. 配置项清单

```python
# core/config.py 新增配置

class RAGConfig:
    # 向量检索
    VECTOR_SEARCH_LIMIT: int = 20
    VECTOR_SCORE_THRESHOLD: float = 0.3
    
    # 重排序
    ENABLE_RERANKER: bool = True
    RERANK_TOP_K: int = 10
    DIVERSITY_THRESHOLD: float = 0.8
    
    # 混合检索
    ENABLE_HYBRID_SEARCH: bool = True
    VECTOR_WEIGHT: float = 0.5
    KEYWORD_WEIGHT: float = 0.5
    
    # 分块
    CHUNK_MAX_TOKENS: int = 512
    CHUNK_OVERLAP_TOKENS: int = 50
    CHUNK_MIN_TOKENS: int = 100
    
    # 可信度
    ENABLE_CREDIBILITY_SCORING: bool = True
    MIN_CREDIBILITY_THRESHOLD: float = 0.5
```

### C. 测试清单

| 测试文件 | 测试内容 |
|---------|---------|
| `tests/test_vector_adapter.py` | 向量存储适配器 |
| `tests/test_rag_retriever.py` | RAG检索器 |
| `tests/test_chunker.py` | 语义分块器 |
| `tests/test_reranker.py` | 重排序器 |
| `tests/test_hybrid_retriever.py` | 混合检索器 |
| `tests/test_credibility.py` | 可信度评估器 |
| `tests/test_version_manager.py` | 版本管理器 |

---

> 文档结束  
> 下一步: 按照优先级顺序开始实施 P0-1 任务
