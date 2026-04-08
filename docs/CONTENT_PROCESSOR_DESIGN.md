# 网页内容处理优化方案设计

> 设计日期: 2026-04-05  
> 目标: 替代截断策略，实现智能内容处理流程

---

## 目录

1. [当前架构分析](#1-当前架构分析)
2. [新架构设计](#2-新架构设计)
3. [三大处理模块设计](#3-三大处理模块设计)
4. [数据流设计](#4-数据流设计)
5. [接口与数据结构](#5-接口与数据结构)
6. [实施计划](#6-实施计划)

---

## 1. 当前架构分析

### 1.1 现有流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           当前数据流                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  MCPGateway.search(query)                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  SearchResult[] (URL列表)                                                   │
│         │                                                                   │
│         ▼                                                                   │
│  SmartScraper.scrape_batch(urls)                                            │
│         │                                                                   │
│         ├── HTTP请求 → HTML                                                 │
│         ├── Jina Reader API → Markdown (简单转换)                           │
│         ├── _level1_heuristic_clean() → 基础降噪                            │
│         │                                                                   │
│         ▼                                                                   │
│  ScrapedData { markdown, title, url }                                       │
│         │                                                                   │
│         ▼                                                                   │
│  DistillerAgent.distill(markdown[:8000])  ← 截断！                          │
│         │                                                                   │
│         ▼                                                                   │
│  AtomicFact[]                                                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 现有问题

| 问题 | 说明 |
|-----|------|
| **截断策略** | 直接截断至8000字符，丢失大量信息 |
| **缺乏元数据** | 没有提取发布日期、作者等关键信息 |
| **无语义分块** | 不按语义边界分割，可能切断完整语义 |
| **无相关性筛选** | 抓取的内容可能只有20%与查询相关 |
| **无跨源对齐** | 多个来源的信息没有归类整合 |

---

## 2. 新架构设计

### 2.1 新数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           新数据流设计                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Step 0: 搜索                                                               │
│  ─────────────────                                                          │
│  MCPGateway.search(query) → SearchResult[]                                  │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  Step 1: 结构化清洗与视图转换 (新增)                                          │
│  ─────────────────────────────────────                                      │
│  ContentTransformer.transform(html) → StructuredContent                     │
│  • HTML → Markdown (Readability/Firecrawl)                                  │
│  • 元数据提取 (日期、作者、标题)                                              │
│  • 导航/广告/脚本过滤                                                        │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  Step 2: 语义分块与相关性筛选 (新增)                                          │
│  ─────────────────────────────────────                                      │
│  SemanticChunker.chunk_and_filter(content, query) → RelevantChunk[]         │
│  • 语义分块 (RecursiveCharacterTextSplitter)                                │
│  • 向量相似度计算                                                            │
│  • 低相关度块过滤                                                            │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  Step 3: 跨源信息对齐 (新增)                                                 │
│  ─────────────────────────────────────                                      │
│  CrossSourceAligner.align(chunks_from_all_sources) → AlignedTopics          │
│  • 子话题识别                                                               │
│  • 按话题归类                                                               │
│  • 信源汇总                                                                 │
│                                                                             │
│         │                                                                   │
│         ▼                                                                   │
│                                                                             │
│  Step 4: 事实提取                                                           │
│  ─────────────────────────────────────                                      │
│  DistillerAgent.distill(aligned_topics) → AtomicFact[]                      │
│  • 无需截断，传入的是已筛选的高质量内容                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 模块职责划分

| 模块 | 职责 | 输入 | 输出 |
|-----|------|------|------|
| **ContentTransformer** | 结构化清洗 | HTML | StructuredContent |
| **SemanticChunker** | 分块+筛选 | StructuredContent + Query | RelevantChunk[] |
| **CrossSourceAligner** | 跨源对齐 | RelevantChunk[][] | AlignedTopics |
| **DistillerAgent** | 事实提取 | AlignedTopics | AtomicFact[] |

---

## 3. 三大处理模块设计

### 3.1 模块一: 结构化清洗与视图转换 (ContentTransformer)

#### 3.1.1 功能设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ContentTransformer 设计                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: HTML (原始网页)                                                       │
│                                                                             │
│  处理步骤:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. HTML → Markdown                                                  │    │
│  │     • 使用 Readability.js 或 Firecrawl                               │    │
│  │     • 自动过滤: <nav>, <footer>, <script>, 广告栏                     │    │
│  │     • 保留: <article>, <main>, 核心内容                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. 元数据提取                                                       │    │
│  │     • 发布日期: <meta name="date">, <time>, JSON-LD                  │    │
│  │     • 作者: <meta name="author">, .author, .byline                   │    │
│  │     • 来源URL: 原始URL                                               │    │
│  │     • 网页标题: <title>, <h1>, <meta property="og:title">            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. 内容清洗                                                         │    │
│  │     • 移除多余空白                                                   │    │
│  │     • 规范化标题层级                                                 │    │
│  │     • 保留图片alt文本                                                │    │
│  │     • 保留链接文本                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  输出: StructuredContent { markdown, metadata }                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.1.2 数据结构

```python
@dataclass
class ContentMetadata:
    """内容元数据"""
    title: str                    # 网页标题
    url: str                      # 来源URL
    author: Optional[str]         # 作者
    publish_date: Optional[str]   # 发布日期
    fetch_date: str               # 抓取日期
    site_name: Optional[str]      # 网站名称
    language: Optional[str]       # 语言


@dataclass
class StructuredContent:
    """结构化内容"""
    markdown: str                 # 清洗后的Markdown
    metadata: ContentMetadata     # 元数据
    original_length: int          # 原始长度
    cleaned_length: int           # 清洗后长度
    sections: List[str]           # 章节列表 (用于后续分块)
```

#### 3.1.3 实现方案

**方案A: 使用 Jina Reader API (当前已有)**

```python
# 当前实现，需要增强元数据提取
class ContentTransformer:
    JINA_READER_URL = "https://r.jina.ai/"
    
    async def transform(self, url: str, html: str = None) -> StructuredContent:
        # 使用 Jina Reader 获取 Markdown
        markdown = await self._fetch_via_jina(url)
        
        # 增强元数据提取
        metadata = await self._extract_metadata(url, html or markdown)
        
        # 内容清洗
        cleaned = self._clean_content(markdown)
        
        return StructuredContent(
            markdown=cleaned,
            metadata=metadata,
            ...
        )
```

**方案B: 使用 Firecrawl (推荐)**

```python
# Firecrawl 提供更完整的元数据
class ContentTransformer:
    FIRECRAWL_URL = "https://api.firecrawl.dev/v1/scrape"
    
    async def transform(self, url: str) -> StructuredContent:
        response = await self._call_firecrawl(url)
        
        return StructuredContent(
            markdown=response["markdown"],
            metadata=ContentMetadata(
                title=response["metadata"]["title"],
                author=response["metadata"]["author"],
                publish_date=response["metadata"]["publishedTime"],
                ...
            ),
            ...
        )
```

**推荐**: 方案B，Firecrawl 提供更完整的元数据提取。

---

### 3.2 模块二: 语义分块与相关性筛选 (SemanticChunker)

#### 3.2.1 功能设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SemanticChunker 设计                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: StructuredContent + Query (搜索意图)                                  │
│                                                                             │
│  处理步骤:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. 语义分块                                                         │    │
│  │     • 使用 RecursiveCharacterTextSplitter                           │    │
│  │     • 按 ["\n\n", "\n", "。", ".", " "] 优先级分割                   │    │
│  │     • 块大小: 1000-2000 字符                                         │    │
│  │     • 重叠: 200 字符 (保持上下文)                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. 向量相似度计算                                                   │    │
│  │     • 计算 Query 的 embedding                                       │    │
│  │     • 计算每个 Chunk 的 embedding                                   │    │
│  │     • 计算余弦相似度                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. 相关性筛选                                                       │    │
│  │     • 设定阈值: 0.3 (可配置)                                         │    │
│  │     • 丢弃低于阈值的块                                               │    │
│  │     • 保留 Top-K 高相关度块                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  输出: RelevantChunk[] (已筛选的高相关度块)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.2.2 数据结构

```python
@dataclass
class RelevantChunk:
    """相关性筛选后的文本块"""
    text: str                     # 文本内容
    chunk_id: str                 # 块ID
    relevance_score: float        # 相关性分数 (0.0-1.0)
    source_url: str               # 来源URL
    source_title: str             # 来源标题
    metadata: ContentMetadata     # 元数据
    position: int                 # 在原文中的位置


@dataclass
class ChunkingResult:
    """分块结果"""
    chunks: List[RelevantChunk]   # 筛选后的块
    total_chunks: int             # 总块数
    filtered_chunks: int          # 过滤掉的块数
    avg_relevance: float          # 平均相关性
```

#### 3.2.3 实现方案

```python
class SemanticChunker:
    """语义分块与相关性筛选器"""
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        relevance_threshold: float = 0.3,
        top_k: int = 20,
        embedding_model: str = "bge-large-zh"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.embedding_model = embedding_model
    
    async def chunk_and_filter(
        self,
        content: StructuredContent,
        query: str
    ) -> ChunkingResult:
        """分块并筛选相关内容"""
        
        # Step 1: 语义分块
        chunks = self._split_by_semantic(content.markdown)
        
        # Step 2: 计算相关性
        query_embedding = await self._get_embedding(query)
        
        scored_chunks = []
        for i, chunk_text in enumerate(chunks):
            chunk_embedding = await self._get_embedding(chunk_text)
            relevance = self._cosine_similarity(query_embedding, chunk_embedding)
            
            if relevance >= self.relevance_threshold:
                scored_chunks.append(RelevantChunk(
                    text=chunk_text,
                    chunk_id=f"{content.metadata.url}#{i}",
                    relevance_score=relevance,
                    source_url=content.metadata.url,
                    source_title=content.metadata.title,
                    metadata=content.metadata,
                    position=i
                ))
        
        # Step 3: 排序并取Top-K
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        top_chunks = scored_chunks[:self.top_k]
        
        return ChunkingResult(
            chunks=top_chunks,
            total_chunks=len(chunks),
            filtered_chunks=len(chunks) - len(scored_chunks),
            avg_relevance=sum(c.relevance_score for c in top_chunks) / len(top_chunks) if top_chunks else 0
        )
    
    def _split_by_semantic(self, text: str) -> List[str]:
        """语义分块 - 使用递归字符分割"""
        separators = ["\n\n", "\n", "。", ".", "！", "!", "？", "?", " "]
        
        # 实现递归分割逻辑
        # 类似 LangChain 的 RecursiveCharacterTextSplitter
        ...
```

---

### 3.3 模块三: 跨源信息对齐 (CrossSourceAligner)

#### 3.3.1 功能设计

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CrossSourceAligner 设计                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入: RelevantChunk[][] (来自多个网页的已筛选块)                             │
│                                                                             │
│  处理步骤:                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. 子话题识别                                                       │    │
│  │     • 使用 LLM 分析所有块的内容                                      │    │
│  │     • 提取关键实体和主题                                             │    │
│  │     • 生成子话题列表                                                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  2. 按话题归类                                                       │    │
│  │     • 将每个块分配到最相关的子话题                                   │    │
│  │     • 同一话题的块合并                                               │    │
│  │     • 去重相似内容                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  3. 信源汇总                                                         │    │
│  │     • 每个话题标注所有来源                                           │    │
│  │     • 标注冲突信息                                                   │    │
│  │     • 生成汇总文本                                                   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  输出: AlignedTopics (按话题归类的信源汇总)                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 3.3.2 数据结构

```python
@dataclass
class SubTopic:
    """子话题"""
    topic_id: str                 # 话题ID
    name: str                     # 话题名称
    description: str              # 话题描述
    chunks: List[RelevantChunk]   # 相关块
    sources: List[str]            # 来源URL列表
    summary: str                  # 汇总摘要


@dataclass
class AlignedTopics:
    """对齐后的话题集合"""
    topics: List[SubTopic]        # 子话题列表
    total_sources: int            # 总来源数
    total_chunks: int             # 总块数
    conflicts: List[dict]         # 冲突信息
```

#### 3.3.3 实现方案

```python
class CrossSourceAligner:
    """跨源信息对齐器"""
    
    def __init__(self, llm_client, max_topics: int = 5):
        self.llm_client = llm_client
        self.max_topics = max_topics
    
    async def align(
        self,
        all_chunks: List[List[RelevantChunk]],
        query: str
    ) -> AlignedTopics:
        """跨源对齐"""
        
        # 展平所有块
        flat_chunks = [chunk for chunks in all_chunks for chunk in chunks]
        
        # Step 1: 子话题识别
        topics = await self._identify_topics(flat_chunks, query)
        
        # Step 2: 按话题归类
        for topic in topics:
            topic.chunks = await self._assign_chunks_to_topic(flat_chunks, topic)
            topic.sources = list(set(c.source_url for c in topic.chunks))
        
        # Step 3: 生成汇总
        for topic in topics:
            topic.summary = await self._summarize_topic(topic)
        
        # Step 4: 检测冲突
        conflicts = await self._detect_conflicts(topics)
        
        return AlignedTopics(
            topics=topics,
            total_sources=len(set(c.source_url for c in flat_chunks)),
            total_chunks=len(flat_chunks),
            conflicts=conflicts
        )
    
    async def _identify_topics(
        self,
        chunks: List[RelevantChunk],
        query: str
    ) -> List[SubTopic]:
        """使用LLM识别子话题"""
        
        # 收集所有块的摘要
        chunk_summaries = [c.text[:200] for c in chunks[:20]]
        
        prompt = f"""分析以下内容片段，识别出 {self.max_topics} 个主要子话题。

查询主题: {query}

内容片段:
{chr(10).join(f'{i+1}. {s}' for i, s in enumerate(chunk_summaries))}

请输出JSON格式的子话题列表:
{{
  "topics": [
    {{"name": "话题名称", "description": "话题描述"}},
    ...
  ]
}}"""

        response = await self.llm_client.generate(prompt)
        # 解析响应...
```

---

## 4. 数据流设计

### 4.1 完整数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           完整数据流                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  researcher_async                                                           │
│  │                                                                          │
│  ├── MCPGateway.search(query)                                               │
│  │         │                                                                │
│  │         ▼                                                                │
│  │    SearchResult[] (5个URL)                                               │
│  │                                                                          │
│  ├── ContentTransformer.transform_batch(urls)                               │
│  │         │                                                                │
│  │         ▼                                                                │
│  │    StructuredContent[] (5个结构化内容)                                    │
│  │                                                                          │
│  ├── SemanticChunker.chunk_and_filter_batch(contents, query)                │
│  │         │                                                                │
│  │         ▼                                                                │
│  │    ChunkingResult[] (5个分块结果)                                         │
│  │    - 每个结果包含 Top-K 高相关度块                                        │
│  │    - 平均过滤掉 80% 的低相关内容                                          │
│  │                                                                          │
│  ├── CrossSourceAligner.align(all_chunks, query)                            │
│  │         │                                                                │
│  │         ▼                                                                │
│  │    AlignedTopics (按话题归类的汇总)                                       │
│  │    - 5个子话题                                                           │
│  │    - 每个话题包含多源信息                                                 │
│  │                                                                          │
│  ▼                                                                          │
│  distiller_async                                                            │
│  │                                                                          │
│  ├── DistillerAgent.distill(aligned_topics)                                 │
│  │         │                                                                │
│  │         ▼                                                                │
│  │    AtomicFact[] (原子事实)                                               │
│  │    - 每个事实带有来源引用                                                 │
│  │    - 无需截断，传入的是高质量内容                                         │
│  │                                                                          │
│  ▼                                                                          │
│  writer_async → FinalReport                                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 数据量变化

```
原始数据流:
  5个网页 × 平均 280,000 字符 = 1,400,000 字符
  → 截断至 8,000 字符 = 40,000 字符 (丢失 97%)

新数据流:
  5个网页 × 平均 280,000 字符 = 1,400,000 字符
  → 清洗后: 5 × 140,000 = 700,000 字符 (降噪 50%)
  → 分块筛选: 5 × 20块 × 1500字符 = 150,000 字符 (筛选 79%)
  → 跨源对齐: 5话题 × 10000字符 = 50,000 字符 (汇总)
  → 事实提取: ~100个事实 × 200字符 = 20,000 字符

信息保留率: 20,000 / 1,400,000 = 1.4% (有效信息)
质量提升: 高相关度内容，带来源引用
```

---

## 5. 接口与数据结构

### 5.1 模块接口

```python
# core/content_processor.py

class ContentProcessor:
    """内容处理器 - 整合三大模块"""
    
    def __init__(self, config: ProcessorConfig = None):
        self.transformer = ContentTransformer()
        self.chunker = SemanticChunker()
        self.aligner = CrossSourceAligner()
    
    async def process(
        self,
        urls: List[str],
        query: str
    ) -> AlignedTopics:
        """处理多个URL的内容"""
        
        # Step 1: 结构化清洗
        contents = await self.transformer.transform_batch(urls)
        
        # Step 2: 分块筛选
        chunk_results = await self.chunker.chunk_and_filter_batch(contents, query)
        
        # Step 3: 跨源对齐
        all_chunks = [result.chunks for result in chunk_results]
        aligned = await self.aligner.align(all_chunks, query)
        
        return aligned
```

### 5.2 配置项

```python
# core/config.py

class ContentProcessorConfig:
    # ContentTransformer
    TRANSFORMER_PROVIDER: str = "firecrawl"  # jina / firecrawl
    EXTRACT_METADATA: bool = True
    
    # SemanticChunker
    CHUNK_SIZE: int = 1500
    CHUNK_OVERLAP: int = 200
    RELEVANCE_THRESHOLD: float = 0.3
    TOP_K_CHUNKS: int = 20
    
    # CrossSourceAligner
    MAX_TOPICS: int = 5
    ENABLE_CONFLICT_DETECTION: bool = True
```

---

## 6. 实施计划

### 6.1 阶段划分

| 阶段 | 任务 | 预估时间 | 优先级 |
|-----|------|---------|--------|
| **阶段1** | ContentTransformer 实现 | 4h | P0 |
| **阶段2** | SemanticChunker 实现 | 6h | P0 |
| **阶段3** | CrossSourceAligner 实现 | 6h | P1 |
| **阶段4** | 集成到 graph.py | 2h | P0 |
| **阶段5** | 测试与优化 | 4h | P1 |

### 6.2 文件变更清单

| 文件 | 操作 | 说明 |
|-----|------|------|
| `core/content_transformer.py` | 新增 | 结构化清洗模块 |
| `core/semantic_chunker.py` | 新增 | 语义分块模块 |
| `core/cross_source_aligner.py` | 新增 | 跨源对齐模块 |
| `core/content_processor.py` | 新增 | 整合处理器 |
| `core/graph.py` | 修改 | 集成新流程 |
| `schemas/state.py` | 修改 | 新增数据结构 |
| `core/config.py` | 修改 | 新增配置项 |

### 6.3 依赖项

| 依赖 | 用途 | 安装命令 |
|-----|------|---------|
| `readability-lxml` | HTML解析 | `pip install readability-lxml` |
| `tiktoken` | Token计算 | `pip install tiktoken` |
| `firecrawl-py` | Firecrawl SDK | `pip install firecrawl-py` (可选) |

---

## 7. 预期效果

### 7.1 效果对比

| 指标 | 当前方案 (截断) | 新方案 (智能处理) |
|-----|----------------|------------------|
| 信息保留率 | 3% | 1.4% (有效信息) |
| 相关性 | 低 (包含大量噪音) | 高 (筛选后高相关) |
| 来源追溯 | 无 | 有 (每个事实带来源) |
| 跨源整合 | 无 | 有 (按话题归类) |
| 冲突检测 | 无 | 有 |
| API成本 | 低 | 中 (+50%) |
| 处理时间 | 快 | 中 (+30%) |

### 7.2 核心优势

1. **无截断**: 不丢失信息，而是智能筛选
2. **高相关性**: 只保留与查询相关的内容
3. **可追溯**: 每个事实都有来源引用
4. **跨源整合**: 多源信息按话题归类
5. **冲突检测**: 自动发现矛盾信息

---

## 8. 风险评估

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|------|---------|
| API成本增加 | 高 | 中 | 配置项控制阈值 |
| 处理时间增加 | 中 | 低 | 并行处理 |
| 相关性判断不准 | 中 | 中 | 可调整阈值 |
| 跨源对齐复杂 | 中 | 中 | 简化话题数量 |

---

> 设计完成  
> 下一步: 根据本方案开始实施阶段1 (ContentTransformer)
