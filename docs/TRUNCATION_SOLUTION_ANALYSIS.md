# 截断问题解决方案分析

> 分析日期: 2026-04-05  
> 分析对象: DistillerAgent 中的8000字符截断

---

## 目录

1. [问题现状](#1-问题现状)
2. [必要性分析](#2-必要性分析)
3. [可行性分析](#3-可行性分析)
4. [解决方案对比](#4-解决方案对比)
5. [推荐方案](#5-推荐方案)

---

## 1. 问题现状

### 1.1 当前截断位置

**文件**: [agents/distiller.py:140-141](file:///e:/mini-deep-research/agents/distiller.py#L140-L141)

```python
def _build_prompt(self, markdown_text: str) -> str:
    truncated = markdown_text[:8000]  # ← 截断至8000字符
```

### 1.2 截断影响分析

| 维度 | 分析 |
|-----|------|
| **截断长度** | 8000字符 |
| **对应Token** | 约4000-5000 tokens (中文约1.5字符/token) |
| **典型网页长度** | 5000-50000字符不等 |
| **信息丢失率** | 长文档可能丢失50-80%内容 |

### 1.3 实际影响示例

```
假设一篇关于台积电2nm的技术文章，长度15000字符:

┌─────────────────────────────────────────────────────────────────┐
│  0-8000字符 (保留)                                               │
│  ├── 背景介绍                                                    │
│  ├── 技术概述                                                    │
│  └── 部分技术细节                                                │
├─────────────────────────────────────────────────────────────────┤
│  8000-15000字符 (丢失)                                           │
│  ├── 具体量产时间表                                              │
│  ├── 投资金额数据                                                │
│  ├── 竞争对手对比                                                │
│  └── 未来规划                                                    │
└─────────────────────────────────────────────────────────────────┘

结果: 关键数据丢失，提取的事实不完整
```

---

## 2. 必要性分析

### 2.1 是否真的需要解决？

#### 场景分析

| 场景 | 文档长度 | 截断影响 | 是否需要解决 |
|-----|---------|---------|-------------|
| 新闻短文 | 2000-5000字符 | 无影响 | ❌ 不需要 |
| 技术博客 | 5000-10000字符 | 轻微影响 | ⚠️ 可选 |
| 深度分析报告 | 10000-30000字符 | 严重丢失 | ✅ 需要 |
| 学术论文 | 20000-50000字符 | 大量丢失 | ✅ 需要 |

#### 用户需求分析

**目标用户**: 行业分析师、投资研究员、学术研究者

**典型调研场景**:
- 调研某公司财报 → 财报文档通常很长
- 分析行业趋势 → 需要综合多篇深度报告
- 技术路线研究 → 技术白皮书内容丰富

**结论**: 对于深度调研场景，解决截断问题**有必要**。

### 2.2 优先级评估

| 维度 | 评分 | 说明 |
|-----|------|------|
| 业务价值 | ⭐⭐⭐⭐ | 直接影响报告质量 |
| 用户痛点 | ⭐⭐⭐⭐ | 长文档场景常见 |
| 实现难度 | ⭐⭐⭐ | 需要一定开发工作 |
| 风险程度 | ⭐⭐ | 可能增加API成本 |

**综合优先级**: **P1 (高优先级)**

---

## 3. 可行性分析

### 3.1 技术约束

#### 3.1.1 LLM上下文限制

| 模型 | 上下文窗口 | 实际可用 |
|-----|-----------|---------|
| DeepSeek Chat | 64K tokens | 约40K tokens (扣除prompt) |
| GPT-4 | 128K tokens | 约100K tokens |
| Claude 3 | 200K tokens | 约150K tokens |

**分析**: 当前使用的DeepSeek Chat理论上支持约40K tokens的输入，8000字符(约5000 tokens)远未达到限制。

#### 3.1.2 API成本考虑

| 方案 | Token消耗 | 成本影响 |
|-----|----------|---------|
| 当前(8000字符) | ~5000 tokens/次 | 基准 |
| 不截断(平均20000字符) | ~13000 tokens/次 | +160% |
| 分块处理(3块) | ~15000 tokens/次 | +200% |

**分析**: 成本增加可控，且可通过配置调整。

#### 3.1.3 处理时间考虑

| 方案 | 处理时间 | 用户体验 |
|-----|---------|---------|
| 当前 | 3-5秒 | 良好 |
| 不截断 | 5-10秒 | 可接受 |
| 分块处理 | 10-20秒 | 需要优化 |

### 3.2 方案可行性矩阵

| 方案 | 技术可行性 | 成本可控性 | 用户体验 | 综合评估 |
|-----|-----------|-----------|---------|---------|
| 方案A: 提高截断阈值 | ✅ 高 | ✅ 高 | ✅ 高 | ⭐⭐⭐⭐⭐ |
| 方案B: 不截断 | ✅ 高 | ⚠️ 中 | ⚠️ 中 | ⭐⭐⭐⭐ |
| 方案C: 分块处理 | ✅ 高 | ⚠️ 中 | ⚠️ 中 | ⭐⭐⭐ |
| 方案D: 摘要后提取 | ⚠️ 中 | ✅ 高 | ✅ 高 | ⭐⭐⭐ |

---

## 4. 解决方案对比

### 4.1 方案A: 提高截断阈值

**方案描述**: 将截断阈值从8000字符提高到更大的值。

```python
def _build_prompt(self, markdown_text: str) -> str:
    truncated = markdown_text[:30000]  # 提高到30000字符
```

**优点**:
- 实现简单，一行代码修改
- 兼容性好，不影响现有逻辑
- 成本增加可控

**缺点**:
- 仍有截断，超长文档仍会丢失信息
- 可能接近上下文限制

**适用场景**: 大多数场景，文档长度<30000字符

**推荐指数**: ⭐⭐⭐⭐⭐

---

### 4.2 方案B: 不截断

**方案描述**: 直接传递完整文档，不做截断。

```python
def _build_prompt(self, markdown_text: str) -> str:
    return f"""请分析以下文本，提取原子事实：

{'-'*60}
{markdown_text}  # 不截断
{'-'*60}
...
```

**优点**:
- 信息完整，不丢失任何内容
- 实现最简单

**缺点**:
- 超长文档可能超过上下文限制
- API成本增加
- 处理时间增加

**适用场景**: 文档长度可控的场景

**推荐指数**: ⭐⭐⭐⭐

---

### 4.3 方案C: 分块处理

**方案描述**: 将长文档分成多个块，分别提取事实，最后合并去重。

```python
async def distill(self, markdown_text: str, source_url: str, ...) -> DistillationResult:
    if len(markdown_text) > 10000:
        # 分块
        chunks = self._split_into_chunks(markdown_text, chunk_size=8000, overlap=500)
        
        all_facts = []
        for chunk in chunks:
            result = await self._distill_single_chunk(chunk, source_url)
            all_facts.extend(result.facts)
        
        # 去重合并
        merged_facts = self._merge_and_deduplicate(all_facts)
        
        return DistillationResult(facts=merged_facts, ...)
    else:
        return await self._distill_single_chunk(markdown_text, source_url)
```

**优点**:
- 支持任意长度文档
- 信息不丢失
- 可并行处理

**缺点**:
- 实现复杂
- API调用次数增加，成本上升
- 需要处理跨块事实合并
- 处理时间增加

**适用场景**: 超长文档(>30000字符)

**推荐指数**: ⭐⭐⭐

---

### 4.4 方案D: 摘要后提取

**方案描述**: 先用LLM生成摘要，再从摘要中提取事实。

```python
async def distill(self, markdown_text: str, source_url: str, ...) -> DistillationResult:
    if len(markdown_text) > 20000:
        # 先生成摘要
        summary = await self._generate_summary(markdown_text)
        # 从摘要提取事实
        facts = await self._extract_facts_from_summary(summary, source_url)
    else:
        facts = await self._extract_facts_directly(markdown_text, source_url)
    
    return DistillationResult(facts=facts, ...)
```

**优点**:
- 成本可控
- 处理速度快

**缺点**:
- 摘要过程可能丢失细节
- 两次LLM调用，增加复杂度
- 摘要质量影响最终结果

**适用场景**: 对细节要求不高的场景

**推荐指数**: ⭐⭐⭐

---

### 4.5 方案E: 智能截断

**方案描述**: 在语义边界处截断，而非固定字符数。

```python
def _smart_truncate(self, markdown_text: str, max_length: int = 30000) -> str:
    if len(markdown_text) <= max_length:
        return markdown_text
    
    # 在段落边界截断
    truncated = markdown_text[:max_length]
    
    # 找到最后一个完整的段落
    last_paragraph_end = truncated.rfind('\n\n')
    if last_paragraph_end > max_length * 0.8:
        return truncated[:last_paragraph_end]
    
    # 找到最后一个完整的句子
    last_sentence_end = max(
        truncated.rfind('。'),
        truncated.rfind('.'),
        truncated.rfind('！'),
        truncated.rfind('?')
    )
    if last_sentence_end > max_length * 0.8:
        return truncated[:last_sentence_end + 1]
    
    return truncated
```

**优点**:
- 语义完整
- 实现相对简单
- 不增加API成本

**缺点**:
- 仍有信息丢失
- 边界判断可能不准确

**适用场景**: 需要语义完整性的场景

**推荐指数**: ⭐⭐⭐⭐

---

## 5. 推荐方案

### 5.1 推荐策略: 分层处理

根据文档长度采用不同策略:

```
┌─────────────────────────────────────────────────────────────────────┐
│                      分层处理策略                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  文档长度 ≤ 10000字符                                               │
│  └── 直接处理，不截断                                               │
│                                                                     │
│  10000字符 < 文档长度 ≤ 30000字符                                   │
│  └── 智能截断至30000字符 (在语义边界)                               │
│                                                                     │
│  文档长度 > 30000字符                                               │
│  └── 分块处理 (可选，根据配置)                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 推荐实现

```python
class DistillerAgent:
    def __init__(
        self,
        ...
        max_content_length: int = 30000,  # 新增配置
        enable_chunking: bool = False      # 新增配置
    ):
        self.max_content_length = max_content_length
        self.enable_chunking = enable_chunking
        ...

    def _build_prompt(self, markdown_text: str) -> str:
        content = self._prepare_content(markdown_text)
        
        return f"""请分析以下文本，提取原子事实：

{'-'*60}
{content}
{'-'*60}
...
"""
    
    def _prepare_content(self, markdown_text: str) -> str:
        """准备内容，根据长度采取不同策略"""
        if len(markdown_text) <= 10000:
            return markdown_text
        
        if len(markdown_text) <= self.max_content_length:
            return self._smart_truncate(markdown_text, self.max_content_length)
        
        if self.enable_chunking:
            # 分块处理逻辑
            return self._chunk_and_merge(markdown_text)
        
        return self._smart_truncate(markdown_text, self.max_content_length)
    
    def _smart_truncate(self, text: str, max_length: int) -> str:
        """在语义边界截断"""
        if len(text) <= max_length:
            return text
        
        truncated = text[:max_length]
        
        # 优先在段落边界截断
        last_para = truncated.rfind('\n\n')
        if last_para > max_length * 0.8:
            return truncated[:last_para]
        
        # 其次在句子边界截断
        for end_char in ['。', '.', '！', '!', '？', '?']:
            last_sent = truncated.rfind(end_char)
            if last_sent > max_length * 0.8:
                return truncated[:last_sent + 1]
        
        return truncated
```

### 5.3 配置项建议

```python
# core/config.py 新增配置

class DistillerConfig:
    MAX_CONTENT_LENGTH: int = 30000      # 最大内容长度
    ENABLE_CHUNKING: bool = False        # 是否启用分块
    CHUNK_SIZE: int = 8000               # 分块大小
    CHUNK_OVERLAP: int = 500             # 分块重叠
```

### 5.4 实施建议

| 阶段 | 任务 | 预估时间 |
|-----|------|---------|
| 阶段1 | 提高截断阈值至30000字符 | 10分钟 |
| 阶段2 | 实现智能截断 | 30分钟 |
| 阶段3 | 添加配置项 | 15分钟 |
| 阶段4 | (可选) 实现分块处理 | 2小时 |

### 5.5 风险评估

| 风险 | 可能性 | 影响 | 应对措施 |
|-----|-------|------|---------|
| API成本增加 | 高 | 中 | 添加配置项，用户可调整阈值 |
| 处理时间增加 | 中 | 低 | 添加进度提示 |
| 超长文档仍丢失信息 | 低 | 中 | 可选启用分块处理 |

---

## 总结

### 最终建议

1. **短期方案 (立即实施)**: 将截断阈值从8000提高到30000字符
2. **中期方案 (下阶段)**: 实现智能截断，在语义边界处截断
3. **长期方案 (可选)**: 支持分块处理，处理超长文档

### 成本效益分析

| 方案 | 开发成本 | 运行成本 | 效果提升 | ROI |
|-----|---------|---------|---------|-----|
| 提高阈值 | 极低 | 低 | 高 | ⭐⭐⭐⭐⭐ |
| 智能截断 | 低 | 无 | 中 | ⭐⭐⭐⭐ |
| 分块处理 | 中 | 中 | 高 | ⭐⭐⭐ |

**结论**: 建议先实施"提高阈值"方案，投入产出比最高。
