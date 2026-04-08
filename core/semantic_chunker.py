"""
SemanticChunker模块
语义分块与相关性筛选
"""

import re
import os
import httpx
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import datetime


@dataclass
class RelevantChunk:
    text: str
    chunk_id: str
    relevance_score: float = 0.0
    source_url: str = ""
    source_title: str = ""
    position: int = 0


@dataclass
class ChunkingResult:
    chunks: List[RelevantChunk]
    total_chunks: int = 0
    filtered_chunks: int = 0
    avg_relevance: float = 0.0


class SemanticChunker:
    """语义分块与相关性筛选器"""
    
    def __init__(
        self,
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        relevance_threshold: float = 0.3,
        top_k: int = 20,
        embedding_api_base: str = "https://api.siliconflow.cn/v1",
        embedding_model: str = "BAAI/bge-large-zh-v1.5"
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.relevance_threshold = relevance_threshold
        self.top_k = top_k
        self.embedding_api_base = embedding_api_base
        self.embedding_model = embedding_model
        self.api_key = os.getenv("SILICONFLOW_API_KEY", "")
    
    def chunk(self, text: str) -> List[str]:
        """
        语义分块 - 使用递归字符分割
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分块后的文本列表
        """
        if not text:
            return []
        
        # 分隔符优先级
        separators = ["\n\n\n", "\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " ", ""]
        
        chunks = self._split_recursive(text, separators, self.chunk_size)
        
        return chunks
    
    def _split_recursive(
        self,
        text: str,
        separators: List[str],
        chunk_size: int
    ) -> List[str]:
        """递归分割"""
        if not text:
            return []
        
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        # 尝试每个分隔符
        for i, separator in enumerate(separators):
            if separator in text:
                parts = text.split(separator)
                
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    if not part.strip():
                        continue
                    
                    if len(current_chunk) + len(part) + len(separator) <= chunk_size:
                        current_chunk += (separator if current_chunk else "") + part
                    else:
                        if current_chunk.strip():
                            chunks.append(current_chunk.strip())
                        
                        if len(part) > chunk_size:
                            # 需要进一步分割
                            sub_chunks = self._split_recursive(
                                part,
                                separators[i+1:] if i+1 < len(separators) else [""],
                                chunk_size
                            )
                            chunks.extend(sub_chunks)
                            current_chunk = ""
                        else:
                            current_chunk = part
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                return chunks
        
        # 没有分隔符，直接按字符切分
        return [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size) if text[i:i+chunk_size].strip()]
    
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """获取文本的embedding向量"""
        if not self.api_key:
            # 如果没有API key，返回None，后续使用简单的关键词匹配
            return None
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.embedding_api_base}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.embedding_model,
                        "input": text[:8000],  # 截断
                        "encoding_format": "float"
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data.get("data", [{}])[0].get("embedding", [])
        except Exception as e:
            print(f"[SemanticChunker] Error getting embedding: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算余弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        
        arr1 = np.array(vec1)
        arr2 = np.array(vec2)
        
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def keyword_relevance(self, text: str, query: str) -> float:
        """基于关键词的相关性计算（fallback）"""
        text_lower = text.lower()
        query_lower = query.lower()
        
        # 提取查询中的关键词
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        if not query_words:
            return 0.5
        
        # 计算关键词覆盖率
        matched = sum(1 for word in query_words if word in text_lower)
        coverage = matched / len(query_words)
        
        # 计算关键词密度
        text_words = len(re.findall(r'\b\w+\b', text_lower))
        if text_words == 0:
            return 0.0
        
        density = matched / min(text_words, 100)  # 限制在100词内计算
        
        # 综合得分
        score = 0.7 * coverage + 0.3 * min(density * 10, 1.0)
        
        return min(1.0, score)
    
    async def chunk_and_filter(
        self,
        text: str,
        query: str,
        source_url: str = "",
        source_title: str = ""
    ) -> ChunkingResult:
        """
        分块并筛选相关内容
        
        Args:
            text: 输入文本
            query: 搜索查询
            source_url: 来源URL
            source_title: 来源标题
            
        Returns:
            ChunkingResult: 分块结果
        """
        # Step 1: 分块
        raw_chunks = self.chunk(text)
        total_chunks = len(raw_chunks)
        
        if not raw_chunks:
            return ChunkingResult(
                chunks=[],
                total_chunks=0,
                filtered_chunks=0,
                avg_relevance=0.0
            )
        
        # Step 2: 获取query的embedding
        query_embedding = await self.get_embedding(query)
        
        # Step 3: 计算每个块的相关性
        scored_chunks = []
        
        for i, chunk_text in enumerate(raw_chunks):
            if len(chunk_text) < 50:  # 跳过太短的块
                continue
            
            # 计算相关性
            if query_embedding:
                chunk_embedding = await self.get_embedding(chunk_text)
                if chunk_embedding:
                    relevance = self.cosine_similarity(query_embedding, chunk_embedding)
                else:
                    relevance = self.keyword_relevance(chunk_text, query)
            else:
                relevance = self.keyword_relevance(chunk_text, query)
            
            if relevance >= self.relevance_threshold:
                scored_chunks.append(RelevantChunk(
                    text=chunk_text,
                    chunk_id=f"{source_url}#chunk_{i}",
                    relevance_score=relevance,
                    source_url=source_url,
                    source_title=source_title,
                    position=i
                ))
        
        # Step 4: 排序并取Top-K
        scored_chunks.sort(key=lambda x: x.relevance_score, reverse=True)
        top_chunks = scored_chunks[:self.top_k]
        
        # 计算平均相关性
        avg_relevance = sum(c.relevance_score for c in top_chunks) / len(top_chunks) if top_chunks else 0.0
        
        return ChunkingResult(
            chunks=top_chunks,
            total_chunks=total_chunks,
            filtered_chunks=total_chunks - len(scored_chunks),
            avg_relevance=avg_relevance
        )
    
    async def chunk_and_filter_batch(
        self,
        items: List[dict],
        query: str
    ) -> List[ChunkingResult]:
        """批量处理"""
        results = []
        
        for item in items:
            result = await self.chunk_and_filter(
                text=item.get("markdown", ""),
                query=query,
                source_url=item.get("url", ""),
                source_title=item.get("title", "")
            )
            results.append(result)
        
        return results
