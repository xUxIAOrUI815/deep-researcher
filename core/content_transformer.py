"""
ContentTransformer模块
结构化清洗与视图转换
"""

import re
import os
from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime


@dataclass
class ContentMetadata:
    title: str = ""
    url: str = ""
    author: Optional[str] = None
    publish_date: Optional[str] = None
    fetch_date: str = field(default_factory=lambda: datetime.now().isoformat())
    site_name: Optional[str] = None
    language: Optional[str] = None


@dataclass
class StructuredContent:
    markdown: str
    metadata: ContentMetadata
    original_length: int = 0
    cleaned_length: int = 0
    sections: List[str] = field(default_factory=list)


class ContentTransformer:
    """结构化清洗与视图转换器"""
    
    def __init__(self, jina_api_base: str = "https://r.jina.ai/"):
        self.jina_api_base = jina_api_base
    
    def transform(self, markdown: str, url: str, title: str = "") -> StructuredContent:
        """
        转换原始markdown为结构化内容
        
        Args:
            markdown: 原始markdown文本
            url: 来源URL
            title: 页面标题
            
        Returns:
            StructuredContent: 结构化内容
        """
        original_length = len(markdown)
        
        # Step 1: 提取元数据
        metadata = self._extract_metadata(markdown, url, title)
        
        # Step 2: 清洗内容
        cleaned = self._clean_content(markdown)
        
        # Step 3: 提取章节
        sections = self._extract_sections(cleaned)
        
        return StructuredContent(
            markdown=cleaned,
            metadata=metadata,
            original_length=original_length,
            cleaned_length=len(cleaned),
            sections=sections
        )
    
    def _extract_metadata(self, markdown: str, url: str, title: str) -> ContentMetadata:
        """从markdown中提取元数据"""
        metadata = ContentMetadata(
            url=url,
            title=title,
            fetch_date=datetime.now().isoformat()
        )
        
        # 尝试从内容中提取作者
        author_patterns = [
            r'[Bb]y\s+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'[Aa]uthor[:\s]+([A-Z][a-z]+\s+[A-Z][a-z]+)',
            r'作者[:\s]+([^\n]+)',
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, markdown[:2000])
            if match:
                metadata.author = match.group(1).strip()
                break
        
        # 尝试提取日期
        date_patterns = [
            r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}[日]?)',
            r'([Jj]anuary|[Ff]ebruary|[Mm]arch|[Aa]pril|[Mm]ay|[Jj]une|[Jj]uly|[Aa]ugust|[Ss]eptember|[Oo]ctober|[Nn]ovember|[Dd]ecember)\s+\d{1,2},?\s+\d{4}',
            r'(Published|Updated|发布)[:\s]+(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, markdown[:3000])
            if match:
                metadata.publish_date = match.group(1) if match.lastindex else match.group(0)
                break
        
        # 提取网站名称
        if url:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            metadata.site_name = parsed.netloc.replace('www.', '')
        
        return metadata
    
    def _clean_content(self, markdown: str) -> str:
        """清洗markdown内容"""
        lines = markdown.split('\n')
        cleaned_lines = []
        
        skip_patterns = [
            r'^#{1,3}\s*(navbar|nav|menu|header|footer|sidebar|advertisement|ad-)',
            r'^\s*(cookie|cookies|privacy policy|terms of service|subscribe|newsletter)',
            r'^\s*(sign in|log in|sign up|register|login)',
            r'^\s*(follow us|share this|share on)',
            r'^\s*(related articles|you may also like|recommended)',
            r'^\s*(advertisement|sponsored|promoted)',
        ]
        
        for line in lines:
            line_stripped = line.strip()
            
            # 跳过匹配的行
            should_skip = False
            for pattern in skip_patterns:
                if re.match(pattern, line_stripped, re.IGNORECASE):
                    should_skip = True
                    break
            
            if not should_skip:
                cleaned_lines.append(line)
        
        # 移除多余空行
        result = '\n'.join(cleaned_lines)
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _extract_sections(self, markdown: str) -> List[str]:
        """提取章节标题"""
        sections = []
        for line in markdown.split('\n'):
            match = re.match(r'^#{1,3}\s+(.+)$', line)
            if match:
                sections.append(match.group(1).strip())
        return sections


def transform_batch(items: List[dict]) -> List[StructuredContent]:
    """批量转换"""
    transformer = ContentTransformer()
    results = []
    
    for item in items:
        result = transformer.transform(
            markdown=item.get("markdown", ""),
            url=item.get("url", ""),
            title=item.get("title", "")
        )
        results.append(result)
    
    return results
