from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime
import uuid


class TaskNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    status: Literal["pending", "running", "completed", "failed"] = "pending"
    depth: int = 0
    priority: float = 0.0
    parent_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True}


class AtomicFact(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    source_url: str
    confidence: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.now)
    task_id: Optional[str] = None

    model_config = {"strict": True}


class TokenUsage(BaseModel):
    planning_tokens: int = 0
    research_tokens: int = 0
    distillation_tokens: int = 0
    writing_tokens: int = 0
    total_tokens: int = 0

    model_config = {"strict": True}


class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    score: float = 0.0
    raw_content: Optional[str] = None

    model_config = {"strict": True}


class ScrapedData(BaseModel):
    url: str
    markdown: str
    title: str = ""
    fetch_method: Literal["jina", "playwright"] = "jina"
    timestamp: datetime = Field(default_factory=datetime.now)
    error: Optional[str] = None

    model_config = {"strict": True}


class ResearchState(BaseModel):
    task_tree: Dict[str, TaskNode] = Field(default_factory=dict)
    fact_pool: List[str] = Field(default_factory=list)
    atomic_facts: List[AtomicFact] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    current_focus: Optional[str] = None
    root_task_id: Optional[str] = None
    completed_tasks: List[str] = Field(default_factory=list)
    failed_tasks: List[str] = Field(default_factory=list)
    raw_scraped_data: List[ScrapedData] = Field(default_factory=list)
    search_results: List[SearchResult] = Field(default_factory=list)

    model_config = {"strict": True}


def compute_priority(relevance: float, depth: int, depth_penalty: float = 0.3) -> float:
    depth_factor = max(0.0, 1.0 - (depth * depth_penalty))
    return relevance * 0.7 + depth_factor * 0.3
