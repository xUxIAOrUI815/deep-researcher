from datetime import datetime
from enum import Enum
import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    DEFERRED = "deferred"
    PRUNED = "pruned"
    MERGED = "merged"


class TaskNodeType(str, Enum):
    QUESTION = "question"
    GAP = "gap"
    CONFLICT = "conflict"
    HYPOTHESIS = "hypothesis"
    VERIFICATION_TASK = "verification_task"
    SOURCE_DISCOVERY = "source_discovery"
    ENTITY_PROFILE = "entity_profile"
    TIMELINE_CHECK = "timeline_check"
    DATA_POINT_CHECK = "data_point_check"
    SECTION_SUPPORT = "section_support"
    FOLLOWUP = "followup"


class TaskTreeOperation(str, Enum):
    ATTACH = "attach"
    MERGE = "merge"
    PRUNE = "prune"
    DEFER = "defer"
    UPDATE = "update"


class TaskNode(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    query: str
    title: str = ""
    rationale: str = ""
    node_type: Literal[
        "question",
        "gap",
        "conflict",
        "hypothesis",
        "verification_task",
        "source_discovery",
        "entity_profile",
        "timeline_check",
        "data_point_check",
        "section_support",
        "followup",
    ] = TaskNodeType.QUESTION.value
    status: Literal[
        "pending",
        "running",
        "completed",
        "failed",
        "deferred",
        "pruned",
        "merged",
    ] = TaskStatus.PENDING.value
    depth: int = 0
    priority: float = 0.0
    parent_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    children_ids: List[str] = Field(default_factory=list)
    source_span_ids: List[str] = Field(default_factory=list)
    related_fact_ids: List[str] = Field(default_factory=list)
    related_claim_ids: List[str] = Field(default_factory=list)
    related_evidence_ids: List[str] = Field(default_factory=list)
    section_ids: List[str] = Field(default_factory=list)
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    merge_into: Optional[str] = None
    defer_reason: Optional[str] = None
    prune_reason: Optional[str] = None
    is_user_triggered: bool = False
    attempts: int = 0
    max_attempts: int = 3
    tags: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True, "use_enum_values": True}


class TaskTreePatch(BaseModel):
    operation: Literal["attach", "merge", "prune", "defer", "update"]
    task_id: Optional[str] = None
    parent_task_id: Optional[str] = None
    target_task_id: Optional[str] = None
    task: Optional[TaskNode] = None
    updates: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)

    model_config = {"strict": True, "use_enum_values": True}
