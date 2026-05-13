from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ResearchCreateRequest(BaseModel):
    """创建研究任务时前端提交的请求体。"""

    query: str
    instructions: str = ""
    depth: Literal["quick", "standard", "deep"] = "standard"

    model_config = {"strict": True}


class ResearchCreateResponse(BaseModel):
    """研究任务创建成功后返回给控制台的入口信息。"""

    research_id: str
    thread_id: str
    session_id: str
    status: str
    console_url: str
    report_url: str

    model_config = {"strict": True}


class TimelineEventSummary(BaseModel):
    """控制台时间线中的单条事件摘要。"""

    event_id: str = ""
    event_type: str
    timestamp: str
    level: str = "info"
    message: str = ""
    node_name: Optional[str] = None
    agent_name: Optional[str] = None
    task_id: Optional[str] = None
    section_id: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class KnowledgeSummary(BaseModel):
    """当前研究会话中已沉淀知识资产的数量概览。"""

    source_count: int = 0
    claim_count: int = 0
    fact_count: int = 0
    evidence_count: int = 0
    conflict_count: int = 0
    open_gap_count: int = 0
    section_pack_count: int = 0

    model_config = {"strict": True}


class ActiveAgentSummary(BaseModel):
    """控制台顶部/侧栏展示的当前活跃 Agent 状态。"""

    name: str = "idle"
    status: str = "idle"
    target: str = ""
    last_output_summary: str = ""

    model_config = {"strict": True}


class ContextPanelSummary(BaseModel):
    """上下文面板中展示给不同角色的摘要信息。"""

    planner: Dict[str, Any] = Field(default_factory=dict)
    researcher: Dict[str, Any] = Field(default_factory=dict)
    writer: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class ConsoleRunSummary(BaseModel):
    """控制台主视图所需的完整运行状态摘要。"""

    # 运行标识与基础状态。
    research_id: str
    thread_id: str
    session_id: str
    query: str
    status: str
    current_stage: str
    current_round: int = 0
    elapsed_seconds: float = 0.0
    resumed: bool = False
    has_report: bool = False

    # 当前任务树和规划器状态。
    root_task_id: Optional[str] = None
    active_task_id: Optional[str] = None
    planner_state: Dict[str, Any] = Field(default_factory=dict)
    report_outline: Dict[str, Any] = Field(default_factory=dict)
    task_tree: Dict[str, Any] = Field(default_factory=dict)

    # 控制台时间线和知识库概览。
    timeline: List[TimelineEventSummary] = Field(default_factory=list)
    knowledge_summary: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
    latest_coverage_snapshot: Optional[Dict[str, Any]] = None

    # 研究过程中的缺口、冲突、证据包和来源列表。
    open_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    section_packs: List[Dict[str, Any]] = Field(default_factory=list)
    sources: List[Dict[str, Any]] = Field(default_factory=list)

    # 当前执行者、角色上下文摘要和额外运行元数据。
    active_agent: ActiveAgentSummary = Field(default_factory=ActiveAgentSummary)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)
    run_metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}


class ReportViewResponse(BaseModel):
    """报告阅读页的响应结构。"""

    # 报告基础信息和正文内容。
    research_id: str
    session_id: str
    query: str
    status: str
    title: str = ""
    markdown: str = ""
    outline: Dict[str, Any] = Field(default_factory=dict)
    report: Dict[str, Any] = Field(default_factory=dict)

    # 报告页侧边栏需要的知识库和上下文摘要。
    knowledge_summary: KnowledgeSummary = Field(default_factory=KnowledgeSummary)
    latest_coverage_snapshot: Optional[Dict[str, Any]] = None
    open_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    section_packs: List[Dict[str, Any]] = Field(default_factory=list)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)

    model_config = {"strict": True}


class DebugViewResponse(BaseModel):
    """调试视图的响应结构，包含更接近运行时内部状态的数据。"""

    research_id: str
    session_id: str
    status: str
    state_summary: Dict[str, Any] = Field(default_factory=dict)
    context_summary: ContextPanelSummary = Field(default_factory=ContextPanelSummary)
    trace: List[TimelineEventSummary] = Field(default_factory=list)
    raw_state: Dict[str, Any] = Field(default_factory=dict)
    snapshot_summary: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"strict": True}
