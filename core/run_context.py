from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import uuid
from typing import Any, Mapping, Optional


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(frozen=True)
class RunContext:
    """Correlation context shared by graph nodes, agents, and observability."""

    research_id: str
    thread_id: str
    run_id: str = field(default_factory=lambda: _new_id("run"))
    trace_id: str = field(default_factory=lambda: _new_id("trace"))
    session_id: Optional[str] = None
    root_query: str = ""
    graph_version: str = "graph.v1"
    prompt_version: str = "prompt.v1"
    started_at: datetime = field(default_factory=datetime.now)

    @classmethod
    def from_config(
        cls,
        config: Optional[Mapping[str, Any]],
        root_query: str = "",
        research_id: Optional[str] = None,
        graph_version: str = "graph.v1",
        prompt_version: str = "prompt.v1",
    ) -> "RunContext":
        configurable = dict((config or {}).get("configurable", {}))
        thread_id = str(configurable.get("thread_id") or "default-thread")
        resolved_research_id = research_id or str(
            configurable.get("research_id") or thread_id or "default"
        )
        session_id = configurable.get("session_id")

        return cls(
            research_id=resolved_research_id,
            thread_id=thread_id,
            session_id=session_id,
            root_query=root_query,
            graph_version=graph_version,
            prompt_version=prompt_version,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["started_at"] = self.started_at.isoformat()
        return data

    @property
    def knowledge_collection(self) -> str:
        return f"research_{self.research_id}"
