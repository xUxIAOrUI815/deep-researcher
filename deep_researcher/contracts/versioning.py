from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier, validate_semantic_version


class ComponentKind(str, Enum):
    RUNTIME = "runtime"
    SCHEDULER = "scheduler"
    GRAPH_ADAPTER = "graph_adapter"
    MODEL = "model"
    AGENT_SPEC = "agent_spec"
    PROMPT = "prompt"
    SKILL = "skill"
    TOOL = "tool"
    TOOL_POLICY = "tool_policy"
    STOP_POLICY = "stop_policy"
    VERIFICATION_POLICY = "verification_policy"
    RUBRIC = "rubric"
    DATASET = "dataset"


class VersionRef(ContractModel):
    version_id: str = Field(default_factory=lambda: new_id("version"))
    kind: ComponentKind
    name: str = Field(min_length=1, max_length=200)
    version: str
    artifact_id: str | None = None
    content_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("version_id", "artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)

    @field_validator("created_at")
    @classmethod
    def _aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value


class ComponentVersionSet(ContractModel):
    runtime: VersionRef
    scheduler: VersionRef
    model: VersionRef | None = None
    agent_spec: VersionRef | None = None
    prompt: VersionRef | None = None
    skill: VersionRef | None = None
    tool_policy: VersionRef | None = None
    stop_policy: VersionRef | None = None
    verification_policy: VersionRef | None = None
    rubric: VersionRef | None = None
    tools: tuple[VersionRef, ...] = ()

    @model_validator(mode="after")
    def _kinds_match(self) -> "ComponentVersionSet":
        expected = {
            "runtime": ComponentKind.RUNTIME,
            "scheduler": ComponentKind.SCHEDULER,
            "model": ComponentKind.MODEL,
            "agent_spec": ComponentKind.AGENT_SPEC,
            "prompt": ComponentKind.PROMPT,
            "skill": ComponentKind.SKILL,
            "tool_policy": ComponentKind.TOOL_POLICY,
            "stop_policy": ComponentKind.STOP_POLICY,
            "verification_policy": ComponentKind.VERIFICATION_POLICY,
            "rubric": ComponentKind.RUBRIC,
        }
        for field_name, kind in expected.items():
            value = getattr(self, field_name)
            if value is not None and value.kind != kind:
                raise ValueError(f"{field_name} must reference a {kind.value} version")
        if any(tool.kind != ComponentKind.TOOL for tool in self.tools):
            raise ValueError("tools must contain only tool versions")
        return self
