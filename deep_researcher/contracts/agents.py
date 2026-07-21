from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, validate_identifier, validate_semantic_version
from .budgets import Budget
from .commands import CommandKind
from .versioning import ComponentKind, VersionRef


class AgentRole(str, Enum):
    RESEARCH_SUPERVISOR = "research_supervisor"
    RESEARCH_WORKER = "research_worker"
    EVIDENCE_VERIFIER = "evidence_verifier"
    SYNTHESIS_WRITER = "synthesis_writer"
    REPORT_REVIEWER = "report_reviewer"


class MiddlewareStage(str, Enum):
    CONTEXT_TRIMMING = "context_trimming"
    REDACTION = "redaction"
    VERSION_INJECTION = "version_injection"
    BUDGET_CHECK = "budget_check"
    SCHEMA_VALIDATION = "schema_validation"
    SCHEMA_REPAIR = "schema_repair"
    COMMAND_NORMALIZATION = "command_normalization"
    POLICY_CHECK = "policy_check"


class MiddlewareSpec(ContractModel):
    stage: MiddlewareStage
    enabled: bool = True
    order: int = Field(ge=0)
    config: dict[str, Any] = Field(default_factory=dict)


class ToolGrant(ContractModel):
    tool_name: str = Field(min_length=1, max_length=200)
    allowed_operations: tuple[str, ...]
    requires_approval: bool = False
    max_calls_per_task: int | None = Field(default=None, gt=0)
    argument_constraints: dict[str, Any] = Field(default_factory=dict)

    @field_validator("allowed_operations")
    @classmethod
    def _operations(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(operation.strip() for operation in value if operation.strip())
        if not normalized:
            raise ValueError("a tool grant requires at least one operation")
        if len(set(normalized)) != len(normalized):
            raise ValueError("tool operations must be unique")
        return normalized


class AgentSpec(ContractModel):
    agent_spec_id: str = Field(default_factory=lambda: new_id("agent_spec"))
    name: str = Field(min_length=1, max_length=200)
    version: str
    role: AgentRole
    description: str = Field(min_length=1, max_length=2000)
    input_schema: str = Field(min_length=1, max_length=255)
    output_schema: str = Field(min_length=1, max_length=255)
    allowed_commands: tuple[CommandKind, ...]
    tool_grants: tuple[ToolGrant, ...] = ()
    model: VersionRef
    prompt: VersionRef
    skill: VersionRef | None = None
    tool_policy: VersionRef
    stop_policy: VersionRef
    verification_policy: VersionRef | None = None
    default_budget: Budget
    middleware: tuple[MiddlewareSpec, ...]
    context_window_tokens: int = Field(gt=0)
    reserved_output_tokens: int = Field(gt=0)
    max_parallel_commands: int = Field(default=1, gt=0)
    supports_delegation: bool = False
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("agent_spec_id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("version")
    @classmethod
    def _version(cls, value: str) -> str:
        return validate_semantic_version(value)

    @field_validator("allowed_commands")
    @classmethod
    def _commands(cls, value: tuple[CommandKind, ...]) -> tuple[CommandKind, ...]:
        if not value:
            raise ValueError("an agent must allow at least one command")
        if len(set(value)) != len(value):
            raise ValueError("allowed commands must be unique")
        return value

    @field_validator("middleware")
    @classmethod
    def _middleware_order(cls, value: tuple[MiddlewareSpec, ...]) -> tuple[MiddlewareSpec, ...]:
        if not value:
            raise ValueError("an agent requires an explicit middleware pipeline")
        orders = [item.order for item in value if item.enabled]
        if len(set(orders)) != len(orders):
            raise ValueError("enabled middleware stages require unique order values")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "AgentSpec":
        expected = {
            "model": ComponentKind.MODEL,
            "prompt": ComponentKind.PROMPT,
            "skill": ComponentKind.SKILL,
            "tool_policy": ComponentKind.TOOL_POLICY,
            "stop_policy": ComponentKind.STOP_POLICY,
            "verification_policy": ComponentKind.VERIFICATION_POLICY,
        }
        for field_name, kind in expected.items():
            value = getattr(self, field_name)
            if value is not None and value.kind != kind:
                raise ValueError(f"{field_name} must reference a {kind.value} version")
        if self.reserved_output_tokens >= self.context_window_tokens:
            raise ValueError("reserved output tokens must be smaller than the context window")
        if CommandKind.DELEGATE in self.allowed_commands and not self.supports_delegation:
            raise ValueError("delegate command requires supports_delegation")
        return self
