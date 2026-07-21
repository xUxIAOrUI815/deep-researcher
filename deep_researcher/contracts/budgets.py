from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Iterable

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, utc_now


class BudgetDimension(str, Enum):
    TOKENS = "tokens"
    COST = "cost"
    WALL_TIME = "wall_time"
    MODEL_CALLS = "model_calls"
    TOOL_CALLS = "tool_calls"
    SEARCH_CALLS = "search_calls"
    RETRIES = "retries"
    ERRORS = "errors"


class StopReason(str, Enum):
    SUCCESS = "success"
    SEMANTIC_COMPLETE = "semantic_complete"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOW_INFORMATION_GAIN = "low_information_gain"
    REPEATED_ERROR = "repeated_error"
    USER_CANCELLED = "user_cancelled"
    APPROVAL_REQUIRED = "approval_required"
    POLICY_DENIED = "policy_denied"
    DEADLINE_REACHED = "deadline_reached"
    VERIFICATION_FAILED = "verification_failed"
    NO_ACTION_AVAILABLE = "no_action_available"


class Budget(ContractModel):
    max_tokens: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0.0)
    max_wall_time_seconds: float | None = Field(default=None, gt=0.0)
    max_model_calls: int | None = Field(default=None, gt=0)
    max_tool_calls: int | None = Field(default=None, gt=0)
    max_search_calls: int | None = Field(default=None, gt=0)
    max_retries: int | None = Field(default=None, ge=0)
    max_errors: int | None = Field(default=None, ge=0)
    deadline: datetime | None = None

    @field_validator("deadline")
    @classmethod
    def _aware_deadline(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("deadline must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _has_limit(self) -> "Budget":
        limits = (
            self.max_tokens,
            self.max_cost_usd,
            self.max_wall_time_seconds,
            self.max_model_calls,
            self.max_tool_calls,
            self.max_search_calls,
            self.max_retries,
            self.max_errors,
            self.deadline,
        )
        if all(value is None for value in limits):
            raise ValueError("a budget must define at least one limit")
        return self

    def exceeded_dimensions(
        self,
        usage: "BudgetUsage",
        *,
        now: datetime | None = None,
    ) -> tuple[BudgetDimension, ...]:
        current = now or utc_now()
        checks: list[tuple[BudgetDimension, float | int, float | int | None]] = [
            (BudgetDimension.TOKENS, usage.total_tokens, self.max_tokens),
            (BudgetDimension.COST, usage.cost_usd, self.max_cost_usd),
            (BudgetDimension.WALL_TIME, usage.wall_time_seconds, self.max_wall_time_seconds),
            (BudgetDimension.MODEL_CALLS, usage.model_calls, self.max_model_calls),
            (BudgetDimension.TOOL_CALLS, usage.tool_calls, self.max_tool_calls),
            (BudgetDimension.SEARCH_CALLS, usage.search_calls, self.max_search_calls),
            (BudgetDimension.RETRIES, usage.retries, self.max_retries),
            (BudgetDimension.ERRORS, usage.errors, self.max_errors),
        ]
        exceeded = [dimension for dimension, actual, limit in checks if limit is not None and actual >= limit]
        if self.deadline is not None and current >= self.deadline:
            exceeded.append(BudgetDimension.WALL_TIME)
        return tuple(dict.fromkeys(exceeded))


class BudgetUsage(ContractModel):
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_usd: float = Field(default=0.0, ge=0.0)
    wall_time_seconds: float = Field(default=0.0, ge=0.0)
    model_calls: int = Field(default=0, ge=0)
    tool_calls: int = Field(default=0, ge=0)
    search_calls: int = Field(default=0, ge=0)
    retries: int = Field(default=0, ge=0)
    errors: int = Field(default=0, ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def plus(self, **increments: int | float) -> "BudgetUsage":
        allowed = set(type(self).model_fields) - {"schema_version"}
        unknown = set(increments) - allowed
        if unknown:
            raise ValueError(f"unknown budget usage fields: {sorted(unknown)}")
        update = {
            name: getattr(self, name) + increments.get(name, 0)
            for name in allowed
        }
        return self.model_copy(update=update)


class StopDecision(ContractModel):
    should_stop: bool
    reason: StopReason
    summary: str = Field(min_length=1, max_length=1000)
    exhausted_dimensions: tuple[BudgetDimension, ...] = ()
    repairable: bool = False
    approval_required: bool = False
    decided_at: datetime = Field(default_factory=utc_now)

    @model_validator(mode="after")
    def _consistent(self) -> "StopDecision":
        if self.reason == StopReason.BUDGET_EXHAUSTED and not self.exhausted_dimensions:
            raise ValueError("budget exhaustion requires at least one exhausted dimension")
        if self.approval_required and self.reason != StopReason.APPROVAL_REQUIRED:
            raise ValueError("approval_required must use the approval_required reason")
        return self


def combine_usage(items: Iterable[BudgetUsage]) -> BudgetUsage:
    combined = BudgetUsage()
    for item in items:
        combined = combined.plus(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cost_usd=item.cost_usd,
            wall_time_seconds=item.wall_time_seconds,
            model_calls=item.model_calls,
            tool_calls=item.tool_calls,
            search_calls=item.search_calls,
            retries=item.retries,
            errors=item.errors,
        )
    return combined
