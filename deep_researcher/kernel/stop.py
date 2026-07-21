from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from deep_researcher.contracts import Budget, BudgetDimension, BudgetUsage, StopDecision, StopReason, utc_now

from .types import CancellationToken


@dataclass
class StopPolicyState:
    consecutive_low_gain: int = 0
    consecutive_same_error: int = 0
    last_error_fingerprint: str = ""


class StopPolicyEngine:
    def __init__(self, *, low_gain_threshold: float = 0.01, max_low_gain_rounds: int = 3, max_repeated_errors: int = 3) -> None:
        self.low_gain_threshold = low_gain_threshold
        self.max_low_gain_rounds = max_low_gain_rounds
        self.max_repeated_errors = max_repeated_errors

    def record_gain(self, state: StopPolicyState, value: float) -> None:
        state.consecutive_low_gain = state.consecutive_low_gain + 1 if value <= self.low_gain_threshold else 0

    def record_error(self, state: StopPolicyState, fingerprint: str) -> None:
        if fingerprint == state.last_error_fingerprint:
            state.consecutive_same_error += 1
        else:
            state.last_error_fingerprint = fingerprint
            state.consecutive_same_error = 1

    def evaluate(
        self,
        *,
        budget: Budget,
        usage: BudgetUsage,
        state: StopPolicyState,
        cancellation: CancellationToken,
        now: datetime | None = None,
        budget_dimensions: tuple[BudgetDimension, ...] | None = None,
    ) -> StopDecision | None:
        if cancellation.cancelled:
            return StopDecision(should_stop=True, reason=StopReason.USER_CANCELLED, summary="Execution was cancelled.")
        exhausted = list(budget.exceeded_dimensions(usage, now=now or utc_now()))
        # Zero is a meaningful "no retries/errors permitted" ceiling, but the
        # initial zero usage is not itself an exhausted run.
        if usage.retries == 0 and budget.max_retries == 0:
            exhausted = [item for item in exhausted if item != BudgetDimension.RETRIES]
        if usage.errors == 0 and budget.max_errors == 0:
            exhausted = [item for item in exhausted if item != BudgetDimension.ERRORS]
        if budget_dimensions is not None:
            allowed = {
                BudgetDimension.TOKENS,
                BudgetDimension.COST,
                BudgetDimension.WALL_TIME,
                BudgetDimension.ERRORS,
                BudgetDimension.RETRIES,
                *budget_dimensions,
            }
            exhausted = [item for item in exhausted if item in allowed]
        if exhausted:
            return StopDecision(should_stop=True, reason=StopReason.BUDGET_EXHAUSTED, summary="Execution budget was exhausted.", exhausted_dimensions=tuple(exhausted))
        if state.consecutive_same_error >= self.max_repeated_errors:
            return StopDecision(should_stop=True, reason=StopReason.REPEATED_ERROR, summary="The same error repeated without recovery.")
        if state.consecutive_low_gain >= self.max_low_gain_rounds:
            return StopDecision(should_stop=True, reason=StopReason.LOW_INFORMATION_GAIN, summary="Repeated actions produced no material information gain.")
        return None
