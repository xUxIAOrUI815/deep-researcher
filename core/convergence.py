from dataclasses import dataclass
from core.config import ResearchConfig


@dataclass
class ConvergenceDecision:
    should_converge: bool
    reason: str
    action: str
    skip_pending_tasks: bool


class ConvergenceChecker:

    @staticmethod
    def check(state: dict) -> ConvergenceDecision:
        if state.get("planner_action") == "finish":
            return ConvergenceDecision(
                should_converge=True,
                reason="Planner决策finish",
                action="finish",
                skip_pending_tasks=True
            )

        current_depth = state.get("current_depth", 0)
        task_tree_size = len(state.get("task_tree", {}))
        total_facts = len(state.get("all_fact_ids", []))

        if current_depth >= ResearchConfig.MAX_DEPTH:
            return ConvergenceDecision(
                should_converge=True,
                reason=f"深度{current_depth}达到上限{ResearchConfig.MAX_DEPTH}",
                action="finish",
                skip_pending_tasks=True
            )

        if task_tree_size >= ResearchConfig.MAX_NODES:
            return ConvergenceDecision(
                should_converge=True,
                reason=f"节点数{task_tree_size}达到上限{ResearchConfig.MAX_NODES}",
                action="finish",
                skip_pending_tasks=True
            )

        if total_facts > ResearchConfig.MAX_FACTS:
            return ConvergenceDecision(
                should_converge=True,
                reason=f"事实数{total_facts}超过上限{ResearchConfig.MAX_FACTS}",
                action="finish",
                skip_pending_tasks=True
            )

        pending_count = len(state.get("pending_tasks", []))

        pending_at_max_depth = sum(
            1 for tid in state.get("pending_tasks", [])
            if state.get("task_tree", {}).get(tid, {}).get("depth", 0) >= ResearchConfig.MAX_DEPTH
        )

        if pending_count > 0 and pending_at_max_depth == pending_count:
            return ConvergenceDecision(
                should_converge=True,
                reason=f"所有{pending_count}个待办任务已达最大深度",
                action="finish",
                skip_pending_tasks=True
            )

        if pending_count > 0:
            return ConvergenceDecision(
                should_converge=False,
                reason=f"有{pending_count}个待办任务",
                action="continue",
                skip_pending_tasks=False
            )

        return ConvergenceDecision(
            should_converge=True,
            reason="无待办任务，默认收敛",
            action="finish",
            skip_pending_tasks=False
        )
