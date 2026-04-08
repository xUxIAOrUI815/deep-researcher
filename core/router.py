from core.convergence import ConvergenceChecker


def should_continue(state: dict) -> str:
    decision = ConvergenceChecker.check(state)

    print(f"[ROUTER] 决策: {decision.action}, 原因: {decision.reason}")

    if decision.should_converge:
        return "writer"
    else:
        return "researcher"
