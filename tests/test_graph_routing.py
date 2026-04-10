from core.graph import route_after_planner, should_continue


def _state(action: str | None, *, pending_count: int = 0) -> dict:
    task_tree = {
        f"task-{index}": {"status": "pending" if index < pending_count else "completed"}
        for index in range(max(1, pending_count))
    }
    return {
        "planner_state": {"action": action},
        "task_tree": task_tree,
    }


def test_route_after_planner_continue_research_with_pending_tasks_goes_to_researcher():
    state = _state("continue_research", pending_count=2)
    assert route_after_planner(state) == "researcher"
    assert should_continue(state) == "researcher"


def test_route_after_planner_continue_research_without_pending_tasks_goes_to_writer():
    state = {
        "planner_state": {"action": "continue_research"},
        "task_tree": {"task-1": {"status": "completed"}},
    }
    assert route_after_planner(state) == "writer"
    assert should_continue(state) == "writer"


def test_route_after_planner_start_writing_goes_to_writer():
    state = _state("start_writing", pending_count=1)
    assert route_after_planner(state) == "writer"
    assert should_continue(state) == "writer"


def test_route_after_planner_stop_goes_to_end():
    state = _state("stop", pending_count=2)
    assert route_after_planner(state) == "end"
    assert should_continue(state) == "end"


def test_route_after_planner_unknown_action_falls_back_to_task_status():
    state_with_pending = _state(None, pending_count=1)
    state_without_pending = {
        "planner_state": {"action": None},
        "task_tree": {"task-1": {"status": "completed"}},
    }

    assert route_after_planner(state_with_pending) == "researcher"
    assert route_after_planner(state_without_pending) == "writer"
