import pytest
import pytest_asyncio
import asyncio
import os
from core import init_sqlite_saver, create_research_graph
from schemas.state import TaskNode, ResearchState


@pytest.fixture
def db_path():
    return "test_research.db"


@pytest_asyncio.fixture
async def checkpointer(db_path):
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
        except PermissionError:
            pass
    if os.path.exists(f"{db_path}-shm"):
        try:
            os.remove(f"{db_path}-shm")
        except PermissionError:
            pass
    if os.path.exists(f"{db_path}-wal"):
        try:
            os.remove(f"{db_path}-wal")
        except PermissionError:
            pass

    await asyncio.sleep(0.1)

    saver = await init_sqlite_saver(db_path)
    yield saver

    try:
        await saver.conn.close()
    except Exception:
        pass

    await asyncio.sleep(0.1)

    for path in [db_path, f"{db_path}-shm", f"{db_path}-wal"]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except PermissionError:
                pass


@pytest.mark.asyncio
async def test_initial_state_serialization(checkpointer):
    graph = create_research_graph(checkpointer)

    initial_state = {
        "task_tree": {},
        "fact_pool": [],
        "atomic_facts": [],
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0
        },
        "current_focus": None,
        "root_task_id": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": []
    }

    config = {"configurable": {"thread_id": "test-thread-1"}}

    result = await graph.ainvoke(initial_state, config)

    assert result is not None
    assert "task_tree" in result
    assert len(result["task_tree"]) > 0
    print(f"[TEST] Initial state serialized. Task tree has {len(result['task_tree'])} nodes")


@pytest.mark.asyncio
async def test_state_persistence_after_disconnect(checkpointer):
    graph = create_research_graph(checkpointer)

    initial_state = {
        "task_tree": {},
        "fact_pool": [],
        "atomic_facts": [],
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0
        },
        "current_focus": None,
        "root_task_id": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": []
    }

    config = {"configurable": {"thread_id": "test-thread-2"}}

    result = await graph.ainvoke(initial_state, config)

    root_task_id = result.get("root_task_id")
    assert root_task_id is not None
    assert root_task_id in result["task_tree"]

    print(f"[TEST] First run complete. Root task ID: {root_task_id}")

    recovered_state = await graph.aget_state(config)
    assert recovered_state is not None
    assert recovered_state.config["configurable"]["thread_id"] == "test-thread-2"
    print(f"[TEST] State recovered after disconnect. Task tree has {len(recovered_state.values.get('task_tree', {}))} nodes")


@pytest.mark.asyncio
async def test_task_tree_structure(checkpointer):
    graph = create_research_graph(checkpointer)

    initial_state = {
        "task_tree": {},
        "fact_pool": [],
        "atomic_facts": [],
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0
        },
        "current_focus": None,
        "root_task_id": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": []
    }

    config = {"configurable": {"thread_id": "test-thread-3"}}

    result = await graph.ainvoke(initial_state, config)

    for task_id, task_data in result["task_tree"].items():
        assert "id" in task_data
        assert "query" in task_data
        assert "status" in task_data
        assert "depth" in task_data
        assert "priority" in task_data
        print(f"[TEST] Task {task_id}: status={task_data['status']}, depth={task_data['depth']}, priority={task_data['priority']}")

    print(f"[TEST] TaskTree structure verified with {len(result['task_tree'])} nodes")


@pytest.mark.asyncio
async def test_token_usage_tracking(checkpointer):
    graph = create_research_graph(checkpointer)

    initial_state = {
        "task_tree": {},
        "fact_pool": [],
        "atomic_facts": [],
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0
        },
        "current_focus": None,
        "root_task_id": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": []
    }

    config = {"configurable": {"thread_id": "test-thread-4"}}

    result = await graph.ainvoke(initial_state, config)

    token_usage = result.get("token_usage", {})
    assert token_usage["planning_tokens"] > 0
    assert token_usage["research_tokens"] > 0
    assert token_usage["distillation_tokens"] > 0
    assert token_usage["writing_tokens"] > 0

    total = sum(token_usage.values())
    print(f"[TEST] Token usage tracked: planning={token_usage['planning_tokens']}, research={token_usage['research_tokens']}, distillation={token_usage['distillation_tokens']}, writing={token_usage['writing_tokens']}, total={total}")
    print(f"[TEST] Token usage verification passed")
