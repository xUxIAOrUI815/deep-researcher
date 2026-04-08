class StateManager:

    @staticmethod
    def skip_pending_tasks(state: dict) -> dict:
        pending_tasks = list(state.get("pending_tasks", []))
        if pending_tasks:
            print(f"[StateManager] 跳过 {len(pending_tasks)} 个待办任务")
            for tid in pending_tasks:
                task_tree = state.get("task_tree", {})
                if tid in task_tree:
                    task_tree[tid]["status"] = "skipped"
            state["pending_tasks"] = []
        return state

    @staticmethod
    def mark_task_completed(state: dict, task_id: str) -> dict:
        task_tree = state.get("task_tree", {})
        if task_id in task_tree:
            task_tree[task_id]["status"] = "completed"

        pending_tasks = state.get("pending_tasks", [])
        if task_id in pending_tasks:
            pending_tasks.remove(task_id)

        completed_tasks = state.get("completed_tasks", [])
        if task_id not in completed_tasks:
            completed_tasks.append(task_id)

        return state

    @staticmethod
    def mark_task_running(state: dict, task_id: str) -> dict:
        task_tree = state.get("task_tree", {})
        if task_id in task_tree:
            task_tree[task_id]["status"] = "running"
        return state
