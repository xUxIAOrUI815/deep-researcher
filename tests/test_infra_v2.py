import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from schemas.state import TaskNode, AtomicFact
from core.knowledge import KnowledgeManager
from core.graph import create_research_graph, init_sqlite_saver, _get_research_id, _update_task_in_tree, _add_child_to_parent


class TestTrinityBinding:
    def test_get_collection_name(self):
        collection1 = KnowledgeManager.get_collection_name("research_001")
        collection2 = KnowledgeManager.get_collection_name("research_002")

        assert collection1 == "research_research_001"
        assert collection2 == "research_research_002"
        assert collection1 != collection2

        print("✅ Trinity binding: thread_id -> collection_name mapping works")


class TestBidirectionalTaskLinking:
    def test_update_task_in_tree(self):
        task1 = TaskNode(id="task_1", query="Parent task", depth=0, priority=1.0)
        task2 = TaskNode(id="task_2", query="Child task", depth=1, priority=0.8, parent_id="task_1")

        tree = {
            "task_1": task1.model_dump(),
            "task_2": task2.model_dump()
        }

        new_tree = _update_task_in_tree(tree, "task_1", {"status": "completed", "children_ids": ["task_2"]})

        assert new_tree["task_1"]["status"] == "completed"
        assert "task_2" in new_tree["task_1"]["children_ids"]

        print("✅ Bidirectional task linking: _update_task_in_tree works")

    def test_add_child_to_parent(self):
        task1 = TaskNode(id="task_1", query="Parent", depth=0)
        tree = {"task_1": task1.model_dump()}

        new_tree = _add_child_to_parent(tree, "task_1", "task_2")

        assert "task_2" in new_tree["task_1"]["children_ids"]

        new_tree2 = _add_child_to_parent(new_tree, "task_1", "task_3")
        assert "task_2" in new_tree2["task_1"]["children_ids"]
        assert "task_3" in new_tree2["task_1"]["children_ids"]

        print("✅ Bidirectional task linking: _add_child_to_parent works")


class TestResearchStateContract:
    def test_task_node_children_field(self):
        task = TaskNode(
            query="Test task",
            parent_id="parent_123",
            children_ids=["child_1", "child_2"],
            depth=1,
            priority=0.8,
            is_user_triggered=True
        )

        assert task.children_ids == ["child_1", "child_2"]
        assert task.parent_id == "parent_123"
        assert task.depth == 1
        assert task.is_user_triggered == True

        print("✅ TaskNode with children_ids and is_user_triggered works")

    def test_research_state_new_fields(self):
        from core.graph import GraphState

        state: GraphState = {
            "task_tree": {},
            "fact_pool": [],
            "atomic_facts": [],
            "token_usage": {"planning_tokens": 0, "research_tokens": 0, "distillation_tokens": 0, "writing_tokens": 0, "total_tokens": 0},
            "current_focus": None,
            "root_task_id": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "messages": [],
            "raw_scraped_data": [],
            "search_results": [],
            "final_report": None,
            "current_status": "正在初始化",
            "ui_logs": ["log1", "log2"],
            "report_sections": {"task_1": "Section content"},
            "pending_tasks": ["task_1", "task_2"]
        }

        assert state["current_status"] == "正在初始化"
        assert state["ui_logs"] == ["log1", "log2"]
        assert state["report_sections"]["task_1"] == "Section content"
        assert state["pending_tasks"] == ["task_1", "task_2"]

        print("✅ ResearchState with new fields works")


class TestKnowledgeCollectionIsolation:
    @staticmethod
    async def test_separate_collections():
        km = KnowledgeManager(base_storage_path="./test_km_data")

        collection_a = KnowledgeManager.get_collection_name("research_a")
        collection_b = KnowledgeManager.get_collection_name("research_b")

        fact1 = AtomicFact(
            text="Fact for research A",
            source_url="https://a.com",
            confidence=0.9
        )

        fact2 = AtomicFact(
            text="Fact for research B",
            source_url="https://b.com",
            confidence=0.8
        )

        await km.add_facts([fact1], collection_a)
        await km.add_facts([fact2], collection_b)

        facts_a = await km.search_facts("Fact", collection_a, limit=10)
        facts_b = await km.search_facts("Fact", collection_b, limit=10)

        assert len(facts_a) == 1
        assert len(facts_b) == 1
        assert "research A" in facts_a[0]["text"]
        assert "research B" in facts_b[0]["text"]

        stats_a = km.get_stats(collection_a)
        stats_b = km.get_stats(collection_b)

        assert stats_a.total_facts == 1
        assert stats_b.total_facts == 1

        print(f"✅ Collection isolation works: {collection_a} and {collection_b} are separate")

        await km.clear_collection(collection_a)
        await km.clear_collection(collection_b)

        import shutil
        if os.path.exists("./test_km_data"):
            try:
                shutil.rmtree("./test_km_data")
            except Exception:
                pass


class TestLangGraphCheckpointer:
    @staticmethod
    async def test_task_tree_persistence():
        for path in ["test_infra.db", "test_infra.db-shm", "test_infra.db-wal"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except PermissionError:
                    pass

        await asyncio.sleep(0.1)

        saver = await init_sqlite_saver("test_infra.db")
        graph = create_research_graph(saver)

        initial_state = {
            "task_tree": {},
            "fact_pool": [],
            "atomic_facts": [],
            "token_usage": {"planning_tokens": 0, "research_tokens": 0, "distillation_tokens": 0, "writing_tokens": 0, "total_tokens": 0},
            "current_focus": None,
            "root_task_id": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "messages": [],
            "raw_scraped_data": [],
            "search_results": [],
            "final_report": None,
            "current_status": "初始化中",
            "ui_logs": [],
            "report_sections": {},
            "pending_tasks": []
        }

        config = {"configurable": {"thread_id": "test-thread-1"}}

        result = await graph.ainvoke(initial_state, config)

        root_id = result.get("root_task_id")
        assert root_id is not None
        assert root_id in result["task_tree"]

        task = result["task_tree"][root_id]
        assert "children_ids" in task
        assert "parent_id" in task
        assert task["depth"] == 0
        assert task["priority"] == 1.0

        print(f"✅ TaskTree persisted with root_id: {root_id}")

        recovered_state = await graph.aget_state(config)
        assert recovered_state is not None
        assert recovered_state.config["configurable"]["thread_id"] == "test-thread-1"

        recovered_task = recovered_state.values.get("task_tree", {}).get(root_id)
        assert recovered_task is not None
        assert "children_ids" in recovered_task

        print(f"✅ TaskTree recovered correctly with children_ids")

        await saver.conn.close()

        for path in ["test_infra.db", "test_infra.db-shm", "test_infra.db-wal"]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass


class TestConfigContext:
    def test_get_research_id(self):
        config1 = {"configurable": {"thread_id": "research_abc"}}
        config2 = {"configurable": {}}

        assert _get_research_id(config1) == "research_abc"
        assert _get_research_id(config2) == "default"

        print("✅ _get_research_id correctly extracts thread_id from config")


async def main():
    print("\n" + "="*60)
    print("INFRA V2 - 三位一体架构测试")
    print("="*60)

    test_trinity = TestTrinityBinding()
    test_trinity.test_get_collection_name()

    test_bidirectional = TestBidirectionalTaskLinking()
    test_bidirectional.test_update_task_in_tree()
    test_bidirectional.test_add_child_to_parent()

    test_state = TestResearchStateContract()
    test_state.test_task_node_children_field()
    test_state.test_research_state_new_fields()

    test_config = TestConfigContext()
    test_config.test_get_research_id()

    await TestKnowledgeCollectionIsolation.test_separate_collections()

    await TestLangGraphCheckpointer.test_task_tree_persistence()

    print("\n" + "="*60)
    print("ALL INFRA V2 TESTS PASSED ✅")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
