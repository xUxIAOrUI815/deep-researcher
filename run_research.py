import asyncio
import os
import sys
sys.path.insert(0, '.')

from core.observability import get_observer, set_observer
from core.graph import run_research_cycle, set_durable_knowledge_ingestor
from deep_researcher.events import build_event_runtime
from deep_researcher.knowledge import build_knowledge_runtime


async def main():
    print("="*60)
    print("Starting AIRE Research Cycle")
    print("="*60)

    previous_observer = get_observer()
    runtime = build_event_runtime(os.getenv("EVENT_STORE_PATH", "event_data/research_events.sqlite3"))
    knowledge_runtime = build_knowledge_runtime(
        os.getenv("KNOWLEDGE_RUNTIME_PATH", "knowledge_v2_data")
    )
    set_observer(runtime.observer)
    previous_ingestor = set_durable_knowledge_ingestor(knowledge_runtime.ingestion)
    try:
        result = await run_research_cycle("Test research query")
    finally:
        set_durable_knowledge_ingestor(previous_ingestor)
        set_observer(previous_observer)
        knowledge_runtime.close()
        runtime.close()

    print("\n" + "="*60)
    print("Final State Summary:")
    print("="*60)
    print(f"Task Tree Size: {len(result.get('task_tree', {}))}")
    print(f"Completed Tasks: {len(result.get('completed_tasks', []))}")
    print(f"Atomic Facts: {len(result.get('atomic_facts', []))}")
    print(f"Messages: {len(result.get('messages', []))}")

    print("\n" + "="*60)
    print("Token Usage Breakdown:")
    print("="*60)
    for key, value in result.get('token_usage', {}).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
