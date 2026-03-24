import asyncio
import sys
sys.path.insert(0, '.')

from core.graph import run_research_cycle


async def main():
    print("="*60)
    print("Starting AIRE Research Cycle")
    print("="*60)

    result = await run_research_cycle("Test research query")

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
