import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from core import KnowledgeManager
from agents import DistillerAgent
from schemas.state import AtomicFact


async def test_knowledge_manager():
    print("=" * 60)
    print("TEST 1: KnowledgeManager Basic Operations")
    print("=" * 60)

    km = KnowledgeManager(storage_path="./test_knowledge_data")

    await km.clear_collection()

    test_facts = [
        AtomicFact(
            text="台积电计划在2025年开始量产2纳米芯片",
            source_url="https://example.com/tsmc-2nm",
            confidence=0.9
        ),
        AtomicFact(
            text="华为推出了自主研发的麒麟9000S芯片",
            source_url="https://example.com/huawei-kirin",
            confidence=0.85
        ),
        AtomicFact(
            text="三星计划在2025年实现3纳米芯片量产",
            source_url="https://example.com/samsung-3nm",
            confidence=0.8
        ),
    ]

    print("\n[1.1] Adding initial facts...")
    added_ids, dup_ids = await km.add_facts(test_facts)
    print(f"  Added: {len(added_ids)} facts")
    print(f"  Duplicates merged: {len(dup_ids)} facts")

    print("\n[1.2] Searching for '芯片制造'...")
    results = await km.search_facts("芯片制造", limit=5)
    print(f"  Found {len(results)} relevant facts")
    for r in results:
        print(f"    - {r['text'][:50]}... (score: {r['score']:.2f})")

    stats = km.get_stats()
    print(f"\n[1.3] Knowledge stats:")
    print(f"    Total facts: {stats.total_facts}")
    print(f"    Verified: {stats.verified_facts}")
    print(f"    Conflicts: {stats.conflicts_detected}")

    await km.clear_collection()
    print("\n  Collection cleared for next test")


async def test_semantic_dedup():
    print("\n" + "=" * 60)
    print("TEST 2: Semantic Deduplication")
    print("=" * 60)

    km = KnowledgeManager(storage_path="./test_knowledge_data")

    await km.clear_collection()

    fact1 = AtomicFact(
        text="台积电宣布将在2025年量产2纳米制程芯片",
        source_url="https://news.example.com/1",
        confidence=0.9
    )

    fact2 = AtomicFact(
        text="台积电宣布将在2025年量产2纳米制程芯片",
        source_url="https://news.example.com/1",
        confidence=0.9
    )

    fact3 = AtomicFact(
        text="台积电将于2025年实现2纳米芯片量产计划",
        source_url="https://news.example.com/2",
        confidence=0.85
    )

    print("\n[2.1] Adding fact 1...")
    added1, _ = await km.add_facts([fact1])
    print(f"  Added: {added1}")

    print("\n[2.2] Adding duplicate fact (same URL)...")
    added2, dup2 = await km.add_facts([fact2])
    print(f"  Added: {added2}, Duplicates: {dup2}")

    print("\n[2.3] Adding similar fact (different URL)...")
    added3, dup3 = await km.add_facts([fact3])
    print(f"  Added: {added3}, Duplicates: {dup3}")

    stats = km.get_stats()
    print(f"\n[2.4] Stats: {stats.duplicates_merged} duplicates merged, {stats.verified_facts} verified")

    await km.clear_collection()


async def test_distiller():
    print("\n" + "=" * 60)
    print("TEST 3: Distiller Agent")
    print("=" * 60)

    distiller = DistillerAgent()

    test_markdown = """
# 台积电2纳米芯片进展

台积电近日宣布，其2纳米制程技术研发进展顺利，预计将在2025年下半年开始量产。
这款芯片将采用全新的Gate-All-Around晶体管架构，与当前的FinFET相比，性能提升可达10-15%。

台积电CEO魏哲家表示，公司已投资超过200亿美元用于2纳米研发。
预计首款2纳米芯片将为苹果的下一代iPhone提供支持。

三星电子也在积极开发3纳米GAA技术，计划在2025年实现量产。
英特尔则宣布其1.8纳米技术将在2024年底开始风险生产。
"""

    print("\n[3.1] Distilling facts from markdown...")
    result = await distiller.distill(test_markdown, "https://test.example.com/tech")

    print(f"\n[3.2] Extracted {len(result.facts)} facts:")
    for i, fact in enumerate(result.facts, 1):
        print(f"    {i}. {fact.text[:60]}...")
        print(f"       Confidence: {fact.confidence:.2f}, Source: {fact.source_url}")

    return result.facts


async def test_full_pipeline():
    print("\n" + "=" * 60)
    print("TEST 4: Full Pipeline (Scrape -> Distill -> Store)")
    print("=" * 60)

    from providers import MCPGateway, SmartScraper

    gateway = MCPGateway()
    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)
    distiller = DistillerAgent()
    km = KnowledgeManager(storage_path="./test_knowledge_data")

    await km.clear_collection()

    print("\n[4.1] Searching for chip news...")
    search_results = await gateway.search("国产芯片进展 2024", max_results=2, provider="tavily")

    if not search_results:
        print("  No search results, using mock data")
        test_markdown = """
台积电是全球最大的芯片代工厂，其2纳米制程技术进展备受关注。
公司计划在2025年实现2纳米量产，这将显著提升芯片性能。

华为自研的麒麟芯片在2023年取得突破，麒麟9000S采用7纳米工艺。
中芯国际也在不断提升产能，计划在2024年实现14纳米大规模量产。

三星电子宣布将在2025年推出3纳米GAA芯片，功耗将降低30%。
        """
        scraped_urls = ["https://mock.example.com/1"]
        scraped_data = [{"url": scraped_urls[0], "markdown": test_markdown, "title": "Mock Article"}]
    else:
        urls = [r.url for r in search_results[:2]]
        print(f"\n[4.2] Scraping {len(urls)} URLs...")
        scraped = await scraper.scrape_batch(urls, force_playwright=False)
        scraped_data = [{"url": s.url, "markdown": s.markdown, "title": s.title} for s in scraped if s.markdown]

    if not scraped_data:
        print("  No scraped data available")
        return

    print(f"\n[4.3] Distilling facts from {len(scraped_data)} pages...")
    dist_results = await distiller.distill_batch(scraped_data)

    all_facts = []
    for i, result in enumerate(dist_results):
        print(f"  Page {i+1}: extracted {len(result.facts)} facts")
        all_facts.extend(result.facts)

    print(f"\n[4.4] Storing {len(all_facts)} facts in Knowledge Base...")
    added_ids, dup_ids = await km.add_facts(all_facts)
    print(f"  Added: {len(added_ids)}, Duplicates merged: {len(dup_ids)}")

    print("\n[4.5] Searching for '台积电'...")
    results = await km.search_facts("台积电 芯片", limit=5)
    print(f"  Found {len(results)} facts:")
    for r in results:
        print(f"    - {r['text'][:50]}... (score: {r['score']:.2f})")
        print(f"      Source: {r['source_url']}")

    stats = km.get_stats()
    print(f"\n[4.6] Final stats:")
    print(f"    Total: {stats.total_facts}, Verified: {stats.verified_facts}, Conflicts: {stats.conflicts_detected}")

    await km.clear_collection()
    await scraper.close()


async def main():
    print("\n" + "=" * 60)
    print("AIRE Knowledge Layer Test Suite")
    print("=" * 60)

    await test_knowledge_manager()
    await test_semantic_dedup()
    await test_distiller()
    await test_full_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)

    import shutil
    if os.path.exists("./test_knowledge_data"):
        try:
            shutil.rmtree("./test_knowledge_data")
            print("\nCleaned up test data directory")
        except Exception as e:
            print(f"\nNote: Could not clean up test data: {e}")


if __name__ == "__main__":
    asyncio.run(main())
