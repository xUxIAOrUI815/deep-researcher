import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from providers import MCPGateway, SmartScraper, DenoiseStats


async def research_topic(topic: str):
    print("=" * 60)
    print(f"研究主题: {topic}")
    print("=" * 60)

    gateway = MCPGateway()
    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    print("\n[1] 调用 Tavily 搜索...")
    search_results = await gateway.search(topic, max_results=5, provider="tavily")

    if not search_results:
        print("未找到相关结果")
        return

    print(f"\n找到 {len(search_results)} 条搜索结果，选择前 3 条:")
    for i, result in enumerate(search_results[:3], 1):
        print(f"\n  [{i}] {result.title}")
        print(f"      URL: {result.url}")
        print(f"      相关度: {result.score:.2f}")
        print(f"      摘要: {result.snippet[:80]}...")

    urls_to_scrape = [r.url for r in search_results[:3]]

    print(f"\n[2] 使用 SmartScraper 抓取 + 三级去噪...")
    print("-" * 60)

    scraped_data, all_stats = await scraper.scrape_batch_with_denoise(
        urls_to_scrape, topic, force_playwright=False
    )

    print("\n" + "=" * 60)
    print(f"[3] 去噪统计报告")
    print("=" * 60)

    total_original = 0
    total_cleaned = 0

    for i, (data, stats) in enumerate(zip(scraped_data, all_stats), 1):
        print(f"\n第 {i} 篇: {data.url}")
        print(f"  获取方式: {data.fetch_method}")
        print(f"  语义相关性得分: {stats.semantic_score:.2f}")
        print(f"  去噪前长度: {stats.original_length} 字符")
        print(f"  去噪后长度: {stats.cleaned_length} 字符")
        print(f"  去噪率: {stats.denoise_rate:.1%}")

        if stats.was_discarded:
            print(f"  状态: ❌ 已丢弃 (Score < 0.5)")
        elif data.error:
            print(f"  状态: ⚠️ 错误 - {data.error}")
        else:
            print(f"  状态: ✅ 保留")

        total_original += stats.original_length
        total_cleaned += stats.cleaned_length

    print("\n" + "-" * 60)
    print(f"总体统计:")
    print(f"  原始总长度: {total_original} 字符")
    print(f"  清洗后总长度: {total_cleaned} 字符")
    print(f"  节省 Token (预估): {scraper.get_token_savings()} 字符")

    valid_data = [d for d, s in zip(scraped_data, all_stats) if not s.was_discarded and d.markdown]
    if valid_data:
        print(f"\n[4] 有效内容 ({len(valid_data)} 篇)，打印 Markdown:")
        print("=" * 60)

        for i, data in enumerate(valid_data, 1):
            print(f"\n{'='*60}")
            print(f"第 {i} 篇: {data.url}")
            print(f"标题: {data.title}")
            print("-" * 60)
            print("【Markdown 内容】")
            print(data.markdown)
            print("-" * 60)
    else:
        print("\n[4] 没有有效内容可显示（所有页面均被语义过滤器丢弃）")

    await scraper.close()

    print("\n" + "=" * 60)
    print("抓取完成")
    print("=" * 60)


async def interactive_mode():
    print("\n" + "=" * 60)
    print("AIRE 智能研究助手 - 内容抓取测试")
    print("=" * 60)
    print("\n请输入一个垂直领域的研究话题")
    print("例如: '分析目前国产 2 纳米芯片的最新流片进展'")
    print("输入 'quit' 退出\n")

    while True:
        try:
            topic = input("研究话题> ").strip()

            if not topic:
                print("请输入有效的话题\n")
                continue

            if topic.lower() in ['quit', 'exit', 'q']:
                print("再见!")
                break

            await research_topic(topic)
            print("\n可以继续输入新话题，或输入 'quit' 退出\n")

        except KeyboardInterrupt:
            print("\n\n已退出")
            break
        except Exception as e:
            print(f"\n错误: {e}\n")


async def test_mcp_gateway():
    print("=" * 60)
    print("TEST 1: MCP Gateway - Tavily Search")
    print("=" * 60)

    gateway = MCPGateway()

    tool_defs = gateway.get_tool_definitions()
    print(f"Available tools: {[t['name'] for t in tool_defs]}")

    results = await gateway.search("artificial intelligence trends 2024", max_results=5, provider="tavily")

    print(f"\nSearch returned {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] {result.title}")
        print(f"      URL: {result.url}")
        print(f"      Score: {result.score}")
        print(f"      Snippet: {result.snippet[:100]}...")

    return results


async def test_exa_provider():
    print("\n" + "=" * 60)
    print("TEST 2: MCP Gateway - Exa Search")
    print("=" * 60)

    gateway = MCPGateway()

    results = await gateway.search("machine learning breakthroughs", max_results=3, provider="exa")

    print(f"\nExa search returned {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n  [{i}] {result.title}")
        print(f"      URL: {result.url}")

    return results


async def test_smart_scraper():
    print("\n" + "=" * 60)
    print("TEST 3: SmartScraper - Jina Reader")
    print("=" * 60)

    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    test_urls = [
        "https://example.com",
        "https://httpbin.org/html",
    ]

    print(f"\nScraping {len(test_urls)} URLs...")
    results = await scraper.scrape_batch(test_urls, force_playwright=False)

    for result in results:
        print(f"\n  URL: {result.url}")
        print(f"  Method: {result.fetch_method}")
        print(f"  Title: {result.title[:50] if result.title else 'N/A'}...")
        print(f"  Content length: {len(result.markdown)} chars")
        print(f"  Error: {result.error if result.error else 'None'}")

        if result.markdown:
            print(f"  Preview: {result.markdown[:150].replace(chr(10), ' ')}...")

    await scraper.close()
    return results


async def test_playwright_fallback():
    print("\n" + "=" * 60)
    print("TEST 4: SmartScraper - Playwright Fallback")
    print("=" * 60)

    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    print("\nForcing Playwright mode (simulating anti-scraping scenario)...")
    result = await scraper.scrape("https://example.com", force_playwright=True)

    print(f"\n  URL: {result.url}")
    print(f"  Method: {result.fetch_method}")
    print(f"  Title: {result.title}")
    print(f"  Content length: {len(result.markdown)} chars")

    await scraper.close()
    return result


async def test_denoiing():
    print("\n" + "=" * 60)
    print("TEST 5: Markdown Denoising")
    print("=" * 60)

    scraper = SmartScraper()

    noisy_md = """
# Navigation

Home | About | Contact | Login

## Main Content

This is the actual article content about AI technology.

## Related Articles

- Article 1
- Article 2

Advertisement

Subscribe to our newsletter!

## Footer

Copyright 2024 | Privacy Policy | Terms
    """

    cleaned = scraper._level1_heuristic_clean(noisy_md)[0]

    print("Original length:", len(noisy_md))
    print("Cleaned length:", len(cleaned))
    print("\nCleaned content:")
    print(cleaned)


async def test_concurrent_scraping():
    print("\n" + "=" * 60)
    print("TEST 6: Concurrent Scraping with Semaphore")
    print("=" * 60)

    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    urls = [
        "https://example.com",
        "https://httpbin.org/html",
        "https://example.org",
        "https://example.net",
    ]

    print(f"\nScraping {len(urls)} URLs concurrently (max 3 at a time)...")

    import time
    start = time.time()
    results = await scraper.scrape_batch(urls, force_playwright=False)
    elapsed = time.time() - start

    print(f"\nCompleted in {elapsed:.2f}s")
    print(f"Successful: {sum(1 for r in results if r.markdown)}")
    print(f"Failed: {sum(1 for r in results if r.error)}")

    await scraper.close()
    return results


async def test_anti_scraping_simulation():
    print("\n" + "=" * 60)
    print("TEST 7: Anti-Scraping Fallback (Zhihu-like)")
    print("=" * 60)

    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    print("\nSimulating a website with anti-scraping measures...")
    print("(Using httpbin to simulate 403 response via Jina)")

    result = await scraper.scrape("https://example.com", force_playwright=True)

    print(f"\n  URL: {result.url}")
    print(f"  Fallback used: {result.fetch_method == 'playwright'}")
    print(f"  Success: {bool(result.markdown)}")

    await scraper.close()
    return result


async def test_denoise_pipeline():
    print("\n" + "=" * 60)
    print("TEST 8: Three-Level Denoise Pipeline")
    print("=" * 60)

    scraper = SmartScraper(timeout=20.0, maxConcurrency=3)

    test_urls = [
        "https://xueqiu.com/3951090421/375939087",
    ]

    query = "2纳米芯片 流片进展"

    print(f"\nScraping with denoise for query: {query}")
    scraped_data, all_stats = await scraper.scrape_batch_with_denoise(
        test_urls, query, force_playwright=False
    )

    for data, stats in zip(scraped_data, all_stats):
        print(f"\n  URL: {data.url}")
        print(f"  Original: {stats.original_length} chars")
        print(f"  Cleaned: {stats.cleaned_length} chars")
        print(f"  Denoise Rate: {stats.denoise_rate:.1%}")
        print(f"  Semantic Score: {stats.semantic_score:.2f}")
        print(f"  Discarded: {stats.was_discarded}")

        if data.markdown and not stats.was_discarded:
            print(f"  Preview: {data.markdown[:200]}...")

    await scraper.close()


async def run_all_tests():
    await test_mcp_gateway()
    await test_exa_provider()
    await test_smart_scraper()
    await test_playwright_fallback()
    await test_denoiing()
    await test_concurrent_scraping()
    await test_anti_scraping_simulation()
    await test_denoise_pipeline()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)


async def main():
    if len(sys.argv) > 1:
        topic = " ".join(sys.argv[1:])
        await research_topic(topic)
    else:
        mode = input("选择模式:\n  1. 交互模式 (输入话题抓取内容)\n  2. 运行所有测试\n\n请输入 (1/2): ").strip()

        if mode == "2":
            await run_all_tests()
        else:
            await interactive_mode()


if __name__ == "__main__":
    asyncio.run(main())
