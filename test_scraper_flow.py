import asyncio
import time
import json
from datetime import datetime
from providers import SmartScraper
from agents import DistillerAgent


TEST_URLS = [
    "https://www.reuters.com/technology/tsmc-likely-see-profit-jump-ai-chip-demand-2024-2024-01-18/",
    "https://www.bloomberg.com/news/articles/2024-01-15/tsmc-advances-2nm-chip-production-timeline",
    "https://techcrunch.com/2024/01/10/tsmc-arizona-chip-factory-delay/",
    "https://www.cnbc.com/2024/01/12/tsmc-revenue-forecast-ai-chip-demand.html",
    "https://www.ft.com/content/tsmc-2nm-chip-technology",
    "https://www.wsj.com/tech/tsmc-chip-manufacturing-ai-2024",
    "https://www.bbc.com/news/technology-tsMC-chip-production",
    "https://www.theguardian.com/technology/2024/tsmc-taiwan-semiconductor",
    "https://www.economist.com/business/tsmc-global-chip-shortage",
    "https://www.wired.com/story/tsmc-chip-manufacturing-future/",
]

SIMPLE_TEST_URLS = [
    "https://en.wikipedia.org/wiki/TSMC",
    "https://www.bbc.com/news/technology",
    "https://techcrunch.com",
    "https://www.reuters.com/technology",
    "https://www.cnbc.com/technology",
    "https://www.ft.com/technology",
    "https://www.wsj.com/tech",
    "https://www.theverge.com/tech",
    "https://arstechnica.com/technology",
    "https://www.wired.com",
]


async def test_single_url(scraper: SmartScraper, url: str, index: int):
    print(f"\n[{index+1}/10] Testing: {url}")
    
    start_time = time.time()
    
    try:
        scraped = await scraper.scrape(url, force_playwright=False)
        
        fetch_time = time.time() - start_time
        
        if scraped.error:
            print(f"  ❌ Error: {scraped.error}")
            return {
                "url": url,
                "success": False,
                "error": scraped.error,
                "fetch_time": fetch_time
            }
        
        original_length = len(scraped.markdown)
        
        cleaned_markdown, denoise_stats = scraper._level1_heuristic_clean(scraped.markdown)
        cleaned_length = len(cleaned_markdown)
        
        denoise_time = time.time() - start_time - fetch_time
        
        print(f"  ✅ Original: {original_length} chars")
        print(f"  ✅ Cleaned: {cleaned_length} chars (removed {denoise_stats.denoise_rate:.1%})")
        print(f"  ⏱️ Fetch: {fetch_time:.2f}s, Denoise: {denoise_time:.2f}s")
        
        return {
            "url": url,
            "success": True,
            "title": scraped.title,
            "fetch_method": scraped.fetch_method,
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "denoise_rate": denoise_stats.denoise_rate,
            "fetch_time": fetch_time,
            "denoise_time": denoise_time,
            "total_time": fetch_time + denoise_time,
            "link_density_removed": denoise_stats.link_density_removed,
            "risk_block_removed": denoise_stats.risk_block_removed,
        }
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return {
            "url": url,
            "success": False,
            "error": str(e),
            "fetch_time": time.time() - start_time
        }


async def main():
    print("=" * 70)
    print("网页爬取到Distiller流程测试")
    print("=" * 70)
    print(f"测试时间: {datetime.now().isoformat()}")
    print(f"测试URL数量: {len(SIMPLE_TEST_URLS)}")
    
    scraper = SmartScraper(timeout=30.0, maxConcurrency=3)
    
    results = []
    
    for i, url in enumerate(SIMPLE_TEST_URLS):
        result = await test_single_url(scraper, url, i)
        results.append(result)
        await asyncio.sleep(1)
    
    await scraper.close()
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    
    if successful:
        avg_original = sum(r["original_length"] for r in successful) / len(successful)
        avg_cleaned = sum(r["cleaned_length"] for r in successful) / len(successful)
        avg_denoise_rate = sum(r["denoise_rate"] for r in successful) / len(successful)
        avg_fetch_time = sum(r["fetch_time"] for r in successful) / len(successful)
        avg_denoise_time = sum(r["denoise_time"] for r in successful) / len(successful)
        avg_total_time = sum(r["total_time"] for r in successful) / len(successful)
        
        max_original = max(r["original_length"] for r in successful)
        min_original = min(r["original_length"] for r in successful)
        max_cleaned = max(r["cleaned_length"] for r in successful)
        min_cleaned = min(r["cleaned_length"] for r in successful)
        
        print(f"\n📊 长度统计:")
        print(f"  原始长度 - 平均: {avg_original:.0f} 字符, 最大: {max_original}, 最小: {min_original}")
        print(f"  清洗后长度 - 平均: {avg_cleaned:.0f} 字符, 最大: {max_cleaned}, 最小: {min_cleaned}")
        print(f"  平均降噪率: {avg_denoise_rate:.1%}")
        
        print(f"\n⏱️ 时间统计:")
        print(f"  平均抓取时间: {avg_fetch_time:.2f}s")
        print(f"  平均降噪时间: {avg_denoise_time:.4f}s")
        print(f"  平均总时间: {avg_total_time:.2f}s")
        
        over_8000 = [r for r in successful if r["cleaned_length"] > 8000]
        over_30000 = [r for r in successful if r["cleaned_length"] > 30000]
        over_50000 = [r for r in successful if r["cleaned_length"] > 50000]
        
        print(f"\n📏 长度分布:")
        print(f"  > 8,000字符: {len(over_8000)}/{len(successful)} ({len(over_8000)/len(successful)*100:.1f}%)")
        print(f"  > 30,000字符: {len(over_30000)}/{len(successful)} ({len(over_30000)/len(successful)*100:.1f}%)")
        print(f"  > 50,000字符: {len(over_50000)}/{len(successful)} ({len(over_50000)/len(successful)*100:.1f}%)")
    
    if failed:
        print(f"\n❌ 失败详情:")
        for r in failed:
            print(f"  {r['url']}: {r.get('error', 'Unknown error')}")
    
    report = {
        "test_time": datetime.now().isoformat(),
        "total_urls": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "results": results,
        "summary": {
            "avg_original_length": avg_original if successful else 0,
            "avg_cleaned_length": avg_cleaned if successful else 0,
            "avg_denoise_rate": avg_denoise_rate if successful else 0,
            "avg_fetch_time": avg_fetch_time if successful else 0,
            "avg_denoise_time": avg_denoise_time if successful else 0,
            "avg_total_time": avg_total_time if successful else 0,
            "max_original_length": max_original if successful else 0,
            "min_original_length": min_original if successful else 0,
            "max_cleaned_length": max_cleaned if successful else 0,
            "min_cleaned_length": min_cleaned if successful else 0,
        } if successful else {}
    }
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细结果已保存到 test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
