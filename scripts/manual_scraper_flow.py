"""
测试脚本: 网页爬取到Distiller流程测试
测试10个真实网站，保存原始文件和清洗后文件供对比
"""

import asyncio
import sys
import time
import json
import os
from datetime import datetime

sys.path.insert(0, '.')

from providers import SmartScraper


TEST_URLS = [
    ("https://en.wikipedia.org/wiki/TSMC", "Wikipedia - TSMC"),
    ("https://www.trendforce.com/news/2024/12/16/news-tsmc-reveals-n2-nanosheet-details-35-power-savings-15-performance-gain-densest-sram-cell-yet/", "TrendForce - TSMC N2"),
    ("https://en.wikipedia.org/wiki/Semiconductor", "Wikipedia - Semiconductor"),
    ("https://en.wikipedia.org/wiki/Artificial_intelligence", "Wikipedia - AI"),
    ("https://en.wikipedia.org/wiki/Nvidia", "Wikipedia - Nvidia"),
    ("https://en.wikipedia.org/wiki/Apple_Inc.", "Wikipedia - Apple"),
    ("https://en.wikipedia.org/wiki/Microprocessor", "Wikipedia - Microprocessor"),
    ("https://en.wikipedia.org/wiki/Integrated_circuit", "Wikipedia - IC"),
    ("https://en.wikipedia.org/wiki/Moore%27s_law", "Wikipedia - Moore's Law"),
    ("https://en.wikipedia.org/wiki/FinFET", "Wikipedia - FinFET"),
]

OUTPUT_DIR = "test_output"
ORIGINAL_TRUNCATE_SIZE = 100000  # 原始文件截取大小：10万字符


async def test_single_url(scraper, url, name, index):
    print(f"\n{'='*60}")
    print(f"Testing [{index+1}/10]: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        scraped = await scraper.scrape(url, force_playwright=False)
        
        fetch_time = time.time() - start_time
        
        markdown = scraped.markdown
        original_length = len(markdown) if markdown else 0
        
        if scraped.error:
            print(f"  ❌ Error: {scraped.error}")
            return {
                "name": name,
                "url": url,
                "success": False,
                "error": scraped.error,
                "original_length": original_length,
                "fetch_time": fetch_time,
            }
        
        level1_cleaned, denoise_stats = scraper._level1_heuristic_clean(markdown)
        
        level1_time = time.time() - start_time - fetch_time
        level1_length = len(level1_cleaned)
        
        # 保存文件
        safe_name = name.replace(" ", "_").replace("/", "_")
        
        # 保存原始文件（截取前10万字符）
        original_truncated = markdown[:ORIGINAL_TRUNCATE_SIZE]
        original_file = os.path.join(OUTPUT_DIR, f"{index+1:02d}_{safe_name}_original.txt")
        with open(original_file, "w", encoding="utf-8") as f:
            f.write(original_truncated)
        
        # 保存完整清洗后的文件
        cleaned_file = os.path.join(OUTPUT_DIR, f"{index+1:02d}_{safe_name}_cleaned.txt")
        with open(cleaned_file, "w", encoding="utf-8") as f:
            f.write(level1_cleaned)
        
        print(f"  ✓ Fetch: {fetch_time:.2f}s, Original: {original_length:,} chars")
        print(f"  ✓ Level1 Clean: {level1_time:.4f}s, Cleaned: {level1_length:,} chars")
        print(f"  ✓ Denoise Rate: {denoise_stats.denoise_rate:.1%}")
        print(f"  ✓ Files saved:")
        print(f"    - Original (truncated): {original_file}")
        print(f"    - Cleaned (full): {cleaned_file}")
        
        return {
            "name": name,
            "url": url,
            "success": True,
            "original_length": original_length,
            "cleaned_length": level1_length,
            "denoise_rate": denoise_stats.denoise_rate,
            "fetch_time": fetch_time,
            "level1_time": level1_time,
            "total_time": fetch_time + level1_time,
            "link_density_removed": denoise_stats.link_density_removed,
            "risk_block_removed": denoise_stats.risk_block_removed,
            "original_file": original_file,
            "cleaned_file": cleaned_file,
        }
        
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return {
            "name": name,
            "url": url,
            "success": False,
            "error": str(e),
            "original_length": 0,
            "fetch_time": time.time() - start_time,
        }


async def main():
    print("="*70)
    print("网页爬取到Distiller流程测试")
    print("="*70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试URL数量: {len(TEST_URLS)}")
    print(f"原始文件截取大小: {ORIGINAL_TRUNCATE_SIZE:,} 字符")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    scraper = SmartScraper(timeout=30.0, maxConcurrency=3)
    
    results = []
    
    for i, (url, name) in enumerate(TEST_URLS):
        result = await test_single_url(scraper, url, name, i)
        results.append(result)
        await asyncio.sleep(0.5)
    
    await scraper.close()
    
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    successful = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]
    
    print(f"\n成功: {len(successful)}/{len(results)}")
    print(f"失败: {len(failed)}/{len(results)}")
    
    if successful:
        avg_original = sum(r["original_length"] for r in successful) / len(successful)
        avg_cleaned = sum(r["cleaned_length"] for r in successful) / len(successful)
        avg_denoise_rate = sum(r["denoise_rate"] for r in successful) / len(successful)
        avg_fetch_time = sum(r["fetch_time"] for r in successful) / len(successful)
        avg_level1_time = sum(r["level1_time"] for r in successful) / len(successful)
        avg_total_time = sum(r["total_time"] for r in successful) / len(successful)
        
        max_original = max(r["original_length"] for r in successful)
        min_original = min(r["original_length"] for r in successful)
        max_cleaned = max(r["cleaned_length"] for r in successful)
        min_cleaned = min(r["cleaned_length"] for r in successful)
        
        over_8000 = len([r for r in successful if r["cleaned_length"] > 8000])
        over_30000 = len([r for r in successful if r["cleaned_length"] > 30000])
        over_50000 = len([r for r in successful if r["cleaned_length"] > 50000])
        over_100000 = len([r for r in successful if r["cleaned_length"] > 100000])
        over_150000 = len([r for r in successful if r["cleaned_length"] > 150000])
        
        print(f"\n📊 长度统计:")
        print(f"  原始长度 - 平均: {avg_original:,.0f} 字符")
        print(f"  原始长度 - 最大: {max_original:,} 字符, 最小: {min_original:,} 字符")
        print(f"  清洗后长度 - 平均: {avg_cleaned:,.0f} 字符")
        print(f"  清洗后长度 - 最大: {max_cleaned:,} 字符, 最小: {min_cleaned:,} 字符")
        print(f"  平均降噪率: {avg_denoise_rate:.1%}")
        
        print(f"\n⏱️ 时间统计:")
        print(f"  平均抓取时间: {avg_fetch_time:.2f}s")
        print(f"  平均清洗时间: {avg_level1_time:.4f}s")
        print(f"  平均总时间: {avg_total_time:.2f}s")
        
        print(f"\n📏 长度分布 (清洗后):")
        print(f"  > 8,000字符: {over_8000}/{len(successful)} ({over_8000/len(successful)*100:.1f}%)")
        print(f"  > 30,000字符: {over_30000}/{len(successful)} ({over_30000/len(successful)*100:.1f}%)")
        print(f"  > 50,000字符: {over_50000}/{len(successful)} ({over_50000/len(successful)*100:.1f}%)")
        print(f"  > 100,000字符: {over_100000}/{len(successful)} ({over_100000/len(successful)*100:.1f}%)")
        print(f"  > 150,000字符: {over_150000}/{len(successful)} ({over_150000/len(successful)*100:.1f}%)")
        
        print(f"\n📁 输出文件:")
        print(f"  目录: {OUTPUT_DIR}/")
        print(f"  文件格式: XX_名称_original.txt (原始, 截取前{ORIGINAL_TRUNCATE_SIZE:,}字符)")
        print(f"           XX_名称_cleaned.txt (清洗后, 完整)")
        
        print(f"\n📋 详细结果:")
        print("-"*60)
        for r in successful:
            print(f"\n{r['name']}:")
            print(f"  原始长度: {r['original_length']:,} 字符")
            print(f"  清洗后长度: {r['cleaned_length']:,} 字符")
            print(f"  降噪率: {r['denoise_rate']:.1%}")
            print(f"  抓取时间: {r['fetch_time']:.2f}s")
            print(f"  清洗时间: {r['level1_time']:.4f}s")
    
    if failed:
        print(f"\n❌ 失败的URL:")
        for r in failed:
            print(f"  {r['name']}: {r.get('error', 'Unknown error')}")
    
    report = {
        "test_time": datetime.now().isoformat(),
        "total_urls": len(results),
        "successful": len(successful),
        "failed": len(failed),
        "original_truncate_size": ORIGINAL_TRUNCATE_SIZE,
        "results": results,
        "summary": {
            "avg_original_length": avg_original if successful else 0,
            "avg_cleaned_length": avg_cleaned if successful else 0,
            "avg_denoise_rate": avg_denoise_rate if successful else 0,
            "avg_fetch_time": avg_fetch_time if successful else 0,
            "avg_level1_time": avg_level1_time if successful else 0,
            "avg_total_time": avg_total_time if successful else 0,
            "max_original_length": max_original if successful else 0,
            "min_original_length": min_original if successful else 0,
            "max_cleaned_length": max_cleaned if successful else 0,
            "min_cleaned_length": min_cleaned if successful else 0,
            "over_8000_count": over_8000 if successful else 0,
            "over_30000_count": over_30000 if successful else 0,
            "over_50000_count": over_50000 if successful else 0,
            "over_100000_count": over_100000 if successful else 0,
            "over_150000_count": over_150000 if successful else 0,
        } if successful else {}
    }
    
    with open("test_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n📄 详细结果已保存到 test_results.json")


if __name__ == "__main__":
    asyncio.run(main())
