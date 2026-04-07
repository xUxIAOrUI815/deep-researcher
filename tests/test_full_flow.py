"""
全流程测试脚本
测试从输入搜索问题到输出markdown的完整流程
保存所有中间输出和最终输出到测试目录
"""

import asyncio
import sys
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from providers import MCPGateway, SmartScraper
from agents import DistillerAgent
from core.content_transformer import ContentTransformer, StructuredContent
from core.semantic_chunker import SemanticChunker, ChunkingResult, RelevantChunk


TEST_QUERY = "对比 HBM4（第四代高带宽内存）标准下，SK海力士、三星电子和美光的量产进度表。重点识别关于'样品交付时间'和'量产年份'在各家官宣与产业链传闻（如 Digitimes 或 TrendForce）中的具体差异点。"

REAL_URLS = [
    "https://www.trendforce.com/news/2024/03/15/news-hbm4-development-sk-hynix-samsung-micron/",
    "https://www.digitimes.com/news/a20240315PD201/hbm4-memory-sk-hynix-samsung-micron.html",
    "https://www.koreaherald.com/view.php?ud=20240315000501",
]

OUTPUT_DIR = "test_full_flow"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(data: Any, filename: str, subdir: str = ""):
    path = os.path.join(OUTPUT_DIR, subdir, filename) if subdir else os.path.join(OUTPUT_DIR, filename)
    ensure_dir(os.path.dirname(path))
    
    def json_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=json_serializer)
    print(f"  Saved: {path}")
    return path


def save_text(text: str, filename: str, subdir: str = ""):
    path = os.path.join(OUTPUT_DIR, subdir, filename) if subdir else os.path.join(OUTPUT_DIR, filename)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Saved: {path}")
    return path


async def test_current_flow():
    print("=" * 70)
    print("测试当前流程 (搜索 → 截断 → Distiller)")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试问题: {TEST_QUERY}")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = f"current_{timestamp}"
    ensure_dir(os.path.join(OUTPUT_DIR, current_dir))
    
    all_results = {
        "test_query": TEST_QUERY,
        "test_time": datetime.now().isoformat(),
        "flow_type": "current",
        "steps": {}
    }
    
    start_total = time.time()
    
    # Step 1: 使用预设的真实URL
    print("\n[Step 1] 使用预设URL...")
    step1_start = time.time()
    
    urls = REAL_URLS[:3]
    
    step1_time = time.time() - step1_start
    print(f"  使用 {len(urls)} 个预设URL, 耗时 {step1_time:.2f}s")
    
    save_json(urls, "1_urls.json", current_dir)
    
    all_results["steps"]["urls"] = {
        "time": step1_time,
        "count": len(urls),
        "urls": urls
    }
    
    # Step 2: 抓取
    print("\n[Step 2] 抓取网页...")
    step2_start = time.time()
    
    scraper = SmartScraper(timeout=30.0, maxConcurrency=3)
    
    scraped_data = await scraper.scrape_batch(urls, force_playwright=False)
    
    step2_time = time.time() - step2_start
    print(f"  抓取 {len(scraped_data)} 个网页, 耗时 {step2_time:.2f}s")
    
    scraped_list = []
    for i, scraped in enumerate(scraped_data):
        data = scraped.model_dump()
        scraped_list.append(data)
        
        # 保存原始markdown
        if data.get("markdown"):
            save_text(
                data["markdown"][:50000],  # 截取前50000字符
                f"2_{i+1}_raw_markdown.txt",
                f"{current_dir}/scraped"
            )
    
    save_json(scraped_list, "2_scraped_data.json", current_dir)
    
    all_results["steps"]["scrape"] = {
        "time": step2_time,
        "count": len(scraped_data),
        "urls": urls
    }
    
    # Step 3: 清洗
    print("\n[Step 3] 清洗...")
    step3_start = time.time()
    
    cleaned_data = []
    for i, scraped in enumerate(scraped_data):
        if scraped.markdown and not scraped.error:
            cleaned_markdown, denoise_stats = scraper._level1_heuristic_clean(scraped.markdown)
            
            cleaned_data.append({
                "url": scraped.url,
                "title": scraped.title,
                "original_length": len(scraped.markdown),
                "cleaned_length": len(cleaned_markdown),
                "denoise_rate": denoise_stats.denoise_rate,
                "markdown": cleaned_markdown
            })
            
            # 保存清洗后markdown
            save_text(
                cleaned_markdown,
                f"3_{i+1}_cleaned_markdown.txt",
                f"{current_dir}/cleaned"
            )
    
    step3_time = time.time() - step3_start
    print(f"  清洗 {len(cleaned_data)} 个文档, 耗时 {step3_time:.2f}s")
    
    all_results["steps"]["clean"] = {
        "time": step3_time,
        "count": len(cleaned_data),
        "stats": [{
            "url": d["url"],
            "original_length": d["original_length"],
            "cleaned_length": d["cleaned_length"],
            "denoise_rate": d["denoise_rate"]
        } for d in cleaned_data]
    }
    
    # Step 4: 截断 (当前流程)
    print("\n[Step 4] 截断 (当前流程: 8000字符)...")
    step4_start = time.time()
    
    TRUNCATE_SIZE = 8000
    truncated_data = []
    
    for i, data in enumerate(cleaned_data):
        truncated = data["markdown"][:TRUNCATE_SIZE]
        truncated_data.append({
            "url": data["url"],
            "title": data["title"],
            "original_length": data["cleaned_length"],
            "truncated_length": len(truncated),
            "retention_rate": len(truncated) / data["cleaned_length"] if data["cleaned_length"] > 0 else 0,
            "markdown": truncated
        })
        
        # 保存截断后markdown
        save_text(
            truncated,
            f"4_{i+1}_truncated_markdown.txt",
            f"{current_dir}/truncated"
        )
    
    step4_time = time.time() - step4_start
    print(f"  截断 {len(truncated_data)} 个文档, 耗时 {step4_time:.4f}s")
    
    all_results["steps"]["truncate"] = {
        "time": step4_time,
        "count": len(truncated_data),
        "truncate_size": TRUNCATE_SIZE,
        "stats": [{
            "url": d["url"],
            "original_length": d["original_length"],
            "truncated_length": d["truncated_length"],
            "retention_rate": d["retention_rate"]
        } for d in truncated_data]
    }
    
    # Step 5: Distiller提取
    print("\n[Step 5] Distiller提取事实...")
    step5_start = time.time()
    
    distiller = DistillerAgent()
    all_facts = []
    
    for i, data in enumerate(truncated_data):
        try:
            result = await distiller.distill(
                markdown_text=data["markdown"],
                source_url=data["url"],
                task_id=f"test_{i}"
            )
            
            facts_data = [f.model_dump() for f in result.facts]
            all_facts.extend(facts_data)
            
            # 保存提取的事实
            save_json(
                facts_data,
                f"5_{i+1}_extracted_facts.json",
                f"{current_dir}/facts"
            )
            
            print(f"  从文档 {i+1} 提取 {len(facts_data)} 个事实")
        except Exception as e:
            print(f"  文档 {i+1} 提取失败: {e}")
    
    step5_time = time.time() - step5_start
    print(f"  总计提取 {len(all_facts)} 个事实, 耗时 {step5_time:.2f}s")
    
    all_results["steps"]["distill"] = {
        "time": step5_time,
        "total_facts": len(all_facts),
        "facts": all_facts
    }
    
    # Step 6: 生成报告
    print("\n[Step 6] 生成报告...")
    step6_start = time.time()
    
    # 格式化事实
    facts_text = []
    for i, fact in enumerate(all_facts, 1):
        facts_text.append(f"[{i}] {fact.get('text', '')} (来源: {fact.get('source_url', 'N/A')}, 置信度: {fact.get('confidence', 0):.2f})")
    
    facts_context = "\n\n".join(facts_text)
    
    # 使用DeepSeek生成报告
    report = await generate_report(TEST_QUERY, facts_context)
    
    step6_time = time.time() - step6_start
    print(f"  生成报告, 耗时 {step6_time:.2f}s")
    
    # 保存报告
    save_text(report, "6_final_report.md", current_dir)
    
    all_results["steps"]["report"] = {
        "time": step6_time,
        "report_length": len(report)
    }
    
    total_time = time.time() - start_total
    
    all_results["total_time"] = total_time
    all_results["final_report"] = report
    
    # 保存完整结果
    save_json(all_results, "full_results.json", current_dir)
    
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print(f"总耗时: {total_time:.2f}s")
    print(f"预设URL: {len(urls)}")
    print(f"抓取网页: {len(scraped_data)}")
    print(f"提取事实: {len(all_facts)}")
    print(f"报告长度: {len(report)} 字符")
    print(f"输出目录: {OUTPUT_DIR}/{current_dir}/")
    
    await scraper.close()
    
    return all_results


async def test_new_flow():
    """
    测试新流程: 搜索 → 结构化清洗 → 语义分块+筛选 → Distiller
    """
    print("\n")
    print("=" * 70)
    print("测试新流程 (搜索 → 结构化清洗 → 语义分块+筛选 → Distiller)")
    print("=" * 70)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试问题: {TEST_QUERY}")
    print()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = f"new_{timestamp}"
    ensure_dir(os.path.join(OUTPUT_DIR, new_dir))
    
    all_results = {
        "test_query": TEST_QUERY,
        "test_time": datetime.now().isoformat(),
        "flow_type": "new",
        "steps": {}
    }
    
    start_total = time.time()
    
    # Step 1: 使用预设的真实URL
    print("\n[Step 1] 使用预设URL...")
    step1_start = time.time()
    
    urls = REAL_URLS[:3]
    
    step1_time = time.time() - step1_start
    print(f"  使用 {len(urls)} 个预设URL, 耗时 {step1_time:.2f}s")
    
    save_json(urls, "1_urls.json", new_dir)
    
    all_results["steps"]["urls"] = {
        "time": step1_time,
        "count": len(urls),
        "urls": urls
    }
    
    # Step 2: 抓取
    print("\n[Step 2] 抓取网页...")
    step2_start = time.time()
    
    scraper = SmartScraper(timeout=30.0, maxConcurrency=3)
    
    scraped_data = await scraper.scrape_batch(urls, force_playwright=False)
    
    step2_time = time.time() - step2_start
    print(f"  抓取 {len(scraped_data)} 个网页, 耗时 {step2_time:.2f}s")
    
    scraped_list = []
    for i, scraped in enumerate(scraped_data):
        data = scraped.model_dump()
        scraped_list.append(data)
        
        if data.get("markdown"):
            save_text(
                data["markdown"][:50000],
                f"2_{i+1}_raw_markdown.txt",
                f"{new_dir}/scraped"
            )
    
    save_json(scraped_list, "2_scraped_data.json", new_dir)
    
    all_results["steps"]["scrape"] = {
        "time": step2_time,
        "count": len(scraped_data),
        "urls": urls
    }
    
    # Step 3: 结构化清洗 (新流程)
    print("\n[Step 3] 结构化清洗 (ContentTransformer)...")
    step3_start = time.time()
    
    transformer = ContentTransformer()
    transformed_data = []
    
    for i, scraped in enumerate(scraped_data):
        if scraped.markdown and not scraped.error:
            structured = transformer.transform(
                markdown=scraped.markdown,
                url=scraped.url,
                title=scraped.title or ""
            )
            
            transformed_data.append({
                "url": scraped.url,
                "title": scraped.title,
                "structured": structured
            })
            
            save_text(
                structured.markdown,
                f"3_{i+1}_structured_markdown.txt",
                f"{new_dir}/structured"
            )
            
            save_json(
                {
                    "metadata": structured.metadata.__dict__,
                    "original_length": structured.original_length,
                    "cleaned_length": structured.cleaned_length,
                    "sections": structured.sections
                },
                f"3_{i+1}_metadata.json",
                f"{new_dir}/structured"
            )
    
    step3_time = time.time() - step3_start
    print(f"  结构化清洗 {len(transformed_data)} 个文档, 耗时 {step3_time:.2f}s")
    
    all_results["steps"]["transform"] = {
        "time": step3_time,
        "count": len(transformed_data),
        "stats": [{
            "url": d["url"],
            "original_length": d["structured"].original_length,
            "cleaned_length": d["structured"].cleaned_length,
            "sections_count": len(d["structured"].sections)
        } for d in transformed_data]
    }
    
    # Step 4: 语义分块与相关性筛选 (新流程)
    print("\n[Step 4] 语义分块与相关性筛选 (SemanticChunker)...")
    step4_start = time.time()
    
    chunker = SemanticChunker(
        chunk_size=1500,
        chunk_overlap=200,
        relevance_threshold=0.3,
        top_k=15
    )
    
    all_chunks = []
    chunking_results = []
    
    for i, d in enumerate(transformed_data):
        structured = d["structured"]
        
        result = await chunker.chunk_and_filter(
            text=structured.markdown,
            query=TEST_QUERY,
            source_url=d["url"],
            source_title=d["title"] or ""
        )
        
        chunking_results.append({
            "url": d["url"],
            "result": result
        })
        
        for j, chunk in enumerate(result.chunks):
            all_chunks.append(chunk)
            
            save_text(
                chunk.text,
                f"4_{i+1}_{j+1}_chunk.txt",
                f"{new_dir}/chunks"
            )
        
        print(f"  文档 {i+1}: 总块数={result.total_chunks}, 筛选后={len(result.chunks)}, 平均相关性={result.avg_relevance:.3f}")
    
    step4_time = time.time() - step4_start
    print(f"  总计 {len(all_chunks)} 个相关块, 耗时 {step4_time:.2f}s")
    
    all_results["steps"]["chunk"] = {
        "time": step4_time,
        "total_chunks": len(all_chunks),
        "stats": [{
            "url": cr["url"],
            "total_chunks": cr["result"].total_chunks,
            "filtered_chunks": cr["result"].filtered_chunks,
            "kept_chunks": len(cr["result"].chunks),
            "avg_relevance": cr["result"].avg_relevance
        } for cr in chunking_results]
    }
    
    # 保存所有块的信息
    chunks_data = [{
        "text": c.text,
        "chunk_id": c.chunk_id,
        "relevance_score": c.relevance_score,
        "source_url": c.source_url,
        "source_title": c.source_title,
        "position": c.position
    } for c in all_chunks]
    save_json(chunks_data, "4_all_chunks.json", new_dir)
    
    # Step 5: Distiller提取 (使用筛选后的块)
    print("\n[Step 5] Distiller提取事实...")
    step5_start = time.time()
    
    distiller = DistillerAgent()
    all_facts = []
    
    for i, chunk in enumerate(all_chunks):
        try:
            result = await distiller.distill(
                markdown_text=chunk.text,
                source_url=chunk.source_url,
                task_id=f"new_{i}"
            )
            
            facts_data = [f.model_dump() for f in result.facts]
            all_facts.extend(facts_data)
            
            print(f"  从块 {i+1} 提取 {len(facts_data)} 个事实")
        except Exception as e:
            print(f"  块 {i+1} 提取失败: {e}")
    
    step5_time = time.time() - step5_start
    print(f"  总计提取 {len(all_facts)} 个事实, 耗时 {step5_time:.2f}s")
    
    save_json(all_facts, "5_all_facts.json", new_dir)
    
    all_results["steps"]["distill"] = {
        "time": step5_time,
        "total_facts": len(all_facts),
        "facts": all_facts
    }
    
    # Step 6: 生成报告
    print("\n[Step 6] 生成报告...")
    step6_start = time.time()
    
    facts_text = []
    for i, fact in enumerate(all_facts, 1):
        facts_text.append(f"[{i}] {fact.get('text', '')} (来源: {fact.get('source_url', 'N/A')}, 置信度: {fact.get('confidence', 0):.2f})")
    
    facts_context = "\n\n".join(facts_text)
    
    report = await generate_report(TEST_QUERY, facts_context)
    
    step6_time = time.time() - step6_start
    print(f"  生成报告, 耗时 {step6_time:.2f}s")
    
    save_text(report, "6_final_report.md", new_dir)
    
    all_results["steps"]["report"] = {
        "time": step6_time,
        "report_length": len(report)
    }
    
    total_time = time.time() - start_total
    
    all_results["total_time"] = total_time
    all_results["final_report"] = report
    
    save_json(all_results, "full_results.json", new_dir)
    
    print("\n" + "=" * 70)
    print("新流程测试完成")
    print("=" * 70)
    print(f"总耗时: {total_time:.2f}s")
    print(f"预设URL: {len(urls)}")
    print(f"抓取网页: {len(scraped_data)}")
    print(f"结构化文档: {len(transformed_data)}")
    print(f"筛选块数: {len(all_chunks)}")
    print(f"提取事实: {len(all_facts)}")
    print(f"报告长度: {len(report)} 字符")
    print(f"输出目录: {OUTPUT_DIR}/{new_dir}/")
    
    await scraper.close()
    
    return all_results


async def generate_report(query: str, facts_context: str) -> str:
    import httpx
    import os
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    if not api_key:
        return "Error: DEEPSEEK_API_KEY not set"
    
    prompt = f"""你是一个专业的研究分析师。请根据以下事实信息，撰写一份关于"{query}"的调研报告。

事实信息:
{facts_context}

要求:
1. 结构清晰，使用Markdown格式
2. 包含标题、摘要、正文、结论
3. 每个关键信息标注来源
4. 客观呈现不同来源的信息差异

请开始撰写报告:"""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 4000,
                    "temperature": 0.7
                }
            )
            response.raise_for_status()
            data = response.json()
            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        return f"Error generating report: {e}"


if __name__ == "__main__":
    async def main():
        print("\n" + "=" * 70)
        print("开始全流程对比测试")
        print("=" * 70)
        print(f"测试问题: {TEST_QUERY}")
        print()
        
        current_results = await test_current_flow()
        
        new_results = await test_new_flow()
        
        print("\n" + "=" * 70)
        print("对比结果摘要")
        print("=" * 70)
        print(f"当前流程: 总耗时={current_results['total_time']:.2f}s, 事实数={current_results['steps']['distill']['total_facts']}")
        print(f"新流程: 总耗时={new_results['total_time']:.2f}s, 事实数={new_results['steps']['distill']['total_facts']}")
        print()
    
    asyncio.run(main())
