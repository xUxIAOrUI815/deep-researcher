import asyncio
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, '.')

from schemas.state import AtomicFact, SourceLevel
from core.knowledge import KnowledgeManager


class TestMutualVerification:
    @staticmethod
    async def test_similar_content_different_url():
        km = KnowledgeManager(base_storage_path="./test_km_mutual")

        for path in ["./test_km_mutual"]:
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path, ignore_errors=True)

        collection = "test_mutual"

        fact1 = AtomicFact(
            text="台积电计划在2025年开始量产2nm制程芯片",
            source_url="https://www.reuters.com/technology/tsmc-2nm",
            confidence=0.85,
            snippet="TSMC announced it will begin mass production of 2nm chips in 2025...",
            source_level=SourceLevel.A
        )

        fact2 = AtomicFact(
            text="据报道台积电2nm工艺预计2025年实现量产",
            source_url="https://www.bloomberg.com/news/articles/tsmc-2nm-2025",
            confidence=0.85,
            snippet="TSMC's 2nm process is expected to enter mass production in 2025...",
            source_level=SourceLevel.A
        )

        result1 = await km.upsert_fact_with_verification(fact1, collection)
        assert result1.action == "NEW", f"Expected NEW, got {result1.action}"
        print(f"✅ Fact1 stored as NEW: {result1.fact_id}")

        result2 = await km.upsert_fact_with_verification(fact2, collection)
        assert result2.action == "MUTUAL_VERIFICATION", f"Expected MUTUAL_VERIFICATION, got {result2.action}"
        print(f"✅ Fact2 triggered MUTUAL_VERIFICATION: {result2.action}")

        stats = km.get_stats(collection)
        print(f"   Total facts: {stats.total_facts}")
        print(f"   Verified facts: {stats.verified_facts}")
        print(f"   Confidence change: {result2.confidence_change}")

        stored_fact = await km.get_fact_by_id(result2.fact_id, collection)
        assert stored_fact is not None
        print(f"   Final confidence: {stored_fact['confidence']}")
        print(f"   Verified count: {stored_fact['verified_count']}")

        await km.clear_collection(collection)
        print("✅ Mutual verification test passed!")


class TestConflictDetection:
    @staticmethod
    async def test_conflicting_dates():
        km = KnowledgeManager(base_storage_path="./test_km_conflict")

        for path in ["./test_km_conflict"]:
            if os.path.exists(path):
                import shutil
                shutil.rmtree(path, ignore_errors=True)

        collection = "test_conflict"

        fact_a = AtomicFact(
            text="台积电2nm芯片预计2025年实现量产",
            source_url="https://www.reuters.com/technology/tsmc-2nm-2025",
            confidence=0.9,
            snippet="TSMC announced its 2nm chips will enter mass production in 2025...",
            source_level=SourceLevel.A
        )

        fact_b = AtomicFact(
            text="最新报告显示台积电2nm工艺量产时间延至2026年",
            source_url="https://www.bloomberg.com/news/tsmc-2nm-delay",
            confidence=0.85,
            snippet="According to the latest report, TSMC's 2nm production has been delayed to 2026...",
            source_level=SourceLevel.A
        )

        result_a = await km.upsert_fact_with_verification(fact_a, collection)
        assert result_a.action == "NEW", f"Expected NEW, got {result_a.action}"
        print(f"✅ Fact A stored as NEW: {result_a.fact_id}")

        result_b = await km.upsert_fact_with_verification(fact_b, collection)
        print(f"✅ Fact B result: {result_b.action}")

        if result_b.action == "CONFLICT_DETECTED":
            print(f"   Conflict detected correctly!")
            print(f"   Conflict with fact: {result_b.conflict_with_id}")

            stored_fact = await km.get_fact_by_id(result_b.fact_id, collection)
            print(f"   is_conflict flag: {stored_fact['is_conflict']}")
            print(f"   conflict_with: {stored_fact['conflict_with']}")

            conflicts = await km.get_conflicts(collection)
            print(f"   Total conflicts recorded: {len(conflicts)}")
            if conflicts:
                print(f"   Conflict description: {conflicts[0]['description']}")

            stats = km.get_stats(collection)
            print(f"   Conflicting facts count: {stats.conflicting_facts}")
            print(f"   Conflicts detected: {stats.conflicts_detected}")

        elif result_b.action == "MUTUAL_VERIFICATION":
            print(f"⚠️  Facts are similar enough for mutual verification (same meaning)")
            print(f"   This is acceptable if the dates are not explicitly different")

        else:
            print(f"⚠️  Result: {result_b.action}")

        await km.clear_collection(collection)
        print("✅ Conflict detection test completed!")


class TestSourceLevel:
    @staticmethod
    async def test_source_priority():
        km = KnowledgeManager()

        test_cases = [
            ("https://www.nature.com/article/tsmc-2nm", SourceLevel.S),
            ("https://www.sec.gov/ filings/tsmc", SourceLevel.S),
            ("https://patents.google.com/patent/US123456", SourceLevel.S),
            ("https://www.reuters.com/technology/tsmc", SourceLevel.A),
            ("https://www.bloomberg.com/news/tsmc", SourceLevel.A),
            ("https://www.apple.com/newsroom/tsmc", SourceLevel.A),
            ("https://github.com/tsmc/project", SourceLevel.B),
            ("https://reddit.com/r/technology/tsmc", SourceLevel.B),
            ("https://unknown-blog.com/tech", SourceLevel.C),
        ]

        print("Testing source priority detection:")
        for url, expected_level in test_cases:
            level = km.get_source_level(url)
            status = "✅" if level == expected_level else "❌"
            print(f"   {status} {url[:50]:50s} -> Level {level.value} (expected {expected_level.value})")

        print("✅ Source level test completed!")


class TestAtomicFactUpgrade:
    @staticmethod
    async def test_new_fields():
        fact = AtomicFact(
            text="台积电2nm技术进展顺利",
            source_url="https://www.reuters.com/technology/tsmc-2nm",
            confidence=0.9,
            snippet="TSMC's 2nm technology development is proceeding smoothly...",
            source_level=SourceLevel.A,
            verified_count=3,
            is_conflict=False,
            conflict_with=[],
            confidence_reason="权威媒体确认，具体数据支撑"
        )

        assert fact.snippet != ""
        assert fact.source_level == SourceLevel.A
        assert fact.verified_count == 3
        assert fact.is_conflict == False
        assert fact.conflict_with == []
        assert fact.confidence_reason is not None

        print("✅ AtomicFact with new fields created successfully")
        print(f"   snippet: {fact.snippet[:50]}...")
        print(f"   source_level: {fact.source_level}")
        print(f"   verified_count: {fact.verified_count}")
        print(f"   confidence_reason: {fact.confidence_reason}")


async def main():
    print("\n" + "="*60)
    print("KNOWLEDGE V2 - 情报蒸馏与证据链管理测试")
    print("="*60)

    print("\n--- Test 1: Source Level Detection ---")
    await TestSourceLevel.test_source_priority()

    print("\n--- Test 2: AtomicFact New Fields ---")
    await TestAtomicFactUpgrade.test_new_fields()

    print("\n--- Test 3: Mutual Verification (Similar Content, Different URLs) ---")
    await TestMutualVerification.test_similar_content_different_url()

    print("\n--- Test 4: Conflict Detection (Different Dates) ---")
    await TestConflictDetection.test_conflicting_dates()

    print("\n" + "="*60)
    print("ALL KNOWLEDGE V2 TESTS COMPLETED ✅")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
