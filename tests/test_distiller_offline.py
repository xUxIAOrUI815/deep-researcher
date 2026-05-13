import pytest

import agents.distiller as distiller_module
from agents.distiller import run_distiller
from tests.fixtures.offline_research_inputs import (
    MOCK_KNOWLEDGE_REFS,
    MOCK_REPORT_OUTLINE,
    MOCK_RESEARCHER_OUTPUTS,
    MOCK_SECTION_GOALS,
)


@pytest.mark.asyncio
async def test_distiller_produces_structured_outputs_from_mock_research_materials():
    outputs = await run_distiller(
        task_id="task-1",
        task={"id": "task-1", "query": "AI chip market 2026"},
        researcher_outputs=MOCK_RESEARCHER_OUTPUTS,
        report_outline=MOCK_REPORT_OUTLINE,
        section_goals=MOCK_SECTION_GOALS,
        knowledge_refs=MOCK_KNOWLEDGE_REFS,
    )

    assert outputs.task_id == "task-1"
    assert len(outputs.atomic_facts) > 0
    assert len(outputs.claims) > 0
    assert len(outputs.evidence) > 0
    assert isinstance(outputs.coverage_summary, dict)
    assert "sufficiency_level" in outputs.coverage_summary
    assert "section_status" in outputs.coverage_summary
    assert len(outputs.section_evidence_packs) > 0
    assert len(outputs.unresolved_gaps) > 0
    assert not hasattr(outputs, "suggested_followups")
    assert outputs.summary


@pytest.mark.asyncio
async def test_distiller_uses_deepseek_agent_for_grounded_transformer_outputs(monkeypatch):
    quote = (
        "The Transformer, a model architecture eschewing recurrence and instead relying entirely "
        "on an attention mechanism, draws global dependencies between input and output."
    )
    passage_text = (
        "Attention Is All You Need introduced the Transformer. "
        f"{quote} "
        "This attention-based design enables significantly more parallelization. "
        "Modern large language models are commonly based on Transformer architectures."
    )
    llm_payload = {
        "evidence_units": [
            {
                "quote": quote,
                "summary": "该证据定义 Transformer，并说明其使用注意力机制而不是循环结构。",
                "evidence_type": "definition",
                "quality_score": 0.92,
                "facts": [
                    {"text": "Transformer 是一种神经网络架构。", "confidence": 0.91, "fact_type": "definition"},
                    {"text": "Transformer 架构以注意力机制为核心。", "confidence": 0.91, "fact_type": "mechanism"},
                    {"text": "Transformer 不依赖循环网络作为主要序列建模机制。", "confidence": 0.89, "fact_type": "mechanism"},
                    {"text": "Transformer 最初由论文 Attention Is All You Need 提出。", "confidence": 0.86, "fact_type": "timeline"},
                ],
                "claims": [
                    {"text": "来源认为 Transformer 的注意力机制使模型更容易并行训练。", "confidence": 0.82, "claim_type": "mechanism"},
                    {"text": "来源认为 Transformer 架构是现代大语言模型的基础之一。", "confidence": 0.78, "claim_type": "evaluation"},
                ],
            }
        ]
    }

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"message": {"content": __import__("json").dumps(llm_payload, ensure_ascii=False)}}]}

    class FakeClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def post(self, *args, **kwargs):
            return FakeResponse()

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DISTILLER_USE_LLM", "true")
    monkeypatch.setattr(distiller_module.httpx, "AsyncClient", FakeClient)

    outputs = await run_distiller(
        task_id="task-transformer",
        task={"id": "task-transformer", "query": "what is Transformer in LLM?"},
        researcher_outputs={
            "sources": [
                {
                    "source_id": "src_transformer",
                    "url": "https://example.test/attention-is-all-you-need",
                    "title": "Attention Is All You Need",
                }
            ],
            "passages": [
                {
                    "passage_id": "p_transformer",
                    "source_id": "src_transformer",
                    "url": "https://example.test/attention-is-all-you-need",
                    "title": "Attention Is All You Need",
                    "query": "what is Transformer in LLM?",
                    "text": passage_text,
                }
            ],
            "metadata": {"search_mode": "live", "scraper_mode": "live"},
        },
        report_outline={
            "title": "Transformer in LLM",
            "sections": [
                {"section_id": "sec_context", "title": "背景与定义", "goal": "解释 Transformer 是什么", "order": 1},
                {"section_id": "sec_findings", "title": "核心发现", "goal": "解释 Transformer 在 LLM 中的作用", "order": 2},
            ],
        },
        section_goals=[
            {"section_id": "sec_context", "goal": "解释 Transformer 是什么", "priority": 1.0},
            {"section_id": "sec_findings", "goal": "解释 Transformer 在 LLM 中的作用", "priority": 0.9},
        ],
        knowledge_refs={},
    )

    fact_texts = {fact.text for fact in outputs.atomic_facts}
    claim_texts = {claim.text for claim in outputs.claims}
    evidence_quotes = {item.quote for item in outputs.evidence}

    assert "Transformer 是一种神经网络架构。" in fact_texts
    assert "Transformer 架构以注意力机制为核心。" in fact_texts
    assert "Transformer 不依赖循环网络作为主要序列建模机制。" in fact_texts
    assert "Transformer 最初由论文 Attention Is All You Need 提出。" in fact_texts
    assert "来源认为 Transformer 的注意力机制使模型更容易并行训练。" in claim_texts
    assert "来源认为 Transformer 架构是现代大语言模型的基础之一。" in claim_texts
    assert quote in evidence_quotes
    assert all(item.fact_ids or item.claim_ids for item in outputs.evidence)
