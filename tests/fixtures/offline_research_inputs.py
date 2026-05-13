from __future__ import annotations

from typing import Any, Dict

from core.run_context import RunContext
from schemas.state import DistillerOutputs, KnowledgeRefs, PlannerState, ResearcherOutputs, RunMetadata


ROOT_QUERY = "What is the current state of the AI chip market in 2026?"

MOCK_TASK: Dict[str, Any] = {
    "id": "task-1",
    "query": "AI chip market 2026",
    "title": "Current AI chip market",
    "rationale": "Find evidence about market size, growth drivers, and disagreements in current estimates.",
    "node_type": "question",
    "priority": 0.9,
    "related_fact_ids": [],
    "related_evidence_ids": [],
}

MOCK_REPORT_OUTLINE: Dict[str, Any] = {
    "title": "AI Chip Market 2026",
    "sections": [
        {"section_id": "sec_summary", "title": "Market Outlook", "goal": "Summarize the current market outlook", "order": 1},
        {"section_id": "sec_conflict", "title": "Conflicting Estimates", "goal": "Compare conflicting market size estimates", "order": 2},
        {"section_id": "sec_risks", "title": "Open Risks", "goal": "Describe unresolved risks and evidence gaps", "order": 3},
        {"section_id": "sec_conclusion", "title": "Conclusion", "goal": "State what can be concluded from the current evidence", "order": 4},
    ],
}

MOCK_SECTION_GOALS = [
    {"section_id": "sec_summary", "goal": "Summarize the current market outlook", "priority": 1.0},
    {"section_id": "sec_conflict", "goal": "Compare conflicting market size estimates", "priority": 0.95},
    {"section_id": "sec_risks", "goal": "Describe unresolved risks and evidence gaps", "priority": 0.85},
    {"section_id": "sec_conclusion", "goal": "State what can be concluded from the current evidence", "priority": 0.8},
]

MOCK_KNOWLEDGE_REFS = {"collection_name": "offline_test"}

MOCK_RESEARCHER_OUTPUTS: Dict[str, Any] = {
    "task_id": "task-1",
    "queries": [{"query": "AI chip market 2026", "relevance_score": 9, "scoring_method": "heuristic"}],
    "sources": [
        {
            "source_id": "src_1",
            "url": "https://mock.local/ai-chip/doc-1",
            "title": "AI chip market grows in 2026",
            "snippet": "Analysts reported revenue growth and new launches.",
            "score": 0.93,
            "status": "accepted",
            "text_length": 212,
            "extraction_method": "mock",
            "query": "AI chip market 2026",
        },
        {
            "source_id": "src_2",
            "url": "https://mock.local/ai-chip/doc-2",
            "title": "Different outlook on AI chip revenue",
            "snippet": "Another report showed a smaller market size in 2026.",
            "score": 0.88,
            "status": "accepted",
            "text_length": 207,
            "extraction_method": "mock",
            "query": "AI chip market 2026",
        },
    ],
    "passages": [
        {
            "passage_id": "p1",
            "source_id": "src_1",
            "url": "https://mock.local/ai-chip/doc-1",
            "title": "AI chip market grows in 2026",
            "query": "AI chip market 2026",
            "text": (
                "The AI chip market reached 120 billion dollars in 2026. "
                "Analysts reported that demand increased because cloud providers expanded GPU spending. "
                "The report also said leading vendors launched new accelerator products."
            ),
        },
        {
            "passage_id": "p2",
            "source_id": "src_2",
            "url": "https://mock.local/ai-chip/doc-2",
            "title": "Different outlook on AI chip revenue",
            "query": "AI chip market 2026",
            "text": (
                "A separate industry report stated the AI chip market was 95 billion dollars in 2026. "
                "The authors argued that enterprise adoption grew more slowly than expected. "
                "The report defined AI chips mainly as training accelerators."
            ),
        },
    ],
    "metadata": {
        "stop_reason": "max_search_iterations_reached",
        "follow_up_hints": ["verify market size methodology", "check vendor launch timeline"],
        "scraper_mode": "mock",
        "search_mode": "mock",
    },
}


def build_initial_graph_state() -> Dict[str, Any]:
    config = {"configurable": {"thread_id": "offline-thread", "research_id": "offline-research"}}
    context = RunContext.from_config(config, root_query=ROOT_QUERY)
    return {
        "user_query": ROOT_QUERY,
        "normalized_query": ROOT_QUERY,
        "run_metadata": RunMetadata(
            research_id=context.research_id,
            thread_id=context.thread_id,
            run_id=context.run_id,
            trace_id=context.trace_id,
            session_id=context.session_id,
            graph_version=context.graph_version,
            prompt_version=context.prompt_version,
            root_query=context.root_query,
        ).model_dump(),
        "task_tree": {},
        "root_task_id": None,
        "active_task_id": None,
        "planner_state": PlannerState().model_dump(),
        "researcher_outputs": ResearcherOutputs().model_dump(),
        "distiller_outputs": DistillerOutputs().model_dump(),
        "knowledge_refs": KnowledgeRefs(collection_name=context.knowledge_collection).model_dump(),
        "report_outline": {},
        "section_goals": [],
        "section_evidence_packs": [],
        "final_report": None,
        "token_usage": {
            "planning_tokens": 0,
            "research_tokens": 0,
            "distillation_tokens": 0,
            "writing_tokens": 0,
            "total_tokens": 0,
        },
        "state_events": [],
        "error_state": None,
        "fact_pool": [],
        "atomic_facts": [],
        "current_focus": None,
        "completed_tasks": [],
        "failed_tasks": [],
        "messages": [],
        "raw_scraped_data": [],
        "search_results": [],
    }
