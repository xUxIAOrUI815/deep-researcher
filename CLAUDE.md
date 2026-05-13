# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Run tests

```bash
# Fast offline agent smoke suite (researcher, distiller, writer, graph):
pytest tests/test_researcher_offline.py tests/test_distiller_offline.py tests/test_writer_offline.py tests/test_graph_offline_smoke.py -q

# Session knowledge & full graph integration suite:
pytest tests/test_session_knowledge_store.py tests/test_knowledge_manager_integration.py tests/test_session_knowledge_flow.py tests/test_planner_session_consumption.py tests/test_researcher_session_dedup.py tests/test_writer_session_pack_priority.py tests/test_session_resume_recovery.py tests/test_graph_session_e2e_offline.py -q

# Single test file or specific test:
pytest tests/test_researcher_offline.py::test_name -q
```

Tests use `SCRAPER_MODE=mock` and `RESEARCHER_SEARCH_MODE=mock` by default (no external APIs). No .env needed for offline tests.

### Run the research pipeline

```bash
python run_research.py        # Full research cycle via core/graph.py
python run_console.py         # FastAPI web console on http://127.0.0.1:8000
```

### Environment variables

Required for live runs (copy `.env`):
- `DEEPSEEK_API_KEY` — LLM calls (distiller, relevance scoring)
- `SILICON_FLOW_API_KEY` — embedding API (session knowledge)
- `TAVILY_API_KEY` — live web search

Key config flags:
- `RESEARCHER_SEARCH_MODE=mock|live` — toggle mock vs real search
- `SCRAPER_MODE=mock|jina|playwright` — scraper backend
- `RESEARCHER_USE_LLM_SCORING=1` — use DeepSeek for query relevance (off by default)

## Architecture

This is a **multi-agent deep research system** built on LangGraph. The graph (`core/graph.py`) orchestrates a loop with four agent nodes:

```
planner → researcher → distiller → (back to planner) → writer → END
```

### Agents (`agents/`)

Each agent is a stateless function called from the graph. The graph handles state I/O; agents receive their inputs as parameters and return typed outputs.

- **planner** (`agents/planner.py`): Rule-based task decomposer. Breaks the root question into subtasks (`source_discovery`, `question`, `gap`), then on subsequent loops reads distiller outputs to create follow-up tasks from gaps/conflicts. Produces `PlannerState` with action (`continue_research`/`start_writing`/`stop`) and `TaskTreePatch` list. No LLM calls — purely heuristic.

- **researcher** (`agents/researcher.py`): Search-and-scrape pipeline. Generates candidate queries from task fields, scores relevance (heuristic or DeepSeek), deduplicates via embedding cosine similarity, then searches (Tavily via MCPGateway or mock), scrapes pages (Jina/Playwright via SmartScraper), and returns `ResearcherOutputs` with passages/sources/scraped data.

- **distiller** (`agents/distiller.py`): Extracts structured knowledge from researcher outputs. Two modes: local (regex/heuristic) for offline tests, LLM (DeepSeek) for production. Produces `AtomicFact`, `Claim`, `Evidence`, `ConflictRecord` entities plus `SectionEvidencePack` bundles. Pushes results into `KnowledgeManager`.

- **writer** (`agents/writer.py`): Assembles final `FinalReport` markdown. Iterates over report sections, retrieves evidence packs from session knowledge, renders structured sections with citation markers.

### Session knowledge (`core/session_knowledge.py`, `core/session_retrieval.py`)

Persistent knowledge layer using SQLite (`knowledge_data/`). `KnowledgeManager` stores facts, claims, evidence, conflicts, section packs per research session. `SessionRetrievalService` provides typed retrieval queries with semantic scoring for context builders.

This replaces the legacy `core/knowledge.py` (now deleted) and Qdrant vector store (`core/vector_store_qdrant.py`, now unused).

### Context builders (`core/context_builders.py`)

Three builder classes that bridge session knowledge into each agent's input context:
- `PlannerContextBuilder` — section readiness, coverage, gaps, conflicts
- `ResearcherContextBuilder` — seen sources (dedup), gaps, authority hints
- `WriterContextBuilder` — evidence packs per section with fallback chain

### Providers (`providers/`)

- `MCPGateway` — Unified search via Tavily API with retry/error classification
- `SmartScraper` / `MockScraper` — Web scraping with content denoising
- `build_scraper(mode)` / `resolve_scraper_mode(override)` — factory functions

### Schemas (`schemas/`)

- `state.py` — All Pydantic models: `TaskNode`, `AtomicFact`, `Claim`, `Evidence`, `ConflictRecord`, `ResearcherOutputs`, `DistillerOutputs`, `FinalReport`, `ResearchGraphState` (TypedDict for graph), `ResearchState` (BaseModel), and enums
- `task_tree.py` — `TaskNode`, `TaskTreePatch`, `TaskStatus`, `TaskNodeType`, `TaskTreeOperation`
- `session.py` — Session persistence models for SQLite storage
- `retrieval.py` — `PlannerContext`, `ResearcherContext`, `WriterContext`, `SessionRetrievalQuery/Result`
- `console.py` — FastAPI console API request/response models

### Console app (`console_app/`)

FastAPI web UI for creating and monitoring research runs. `app.py` defines routes, `service.py` wraps graph execution, `templates/` and `static/` serve the UI.

### RunContext (`core/run_context.py`)

Frozen dataclass carrying correlation IDs (`research_id`, `thread_id`, `run_id`, `trace_id`, `session_id`) through every graph node and agent call. Created from LangGraph config via `RunContext.from_config()`.

### Observability (`core/observability/`)

Event recording system (`EventType` enum, `get_observer()`) used by graph nodes and agents to emit structured lifecycle events.

## Key patterns

- **Mock everything for testing**: Agents and graph validate `SCRAPER_MODE`/`RESEARCHER_SEARCH_MODE` env vars. No API keys needed for offline tests.
- **State flows through the graph**: `ResearchGraphState` (TypedDict) is the canonical state schema. All defaults are set by `_ensure_state_defaults()` in graph.py.
- **Task tree is the backbone**: `planner` creates/deletes tasks; `researcher` executes the active task; `distiller` marks it complete. Routing decisions use `_pending_task_ids()`.
- **Durability via SQLite checkpointer**: LangGraph's `AsyncSqliteSaver` provides graph state persistence. Session knowledge has its own separate SQLite store.
- **Thin graph, thick agents**: `core/graph.py` only manages state flow, routing, and orchestration. Agent logic lives entirely in `agents/*.py`.
