# CLAUDE.md

Repository guidance for coding agents. `AGENTS.md`, when present, has the
authoritative command and architecture details.

## Production entry points

```bash
python run_research.py "research question" --depth standard
python run_console.py
```

Both entry points use `deep_researcher.application.ApplicationRuntime`. They do
not use a legacy graph, module-global manager, or compatibility state.

## Tests

```bash
pytest tests -q
pytest tests/test_bg001_application_runtime.py tests/test_bg001_integration_hardening.py -q
```

Tests inject deterministic model/search/scraper adapters. Do not introduce
implicit production mocks or require live credentials for the offline suite.

## Architecture

- `deep_researcher/application/`: production composition root and durable run
  catalog.
- `deep_researcher/contracts/`: versioned domain contracts with no framework or
  provider imports.
- `deep_researcher/orchestration/`: native event-sourced scheduler and
  rebuildable projections.
- `deep_researcher/kernel/`: versioned AgentSpec execution, middleware,
  validation, repair, budgets, and stop policy.
- `deep_researcher/gateway/`: governed tools plus pinned MCP/A2A protocol
  adapters.
- `deep_researcher/providers/`: live model, search, and scraper adapters used by
  the application composition.
- `deep_researcher/research/`: Supervisor, Worker pool, convergence, merging,
  and coordination.
- `deep_researcher/evidence/` and `deep_researcher/knowledge/`: candidate
  ingestion, independent verification, immutable artifacts, and typed evidence
  graph.
- `deep_researcher/reporting/`: Writer, Reviewer, repair, citation, and targeted
  research loop.
- `deep_researcher/events/` and `deep_researcher/studio/`: append-only
  observability and read-only projections.
- `deep_researcher/evaluation/`, `deep_researcher/evolution/`, and
  `deep_researcher/version_registry/`: gated evaluation, offline evolution, and
  release/rollback control.
- `console_app/`: FastAPI Console and Studio API/UI over application
  projections.

## Invariants

- Thin orchestration, thick role runtimes; no fat shared GraphState.
- Scheduler, events, artifacts, knowledge, reporting, evaluation, versions, and
  application status each have one explicit owner.
- Workers create candidate knowledge only. Verification promotes it. Writers
  consume verified evidence only.
- All tools cross the governed gateway. Production never fabricates search
  results when credentials are missing.
- Every mutable workflow is restart-safe, idempotent where required, and fenced
  against stale writes and post-cancellation completion.
- Secrets and hidden reasoning are excluded from durable artifacts/events.
- Reinforcement learning and autonomous online self-modification remain out of
  scope.
