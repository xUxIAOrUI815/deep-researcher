# DeepResearcher

DeepResearcher is a durable, evidence-governed multi-agent research runtime.
The production path follows the Background001 design: a Supervisor decomposes
work, bounded Research Workers use governed tools, an independent verifier
promotes candidate knowledge, and separate Writer and Reviewer roles produce
and gate the report.

This repository does not claim state-of-the-art quality. It provides explicit
contracts, persistence, verification, evaluation, and release gates so quality
can be measured rather than asserted.

## Capability boundary

| Status | Capability |
| --- | --- |
| Current | Native event-sourced scheduler with DAG dependencies, leases, concurrency limits, retries, budgets, cancellation, approval, recovery, and projection rebuild |
| Current | Versioned Agent Kernel for Supervisor, Research Worker, Evidence Verifier, Writer, Reviewer, evaluator, and offline evolution roles |
| Current | Governed Tavily search with Exa fallback, SmartScraper reads, structured extraction/comparison/source verification, durable idempotency, audit, cache, rate limit, and circuit state |
| Current | Artifact-backed Source, Passage, Evidence, AtomicFact, Claim, Citation, Conflict, Section, and Report graph with independent verification |
| Current | Evidence-bounded writing, report review/repair loop, targeted research, citation map, and explicit gap/conflict disclosure |
| Current | Append-only runtime events, Console, Studio V1/V2/V4 projections, replay approval, comparisons, badcases, and version inspection |
| Current | Frozen replay, live-web evaluation, semantic judges, release gates, and offline evolution candidate/release/rollback workflows |
| Optional | Exa fallback, Jina credentials, Playwright rendering, MCP/A2A adapters, vector ranking acceleration, and external telemetry exporters |
| Unsupported | Reinforcement learning; it is explicitly outside this refactor |
| Unsupported | Autonomous online self-modification, automatic production release, or bypassing evaluation and human approval gates |
| Unsupported | The deleted `agents/`, `core/`, `providers/`, `schemas/`, LangGraph checkpoint, fake MCP dispatcher, or legacy graph-state compatibility APIs |
| Unsupported | Arbitrary provider/tool access outside registered grants and permissions |

## Runtime ownership

```text
ApplicationRuntime
  -> NativeEventSourcedScheduler
  -> Research Supervisor
       -> bounded Research Worker pool
       -> governed ProtocolToolGateway
       -> candidate knowledge ingestion
  -> independent Evidence Verifier
  -> Writer
  -> Reviewer / repair / targeted research loop
  -> durable report + citation artifacts
  -> Event, Console, and Studio projections
```

The scheduler journal owns scheduling truth. The event store owns observable
runtime history. Artifact, knowledge, reporting, evaluation, version, and
application stores own their respective domain data. Studio and Console are
read-only projections over those APIs; no fat graph-state dictionary is an
authoritative store.

## Run locally

Install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
```

Configure the live production path:

```text
DEEPSEEK_API_KEY=...
TAVILY_API_KEY=...
```

Optional live configuration:

```text
EXA_API_KEY=...
JINA_API_KEY=...
DEEPSEEK_API_BASE=https://api.deepseek.com
DEEPSEEK_MODEL_NAME=deepseek-chat
DEEP_RESEARCH_RUNTIME_DIR=.research_runtime
```

Run a durable research request:

```bash
python run_research.py "What evidence supports the requested claim?" \
  --instructions "Prefer primary sources and disclose conflicts." \
  --depth standard
```

Run the Console and Studio web application:

```bash
python run_console.py
```

Then open `http://127.0.0.1:8000`. Missing live model or search credentials
produce explicit failed runs; production entry points never silently substitute
demo research.

## Verify

The full suite is offline and uses deterministic injected adapters:

```bash
pytest tests -q
```

Focused production-composition and hardening gates:

```bash
pytest tests/test_bg001_application_runtime.py \
  tests/test_bg001_integration_hardening.py \
  tests/test_console_api.py \
  tests/test_bg001_studio_api.py -q
```

The suite covers single and concurrent execution, crash recovery, provider and
protocol failures, cancellation, approval, budgets, conflict preservation,
report repair, replay/fork controls, evaluation/release gates, offline
evolution release/rollback, database contention, large artifacts, long traces,
redaction, and dataset permissions.

## Documentation

- `docs/BACKGROUND001_EXECUTION_PLAN.md`: authoritative 16-branch execution
  record and acceptance boundary.
- `docs/adr/0002-native-scheduler-is-production-runtime.md`: decision and gates
  for removing the provisional LangGraph adapter.
- `docs/ORCHESTRATION_RUNTIME.md`: scheduler ownership and recovery behavior.
- `docs/EVIDENCE_ENGINE.md`: candidate-to-verified knowledge rules.
- `docs/BACKGROUND001_REPORTING_RUNTIME.md`: report synthesis and repair.
- `docs/BACKGROUND001_SEMANTIC_EVALUATION_RELEASE_GATES.md`: evaluation and
  release policy.
- `docs/OFFLINE_EVOLUTION_LAB.md`: offline-only evolution boundary.
