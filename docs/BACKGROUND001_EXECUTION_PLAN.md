# Background001 Refactor Execution Plan

This document is the durable source of truth for the Background001 refactor.
The source design is `C:\Users\15317\Desktop\背景001.pdf`. The current `main`
implementation is a draft and is not a compatibility contract.

The refactor includes every capability in Background001 except reinforcement
learning. RL training, reward-model training, GRPO/PPO, Agent Lightning, verl,
and GPU training infrastructure are explicitly out of scope for this program.

## Non-negotiable implementation rules

1. Do not implement an MVP, placeholder, no-op production path, fake protocol,
   or interface-only approximation of an in-scope capability.
2. Each branch must deliver its complete bounded capability: domain logic,
   persistence where applicable, failure handling, observability, tests,
   documentation, and integration contracts.
3. Domain contracts must not depend on LangGraph, FastAPI, SQLite, or a concrete
   model/provider SDK.
4. LangGraph is retained only as a replaceable durable scheduler adapter. The
   domain model, event model, artifacts, evidence, evaluation, and evolution
   remain framework-independent.
5. Runtime state stores identifiers and compact scheduling snapshots. Event,
   artifact, evidence, and evaluation stores own durable data.
6. Do not record or expose hidden chain-of-thought. Persist structured decision
   summaries, commands, observations, verification feedback, and artifacts.
7. Agent communication uses TaskEnvelope, TaskResult, Command, Event, and
   Artifact contracts instead of unconstrained agent-to-agent chat.
8. Every loop has semantic termination plus token, cost, time, tool-call,
   search, retry, and error budgets.
9. Evaluation data is strictly partitioned as
   `train -> dev -> selection -> test -> hidden-test`.
10. Production traces may enter a candidate dataset but may not automatically
    promote a global Skill, Prompt, Policy, or model version.
11. No branch may silently stage or commit the user's unrelated untracked
    files (`AGENTS.md`, `console_stdout.log`, `console_stderr.log`).

## Mandatory branch procedure

Before starting every development branch:

1. Read this document from start to finish.
2. Confirm the branch status table and prerequisites.
3. Run `git status -sb` and preserve unrelated user files.
4. Run the relevant existing baseline tests.
5. Create the named branch from `codex/bg001-integration` after all prerequisite
   branches have been merged.
6. Restate the branch boundary in the implementation notes or commit message.

Before completing every development branch:

1. Read this document again from start to finish.
2. Check every deliverable and every explicit non-goal for that branch.
3. Run unit, integration, restart/recovery, concurrency, and failure-path tests
   appropriate to the capability.
4. Confirm no production no-op, fake adapter, unreferenced schema, or unresolved
   core TODO was introduced.
5. Update the branch status table and append an execution record.
6. Commit only branch-owned files, merge into `codex/bg001-integration`, rerun
   the integration suite, and push the integration branch.

## Target architecture and ownership

| Layer | Owns | Must not own |
| --- | --- | --- |
| Domain Contracts | Task, Command, Event, Artifact, Evidence, Citation, Budget, AgentSpec and versioned schemas | Framework, database, provider, or UI logic |
| Agent Kernel | Observe-Decide-Act-Verify loop, model interaction, budgets, stop policy, middleware pipeline | Task persistence, evidence persistence, UI |
| Orchestration Runtime | Durable queue, task DAG scheduling, concurrency, recovery, cancellation, approval | Evidence semantics or report prose |
| Protocol & Tool Gateway | Function Calling, MCP, A2A, tool governance and protocol normalization | Planning, task status ownership, evidence verification |
| Event Store | Append-only truth of what happened during a run | Current-state mutation or artifact bodies |
| Artifact Store | Immutable snapshots, tool outputs, prompts, skills, reports and content hashes | Scheduling or verification decisions |
| Evidence Engine | Source-Passage-Evidence-Fact-Claim-Citation-Section-Report graph | Free-form agent loops or scheduler state |
| Projection Store | Rebuildable Thread, Run, Span, Task, budget and UI read models | Authoritative execution or evidence data |
| Evaluation Lab | Datasets, evaluators, experiments, scores and release gates | Online self-modification |
| Version Registry | AgentSpec, Skill, Prompt, Tool Policy, Stop Policy and Rubric versions | Runtime memory |
| Evolution Lab | Offline candidate analysis, structured patches, selection and release/rollback | Online code/weight modification |
| Studio | Trace, task, evidence, diff, replay, comparison and badcase interfaces | Direct LangGraph-state or database-table access |

## Required loops and roles

All four loops are mandatory:

1. Agent loop: model selects structured search/read/extract/delegate commands
   until success or a budget/semantic stop.
2. Verification loop: an independent verifier checks evidence, citation,
   coverage, conflicts, and repair feedback.
3. Task/report loop: a supervisor and report reviewer schedule targeted research
   or rewriting until quality and convergence gates pass.
4. Cross-task optimization loop: scored traces improve Skill, Prompt, Tool
   Policy, Stop Policy, and Rubric through offline selection gates.

All five logical roles are mandatory and share AgentKernel + AgentSpec:

1. Research Supervisor
2. Research Worker Pool
3. Evidence Verifier
4. Synthesis Writer
5. Report Reviewer

## Branch dependency graph

```text
00 foundation
  |-- 01 event trace store -- 03 Studio V1
  |-- 02 artifact/knowledge -- 03 Studio V1
  |-- 01 + 02 -- 04 Agent Kernel -- 05 Protocol Gateway -- 06 Orchestration
  |-- 02 + 04 -- 07 Evidence Engine/Verifier
  |-- 06 + 07 -- 08 Supervisor/Workers
  |-- 07 + 08 -- 09 Writer/Reviewer
  |-- 01 + 02 + 09 -- 10 Evaluation Core -- 11 Evaluation Gates
  |-- 03 + 07 + 08 + 09 -- 12 Studio V2
  |-- 11 + 12 -- 13 Studio V3/V4
  |-- 11 + 13 -- 14 Offline Evolution
  `-- all branches -- 15 Integration/Hardening
```

## Branch status

| # | Branch | Status | Integration commit | Notes |
| --- | --- | --- | --- | --- |
| 00 | `codex/bg001-00-foundation-contracts-baseline` | integrated | e406407 | Contracts, Frozen Replay, baseline, ADRs, and order-independent tests complete. |
| 01 | `codex/bg001-01-event-trace-store` | integrated | 7a90c9a | Append-only SQLite event store, instrumentation, redaction, and durable OTLP outbox complete. |
| 02 | `codex/bg001-02-artifact-knowledge-storage` | integrated | 58de1c2 | Immutable content-addressed artifacts, revisioned evidence repositories, ingestion/retrieval/projection split, and durable pipeline dual-write complete. |
| 03 | `codex/bg001-03-studio-v1` | integrated | 56224fe | Persistent event-derived Thread/Run/Span projections, searchable/paginated SSE timeline, trace export, masking, and Console trace migration complete. |
| 04 | `codex/bg001-04-agent-kernel` | integrated | 453c564 | Complete Observe-Decide-Act-Verify kernel, middleware, budgets, stops, policies, normalized observations, and durable trace adapter. |
| 05 | `codex/bg001-05-protocol-tool-gateway` | feature complete; integration pending | | Function Calling, real MCP/A2A transports, governed versioned tools, and Tavily/Exa/scraper adapters complete. |
| 06 | `codex/bg001-06-orchestration-runtime` | pending | | |
| 07 | `codex/bg001-07-evidence-engine-verifier` | pending | | |
| 08 | `codex/bg001-08-supervisor-worker-pool` | pending | | |
| 09 | `codex/bg001-09-synthesis-writer-reviewer` | pending | | |
| 10 | `codex/bg001-10-evaluation-lab-core` | pending | | |
| 11 | `codex/bg001-11-evaluation-semantic-gates` | pending | | |
| 12 | `codex/bg001-12-studio-v2` | pending | | |
| 13 | `codex/bg001-13-studio-v3-v4` | pending | | |
| 14 | `codex/bg001-14-offline-evolution` | pending | | |
| 15 | `codex/bg001-15-integration-hardening` | pending | | |

## 00 - Foundation contracts and baseline

Branch: `codex/bg001-00-foundation-contracts-baseline`

Complete deliverables:

- Establish the new framework-independent domain package.
- Define and version TaskEnvelope, TaskResult, Command, Observation,
  ArtifactEnvelope, RunEvent, Budget, BudgetUsage, StopDecision, AgentSpec,
  ComponentVersionSet, VerificationResult, and EvaluationResult.
- Define Source, SourceSnapshot, Passage, Evidence, AtomicFact, Claim, Citation,
  Conflict, Section, Report, and their state machines.
- Standardize identifiers, timestamps, schema versions, producers, causation,
  artifact references, terminal states, and error taxonomy.
- Define schema evolution and serialization rules.
- Define dataset split models and access constraints.
- Create Frozen Replay fixtures from the current mock pipeline.
- Record the current baseline for retrieval, citations, claims, report quality,
  trace coverage, token/cost placeholders, latency, and failures.
- Eliminate test-order dependence and accidental collection of script helpers.
- Add ADRs for owned contracts, LangGraph adapter status, event sourcing,
  chain-of-thought exclusion, and dataset separation.

Boundary:

- Do not change planner/researcher/writer behavior.
- Do not implement stores, kernel, gateway, scheduler, or Studio.

Merge gate:

- Domain package has no framework/provider/storage imports.
- Contract serialization, invalid transition, and version tests pass.
- The full automated test command is order-independent.
- Frozen Replay and baseline outputs are reproducible.

## 01 - Persistent event and trace store

Branch: `codex/bg001-01-event-trace-store`

Complete deliverables:

- Implement a storage-neutral append-only EventStore with a complete SQLite
  adapter, migrations, transactions, indexes, pagination, backup/recovery, and
  corruption checks.
- Persist sequence number, trace/span hierarchy, causation/correlation, actor,
  task, schema version, input/output/state refs, token, cost, latency, status,
  error, and component versions.
- Enforce idempotent append, run-local ordering, span lifecycle, and one terminal
  run outcome.
- Instrument the current pipeline without changing behavior so a complete run
  emits model, tool, task, evidence, report, failure, retry, and budget events.
- Add redaction and structured-decision-summary rules.
- Implement an OTel exporter whose failure cannot lose local events.

Boundary:

- No artifact bodies, Studio UI, replay, agent rewrite, or state thinning.

Merge gate:

- Restart, concurrent append, event ordering, orphan span, duplicate terminal,
  pagination, redaction, migration, and OTel-failure tests pass.

## 02 - Artifact and knowledge storage

Branch: `codex/bg001-02-artifact-knowledge-storage`

Complete deliverables:

- Implement immutable, content-addressed ArtifactStore and migrations.
- Store search responses, source snapshots, cleaned passages, tool outputs,
  structured model outputs, evidence packs, reports, prompts, skills, policies,
  and dataset samples with provenance and content hashes.
- Persist web snapshots rather than URL-only references.
- Implement repositories and relationship storage for the evidence domain.
- Split the responsibilities currently concentrated in session_knowledge.py into
  storage adapter, repository, ingestion, normalization, deduplication,
  retrieval, and projection adapters.
- Enforce transactionality, referential integrity, run isolation, canonical URL
  handling, source versions, backup/recovery, and corruption detection.
- Keep embedding/vector retrieval behind an optional adapter.

Boundary:

- No claim verification, planning, writing, scheduler, or Studio.

Merge gate:

- Hash, referential integrity, idempotency, restart, backup/restore, migration,
  concurrent run isolation, and corruption tests pass.

## 03 - Studio V1

Branch: `codex/bg001-03-studio-v1`

Complete deliverables:

- Build persistent Thread -> Run -> Span projections from events.
- Replace MemoryObserver and last-50-event behavior with projection queries.
- Implement real-time timeline transport, filtering, search, pagination, and
  trace export.
- Display model/tool details, artifact references, versions, tokens, cost,
  latency, retries, permissions, errors, and run terminal state.
- Apply server-side sensitive-field masking.

Boundary:

- No task DAG, evidence graph, state diff, replay, fork, A/B, or badcase UI.

Merge gate:

- Restart persistence, real-time ordering, long-trace pagination, masking, and
  projection rebuild tests pass. Studio reads no LangGraph state.

## 04 - Agent Kernel

Branch: `codex/bg001-04-agent-kernel`

Complete deliverables:

- Implement the Observe -> Decide -> Act -> Verify kernel.
- Implement ModelAdapter, structured Command generation, Observation
  normalization, context construction, schema repair, policy checking, and
  AgentSpec registry.
- Implement model pre/post middleware for trimming, redaction, version injection,
  budget checks, validation, repair, and command normalization.
- Enforce token, cost, wall-time, model-call, tool-call, search, retry, and error
  budgets.
- Implement success, semantic completion, no-gain, budget, repeated-error,
  cancellation, and approval stop policies.

Boundary:

- No scheduler, provider-specific MCP/A2A transport, role business logic,
  evaluation, or direct evidence persistence.

Merge gate:

- Multiple AgentSpecs run on one kernel; deterministic mock runs and every stop,
  repair, budget, and command/observation path are covered.

## 05 - Protocol and Tool Gateway

Branch: `codex/bg001-05-protocol-tool-gateway`

Complete deliverables:

- Replace the fake MCP naming with a real Protocol & Tool Gateway.
- Implement Function Calling -> Command normalization.
- Implement MCP host/client lifecycle, capability discovery, tools/resources/
  prompts, stdio and streaming HTTP transports, schema conversion, cancellation,
  timeouts, and health state against a pinned protocol version.
- Implement A2A capability discovery, TaskEnvelope submission/status/cancel,
  Artifact handoff, failure classification, and cross-deployment correlation.
- Implement versioned Tool Registry, risk/permission/idempotency metadata,
  approval, rate limiting, caching, normalization, safety scan, cost recording,
  retry, fallback, and circuit breaking.
- Convert Tavily, Exa, and scraper implementations into normal tool adapters.

Boundary:

- No task DAG, research planning, task status ownership, or direct evidence
  mutation.

Merge gate:

- Real protocol conformance and lifecycle tests pass; Function Calling, MCP,
  and A2A map consistently to internal contracts; cancellation, approval,
  idempotency, timeout, rate-limit, retry, and circuit-breaker paths pass.

## 06 - Orchestration runtime

Branch: `codex/bg001-06-orchestration-runtime`

Complete deliverables:

- Implement durable task queue and dynamic task-DAG projection.
- Implement dependency, priority, deadline, and budget scheduling.
- Implement controlled concurrency, worker slots, task create/split/merge/defer/
  prune/cancel/retry/fail/complete transitions, crash recovery, user cancellation,
  pause/approve/reject/edit/resume HITL, and idempotent scheduling.
- Implement both LangGraphRuntimeAdapter and native event-sourced scheduler under
  one Scheduler contract and conformance suite.
- Thin LangGraph state to IDs, projection revision, compact active-task/budget
  snapshot, artifact IDs, error ref, and final-report artifact ID.

Boundary:

- No claim/citation semantics, model reasoning, artifact bodies, or role logic.

Merge gate:

- Concurrency, restart, crash recovery, cancellation, approval, idempotency, and
  adapter conformance tests pass; graph state is not a domain database.

## 07 - Evidence Engine and independent verifier

Branch: `codex/bg001-07-evidence-engine-verifier`

Complete deliverables:

- Complete Source -> Snapshot -> Passage -> Evidence -> Fact -> Claim ->
  Citation -> Section -> Report graph.
- Separate candidate and verified knowledge.
- Persist source snapshot, quote offsets, content hash, extraction method,
  timestamp, source level, and citation location.
- Implement independent verifier AgentSpec for quote grounding, claim support,
  overreach, contradiction, source independence, authority/freshness, and
  citation correctness/completeness.
- Implement supported, partially_supported, contradicted, conflicted,
  unsupported, and stale claim states.
- Implement conflict severity/resolution, high-impact claims, section coverage,
  and bounded repair loops with artifact/event feedback.

Boundary:

- No search scheduling, final writing, source snapshot mutation, or offline
  skill optimization.

Merge gate:

- Every verified claim traces to source text; unsupported high-impact claims
  block definitive writing; conflicts remain visible; Frozen Replay verification
  is stable.

## 08 - Research Supervisor and Worker Pool

Branch: `codex/bg001-08-supervisor-worker-pool`

Complete deliverables:

- Implement model-driven Supervisor AgentSpec and validated/repaired dynamic
  task decomposition rather than fixed templates.
- Implement structured TaskEnvelope inputs/outputs, constraints, artifacts,
  schemas, budgets, deadlines, and dependencies.
- Implement Worker AgentSpec commands for search, read, extract, delegate,
  compare, and source verification.
- Implement controlled worker concurrency, source/query/task deduplication,
  cross-worker result merging, information-gain estimation, retry/fallback, and
  dynamic replanning.
- Implement semantic convergence for required sections, high-impact evidence,
  unresolved severe conflicts, repeated low gain, all budgets, cancellation,
  and approval.

Boundary:

- Workers do not write final reports; Supervisor does not call providers
  directly; no free-form agent chat, report reviewer, or offline optimization.

Merge gate:

- Dynamic DAG, dependency-safe concurrency, deduplication, replanning, fallback,
  and bounded semantic convergence tests pass.

## 09 - Synthesis Writer and Report Reviewer

Branch: `codex/bg001-09-synthesis-writer-reviewer`

Complete deliverables:

- Implement Writer AgentSpec that reads only verified evidence.
- Implement section synthesis, multi-source citations, conflict presentation,
  uncertainty language, citation placement/completeness, and cross-section
  consistency.
- Implement Reviewer AgentSpec and rubric for completeness, support, citation,
  conflicts, instruction following, depth, organization, and readability.
- Implement bounded targeted-research, citation-repair, local-rewrite, structural
  rewrite, accept, and reject report-loop commands.
- Persist report artifacts, versions, revision history, and citation map.

Boundary:

- Writer cannot search or use unverified evidence; Reviewer cannot change
  verification conclusions; no release gate or skill optimization.

Merge gate:

- Every factual report claim is traceable; unsupported claims appear only as
  explicit gaps/uncertainty; reviewer feedback maps to bounded repair actions;
  report citations and citation map agree.

## 10 - Evaluation Lab core

Branch: `codex/bg001-10-evaluation-lab-core`

Complete deliverables:

- Implement versioned Dataset Registry with enforced train/dev/selection/test/
  hidden-test access.
- Implement deterministic Frozen Replay and repeated Live Web modes with source
  change and variance reporting.
- Implement deterministic evaluators for URL/citation/quote/schema/coverage,
  source type/authority/freshness/diversity/primary share, tokens, cost, latency,
  failure, recovery, idempotency, and protocol compliance.
- Implement trace metrics: evidence/tool call, redundant search, recovery rate,
  convergence turns, invalid tool calls, and budget violations.
- Implement Experiment Registry with complete component/environment/artifact
  provenance and compare legacy, fixed-workflow, and new runtime baselines.

Boundary:

- No LLM judge, skill patches, release promotion, or test/hidden tuning.

Merge gate:

- Frozen Replay is reproducible; Live Web reports mean/variance; experiments are
  reproducible by component version; split violations are rejected.

## 11 - Semantic evaluation and release gates

Branch: `codex/bg001-11-evaluation-semantic-gates`

Complete deliverables:

- Implement claim support and supported/contradicted/unsupported/uncited-fact
  metrics; retrieval precision/recall/authority/freshness/diversity; report
  completeness/depth/instruction/organization/readability metrics.
- Implement blind, randomized, fixed-version, multi-judge voting with disagreement
  records and human-correlation calibration.
- Use deterministic evaluators for URL/schema/citation position/cost/source time.
- Implement gate policy for required improvements, non-regression, cost, variance,
  safety, and protocol thresholds.
- Implement immutable Version Registry and promote/reject/rollback states for
  AgentSpec, Skill, Prompt, Tool Policy, Stop Policy, and Rubric.
- Restrict test/hidden-test to final promotion.

Boundary:

- Gates decide promotion but do not generate changes; judges do not search or
  rewrite; no automatic skill optimization.

Merge gate:

- Human calibration is recorded; hidden-test leakage is blocked; every decision
  is auditable; regressions cannot publish; rollback is complete.

## 12 - Studio V2

Branch: `codex/bg001-12-studio-v2`

Complete deliverables:

- Add task DAG split/merge/skip/fail/retry/dependency visualization.
- Add Evidence Graph for claims, evidence, sources, citations, conflicts, and
  sections with source-snapshot navigation.
- Add event-derived state diff for tasks, budgets, and evidence status.
- Add error/retry chains, conflict navigation, current component versions, and
  cost/token/latency/budget views.

Boundary:

- Read-only; no replay, fork, A/B, badcase, or direct table access.

Merge gate:

- Every view node resolves to events/artifacts; state diff rebuilds from events;
  large graphs are incrementally queryable; no hidden chain-of-thought appears.

## 13 - Studio V3 and V4

Branch: `codex/bg001-13-studio-v3-v4`

Complete deliverables:

- Fork/replay from any eligible span into a new immutable run.
- Support explicit saved-tool-result and live-environment replay modes, component
  version selection, failed-span restart, and reapproval for side-effect tools.
- Implement A/B run, span, task DAG, evidence graph, component diff, metric,
  cost, latency, and convergence comparison.
- Implement Prompt/Skill/Policy diff and one-click badcase creation with original
  run/span/input/component/evaluation/human-note provenance.

Boundary:

- Never rewrite historical events; badcase creation does not trigger changes;
  A/B does not publish; no optimizer.

Merge gate:

- Original runs are immutable; saved replay is network-free; live replay is
  labeled; A/B aligns dataset samples; badcases preserve the full provenance.

## 14 - Offline evolution

Branch: `codex/bg001-14-offline-evolution`

Complete deliverables:

- Implement optimizer inputs from scored success/failure traces, badcases,
  evaluations, and current versions.
- Optimize planning, query generation, source selection, extraction/citation,
  report writing, tool routing, stop policy, and grader rubric.
- Permit only structured add/delete/replace patches with per-round edit budgets
  (text learning rate).
- Implement candidate versions, strict selection improvement, rejected-edit
  memory, static versioned best_skill, final test/hidden promotion, human gate,
  publish, and rollback.
- Separate runtime memory, cross-task experience, and formal Skill Registry.
- Production traces enter only a candidate pool and cannot self-promote.

Boundary:

- No online code/weight/skill modification and no RL frameworks or training.

Merge gate:

- Every patch is auditable; only strict selection improvement is accepted;
  leakage is blocked; rejected history affects future generation; rollback and
  no-extra-online-inference guarantees pass.

## 15 - Integration and hardening

Branch: `codex/bg001-15-integration-hardening`

Complete deliverables:

- Switch run_research and Console/Studio entry points to the new runtime and
  projection APIs.
- Remove module-global managers/builders/observers, fat GraphState, fake MCP,
  and superseded planner/writer/router/state-manager/vector-store draft code.
- Run full single/concurrent run, crash recovery, tool/model/protocol failure,
  cancellation, approval, budget, conflict, report repair, replay/fork,
  evaluation gate, and evolution release/rollback tests.
- Run database-lock, artifact-size, long-trace, long-report, security/redaction,
  replay-side-effect, and dataset-permission stress tests.
- Complete the scheduler ADR: remove LangGraph only if the native scheduler
  meets recovery/concurrency/cancel/approval/consistency gates; otherwise retain
  it strictly as an adapter.
- Update README with implemented/current versus optional/unsupported capability
  labels and no unsupported SOTA/autonomous-evolution claims.

Boundary:

- No new role, protocol, domain capability, compatibility layer, or RL work.

Final gate:

- All seven Background001 architecture layers, four loops, five roles, protocol
  adapters, Studio V1-V4, Frozen/Live evaluation, release gates, and offline
  evolution are fully implemented and exercised end to end.

## Execution records

Append one record after every branch:

```text
Branch:
Started from integration commit:
Scope delivered:
Boundary check:
Tests:
Failure/recovery tests:
Integration merge commit:
Remote push:
Remaining risks:
```

### Branch 00 execution record

```text
Branch: codex/bg001-00-foundation-contracts-baseline
Started from integration commit: 84dd6f6
Scope delivered: Framework-independent versioned contracts and state machines;
  schema migration/canonical serialization; dataset access separation; frozen
  offline replay and deterministic draft baseline; five architecture ADRs;
  test collection and process-global isolation corrections.
Boundary check: No planner, researcher, or writer behavior changed. No event or
  artifact store, agent kernel, protocol gateway, scheduler, or Studio was
  implemented. The only application correction makes a completed Console run
  report the completed stage consistently.
Tests: 92 passed in normal order; 92 passed with test files in reverse order;
  21 foundation/baseline cases passed; committed baseline replay reproduced.
Failure/recovery tests: Existing session restart/recovery, graph persistence,
  error-path, and offline integration cases are included in both 92-test runs.
Integration merge commit: e406407
Remote push: verified successful to origin/codex/bg001-integration on 2026-07-22
Remaining risks: The captured draft baseline intentionally records missing
  model/tool trace coverage and uninstrumented token, cost, and latency. These
  are owned by branches 01, 04, and 10 rather than disguised in foundation.
```

### Branch 01 execution record

```text
Branch: codex/bg001-01-event-trace-store
Started from integration commit: a8f0b47
Scope delivered: Storage-neutral EventStore API; transactional SQLite schema
  and migrations; indexed run/event catalog and pagination; idempotency,
  continuous ordering, span and terminal invariants; checksums, integrity audit,
  backup/restore; redaction and bounded decision summaries; durable export
  outbox and OTLP/HTTP JSON exporter; persistent compatibility observer; actual
  graph, model, search, scraper, retry, budget, evidence, report, and failure
  instrumentation; production run_research event runtime wiring.
Boundary check: No artifact bodies, Studio projection/UI, replay, AgentKernel,
  task scheduler, GraphState thinning, planner strategy, research decisions, or
  writer synthesis behavior was added or changed. Provider changes are limited
  to emitting retry telemetry from the existing retry path.
Tests: 111 passed in normal order and 111 passed with test files in reverse
  order; Frozen Replay exactly matched the branch-00 baseline.
Failure/recovery tests: Same-process and cross-connection concurrent appends,
  observer/store restart, v1 migration with existing data, checksum corruption,
  orphan/open-child spans, sequence conflicts, duplicate terminal, long-run
  pagination, backup/restore, redaction, graph failure, retry callback, OTLP
  transport failure, durable outbox restart/retry, and real model usage paths.
Integration merge commit: 7a90c9a
Remote push: verified successful to origin/codex/bg001-integration on 2026-07-22
Remaining risks: Console continues to use its draft MemoryObserver until Studio
  V1 branch 03 replaces it with event projections. Artifact bodies and replay
  remain deliberately absent per this branch boundary.
```

### Branch 02 execution record

```text
Branch: codex/bg001-02-artifact-knowledge-storage
Started from integration commit: 6b72c55
Scope delivered: Immutable content-addressed ArtifactStore for every artifact
  kind with provenance, explicit redaction, typed links, migrations, pagination,
  checksums, integrity audit, backup/restore, and concurrent access; revisioned
  Source -> Snapshot -> Passage -> Evidence -> Fact -> Claim -> Citation ->
  Conflict/Section/Report repositories with typed relationships and run-scoped
  natural identities; deterministic normalization and deduplication; optional
  vector adapter; complete retrieval and rebuildable projection adapters;
  manifest-verified KnowledgeRuntime backups; actual source-body persistence,
  source versioning, replay-idempotent ingestion, and controlled graph dual-write.
Boundary check: No claim verification, evidence judgment, planning strategy,
  writer synthesis, scheduler, Studio, protocol gateway, AgentKernel, evaluation,
  evolution, or RL behavior was implemented. Existing agent outputs and Frozen
  Replay remain unchanged; legacy session knowledge is retained only as a
  transitional draft input path.
Tests: 126 passed in normal order and 126 passed with test files in reverse
  order; 15 branch-specific storage/ingestion cases passed; Frozen Replay
  exactly matched the branch-00 baseline; integration branch rerun passed 126.
Failure/recovery tests: Same-process and cross-connection artifact/knowledge
  writes, cross-run collision and relationship rejection, missing/wrong-type
  references, immutable-row enforcement, stable cursor pagination, v1 migration
  with retained data, restart, checksummed backup/restore, manifest validation,
  body/envelope/entity corruption, redaction, source-version changes, idempotent
  replay, full typed evidence traversal, optional vector ranking, and graph
  dual-write are covered.
Integration merge commit: 58de1c2
Remote push: feature and integration commits verified successful to origin on
  2026-07-22.
Remaining risks: Console still reads the draft MemoryObserver/session projection
  until branch 03 installs persistent Studio projections. The graph dual-write
  setter and legacy session manager remain transitional and are removed when
  branch 15 switches every entry point to the new runtime.
```

### Branch 03 execution record

```text
Branch: codex/bg001-03-studio-v1
Started from integration commit: be3ddc8
Scope delivered: Independent persistent Thread -> Run -> Span -> Timeline
  projection schema with migrations, sequence/idempotency validation, cumulative
  model/tool usage, artifacts, versions, latency, attempts, retry, permission,
  error and terminal-state fields; event-outbox projection delivery and catch-up;
  deterministic rebuild; thread/run/span catalogs; filtered full-text timeline
  queries with stable cursors; SSE transport; complete JSON/NDJSON trace export;
  server-side defense-in-depth masking; Console replacement of MemoryObserver
  and state-event fallback plus searchable/paginated detailed trace UI.
Boundary check: No task-DAG, evidence-graph, state-diff, replay, fork, A/B,
  badcase, scheduler, AgentKernel, evaluation, evolution, or RL behavior was
  added. Studio-specific service and HTTP tests pass when LangGraph-state reads
  are forced to fail; source events remain immutable.
Tests: 132 passed in normal order and 132 passed with test files in reverse
  order; eight Studio/Console cases passed; JavaScript syntax validation passed;
  Frozen Replay exactly matched the branch-00 baseline; integration rerun passed
  all 132 tests.
Failure/recovery tests: 2,052-event pagination and complete export, restart
  persistence, projection deletion/rebuild equality, sequence-gap rejection,
  checksum corruption, delivery failure/outbox retry, filter/search cursors,
  span lifecycle details, run terminal status, source-event and projection
  masking, SSE ordering, HTTP export, and observer restoration are covered.
Integration merge commit: 56224fe
Remote push: feature and integration commits verified successful to origin on
  2026-07-22.
Remaining risks: Existing non-Studio Console report/task/context panels still
  use transitional draft graph/session data. Branches 06, 12, and 15 replace
  those panels with task/evidence projections and complete entry-point migration.
```

### Branch 04 execution record

```text
Branch: codex/bg001-04-agent-kernel
Started from integration commit: acf00ef
Scope delivered: Complete framework-independent Observe -> Decide -> Act ->
  Verify task loop; required immutable AgentSpec registry; typed model, action,
  verification, cancellation, and event interfaces; deterministic structured
  command parsing and identity normalization; bounded schema repair; context
  trimming, recursive redaction, complete component-version injection, hard
  policy checks, argument constraints, risk approval, expiry, and tool ceilings;
  normalized observations and errors; semantic repair feedback; all eight
  budget dimensions plus deadline interruption; every required stop policy;
  structured TaskResult/KernelRunResult; and a persistent EventRecorder adapter
  with balanced Agent/Model/Tool spans on success, failure, repair, timeout, and
  cancellation paths.
Boundary check: No scheduler, durable task queue/DAG, task-state ownership,
  provider-specific Function Calling/MCP/A2A transport, research-role business
  logic, claim/evidence verification semantics, evidence persistence, report
  writing, evaluation, evolution, UI, or RL work was added. The kernel depends
  only on contracts and injected interfaces; event persistence is isolated in
  the explicit event-sink adapter.
Tests: 177 tests passed in normal order and 177 passed with test files in
  reverse order; 45 deterministic branch-specific cases cover every command
  kind, observation status, required stop, schema repair, policy, middleware,
  budget, deadline, and multiple-AgentSpec path; the integration branch rerun
  passed all 177 tests.
Failure/recovery tests: Model/action/verifier retry and permanent failure,
  malformed and repeatedly invalid schema, repair-provider failure, context
  exhaustion, active cancellation, wall-time interruption, repeated errors,
  no-gain convergence, invalid executor output, identity injection, redaction,
  approval/denial, exact budget ceilings, event-store span balancing, sequence
  ordering, and persistent success/failure trace integrity are covered.
Integration merge commit: 453c564
Remote push: feature commit d8f9cc1 and integration commits verified successful
  to origin on 2026-07-22.
Remaining risks: The legacy draft graph intentionally does not instantiate the
  new kernel yet. Branches 05-09 provide real protocol adapters, durable
  scheduling, evidence verification, and the five role AgentSpecs before branch
  15 switches production entry points. Persistent cross-release version
  promotion remains owned by branch 11; this branch registry is the immutable
  task-runtime resolver.
```

### Branch 05 execution record

```text
Branch: codex/bg001-05-protocol-tool-gateway
Started from integration commit: 1fddfe8
Scope delivered: Deterministic Function Calling normalization for generic,
  OpenAI, Anthropic, and choice-envelope shapes; immutable versioned Tool
  Registry; durable SQLite idempotency leases/results, cache, rate-limit
  windows, circuit state, and checksummed audit events; permission, operation,
  risk, approval, schema, SSRF/control-character safety, cost, timeout,
  cancellation, retry, fallback, circuit, input/output normalization, and
  AgentKernel Observation integration; official-SDK MCP 2025-11-25 host/client
  lifecycle over stdio and Streamable HTTP with pagination-aware tools,
  resources, prompts, schema mapping, health, session termination, timeout, and
  cancellation; official-SDK A2A 1.0 discovery, binding/version validation,
  TaskEnvelope submission, task status/cancel, artifact handoff and stream
  accumulation, correlation/trace propagation, timeout/cancellation, and typed
  failure classification; governed Tavily, Exa, and scraper Tool Adapters with
  no production demo fallback; migration of the draft researcher and manual
  smoke path away from fake MCP naming; protocol/deployment documentation.
Boundary check: No task queue/DAG, dependency or priority scheduling, research
  planning, task-state ownership, evidence/claim/citation mutation or
  verification, report writing, role business logic, evaluation, evolution,
  UI capability, or RL work was added. The only cross-boundary hardening is
  SQLite checkpointer connection initialization: busy timeout is applied before
  WAL negotiation with bounded lock retry and failed-connection cleanup; it
  carries no scheduler semantics.
Tests: 214 tests passed in normal order and 214 passed with test files in
  reverse order; 43 gateway/protocol/adapter/researcher cases passed. Real MCP
  stdio subprocess and Streamable HTTP servers and official A2A protobuf/SDK
  transport paths are exercised. The Studio polling/WAL concurrency regression
  passed five consecutive repetitions after bounded connection hardening.
Failure/recovery tests: Registry immutability/version activation, malformed
  Function Calls, permission and approval rejection, SSRF and output safety,
  durable idempotent replay and cache restart, concurrent duplicate calls,
  restart-safe rate limiting and circuit fallback, retry accounting, timeout,
  cancellation, schema failures, audit corruption, MCP invalid input/version/
  close/health/session/timeout/cancellation, A2A version/binding/auth/timeout/
  cancellation/message-only response/stream identity, missing provider keys,
  HTTP failures, provider fallback, and isolated offline state are covered.
Integration merge commit: pending
Remote push: pending
Remaining risks: The transitional draft graph invokes the governed research
  facade but does not yet make MCP/A2A transport or AgentKernel the universal
  production entry point. Branch 06 owns durable orchestration and state
  thinning; branches 07-09 own evidence and role semantics; branch 15 removes
  the remaining draft paths. Cross-release policy promotion remains branch 11.
```
