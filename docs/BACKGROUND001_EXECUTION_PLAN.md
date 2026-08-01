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
  |-- all branches -- 15 Integration/Hardening
  `-- 15 -- 16 Console Runtime Alignment -- 17 Live Runtime Contract Recovery
```

## Branch status

| # | Branch | Status | Integration commit | Notes |
| --- | --- | --- | --- | --- |
| 00 | `codex/bg001-00-foundation-contracts-baseline` | integrated | e406407 | Contracts, Frozen Replay, baseline, ADRs, and order-independent tests complete. |
| 01 | `codex/bg001-01-event-trace-store` | integrated | 7a90c9a | Append-only SQLite event store, instrumentation, redaction, and durable OTLP outbox complete. |
| 02 | `codex/bg001-02-artifact-knowledge-storage` | integrated | 58de1c2 | Immutable content-addressed artifacts, revisioned evidence repositories, ingestion/retrieval/projection split, and durable pipeline dual-write complete. |
| 03 | `codex/bg001-03-studio-v1` | integrated | 56224fe | Persistent event-derived Thread/Run/Span projections, searchable/paginated SSE timeline, trace export, masking, and Console trace migration complete. |
| 04 | `codex/bg001-04-agent-kernel` | integrated | 453c564 | Complete Observe-Decide-Act-Verify kernel, middleware, budgets, stops, policies, normalized observations, and durable trace adapter. |
| 05 | `codex/bg001-05-protocol-tool-gateway` | integrated | aa4ab3e | Function Calling, real MCP/A2A transports, governed versioned tools, and Tavily/Exa/scraper adapters complete. |
| 06 | `codex/bg001-06-orchestration-runtime` | integrated | 0c2ba88 | Durable event-sourced task DAG, leases/recovery, HITL controls, hard scheduling budgets, and real thin-state LangGraph adapter complete. |
| 07 | `codex/bg001-07-evidence-engine-verifier` | integrated | e9880da | Complete evidence graph, independent verification, six claim states, conflicts, coverage, bounded repair, and verified-only read boundary complete. |
| 08 | `codex/bg001-08-supervisor-worker-pool` | integrated | 2b54206 | Dynamic Supervisor planning, governed Worker Pool, durable dedup/merge/recovery, and semantic convergence complete. |
| 09 | `codex/bg001-09-synthesis-writer-reviewer` | integrated | 5838f81 | Verified-only Writer, deterministic full-rubric Reviewer, bounded report repairs, and durable revisions/citation maps complete. |
| 10 | `codex/bg001-10-evaluation-lab-core` | integrated | b5834e9 | Five-split datasets, Frozen/Live modes, deterministic/trace metrics, and reproducible baseline experiments complete. |
| 11 | `codex/bg001-11-evaluation-semantic-gates` | integrated | b9ad3fa | Semantic metrics, calibrated blind judging, release gates, and immutable version promotion/rollback complete. |
| 12 | `codex/bg001-12-studio-v2` | integrated | e363823 | Read-only paginated Task/Evidence graphs, event-rebuilt state diffs, recovery/resource/version views, snapshot navigation, and Studio UI complete. |
| 13 | `codex/bg001-13-studio-v3-v4` | integrated | b537426 | Immutable AgentKernel replay/fork, fresh approvals, aligned A/B, component diff, badcase provenance, durable recovery, APIs and Studio V4 UI complete. |
| 14 | `codex/bg001-14-offline-evolution` | integrated | 80dbde8 | Reviewed trace/badcase/evaluation pool, bounded structured patches, strict gated selection, rejected-edit memory, human release, best_skill and rollback complete. |
| 15 | `codex/bg001-15-integration-hardening` | integrated | c241efc | Production composition, draft removal, native-scheduler ADR, full recovery/security/stress gates, and capability-boundary documentation complete. |
| 16 | `codex/bg001-16-console-runtime-alignment` | integrated | 1d9c3f4 | Complete root Console projection, operational workspace, report lifecycle, actions, accessibility, and browser/API contract alignment. |
| 17 | `codex/bg001-17-live-runtime-contract-recovery` | completed locally | pending merge | Strict live Worker contracts, governed phase protocol, grounded ingestion, bounded recovery/convergence, verified-only reporting repair, causal errors, and Console role/stage projection complete. |

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

## 16 - Console runtime alignment

Branch: `codex/bg001-16-console-runtime-alignment`

Complete deliverables:

- Replace the draft root Console with a complete operational workspace aligned
  to the production ApplicationRuntime, native scheduler, five roles,
  independent verification, reporting loop, and Studio projections.
- Add the versioned typed `ConsoleWorkspace@2` projection for identity,
  runtime, actions, scheduler, tasks, evidence readiness, reporting lifecycle,
  timeline, and exact-run Studio navigation.
- Implement every queued/running/approval/reporting/completed/failed/cancelled
  state, visibility-aware polling, approve/cancel flows, retry handling, and
  preservation of the last good view.
- Implement complete landing, run overview, task DAG/inspector, evidence,
  report, and timeline workspaces with responsive and accessible interaction.
- Expose report revisions, reviewer scores/findings/repairs, terminal outcome,
  citation metadata, verified writer packets, sources, coverage gaps, high
  impact blockers, severe conflicts, task budgets, leases, artifacts, and
  errors.
- Add typed projection/API tests, JavaScript module and interaction tests,
  security/accessibility/responsive checks, deterministic end-to-end browser
  validation, and full integration regression coverage.

Boundary:

- No change to runtime/scheduler/evidence/reporting decisions, Studio replay or
  evaluation/evolution semantics, provider policy, release process, or RL.
- No compatibility facade for the draft Console response and no duplication of
  advanced Studio mutation workflows.

Merge gate:

- Every production run state and five-role stage is represented correctly;
  approve/cancel and polling are fenced by backend action availability; task,
  evidence, coverage, report, timeline and Studio links resolve to the exact
  run; payloads are escaped and Markdown is allowlisted; desktop/tablet/mobile
  browser checks and the full normal/reverse test suites pass.

## 17 - Live runtime contract recovery

Branch: `codex/bg001-17-live-runtime-contract-recovery`

Complete deliverables:

- Make every Research Worker model request expose a strict command schema
  derived from its AgentSpec and governed tool contracts, including allowed
  command kinds, canonical tool names, argument objects, and command-specific
  required fields.
- Normalize only unambiguous, allowlisted model omissions: infer a canonical
  tool name from a permitted command kind and infer a command kind from a
  canonical permitted tool name. Reject ambiguous, unknown, or conflicting
  command identities with structured repair feedback.
- Use the same strict contract for initial generation and bounded schema repair,
  and preserve the first causal validation/tool error in task and run outcomes.
- Route schema exhaustion, partial Worker outcomes, retryable tool failures and
  budget-limited attempts through the scheduler's bounded retry/replan policy
  instead of allowing empty convergence cycles.
- Gate the research-to-report transition on scheduler work state, required
  section coverage, verified evidence readiness, blockers and convergence
  semantics. Represent maximum-cycle and no-evidence outcomes explicitly as
  incomplete or failed rather than successful research.
- Build Writer gap disclosures from exact persisted Writer-packet gap identities;
  permit honest incomplete reports only through that contract; route reparable
  Writer domain-validation failures through bounded repair and persist their
  revisions/outcomes.
- Project the complete causal error chain and derive Console stages and five-role
  statuses from authoritative scheduler, trace, evidence and report lifecycle
  state, including Worker and Writer failures.
- Add regression coverage for live DeepSeek-shaped command responses, canonical
  tool resolution, bounded recovery, convergence/report gates, empty evidence,
  exact gaps, report repair, causal errors and Console stage projection; finish
  with a real configured DeepSeek/Tavily run of the reproduced research query.

Boundary:

- No new role, tool, provider, protocol, evidence state, reviewer rubric,
  scheduler transition type, evaluation/evolution capability, compatibility
  facade, deployment, main merge, release, online self-modification, or RL work.
- Do not weaken Writer verified-only or exact-gap invariants. Do not synthesize
  fabricated evidence or treat provider HTTP success as semantic task success.

Merge gate:

- DeepSeek-shaped malformed and partially specified command responses are
  either normalized to an allowlisted canonical command or repaired/rejected
  deterministically; an unknown short tool name cannot reach the gateway.
- Retryable Worker failures consume bounded attempts and either recover or
  terminate with their earliest cause; convergence cannot spin through empty
  cycles or mark research complete with unresolved mandatory work.
- Reporting receives a verified packet or an explicit exact-gap incomplete
  packet, and Writer validation failures receive bounded repair feedback.
- The Console shows authoritative role/stage failure and the complete causal
  chain. Focused, full normal/reverse, restart/failure-path, and real live-run
  browser verification pass.

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
  passed five consecutive repetitions after bounded connection hardening; the
  post-merge integration rerun also passed all 214 tests in both orders.
Failure/recovery tests: Registry immutability/version activation, malformed
  Function Calls, permission and approval rejection, SSRF and output safety,
  durable idempotent replay and cache restart, concurrent duplicate calls,
  restart-safe rate limiting and circuit fallback, retry accounting, timeout,
  cancellation, schema failures, audit corruption, MCP invalid input/version/
  close/health/session/timeout/cancellation, A2A version/binding/auth/timeout/
  cancellation/message-only response/stream identity, missing provider keys,
  HTTP failures, provider fallback, and isolated offline state are covered.
Integration merge commit: aa4ab3e
Remote push: feature commit 2de448b and integration commits verified successful
  to origin on 2026-07-23.
Remaining risks: The transitional draft graph invokes the governed research
  facade but does not yet make MCP/A2A transport or AgentKernel the universal
  production entry point. Branch 06 owns durable orchestration and state
  thinning; branches 07-09 own evidence and role semantics; branch 15 removes
  the remaining draft paths. Cross-release policy promotion remains branch 11.
```

### Branch 06 execution record

```text
Branch: codex/bg001-06-orchestration-runtime
Started from integration commit: da05c6d
Scope delivered: Storage-neutral asynchronous Scheduler contract; transactional
  append-only SQLite scheduler event journal and fully rebuildable run, task,
  dependency, and lease projections; checksums, migrations, integrity audit,
  online backup, restart and deterministic rebuild; atomic dependency,
  priority, deadline, task-budget, actor-assignment, and global worker-slot
  scheduling; complete create/split/merge/defer/prune/claim/heartbeat/complete/
  fail/retry/cancel/edit transitions; worker lease fencing and crash recovery;
  task/run pause, resume, cancellation, approval request/edit/approve/reject;
  mutation fingerprint idempotency including empty claims; native event-sourced
  scheduler and real AsyncSqliteSaver-backed LangGraph adapter under one
  conformance suite; exact thin checkpoint with IDs, projection revision,
  compact task/budget snapshot, artifact/error refs, and final-report artifact
  ref. The reverse-order gate also exposed and fixed an AgentKernel race so
  timed-out/cancelled actions terminate before independent verification.
Boundary check: No claim/evidence/citation semantics, research decomposition,
  model decision logic, artifact bodies, source content, report prose, role
  business logic, evaluation, evolution, Studio capability, or RL work was
  added. Existing graph/Console entry points remain transitional until branch
  15; the kernel correction only stabilizes an already-required budget stop.
Tests: 225 passed in normal order and 225 passed with test files in reverse
  order before merge; 225 passed in both orders after integration; 11 focused
  orchestration cases and 20 consecutive wall-time regression repetitions
  passed. Only upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Cross-connection atomic claim and global-slot fencing,
  mutation replay/conflicting reuse, stale and wrong-owner lease rejection,
  restart recovery, exhausted attempts, deadline precedence, intermediate and
  completion-time budget exhaustion, split/merge/defer/prune/cycle rejection,
  approval approve/edit/reject, task/run pause/resume/cancel, terminal-run
  protection, checksum corruption, projection deletion/rebuild, backup/restore,
  event revision integrity, and missing-thick-state LangGraph checkpoints are
  covered for native and LangGraph paths.
Integration merge commit: 0c2ba88
Remote push: feature commit de95102 and integration commits verified successful
  to origin on 2026-07-24.
Remaining risks: The draft production graph does not instantiate this scheduler
  before branches 07-09 supply evidence and five-role consumers; branch 15 owns
  entry-point migration and deletion of fat GraphState paths. Scheduler DAG and
  budget events are not exposed in Studio V1; branch 12 owns their event-derived
  Studio V2 projections. Evidence-domain scheduling policies remain branch 08.
```

### Branch 07 execution record

```text
Branch: codex/bg001-07-evidence-engine-verifier
Started from integration commit: 1ba61a2
Scope delivered: Complete revisioned Source -> Snapshot -> Passage -> Evidence
  -> Fact -> Claim -> Citation -> Section -> Report contracts and resolver;
  persisted source level, publication/fetch/extraction timestamps and methods,
  immutable hashes, exact quote/citation offsets and locations; strict candidate
  and verified read boundaries; independent Evidence Verifier AgentSpec with
  deterministic and shared-ModelAdapter semantic implementations, bounded
  schema repair and full usage accounting; passage/snapshot integrity, quote
  grounding, relation, fact, overreach, contradiction, source access,
  publisher/domain independence, authority, primary-source, freshness, citation
  path/accuracy/completeness, severe-conflict and section-coverage checks; exact
  supported/partially_supported/contradicted/conflicted/unsupported/stale claim
  outcomes; high-impact writing blockers; conflict severity, definitive
  verified-evidence resolution and accepted-unresolved visibility; immutable
  verification/repair artifacts, durable domain events, normalized graph-input
  invalidation, idempotent replay, and crash reconstruction of missing feedback
  and projections.
Boundary check: No search/query scheduling, worker or supervisor task logic,
  final report prose or writer behavior, source refetch or snapshot mutation,
  scheduler state, Studio feature, evaluation/release gate, offline Skill/Prompt
  optimization, online self-modification, or RL work was added. The draft graph
  is unchanged except for its existing artifact/knowledge ingestion gaining the
  evidence metadata required by this branch.
Tests: 240 passed in normal order and 240 passed with test files in reverse
  order before integration; 240 passed in both orders after the no-ff
  integration merge. Fifteen branch-specific cases cover graph traceability,
  all six claim states, high-impact gates, source independence, input-revision
  invalidation, citation-only quote backfill, conflict resolution, section
  coverage, verifier schema repair, durable events, and candidate/verified
  isolation. Only upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Passage and snapshot hash mismatch, quote and citation
  grounding/location failure, repair-budget exhaustion, duplicate event and
  revision replay, unchanged distiller replay after verification, process
  restart, verification-result commit followed by repair/projection crash,
  missing feedback reconstruction, changed source authority re-verification,
  rejected unverified conflict resolution, accepted-unresolved conflicts, and
  immutable snapshot-history checks are covered.
Integration merge commit: e9880da
Remote push: feature commit ec65dc4 pushed successfully; integration push is
  recorded by the immediately following plan commit on 2026-07-24.
Remaining risks: Branch 08 must make Supervisor/Worker convergence consume
  high-impact blockers, coverage gaps, conflict state and structured repair
  feedback through the durable scheduler. Branch 09 must enforce the verified
  read boundary in Writer/Reviewer role loops. Production entry points remain
  on the transitional draft graph until branch 15, as planned.
```

### Branch 08 execution record

```text
Branch: codex/bg001-08-supervisor-worker-pool
Started from integration commit: 4b237f4
Scope delivered: Model-driven Research Supervisor AgentSpec with bounded
  structured-output repair and authoritative early-stop rejection; complete
  research-only SupervisorPlan/TaskProposal contracts with constraints,
  artifacts, schemas, all budgets, priority, deadline, attempts, worker
  assignment and dependency-cycle validation; dependency-semantic task
  fingerprinting and crash-safe immutable plan application to the scheduler
  DAG; shared-AgentKernel Worker AgentSpec for governed search/read/extract/
  delegate/compare/verify-source commands; configured-pool assignment fencing;
  WAL/checksummed coordination storage for leased query/source deduplication,
  semantic task reservations, novelty, Worker attempts, merges and convergence
  decisions; canonical source/query handling; independent information-gain
  verification; scheduler-controlled pool concurrency; immutable observation,
  Worker, delegation, merge and convergence artifacts; real ProtocolToolGateway
  retry/fallback integration; cross-store Worker-result reconciliation after a
  process crash; retry, cancellation and resumable HITL approval paths; dynamic
  convergence-driven replanning; and bounded convergence over required section
  and citation coverage, high-impact blockers, high/critical conflicts, failed
  or pending work, repeated low gain, all aggregate budgets, maximum cycles,
  cancellation and approval. Command normalization also received a one-line
  deterministic proposed-at correction after the full gate exposed a pre-existing
  timestamp race.
Boundary check: Workers cannot synthesize or review final reports and execute
  providers only through the injected governed command boundary. The Supervisor
  has no tool grants/provider access and emits only structured delegation,
  approval or stop commands. No free-form agent chat, Writer/Reviewer behavior,
  evaluation/release gate, Studio feature, offline optimization, online
  self-modification, compatibility layer, or RL work was added. Production
  entry points remain transitional until branch 15.
Tests: 256 passed in normal order and 256 passed with test files in reverse
  order on the feature branch; 256 passed in both orders after the no-ff
  integration merge. Sixteen branch-specific cases cover role and assignment
  boundaries, schema repair, cyclic/early-stop rejection, dynamic dependency
  DAGs, semantic task/query/source deduplication, every Worker command,
  structured delegation, bounded concurrency, information gain, real gateway
  fallback, cross-worker merge, dynamic replanning, semantic blockers, approval
  and resume, retry, cancellation, run budgets, low gain, restart and end-to-end
  convergence. Only upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Concurrent cross-connection claims, checksum corruption,
  online backup/reopen, identity conflict, immutable plan replay, simulated
  crash after plan artifact but before DAG split, simulated crash after Worker
  intent but before scheduler commit, idempotent merge/decision replay, task
  retry, pending approval and post-approval resume, primary-provider failure
  with governed fallback, runnable-work bounded-round protection, pool
  concurrency fencing, low-gain/max-cycle/budget/cancellation stops, severe
  conflict/high-impact blocking, complete-run restart, and test-order
  independence are covered.
Integration merge commit: 2b54206
Remote push: feature commit b71ba68 pushed successfully; integration push is
  recorded by the immediately following plan commit on 2026-07-26.
Remaining risks: Branch 09 must consume the verified-only evidence view and
  these research convergence outputs in the Writer/Reviewer report loop.
  Branches 10-14 still own evaluation, release gates, advanced Studio, and
  offline evolution. Branch 15 still owns production entry-point migration and
  removal of the transitional draft graph.
```

### Branch 09 execution record

```text
Branch: codex/bg001-09-synthesis-writer-reviewer
Started from integration commit: 0cc2ba7
Scope delivered: Shared-AgentKernel Synthesis Writer and Report Reviewer
  AgentSpecs with no tool/search grants; persisted verified-only Writer packet
  assembled solely through the strict evidence view; unsupported required
  claims converted to explicit gaps; typed section synthesis with independent
  semantic support checks, per-claim citation relationship validation,
  configurable normal/high-impact source independence, deterministic Markdown
  rendering, immediate citation placement, explicit conflict/uncertainty
  presentation and cross-section consistency; complete eight-dimension Reviewer
  rubric whose deterministic artifact audit cannot be overridden by model
  scores; typed and bounded targeted-research, citation-repair, local-rewrite,
  structural-rewrite, accept and reject decisions; concrete Scheduler/Research
  Worker Pool targeted-research dispatch followed by Evidence Engine
  verification and verified-packet rebuild; aggregate report-loop budgets,
  cancellation, approval, revision and research-round bounds; immutable
  checksummed WAL report revision, citation-map, review and terminal-outcome
  journal with restart, backup and corruption checks; structured draft,
  section, report, citation, review and loop-decision artifacts; domain
  Report/Section lifecycle projection; and composition/runtime documentation.
Boundary check: Writer has no search/provider tools and cannot receive candidate
  knowledge as factual claims. Reviewer changes only Report/Section lifecycle
  and quality projections, never Fact/Claim/Evidence/Citation/Conflict
  verification conclusions. Research repairs remain candidate-only until the
  Evidence Engine verifies them. No production entry-point migration,
  evaluation or release gate, Skill/Prompt optimization, online
  self-modification, compatibility layer, or RL work was added.
Tests: 271 passed in normal order and 271 passed with test files in reverse
  order on the feature branch; 271 passed in both orders after the no-ff
  integration merge. Fifteen branch-specific cases cover role boundaries,
  complete accepted synthesis, verified/candidate isolation, explicit gaps,
  conflict and uncertainty presentation, high-impact multi-source enforcement,
  unverified/undercited rejection, citation/local/structural rewrites,
  deterministic false-accept override, explicit reject, revision exhaustion,
  cancellation, bounded model schema repair, reporting-store durability and
  real scheduler targeted research. The combined Evidence/Supervisor/Reporting
  suite passed 46 cases. Only upstream websockets/uvicorn deprecation warnings
  remain.
Failure/recovery tests: Invalid and repeatedly repaired structured output,
  unknown/unverified claim and citation references, insufficient independent
  sources, missing gaps/conflicts, inconsistent cross-section claim rendering,
  false model acceptance, immutable revision conflicts, concurrent idempotent
  writes, restart reads, online backup/reopen, checksum corruption, terminal
  outcome replay, cancellation before model execution, aggregate revision
  bounds, active scheduler research-task materialization, and terminal
  scheduler approval fencing are covered.
Integration merge commit: 5838f81
Remote push: feature commit bb6b6b4 pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Branches 10-11 must turn persisted reports, citations, reviews,
  outcomes and traces into deterministic/semantic evaluation and release
  gates. Branches 12-13 still own report/evidence visualization, replay and
  comparison UI. Branch 14 owns offline evolution only. Branch 15 must compose
  scheduler lifecycle with report-time targeted research, switch production
  entry points to this runtime, exercise long-report stress, and remove the
  transitional draft Writer/graph paths.
```

### Branch 10 execution record

```text
Branch: codex/bg001-10-evaluation-lab-core
Started from integration commit: bc3358d
Scope delivered: Checksummed WAL Evaluation Lab registry and immutable result
  journal with online backup, integrity verification and concurrent connection
  safety; sealed versioned train/dev/selection/test/hidden-test Dataset Bundles
  with content-hash leakage prevention, lineage, deterministic manifests,
  purpose/actor/sample authorization, immutable access audits and crash/retry
  reconstruction; durable Evidence/Report/Event projection into complete
  evaluation snapshots; fixed-version non-LLM evaluators for URL validity and
  uniqueness, citation relationship and quote grounding, citation completeness,
  schema and section coverage, source type/authority/freshness/publisher/domain
  diversity/primary share, token/cost/latency, failure/recovery/idempotency/
  protocol compliance, evidence per tool call, redundant search, trace recovery,
  convergence turns, invalid tool calls and budget violations; network-forbidden
  repeated Frozen Replay with canonical expected-output/event and repeat-metric
  determinism checks; authorized repeated Live Web runs with per-metric mean and
  population variance plus source content, identity, addition and disappearance
  reporting; immutable Experiment definitions/runs/comparisons with subject,
  component, environment, dependency/configuration fingerprint, input and result
  artifact provenance and aligned legacy/fixed-workflow/new-runtime conditions;
  and complete metric/access/recovery documentation.
Boundary check: No LLM or semantic judge, judge calibration, release gate,
  Version Registry promotion/rejection/rollback, Skill/Prompt/Policy patches,
  candidate generation, online self-modification, test/hidden-test tuning, UI,
  production entry-point migration, compatibility layer, or RL work was added.
Tests: 282 passed in normal order and 282 passed with test files in reverse
  order on the feature branch and after the final no-ff integration merge.
  Eleven branch-specific cases and 79 combined contract,
  artifact, event, evidence, reporting and evaluation cases passed. Only
  upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Cross-split relabel leakage, immutable version reuse,
  every split-purpose denial, wrong Live Web subject/sample authorization,
  same-request concurrent access, cross-connection concurrent registration,
  restart restoration, interrupted manifest reconstruction, missing artifacts,
  fixture fingerprint mismatch, attempted Frozen Replay network access,
  reproducible and deliberately divergent replay outputs, Live source mutation
  and set drift, metric variance, misaligned/failed baseline protection,
  immutable record conflict, online backup/reopen, direct checksum corruption,
  and end-to-end verified report projection are covered.
Integration merge commit: b5834e9
Remote push: feature commits ff1d2d3 and 5bb3c47 pushed successfully;
  integration push is recorded by this plan commit on 2026-07-26.
Remaining risks: Branch 11 must add semantic metrics, blind calibrated
  multi-judge evaluation, release policies and immutable Version Registry
  promotion/rollback. Branches 12-13 own Evaluation/Experiment Studio views and
  replay/A-B/badcase interaction. Branch 14 owns offline patch generation and
  selection. Branch 15 owns production entry-point migration and stress
  hardening; this branch intentionally does not make release decisions.
```

### Branch 11 execution record

```text
Branch: codex/bg001-11-evaluation-semantic-gates
Started from integration commit: c3d20bd
Scope delivered: Authorized semantic Evaluation inputs/results with thresholded
  claim support, supported/contradicted/unsupported/uncited-fact, retrieval
  precision/recall/authority/freshness/domain-and-type diversity, and report
  completeness/depth/instruction/organization/readability/safety metrics joined
  to deterministic URL/schema/immediate-citation-position/cost/source-time
  metrics; concrete fixed-ModelAdapter Judge integration with strict structured
  output and bounded repair; odd multi-Judge panels with per-Judge randomized
  candidate ordering, panel-scoped blind labels, fixed Judge/Rubric versions,
  no-search/no-rewrite requests, median score and majority voting, immutable
  ballots and per-dimension range/variance/mixed-vote disagreement records;
  multi-human aggregation plus tie-aware Pearson/Spearman, MAE, pass-agreement
  and accepted/rejected calibration records; complete required-improvement,
  non-regression, absolute/relative-cost, population-variance, safety, protocol,
  semantic-pass and calibration release policy; exact selection versus final
  test/hidden-test access enforcement; immutable gate decisions and separately
  recoverable application records; and a checksummed WAL event-sourced Version
  Registry for AgentSpec, Skill, Prompt, Tool Policy, Stop Policy and Rubric
  manifests with candidate/promoted/rejected/superseded/rolled-back states,
  atomic active-version replacement and full rollback, projection rebuild,
  concurrent fencing, online backup and corruption audit.
Boundary check: Gates select/promote/reject/keep/rollback but create or edit no
  component content. Judges receive no system/baseline identity and cannot
  search, call tools, rewrite reports or mutate evidence. No patch generation,
  automatic Skill optimization, badcase optimizer, online self-modification,
  test/hidden-test tuning, Studio UI, production entry-point migration,
  compatibility layer, or RL work was added.
Tests: 293 passed in normal order and 293 passed with test files in reverse
  order on the feature branch and after the no-ff integration merge. Eleven
  branch-specific cases (including six parametrized releasable component
  kinds) and 76 combined contract/artifact/evidence/reporting/evaluation cases
  passed. Only upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Invalid Judge schema and bounded repair, blind identity
  exclusion, distinct randomized orders, score and vote disagreement,
  accepted and adversarially rejected human calibration, audited semantic
  access binding, selection-stage hidden-test leakage, exact final test/hidden
  coverage, cost/variance/safety/protocol regression rejection,
  failed-candidate rejection, decision
  commit followed by application interruption and idempotent recovery,
  promotion/supersession, post-release regression rollback and replay,
  transition-journal projection rebuild without journal rewrite,
  cross-connection concurrent registration, restart, online backup and direct
  checksum corruption are covered.
Integration merge commit: b9ad3fa
Remote push: feature commit dbfb107 pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Branch 12 must expose task/evidence/state-diff/error/budget
  projections in read-only Studio V2. Branch 13 owns replay, A/B and badcase UI
  over these evaluations and versions. Branch 14 owns offline structured patch
  generation and strict selection using these gates. Branch 15 owns production
  entry-point migration, stress/security hardening and complete composition.
```

### Branch 12 execution record

```text
Branch: codex/bg001-12-studio-v2
Started from integration commit: 4dd09f4
Scope delivered: Typed read-only Studio V2 service over public EventStore,
  scheduler, knowledge, artifact and Version Registry APIs; truly paginated
  scheduler task/event queries; incrementally queryable Task DAG with
  split/merge/prune-as-skip/fail/retry/dependency history and cross-page
  frontiers; paginated Source/Snapshot/Passage/Evidence/Fact/Claim/Citation/
  Conflict/Section/Report graph with complete typed edges, conflict navigation
  and governed source-snapshot content; structured evidence-domain before/after
  status events; deterministic event replay for task, run, budget, evidence and
  section state diffs independent of current projections; separate runtime and
  scheduler error/retry chains; run-pinned and promoted-registry component
  versions; per-run/actor/task model token, cost, latency, error/retry and
  task-budget health views; resolvable event/artifact provenance invariants;
  read-only FastAPI endpoints; and an actual desktop/mobile Studio UI with six
  inspectable panels and provenance details.
Boundary check: StudioV2Service executes no SQL and reads no LangGraph state,
  private connection or legacy state dictionary. The UI and API expose only GET
  operations. No replay, failed-span restart, fork, A/B comparison, component
  diff, badcase creation, release action, optimizer, online modification,
  compatibility layer or RL work was added. Arbitrary entity metadata is not
  projected and immutable Studio contracts reject hidden-reasoning keys.
Tests: 297 passed in normal order and 297 passed with test files in reverse
  order on the feature branch and again after the no-ff integration merge.
  Fifty-one combined Studio, scheduler, evidence and Writer/Reviewer cases
  passed in both orders. Four dense Branch-12 scenarios cover the complete view
  surface. JavaScript syntax validation, Python compilation, desktop browser
  interaction, console-log audit, and 390px responsive visual inspection also
  passed. Only upstream websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Scheduler projection deletion/rebuild produces exactly
  the same historical task/budget diffs; evidence diffs rebuild only from
  structured runtime events; 1,205 task nodes paginate as 500/500/205 without
  duplication; cursors are view-bound; cross-page graph edges expose frontiers;
  every graph node/edge has a resolvable event/artifact; runtime and scheduler
  fail/retry causation is preserved; source content is restricted by artifact
  kind/status; arbitrary internal metadata is excluded; invalid cursors,
  filters, missing resources, content permissions and unsupported HTTP
  mutations are rejected; and existing Studio restart, projection corruption,
  outbox recovery and full scheduler/evidence recovery suites remain green.
Integration merge commit: e363823
Remote push: feature commit 5894c09 pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Branch 13 must build immutable replay/fork, saved-result versus
  live replay, failed-span restart/reapproval, A/B comparisons, component
  diffs and provenance-complete badcase creation on these read models. Branch
  14 owns offline structured candidate optimization and selection. Branch 15
  still owns production entry-point migration, removal of transitional draft
  paths, and stress/security hardening.
```

### Branch 13 execution record

```text
Branch: codex/bg001-13-studio-v3-v4
Started from integration commit: 8315c91
Scope delivered: Immutable ReplayCapsule capture and eligibility over exact
  durable span subtrees; explicit saved-tool-result and live-environment modes;
  real AgentKernel execution into distinct immutable target runs; sealed model,
  tool and verification cassettes with new run-local ToolResult aliases and
  exact network-free enforcement; immutable component selection with Version
  Registry validation; failed-span and root-run-span restart; request-scoped
  preflight and dynamically discovered side-effect reapproval; labeled live
  bindings with exact or explicitly incomplete network accounting; checksummed
  WAL replay request/approval/attempt journal, rebuildable projection, atomic
  claim, cursor pagination, restart recovery into a new run, online backup and
  corruption audit; same-DatasetSample A/B snapshots for runs/spans, Task DAG,
  evidence graph, components, metrics, cost, latency and convergence with event
  and read-model fingerprints; immutable Prompt/Skill/Tool/Stop/Verification
  Policy line diffs; provenance-complete badcases with evaluation-ID-to-artifact
  validation; typed FastAPI preparation/detail/approval/execution/comparison/
  diff/badcase surfaces; and a responsive Studio V4 control UI.
Boundary check: Source events and artifacts are never rewritten; every replay
  attempt receives a new run and a crash or approval stop is retained rather
  than resumed in place. Saved side effects consume only sealed observations
  and are never executed. Live execution requires injected governed bindings
  and an explicit label. A/B artifacts set publishes_versions=false; badcases
  set triggers_change=false and optimizer_invoked=false. No candidate patch,
  optimizer, release transition, online self-modification, production entry
  point migration, compatibility layer, RL work, or hidden chain-of-thought
  exposure was added.
Tests: 302 passed in normal order and 302 passed with test files in reverse
  order on the feature branch, then both full-suite orders passed again after
  the no-ff integration merge. Eighty-one combined Studio, evaluation, event,
  artifact, scheduler and Tool Gateway cases passed in both orders on the
  feature and integration branches. Five dense Branch-13 scenarios cover
  saved/live/dynamic-approval execution, durability, A/B/diff/badcase
  boundaries and HTTP/UI surfaces. Python compilation, desktop browser
  interaction, root-span error feedback, console-log audit and 375px
  responsive visual/layout inspection passed. Only upstream websockets/uvicorn
  deprecation warnings remain.
Failure/recovery tests: Missing/stale replay capsules, nonterminal and root run
  spans, exact subtree fingerprinting, source-run immutability, failed-span
  gating, sealed model/version mismatch, missing preflight approval, newly
  generated side-effect reapproval, approval-stop run retention, distinct
  follow-up run, live-label and saved-network checks, concurrent atomic claims,
  worker-restart abandonment/requeue, projection deletion/rebuild, projection
  checksum corruption, append-only trigger enforcement, cursor pagination,
  backup/reopen, aligned and deliberately misaligned dataset samples,
  evaluation-ID mismatch, unsupported component diff and immutable HTTP
  mutation rejection are covered.
Integration merge commit: b537426
Remote push: feature commit 5083691 pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Branch 14 must consume scored traces, evaluations and these
  inert badcases to generate bounded offline candidate patches and pass strict
  selection/release gates. Branch 15 still owns production entry-point
  migration, removal of transitional draft paths, and complete stress/security
  hardening. Studio intentionally reports legacy runs without ReplayCapsule
  material as inspectable but ineligible instead of fabricating replay inputs.
```

### Branch 14 execution record

```text
Branch: codex/bg001-14-offline-evolution
Started from integration commit: 13d6577
Scope delivered: Framework-independent contracts for all eight optimization
  targets and their allowed Skill/Prompt/Tool Policy/Stop Policy/Rubric
  boundaries; immutable candidate-pool entries and separate human reviews for
  scored success/failure traces, Studio badcases and evaluations; train/dev
  split discovery and relabel/leakage rejection; sealed current-promoted-version
  optimizer inputs with all required source categories; explicitly separated
  runtime, cross-task-experience and formal Skill Registry memory layers; a
  concrete deterministic network-free trace-signal optimizer; exact line-level
  add/delete/replace patches with operation fingerprints, stale/ambiguous/
  overlap rejection and per-round operation/line/character/text-learning-rate
  budgets; immutable patch/content/candidate artifacts and parent-linked Version
  Registry candidates; cross-campaign rejected-edit memory that excludes prior
  operations from later generation; strict positive selection improvement over
  the existing semantic/deterministic/cost/variance/safety/protocol gates;
  explicit human approval/rejection; exact test plus hidden-test final
  promotion; static versioned best_skill history; post-release keep/rollback;
  and a checksummed WAL journal with atomic claims, restart recovery, cursor
  pagination, rebuildable projections, immutable triggers, backup and audit.
Boundary check: Pool submission and review set optimizer/publication triggers to
  false, and no production trace or badcase can create a campaign or version
  without explicit reviewed input, generation and release calls. Candidate
  generation uses no runtime memory or online inference. Human decisions may
  reject a candidate but cannot promote, supersede or roll back a version;
  promotion remains owned by the existing final release gate. No production
  entry-point migration, online code/weight/Skill mutation, model training,
  compatibility layer, hidden chain-of-thought, RL framework or RL training was
  added.
Tests: 318 passed in normal order and 318 passed with test files in reverse
  order on the feature branch, then both full-suite orders passed again after
  the no-ff integration merge. Sixteen Branch-14 scenarios cover all eight
  target boundaries and dense end-to-end/durability paths; 56 combined
  Evolution, semantic gate, advanced Studio, Artifact and Contract cases passed
  in both orders on the integration branch. Python compilation, pyflakes on
  non-export modules and git diff validation passed. Only upstream
  websockets/uvicorn deprecation warnings remain.
Failure/recovery tests: Production trace non-self-promotion, missing source
  categories, unreviewed inputs, observed split relabel and hidden-test leakage,
  non-strict selection, cost/safety final rejection, human rejection, round
  exhaustion, stale base content/version, stale/overlapping edits, every text
  budget, rejected-operation regeneration, malicious online/network generator
  reporting, cross-connection atomic claims, worker restart/abandon/new attempt,
  candidate-pool review artifact/store interruption, campaign input-seal
  interruption, human-decision/Version-Registry/store interruption, release
  transition replay, projection deletion/rebuild, projection corruption,
  append-only trigger enforcement, cursor pagination, online backup/reopen,
  promotion, post-release regression rollback and best_skill restoration are
  covered.
Integration merge commit: 80dbde8
Remote push: feature commit afa695e pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Branch 15 must compose this offline loop with the fully migrated
  runtime entry points without adding online inference, remove all transitional
  draft execution paths, and run the specified database-lock, artifact-size,
  long-trace/report, redaction/security, replay-side-effect and dataset-access
  stress suites. The built-in optimizer is deliberately deterministic and
  network-free; a future offline model-backed generator must satisfy the same
  OfflineGeneratorResult and isolation contract before it can be enabled.
```

### Branch 15 execution record

```text
Branch: codex/bg001-15-integration-hardening
Started from integration commit: 7fd26d2
Scope delivered: A production ApplicationRuntime composition root over the
  native event-sourced Scheduler, five AgentKernel roles, governed research
  tools, candidate ingestion, independent Evidence verification, bounded
  Supervisor/Worker convergence, Writer/Reviewer reporting and Studio
  projections; a checksummed WAL application run catalog and append-only
  transition journal; balanced root/evidence event lifecycle across start,
  approval resume, failure and queued/active cancellation; OpenAI-compatible
  structured model adapter plus governed Tavily/Exa/SmartScraper Worker
  registry; real run_research CLI and projection-only Console/Studio service
  with an application factory instead of module-global runtime state; restart
  recovery, cancellation fencing, approval resolution and report-time scheduler
  ownership; native candidate-to-Citation ingestion; immutable committed draft
  baseline reads; removal of fat GraphState, module-global observers/managers,
  fake/compatibility research gateway, legacy session projection, LangGraph
  adapter/dependencies, Qdrant draft, and superseded agents/core/providers/
  schemas/router/state-manager/vector-store execution paths and tests; accepted
  native-scheduler ADR; current/optional/unsupported README capability matrix;
  and production source, provider, database-lock, large artifact/report, long
  trace, redaction and terminal-state hardening gates.
Boundary check: No new logical role, external protocol, domain capability,
  compatibility layer, online self-modification, model/weight training or RL
  work was added. Existing MCP/A2A, replay/fork, evaluation, release and offline
  evolution capabilities were composed and exercised but not expanded.
Tests: 253 passed in normal order and 253 passed with test files in reverse
  order on the feature branch; both 253-test orders passed again after the
  no-ff integration merge. Seven production-application scenarios and four
  integration-hardening stress scenarios passed. Python compilation, pyflakes
  on non-export production modules, forbidden-import/dependency scan and
  git-diff validation passed. Only upstream websockets/uvicorn deprecation
  warnings remain.
Failure/recovery tests: Full end-to-end verified report and citation creation,
  concurrent run isolation, durable in-progress restart, active and queued
  cancellation with late-completion fencing, model failure and secret
  redaction, approval pause/resume with distinct balanced evidence spans,
  provider credential/protocol failure and governed fallback, research/report
  budgets, conflict and high-impact evidence blocking, bounded report repair
  and targeted research, scheduler leases/projection rebuild/backup/corruption,
  replay/fork side-effect reapproval and immutable source runs, dataset split
  permissions, semantic release keep/reject/rollback, offline evolution
  release/rollback, 24-writer WAL contention, >2 MB report artifacts, 1,507
  event trace pagination/redaction, and production-source boundary checks are
  covered.
Integration merge commit: c241efc
Remote push: feature commit 3a2c291 pushed successfully; integration push is
  recorded by the immediately following plan commit on 2026-07-26.
Remaining risks: Live output quality and availability depend on configured
  model/search providers and are measured by the existing Frozen/Live
  evaluation and release gates rather than claimed. RL remains explicitly out
  of scope. The two test warnings are upstream websocket API deprecations and
  do not affect current MCP protocol behavior.
```

### Branch 16 execution record

```text
Branch: codex/bg001-16-console-runtime-alignment
Started from integration commit: fd6130a
Scope delivered: Versioned typed ConsoleRunListItem@2,
  ConsoleWorkspace@2 and ReportWorkspace@2 read contracts assembled from
  ApplicationRuntime, native Scheduler, Knowledge, Reporting, Artifact and
  Studio public projections; complete identity, stage, five-role progress,
  action availability, task DAG/inspector, budgets, leases, attempts,
  approvals, errors and artifact visibility; section coverage, blockers,
  verified Writer packets, citations, governed sources, gaps and conflicts;
  report outline, revisions, eight-dimension review, findings, bounded repairs
  and terminal outcome; exact-run Studio/timeline/export navigation; provider
  readiness and response security headers; a complete same-origin no-build
  landing, operational Console and safe-Markdown report workspace with
  accessible dialogs, focus, reduced-motion, print and desktop/tablet/mobile
  layouts; visibility-aware polling, last-good-view recovery, filtering,
  pagination, approve/cancel actions and explicit failed/cancelled/not-found
  states. Cancellation reasons now remain available before scheduler creation,
  approval actor IDs are validated at both browser and HTTP boundaries, and
  the typed non-empty run catalog serializes through FastAPI correctly.
Boundary check: No Supervisor planning, Worker search/extraction, Evidence
  verification conclusion, Writer prose, Reviewer scoring, Scheduler
  transition, Studio replay/comparison, Evaluation, Evolution, provider
  policy, release or RL semantics changed. The Console is projection-only,
  contains no draft-response compatibility facade, duplicates no advanced
  Studio mutation workflow, and exposes no hidden reasoning.
Tests: 259 passed in normal order and 259 passed with test files in reverse
  order on the feature branch, then both 259-test orders passed again after
  the no-ff integration merge; 26 focused Console/Application/Studio cases
  passed; JavaScript syntax validation and Python compilation passed. Safe
  Markdown, external/internal URL policy, schema normalization, polling,
  filtering, namespaced actor identity, accessibility, reduced-motion,
  responsive breakpoints and print contracts are covered.
Failure/recovery tests: Browser checks covered provider-not-configured landing,
  a retained/redacted failed live run, completed deterministic run, task,
  evidence, report and filtered error timeline views, approval pause/resume
  through report revision r1, cancellation from approval with retained reason,
  and return to a non-empty durable run catalog. API tests cover empty and
  non-empty catalogs, missing runs, approval/cancel action availability,
  invalid approval actor and blank cancel input rejection, queued
  cancellation, failure redaction, terminal action fencing, security headers
  and exact Console/Report/Studio links. Polling preserves the last good
  projection and exposes bounded retry after request failure.
Integration merge commit: 1d9c3f4
Remote push: feature commit a88229a pushed successfully; integration push is
  recorded by this plan commit on 2026-07-26.
Remaining risks: Live research still requires configured model and search
  provider credentials; the Console reports that boundary instead of
  fabricating results. The two suite warnings are upstream websocket API
  deprecations and do not affect current MCP behavior.
```

### Branch 17 execution record

```text
Branch: codex/bg001-17-live-runtime-contract-recovery
Started from integration commit: 8548453
Scope delivered: Strict AgentSpec/tool-derived Worker command schemas and
  canonical identity normalization; bounded malformed-JSON and structured
  output repair; provider-valid typed observations; governed search/read/
  extract phase protocol; bounded Tavily and scraper payloads; deterministic
  source authority and read-replay preservation; exact persisted-passage quote
  grounding; ungrounded entity rejection; rejected-candidate evidence
  quarantine with valid-support-path verification; realistic per-task planning
  reserves and dynamic plan admission; durable root lease/attempt recovery;
  explicit reportable complete-with-gaps convergence plus honest zero-evidence
  failure; exact-gap and empty-section Writer normalization without relaxing
  factual traceability; causal failure prioritization; balanced evidence/report
  spans; and authoritative Console five-role/stage projection.
Boundary check: No new role, provider, external protocol, evidence state,
  reviewer rubric, scheduler transition, evaluation/evolution release action,
  compatibility facade, deployment, main/release merge, online modification or
  RL work was added. Search relevance never becomes source authority; rejected
  evidence never becomes factual; Writer verified-only, exact-gap and citation
  invariants remain strict. A new convergence result names the already-required
  bounded, cited report-with-gaps outcome without weakening no-evidence failure.
Tests: 289 tests passed in normal order and 289 passed with test files in
  reverse order; JavaScript syntax, Python compilation and git diff checks
  passed. Focused coverage includes command identity/JSON repair, scheduler
  retry and root completion, phase transitions, provider payload bounds,
  source authority/read replay, exact quote selection, ungrounded rejection,
  evidence quarantine, bounded reportable gaps, Writer empty-section handling,
  causal error ordering and Console projections.
Failure/recovery tests: Live traces covered schema exhaustion, malformed model
  JSON, provider/tool failures, task budget exhaustion, partial Worker results,
  source-discovery supersession, root-lease completion, research-budget stops,
  repair-round ownership, rejected noisy evidence, Writer domain validation and
  downstream role attribution. SQLite restart/replay, cancellation, approval,
  concurrency, protocol, reporting and projection recovery remain covered by
  both complete suite orders.
Integration merge commit: pending
Remote push: pending
Remaining risks: The final post-fix configured live retry received permanent
  DeepSeek HTTP 402 on its first Supervisor call, so a fresh terminal online
  report requires restored provider quota. Earlier same-session live runs
  verified real Tavily/Jina/DeepSeek search, read, grounded ingestion and
  independent verification through 100% section/citation coverage and exposed
  the repaired Writer/convergence defects. The two suite warnings are upstream
  websocket API deprecations and do not affect current MCP behavior.
```
