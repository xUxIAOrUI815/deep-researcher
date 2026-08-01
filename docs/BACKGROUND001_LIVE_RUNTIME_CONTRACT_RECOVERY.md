# Background001 Live Runtime Contract Recovery

Branch: `codex/bg001-17-live-runtime-contract-recovery`

This is the durable implementation contract for the live failure reproduced on
2026-08-01. It extends `BACKGROUND001_EXECUTION_PLAN.md` without changing the
Background001 architecture or weakening its research and evidence boundaries.

## Reproduced failure

Two configured live runs reached the model and search-provider environment but
failed before any governed tool execution. DeepSeek returned Worker commands
without `kind`; a later response supplied `kind=search` without `name`, which
was defaulted to the non-canonical name `search` instead of the registered
`research.search`. The Worker exhausted schema repair with zero tool calls.

The scheduler then left failed research work unresolved, convergence advanced
to its maximum cycle, the root research task was completed with zero verified
evidence, and reporting began. The Writer invented a gap claim identity that
did not equal the persisted section gap set. Strict revision persistence
correctly rejected it, but the failure was classified without an allowed repair
path. The Console surfaced only the downstream Writer failure and marked
research and synthesis roles as completed.

## Required invariants

1. Model-facing command schemas and runtime validation are the same contract.
2. A Worker can execute only canonical tool names granted by its AgentSpec and
   task tool contracts.
3. Missing identity fields may be normalized only when the mapping is unique;
   conflicting and unknown identities are repairable validation errors.
4. The earliest causal failure remains visible through Worker, task, run, trace
   and Console projections.
5. Retry, replan, low-gain and maximum-cycle outcomes are bounded and cannot
   silently become successful research.
6. Normal reporting requires resolved mandatory scheduler work and a verified
   evidence packet. An incomplete report, when allowed, uses exact persisted
   gap identities and explicitly records its incomplete outcome.
7. Writer semantic/domain validation is eligible for bounded model repair;
   provider credentials, policy denials and permanent store corruption remain
   non-repairable.
8. Console role and stage status comes from authoritative lifecycle facts, not
   the presence of later events or guessed ordering.

## Implementation areas

### Worker command contract

- Derive strict JSON Schema from `AgentSpec.allowed_commands` and the task's
  governed tool contracts.
- Require an object command list and structured arguments. Restrict `kind` and
  canonical `name` to allowlisted values and include per-command required
  arguments for research tools.
- Normalize canonical name/kind pairs before Pydantic parsing and policy
  evaluation. Record normalization without hiding the original response.
- Feed identical schemas and precise validation errors to the bounded repair
  request.

### Worker recovery and convergence

- Classify schema exhaustion, provider/tool failures, partial results, budget
  stops and permanent errors separately.
- Retry only retryable outcomes within the task attempt limit, then request
  Supervisor replanning where meaningful. Avoid scheduler cycles with neither
  runnable work nor a terminal decision.
- Block successful research completion while required tasks are failed,
  pending, waiting approval or unverified, or required coverage remains open.
- Persist an explicit incomplete/failure result for maximum-cycle or
  no-evidence termination.

### Reporting and gaps

- Carry exact section gap identifiers from the verified Writer packet into the
  model request and revision validation.
- Provide deterministic incomplete-section material when evidence is empty and
  incomplete reporting is explicitly selected by the runtime.
- Treat malformed or invariant-breaking Writer proposals as bounded repairable
  model output; never relax the persisted exact-gap check.

### Error and Console projection

- Preserve and order causal errors from Worker validation through scheduler,
  convergence and reporting.
- Derive Research Worker, Evidence Verifier, Synthesis Writer and Report
  Reviewer status from their real spans and terminal outcomes.
- Do not render a failed or cancelled research stage as completed because the
  root task or a downstream stage emitted a later event.

## Verification matrix

- Kernel: missing `kind`, missing `name`, canonical inference, conflicting
  identity, unknown tool, repair success and repair exhaustion.
- Worker/runtime: scheduler retry, replan, exhausted attempts, no-progress
  convergence, maximum cycles, zero evidence and earliest-error preservation.
- Writer/reporting: exact persisted gaps, invented gap rejection plus repair,
  incomplete packet, revision persistence and terminal outcome.
- Console/API: original reproduced trace shape, Worker-first causal error,
  Writer failure, role status and research/report stage accuracy.
- Regression: focused tests, complete suite in normal and reverse file order,
  restart/failure paths, JavaScript syntax, then configured live DeepSeek and
  Tavily research observed through the Console and trace.

## Explicit non-goals

- No minimum implementation, permissive catch-all schema or silent fallback.
- No fabricated sources, claims, citations, tool results or success state.
- No change to verified-only writing, independent verification, five roles,
  Studio ownership, provider selection, evaluation/evolution release policy,
  main/release branches or reinforcement learning.

## Delivered recovery

- Worker model requests now use the strict AgentSpec/tool-derived command
  schema for initial generation and repair. Canonical name/kind inference is
  allowlisted, malformed provider JSON receives bounded structural repair, and
  typed user observations replace provider-invalid chat roles.
- The Worker protocol is phase-aware (`search -> read -> extract`): search
  snippets discover sources but cannot become quote-ready evidence; original
  reads are bounded, persisted and preferred when resolving exact quotes.
- Candidate ingestion rejects ungrounded quotes and cannot create facts,
  claims or citations without a persisted passage match. Source authority is
  derived from canonical publisher/domain policy rather than provider search
  relevance, and the strongest source classification survives read replay.
- Evidence verification quarantines rejected extra candidate evidence instead
  of allowing it to poison a claim whose remaining evidence path independently
  satisfies support, authority, citation, freshness and grounding gates. The
  rejected evidence and its issue remain visible and non-factual.
- Supervisor planning enforces bounded plan size plus realistic per-extraction
  model-call/token reserves. Root control tasks retain enough attempts for all
  cycles and terminal commit, and failed/paused roots can be recovered for a
  valid terminal convergence artifact.
- Research convergence reserves replan capacity and has an explicit
  `complete_with_gaps` outcome after the configured semantic-gap repair bound,
  but only when at least one verified claim with a verified citation is
  reportable. Zero-evidence and exhausted-budget runs remain failures.
- Writer proposals remove only uncitable model narrative from packet sections
  that contain no verified claims; they never borrow unrelated claim/citation
  IDs. The renderer writes a deterministic evidence-boundary notice, while
  partially linked or verified-section statements still fail strict validation
  and bounded repair.
- Console projections expose the authoritative five roles, scheduler/evidence/
  report lifecycle, earliest causal error and complete downstream chain.
  Evidence verification is not marked failed merely because a later Writer or
  report stage fails.

## Live verification evidence (2026-08-01)

- `research_cf7c476183fb43d3`: 32 governed sources; one persisted claim,
  evidence item and citation; independent verification score `0.9875`; required
  section coverage and citation coverage both `100%`. The run reached reporting
  and reproduced the empty-ID boundary-section Writer defect, which is covered
  by the new strict normalization regression.
- `research_5c0a2ec6d74149d0`: 36 sources, 11 claims and 12 evidence items; four
  claims passed verification before the old global-cycle/repair-round mismatch
  raised `repair round 4 exceeds policy maximum 3`. Repair rounds are now
  clamped at their owning boundary and reportable bounded convergence is typed.
- `research_13396d6f7f1c4c38`: 38 sources and three grounded/cited candidates;
  verification exposed that rejected extra evidence incorrectly blocked valid
  support paths. The quarantine regression now proves valid support can pass
  while the unrelated evidence remains rejected and auditable.
- `research_22aa63a651a148f0`: the final post-fix online retry was stopped on the
  first Supervisor call by provider HTTP 402, before any search or evidence
  mutation. This is retained as the honest external live-verification blocker;
  the runtime did not fabricate a success or retry a permanent billing error.

Local deterministic verification is complete in normal and reverse file order,
including strict Worker recovery, grounded ingestion, evidence quarantine,
bounded gap completion, Writer normalization, report validation and Console
projection. A fresh configured online terminal-report run remains dependent on
restoring model-provider quota.
