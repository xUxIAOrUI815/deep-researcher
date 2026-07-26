# Research Supervisor and Worker Pool

This package implements Background001 branch 08. It is the complete research
coordination layer between the durable scheduler and the independent evidence
engine. It does not write a report.

## Ownership and hard boundaries

The Research Supervisor owns dynamic decomposition and replanning. It consumes
the root objective, prior convergence decisions, section gaps, evidence
blockers, conflicts, task state, and budgets. It emits a validated
`SupervisorPlan`; it has no tool grant and never calls a provider.

Research Workers own bounded execution of:

- `search`
- `read`
- `extract`
- structured `delegate`
- `compare`
- `verify_source`

Provider execution always crosses the injected `CommandExecutionBoundary`. A
`ProtocolToolGateway` is directly compatible with this boundary, so registry
versioning, permissions, approval, rate limiting, idempotency, retries,
fallbacks, circuit breaking, safety scans, cost accounting, timeout, and
cancellation remain centralized in the gateway.

Neither role may synthesize or review final report prose. Worker delegation is
another `TaskEnvelope`, not free-form agent chat. Report writing/review belongs
to branch 09; evaluation and offline optimization belong to branches 10-14.

## Runtime flow

```text
root TaskEnvelope
    |
    v
ResearchSupervisorRunner --validated plan--> durable scheduler DAG
    |                                             |
    | pauses root                                 v
    |                                bounded ResearchWorkerPool
    |                                             |
    |                               governed commands + artifacts
    |                                             |
    |                             ResearchWorkerResult journal
    |                                             |
    +<-- convergence decision <-- evidence verify + cross-worker merge
             |
             +-- replan: edit/resume the root with the exact gap snapshot
             +-- approval: preserve WAITING_APPROVAL without bypass
             +-- complete: complete root and scheduler run
             +-- budget/cancel: cancel the run
             `-- low gain/max cycles: cancel residual work and stop boundedly
```

Both roles run through the shared `AgentKernel` and immutable `AgentSpec`
contracts. The scheduler worker identity is adapted to the logical AgentSpec
identity only at the kernel boundary; lease ownership remains fenced by the
scheduler.

## Dynamic plan contract

`SupervisorPlan` is structured, schema-validated, and repaired with a bounded
number of model calls. Every proposed task carries:

- task kind, title, goal, and constraints;
- input artifact references and expected output schema;
- token/cost/time/model/tool/search/retry/error budget;
- priority, deadline, maximum attempts, tags, and optional worker assignment;
- explicit dependency keys.

Validation rejects unknown dependencies, dependency cycles, writing/review
tasks, empty decomposition, malformed approval actions, and early terminal
actions. Only the authoritative convergence gate may permit termination.

Semantic task fingerprints include the goal, kind, constraints, inputs,
output schema, and recursively fingerprinted prerequisites. Scheduling-only
changes such as priority or a larger retry budget therefore do not duplicate
the same research, while a different prerequisite graph remains distinct.

Plan application first records an immutable `SUPERVISOR_PLAN` artifact and then
atomically splits the durable DAG. Replays validate the recorded plan and repair
a crash between artifact persistence and DAG mutation without rewriting the
artifact.

## Worker execution and information gain

Query keys are whitespace-normalized and case-folded. Source keys use canonical
URLs. Cross-worker claims have durable leases with the states `claimed`,
`completed`, and `failed`; completed work reuses artifact references, in-flight
duplicates do not execute twice, and expired/failed claims may be retried.

Task reservations deduplicate semantic work across Supervisor cycles and Worker
delegation. Novelty is independently registered for query, source, artifact,
and extracted-knowledge keys. The verifier produces an
`InformationGainAssessment` for every successful observation and returns zero
gain for a duplicate.

Every command observation is persisted as a redacted `TOOL_RESULT`. Every
leased task produces an immutable `WORKER_RESULT` artifact and a checksummed
coordination record. Scheduler completion, failure/retry, cancellation, and
approval transitions preserve the task usage. Hitting a hard task budget is
allowed to fail the task before a pause or approval transition.

The Worker attempt record is committed before its scheduler transition.
`ResearchWorkerResultReconciler` repairs a crash in that cross-store window by
replaying the same idempotent complete/fail/retry/approval/cancel mutation while
the lease is valid. If the lease has expired, scheduler recovery fences it and
the next attempt reuses the governed query/source dedup results instead of
silently losing or duplicating provider work.

The Worker Pool claims one lease per configured worker in parallel while the
scheduler enforces the run-wide slot limit, dependencies, priorities,
deadlines, assignments, and task budgets. It clears cancellation tokens even
when a runner raises and fails closed if its bounded claim rounds end while
runnable tasks remain.

`CrossWorkerResultMerger` creates an immutable, replay-idempotent
`RESEARCH_MERGE` artifact containing the complete Worker result set and
deduplicated task, artifact, query, source, and knowledge references.

## Authoritative semantic convergence

`ConvergenceEvaluator` reads only durable scheduler state, verified evidence
state, and the coordination journal. Its decision precedence covers:

1. caller/scheduler cancellation;
2. aggregate run budget exhaustion;
3. pending human approval;
4. runnable/active tasks and the hard cycle ceiling;
5. required section coverage and citation thresholds;
6. unsupported high-impact claims;
7. unresolved high/critical conflicts;
8. failed tasks requiring replanning;
9. consecutive low-information-gain cycles;
10. semantic completion.

Every `ConvergenceDecision` includes the complete captured snapshot and policy,
is persisted once per run/cycle, and has an immutable
`CONVERGENCE_DECISION` artifact. Restart returns the recorded decision instead
of recomputing it with new timestamps. An approval decision can resume after
the scheduler records explicit approval; replay while approval is still
pending cannot bypass the gate.

## Persistence and restart

`SQLiteResearchCoordinationStore` uses WAL transactions, checksums, indexes,
and a versioned schema for:

- deduplication claims and leases;
- semantic task reservations;
- novelty keys;
- Worker results;
- cross-worker merges;
- convergence decisions.

It supports concurrent processes, integrity audit, online backup, restart, and
conflict-safe immutable identities. Artifact bodies remain in the
`ArtifactStore`; task state remains in the `Scheduler`; evidence semantics
remain in `EvidenceRuntime`.

## Composition

`build_research_runtime(...)` requires real injected dependencies:

- a durable `Scheduler`;
- `EvidenceRuntime`;
- Supervisor and Worker `ModelAdapter`s;
- a governed command executor (normally `ProtocolToolGateway`);
- a durable `KernelEventSink`;
- a `ConvergencePolicy`.

It deliberately provides no fake provider, no no-op event sink, and no
production fallback that hides missing dependencies. The returned
`ResearchRuntime.coordinator.run(...)` is restart-aware and stops only through
the bounded convergence actions.

## Verification coverage

`tests/test_bg001_research_supervisor_workers.py` covers role boundaries,
invalid/cyclic plan repair, early-stop rejection, dependency-safe DAG
materialization, crash-safe replay, semantic task deduplication, concurrent
query/source deduplication, all Worker command kinds, structured delegation,
information gain, real governed gateway fallback, bounded global concurrency,
retry, approval/resume, dynamic replanning, cross-worker merge replay,
required-section convergence, low gain, budget exhaustion, cancellation,
restart, corruption detection, backup, and end-to-end run completion.
