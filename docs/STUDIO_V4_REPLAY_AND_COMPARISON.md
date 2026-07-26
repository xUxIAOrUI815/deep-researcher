# Studio V3/V4 replay, comparison, and badcase contract

Studio V4 adds controlled execution and evaluation actions to the read models
introduced by Studio V2. It does not make an event, artifact, component
version, comparison, or badcase mutable.

## Replay eligibility and capsule boundary

A span is replay-eligible only when all of the following are true:

1. The run and span exist in the durable event store.
2. The span has a durable terminal event.
3. The artifact store contains an immutable `ReplayCapsule@1` for the span.
4. The capsule's ordered event IDs and SHA-256 event fingerprint still match
   the durable source span.
5. The capsule contains the original task, AgentSpec, component version set,
   model exchanges, tool observations, and verification exchanges needed for a
   complete offline execution.

The capsule is a runtime capture contract, not a Studio reconstruction of
private LangGraph state. `ReplayCapsuleRepository.create()` rejects missing or
changed source events and missing tool-result artifacts. Existing runs without
this material remain inspectable but are explicitly reported as ineligible.

## Saved-tool-result replay

`saved_tool_results` executes the task through the real `AgentKernel` with:

- sealed model responses;
- sealed tool observations copied into new run-local `ToolResult` alias
  artifacts;
- sealed verification feedback;
- the ordinary command schema, command policy, budgets, retries, stop policy,
  and durable `EventRecorderKernelSink`.

The mode is labeled `sealed-network-free`. It reports zero network calls and
rejects model, Prompt, or Skill changes because a sealed model response is not
evidence for a different model input. Runtime controls and governed policies
may be selected when their version identity is valid. To evaluate a changed
model, Prompt, or Skill, use a labeled live environment.

Saved side-effect results are never executed again. They still require a fresh,
request-scoped approval before the sealed observation can be consumed.

## Live-environment replay

`live_environment` requires an injected `LiveReplayBindings` implementation
containing the selected model adapter, governed action executor, independent
verifier, and an explicit environment label. There is no silent fallback from
live execution to sealed results.

Bindings may expose an exact network-call counter. Without one, Studio reports
the model/action provider-invocation upper bound and marks network accounting
incomplete; it never presents that fallback count as exact telemetry.

The selected Prompt, Skill, AgentSpec, and Policy version references must match
their immutable Version Registry manifests. The live runtime should bind its
action executor to the Tool Gateway so tool permission, schema, safety,
idempotency, rate-limit, retry, circuit-breaker, and provider approval rules
remain active.

The replay policy consumes only approvals written to the new replay request.
Approval events from the source run are never inherited. If a changed component
generates a new governed command, the attempt closes as
`waiting_approval`. After a fresh approval, execution starts a second immutable
run; it does not resume or rewrite the first attempt.

## Persistence and recovery

Replay requests use the checksummed, append-only `studio_replay_journal`.
`studio_replay_projection` is rebuildable from that journal. Requests,
comparisons, and badcases have immutable-table triggers. SQLite uses WAL,
`synchronous=FULL`, foreign keys, and an immediate transaction for claim and
transition operations.

If a process restarts with an active attempt:

1. The old target run and its events are retained.
2. The journal appends `attempt_abandoned`.
3. The request is queued again.
4. The next claim receives a distinct attempt ID and target run ID.

Concurrent claims serialize transactionally; only one can move a queued request
to running. The store supports integrity verification, projection rebuild,
cursor pagination, backup, and restart.

## A/B comparison

Run or span comparison is allowed only when both runs reference the exact same
immutable `DATASET_SAMPLE` artifact. Dataset IDs supplied by a caller are not
accepted as proof of alignment.

The immutable comparison includes:

- terminal run status;
- span graph additions, removals, and changes;
- complete paginated Task DAG and evidence graph differences;
- component version differences;
- token, cost, latency, retry, error, model-call, tool-call, and search-call
  deltas;
- convergence decision and retry summaries.

Comparison creates a `StudioABComparison@1` artifact. Its contract sets
`publishes_versions=false` and `optimizer_invoked=false`. It cannot promote,
reject, roll back, or edit a component.

## Prompt, Skill, and Policy diff

Diff reads the two registered immutable component artifacts and requires the
same component kind and name. It supports Prompt, Skill, Tool Policy, Stop
Policy, and Verification Policy. The response includes structured line opcodes
and a unified text diff. It does not create or publish a candidate version.

## Badcase creation

A badcase can be created only for a terminal durable span and requires:

- original run, span, and ordered event IDs;
- at least one original input artifact;
- the exact component version set and all component version IDs;
- evaluation IDs and supported evaluation artifacts;
- the original dataset-sample artifact;
- a non-empty human note and creator identity.

The resulting `StudioBadcase@1` artifact and Studio record are immutable.
`triggers_change=false` and `optimizer_invoked=false` are enforced. Branch 14
may consume badcases as offline optimizer inputs, but Branch 13 never starts
optimization or changes production versions.

## HTTP surface

The console exposes:

- replay eligibility, preparation, listing, detail, approval, and execution;
- aligned A/B comparison;
- immutable component diff;
- one-click badcase creation.

No delete, event-edit, comparison-publish, badcase-apply, optimizer, or version
transition route exists. The Studio V4 page exposes the same operations with
explicit modes, labels, component selection, approval fingerprints, and
provenance fields.
