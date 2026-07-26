# Background001 Studio V2

Studio V2 is the read-only operational inspection surface defined by
Background001. It does not execute, resume, replay, fork, compare, publish, or
mutate a research run. Those capabilities remain outside this branch.

## Public read boundaries

Studio V2 composes five public stores/services:

- the append-only `EventStore` for runtime events, usage, errors, component
  versions, causation, and artifact references;
- the event-sourced scheduler store for scheduler events and paginated task
  records;
- the knowledge storage/repository for immutable evidence-graph revisions and
  typed relationships;
- the artifact store for immutable metadata and governed source-snapshot
  content;
- the Version Registry for active promoted component versions.

`StudioV2Service` contains no SQL and never reads LangGraph checkpoints,
legacy state dictionaries, or private store connections. The scheduler now
exposes typed task/event page queries so a caller does not need table access.

## Views

### Task DAG

The task graph displays:

- parent/split edges;
- dependency edges;
- merge edges;
- pruned tasks as skipped work;
- failed and retried task history;
- current priority, assignment, attempt, result/error reference, budget limit,
  budget usage, and immutable input/output artifacts.

Task pages use an opaque, view-bound cursor. Edges whose other endpoint is not
in the current page identify that endpoint in `frontier_node_ids`, allowing a
client to request more nodes without receiving a truncated or false graph.

### Evidence Graph

The evidence graph includes `Source`, `SourceSnapshot`, `Passage`, `Evidence`,
`AtomicFact`, `Claim`, `Citation`, `Conflict`, `Section`, and `Report` nodes.
It preserves every typed relationship needed to navigate:

`Report -> Section -> Claim -> Evidence -> Passage -> SourceSnapshot -> Source`

Citations retain their direct claim, evidence, passage, snapshot, and source
paths. Conflicts retain claim/fact and resolution-evidence paths. Snapshot
nodes link to a governed snapshot endpoint and to immutable content artifacts.
Only available snapshot, cleaned-content, and passage artifacts expose bytes.
Other artifacts expose metadata only.

Evidence pages use the knowledge store's stable `(created_at, entity_id)`
ordering behind an opaque, view-bound cursor. Cross-page relationship targets
are returned as frontier node IDs.

### Event-derived state diff

Scheduler state diffs replay scheduler events from sequence one. They compare
the event-carried task records and budget usage, without consulting the current
task projection. Rebuilding the scheduler projection therefore cannot change a
historical diff.

Evidence state diffs replay only `evidence_changed` runtime events.
Evidence-verification events now carry structured `state_changes` with entity
type, entity ID, field, before value, and after value. Existing historical
events retain a deterministic compatibility parser for the previously defined
claim, conflict, and section payloads.

Both streams have independent sequence cursors. The UI never conflates the
scheduler sequence namespace with the runtime-event sequence namespace.

### Error and retry chains

Two independent chains are available:

- runtime errors and `retry_scheduled` events, linked through causation and
  task/span identity;
- scheduler `task_failed`, `task_retried`, and lease-recovery events, linked by
  scheduler task identity.

Only typed, redacted error records and referenced detail artifacts are shown.
No internal reasoning text is projected.

### Components and resource use

The component view contains the immutable versions pinned to the latest run
event and active promoted records from the Version Registry. Every entry links
to its run event, manifest artifact, component artifact, or transition
artifact.

The metrics view provides model tokens/cost, model/tool latency and call
counts, retry/error counts, per-actor and per-task attribution, and every
scheduler task's budget limits, usage, utilization ratios, and exceeded
dimensions. Model billing totals are counted from model terminal events so
verification lifecycle events cannot double-charge the same model usage.

## Provenance invariant

Every graph node and graph edge must have at least one resolvable runtime event,
scheduler event, artifact, or source-snapshot link. Knowledge nodes first use
their causation event and source artifacts, then a task-scoped runtime event as
a compatibility fallback. Construction fails if no provenance can be
resolved; Studio does not invent untraceable nodes.

The generic node data allowlist excludes arbitrary metadata and rejects hidden
reasoning keys through the immutable contract base. The UI renders only typed
outcomes, status, usage, relationships, errors, and provenance.

## HTTP surface

The read-only API is rooted at `/api/studio/v2`:

- `GET /runs/{run_id}/task-graph`
- `GET /runs/{run_id}/evidence-graph`
- `GET /runs/{run_id}/state-diff?domain=scheduler|evidence`
- `GET /runs/{run_id}/error-retry-chain?domain=runtime|scheduler`
- `GET /runs/{run_id}/conflicts`
- `GET /runs/{run_id}/components`
- `GET /runs/{run_id}/metrics`
- `GET /events/{event_id}`
- `GET /runs/{run_id}/scheduler-events/{sequence_no}`
- `GET /snapshots/{snapshot_id}`
- `GET /artifacts/{artifact_id}`
- `GET /artifacts/{artifact_id}/content`

The user interface is available at `GET /studio/{run_id}`. It has no mutation
controls and issues only `GET` requests.

## Explicit exclusions

Studio V2 does not implement:

- replay or failed-span restart;
- run or span fork;
- saved-result or live-environment replay;
- A/B comparison;
- component, prompt, skill, or policy diff;
- badcase creation;
- release/promotion or optimization;
- direct database-table access.

Replay, fork, A/B, and badcase functions belong to Background001 Branch 13.
