# Background001 Orchestration Runtime

## Scope and ownership

`deep_researcher.orchestration` owns durable task scheduling. It does not own
research planning, model decisions, evidence semantics, artifact bodies, role
behavior, or report text. Tasks cross the boundary as versioned
`TaskEnvelope` contracts and completed work is represented by result and
artifact identifiers.

The runtime has one storage-neutral asynchronous `Scheduler` contract.
`NativeEventSourcedScheduler` is the production implementation and performs
scheduling directly against the durable event journal and rebuildable task/DAG
projection. ADR 0002 records why the provisional LangGraph adapter was removed
after the native implementation passed recovery, concurrency, cancellation,
approval, budget, and consistency gates.

## Durable model

`SQLiteSchedulerStore` uses `BEGIN IMMEDIATE` for each scheduling decision, WAL
mode, a 30-second busy timeout, full synchronous writes, foreign keys, and
checksums. One transaction:

1. rejects or replays the supplied mutation ID;
2. reads the current run and task projection;
3. validates and applies the state transition;
4. writes changed projection rows;
5. appends the immutable scheduler event.

The append-only `scheduler_events` journal is the scheduling source of truth.
`scheduler_runs`, `scheduler_tasks`, and `scheduler_dependencies` are query and
claim projections. They can be deleted and deterministically rebuilt from
event payloads. Every event carries the complete post-mutation run control and
all changed task records; event sequence equals projection revision. Checksums,
continuous run-local sequence numbers, projection/event revision equality,
SQLite integrity checks, backup, restart, and rebuild are enforced.

Mutation IDs are globally unique. Repeating the same mutation and canonical
payload returns the original outcome. Reusing the ID for different content
raises `SchedulerMutationConflict`. Empty claims are recorded too, making
polling calls safely replayable.

## Scheduling rules

Ready tasks are claimed atomically under the run-wide worker-slot ceiling.
Ordering is deterministic:

1. higher priority;
2. earlier deadline;
3. earlier creation time;
4. task ID.

An `assigned_actor_id` restricts a task to that worker. A task becomes ready
only when it is available and all dependency tasks are completed. Parent and
dependency references must resolve inside the same run. Split children are
topologically installed, edits are cycle-checked, and merge rewrites all
source dependencies to the target without losing dependent tasks.

Both `TaskEnvelope.deadline` and `Budget.deadline` are enforced before claim.
Token, cost, wall-time, model-call, tool-call, search-call, retry, and error
limits are enforced on usage updates and final completion. A result whose
reported usage reaches a hard limit is not accepted as completed; it is
atomically recorded as a budget failure.

## Task and run transitions

The scheduler implements these task operations:

| Operation | State effect |
| --- | --- |
| create/submit | `pending`, promoted to `ready` when dependencies allow |
| split | atomically adds a validated child DAG |
| merge | source becomes `merged`; dependent edges move to target |
| defer | `pending`/`ready` to `deferred` until an aware timestamp |
| prune | eligible non-running work to terminal `pruned` |
| claim | `ready` to leased `running`, incrementing the attempt |
| heartbeat | extends a live lease owned by the same worker |
| complete | `running` to `completed`, then promotes dependents |
| fail | `running` to terminal-attempt `failed` |
| retry | `failed` to `pending`/`ready` if attempts remain |
| cancel | eligible nonterminal task to `cancelled` |
| pause/resume | leased `running` to `paused`, then `ready` |
| request approval | leased `running` to `waiting_approval` |
| approve/reject | `waiting_approval` to `ready` or `failed` |
| edit | atomically changes allowed queued/HITL fields and revalidates the DAG |

A run can be active, paused, cancelled, or completed. Pausing releases running
leases into run-owned paused tasks and blocks claims. Resuming requeues only
tasks paused by that run operation. User cancellation terminalizes all
remaining work in the same transaction. Completion is rejected while any task
is nonterminal.

## Lease recovery and fencing

Every claim records owner, expiry, heartbeat, and attempt. All worker writes
verify owner, running state, and an unexpired lease. Thus a stale worker cannot
complete, fail, pause, update usage, or request approval after its lease
expires.

Claim and explicit recovery scan expired leases. If attempts remain, the task
is failed and requeued with retry/error usage recorded. If the attempt limit is
exhausted it remains failed. Recovery itself is idempotent and persists a
`RecoveryReport`.

## Human-in-the-loop behavior

Approval requests persist requester, reason, timestamps, and a deterministic
approval ID, while releasing the worker lease. A human may edit an eligible
task before resolving it. Approval records resolver, note, and time and requeues
the task. Rejection records a supplied error reference and fails it. Task-level
and run-level pause/resume/cancel operations are independent from approval.

## Failure and operating behavior

- Projection checksum failure is surfaced as `SchedulerCorruption`; rebuilding
  from a valid event journal repairs projection-only corruption.
- Event checksum or sequence corruption is not repaired silently.
- Concurrent scheduler processes serialize claims in SQLite, so they cannot
  exceed global slots or lease one task twice.
- A committed scheduler mutation is replayable from its mutation identity and
  append-only journal without a second checkpoint system.
- Backup uses the SQLite online backup API after a full WAL checkpoint.
- Production entry points use the native scheduler through the Background001
  application composition root.
