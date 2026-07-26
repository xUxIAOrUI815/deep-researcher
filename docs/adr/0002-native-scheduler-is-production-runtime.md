# ADR 0002: the native event-sourced scheduler is the production runtime

- Status: Accepted, superseding the provisional adapter decision
- Date: 2026-07-26

## Context

Background001 allowed LangGraph to remain only as a replaceable checkpoint
adapter while the native scheduler was being proved. It never owned task,
artifact, evidence, event, report, evaluation, or Studio truth.

The native scheduler now passes the required decision gates:

- append-only event truth and deterministic projection rebuild;
- concurrent claim fencing and run-wide concurrency limits;
- lease expiry recovery and stale-writer rejection;
- task/run cancellation, pause/resume, retry, and durable approval;
- hard token, cost, time, model, tool, search, retry, and error budgets;
- DAG dependency, split, merge, edit, and cycle consistency;
- mutation idempotency, checksum/integrity audit, restart, and online backup;
- production CLI, Console, Studio, research, and report-repair composition.

The former LangGraph checkpoint contained only a derived thin projection and
did not contribute an additional recovery guarantee. Retaining it would add a
second checkpoint lifecycle and four runtime dependencies without owning any
authoritative state.

## Decision

Production uses `NativeEventSourcedScheduler` directly through the
storage-neutral `Scheduler` protocol. The LangGraph adapter and dependencies
are removed.

Scheduler events are authoritative. Scheduler run/task/dependency tables are
rebuildable projections. Application status, evidence, reports, runtime events,
and Studio views remain separately owned projections or domain stores.

No compatibility layer is provided. A future graph framework may integrate
only by implementing the same `Scheduler` protocol and proving the full gate
set above; it may not introduce thick graph state or another business-data
source of truth.

## Consequences

- Crash recovery has one scheduling journal and one lease-fencing model.
- CLI and Console do not load or inspect graph checkpoints.
- Studio reads Event, Scheduler, Knowledge, Artifact, Reporting, and Version
  Registry APIs only.
- Removing a non-authoritative adapter reduces operational state and does not
  change domain contracts or role behavior.
