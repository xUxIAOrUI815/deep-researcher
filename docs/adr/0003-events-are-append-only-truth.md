# ADR 0003: Append-only events are execution truth

- Status: Accepted
- Date: 2026-07-22

## Context

In-memory observer lists and mutable graph state cannot explain a run after a
restart, reconstruct a timeline, or support reliable replay and evaluation.

## Decision

Every observable runtime change emits an immutable `RunEvent`. Events carry a
run-local sequence number, trace/span hierarchy, correlation and causation,
actor and task identity, artifact/state references, budget usage, latency,
status, structured error, and complete component versions. Event stores are
append-only. Corrections are new events; existing events are never updated.

Projection stores derive current Thread, Run, Span, Task, budget, and Studio
views from events and are always rebuildable. Artifact bodies live in the
ArtifactStore and evidence relationships live in evidence repositories. An
event references them but does not duplicate their authoritative content.

## Consequences

Branch 01 must enforce idempotency, run-local ordering, span lifecycle, and a
single terminal run outcome transactionally. Exporter failure cannot remove or
invalidate locally committed events. Projection loss is recoverable by replay.
