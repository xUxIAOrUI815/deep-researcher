# Studio V1: Persistent Trace Explorer

Studio V1 is a read-only trace surface built from append-only `RunEvent` data. It owns a separate, rebuildable SQLite projection with the hierarchy `Thread -> Run -> Span -> Timeline Event`; it never reads LangGraph state or event-store tables directly.

The projector is attached as a durable event export target. Local event commit always happens first. Projection delivery failures remain in the event outbox, and startup/query synchronization catches up any missing sequence. A run can be deleted from the projection and deterministically rebuilt from sequence 1 without changing source events.

The projection includes terminal state, component versions, input/output/state artifact references, model/tool usage, tokens, cost, latency, attempts, retries, permissions, errors, and span hierarchy. Payloads and error details are masked again server-side before projection even though the source EventStore already applies redaction.

HTTP surfaces under `/api/studio` provide thread/run catalogs, span queries, timeline filtering and text search, stable `after_sequence` pagination, SSE streaming, and complete JSON or NDJSON trace export. The Console timeline uses these projections and exposes searchable pages plus full event details. Long traces are never truncated in storage or export.

Studio V1 intentionally does not implement task-DAG, evidence-graph, state-diff, replay, fork, A/B, or badcase behavior; those belong to later Background001 branches.
