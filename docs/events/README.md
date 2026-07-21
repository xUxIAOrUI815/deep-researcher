# Persistent event and trace store

`deep_researcher.events` is the append-only execution record for Background001.
It stores no artifact bodies and exposes no Studio projection.

## Guarantees

- Every run starts at sequence 1 with `run_started`; sequence numbers are
  contiguous and unique inside the run.
- Event IDs are idempotency keys. Re-appending identical serialized content is
  accepted; reusing an ID for different content is rejected.
- A root span opens with the run. Model, tool, agent, and nested spans require an
  open parent and cannot close while a child remains open.
- A run accepts exactly one terminal event and no later events.
- Event JSON is canonical and protected by SHA-256. `integrity_check()` also
  verifies SQLite, foreign keys, indexed columns, sequence cursors, terminal
  pointers, and span state.
- OTLP exports use a transactional outbox target recorded with the local event.
  Collector failure leaves the event committed and the export restart-retryable.
- Sensitive keys and credential-shaped strings are redacted before commit.
  Decision events retain bounded summaries, command IDs, policy checks, stop
  reasons, and artifact IDs; raw model responses and hidden reasoning are not
  accepted as decision trace data.

## SQLite operations

The adapter enables foreign keys, a 30-second busy timeout, WAL, and FULL
synchronous commits for file databases. It uses `BEGIN IMMEDIATE` for append and
export-state transactions. Indexed queries support run-local pagination plus
event type, trace, span, actor, task, and time filters.

Schema migrations are forward-only and transactional:

1. Run, event, span, uniqueness, lifecycle, and query indexes.
2. Event checksums and durable exporter outbox.

Opening a database newer than the supported schema fails. A v1 database is
upgraded in place and existing event checksums are backfilled.

`backup_to()` uses the SQLite online backup API and verifies the copy before
returning. `restore_backup()` first checks the source database, restores it to
an explicitly named destination, applies supported migrations, and performs a
full event integrity audit.

## Runtime configuration

`run_research.py` installs the persistent observer and writes to
`EVENT_STORE_PATH` (default `event_data/research_events.sqlite3`). If
`OTEL_EXPORTER_OTLP_ENDPOINT` is set, events are exported as OTLP/HTTP JSON logs;
`OTEL_SERVICE_NAME` controls the resource service name. Exporter headers can be
provided programmatically to `build_event_runtime` and are never placed in an
event payload.
