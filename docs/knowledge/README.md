# Artifact-backed knowledge runtime

`deep_researcher.knowledge` owns the durable evidence graph used by the
production application:

- `storage.py` and `sqlite_storage.py` provide append-only revisions, typed
  relationships, scoped natural keys, migrations, integrity checks, pagination,
  backup, and recovery.
- `repository.py` exposes typed repositories for Source, SourceSnapshot,
  Passage, Evidence, AtomicFact, Claim, Citation, Conflict, Section, and Report.
- `ingestion.py` converts governed research observations into artifact-backed
  candidate entities and creates the report scaffold.
- `normalization.py`, `deduplication.py`, and `retrieval.py` provide
  deterministic normalization, run-scoped deduplication, and retrieval.
- `runtime.py` owns the artifact store, knowledge store, repositories,
  ingestion, retrieval, integrity checks, and coordinated backup.

Every entity ID and natural key is scoped by `run_id`. Relationships are
type-checked and cannot cross runs. Source snapshots reference real immutable
artifact bodies. Re-ingesting identical content is idempotent; changed content
creates a new source version.

Research tools may write only candidate observations. The Evidence runtime
independently verifies grounding and semantic support before claims become
supported and before citations are generated. The Writer consumes verified
repositories and evidence packets; it never reads a compatibility projection
or legacy session state.

The production composition is
`deep_researcher.application.ApplicationRuntime`. No dual-write, graph-state
adapter, or legacy projection is supported.
