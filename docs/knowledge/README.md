# Artifact-backed Knowledge Storage

`deep_researcher.knowledge` separates the responsibilities previously concentrated in `core/session_knowledge.py`:

- `storage.py` and `sqlite_storage.py`: immutable identities, append-only revisions, typed relationships, natural keys, migrations, transactions, pagination, audits, backup and recovery.
- `repository.py`: typed repositories for Source, SourceSnapshot, Passage, Evidence, AtomicFact, Claim, Citation, Conflict, Section, and Report.
- `ingestion.py`: normalization of the current draft pipeline into artifact-backed candidate entities. It persists actual web bodies and never fabricates a fact-to-evidence edge when a source cannot be matched.
- `normalization.py`: deterministic NFKC text and canonical HTTP(S) URLs.
- `deduplication.py`: exact and lexical duplicate decisions with an optional vector similarity adapter.
- `retrieval.py`: deterministic lexical retrieval with an optional vector ranking adapter. The vector adapter is an acceleration boundary, not a correctness dependency.
- `projection.py`: a rebuildable compatibility projection; it does not read database tables directly.
- `runtime.py`: owns both stores, repositories, ingestion/retrieval/projection services, integrity checks, and manifest-verified coordinated backups.

All entity IDs are namespaced by `run_id`, even when two runs ingest the same URL or upstream draft ID. Natural keys are also scoped by run. Relationships are type checked and cannot cross runs. Captured snapshots must reference a real ArtifactStore body whose run matches the entity provenance. Re-ingesting the same payload is revision-idempotent; changed source content creates a new immutable `SourceSnapshot.source_version`.

Branch 02 uses controlled dual-write from the current graph so the frozen behavior remains unchanged while the new stores receive durable data. The legacy session store remains a transitional draft input path and is not authoritative. Later branches move planning, verification, writing, and orchestration to the new contracts; those behaviors intentionally do not live here.

Production entry points install the runtime through `run_research.py`. Tests and embedded callers may install `KnowledgeIngestionService` with `core.graph.set_durable_knowledge_ingestor` and must restore the prior value after the run.
