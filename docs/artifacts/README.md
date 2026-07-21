# Immutable Artifact Store

`deep_researcher.artifacts` is the authoritative store for immutable bodies produced or consumed by a run. It accepts every `ArtifactKind`, including search responses, source snapshots, cleaned passages, tool results, model inputs/outputs, evidence packs, reports, prompts, skills, policies, rubrics, dataset samples, evaluation results, trace exports, and state patches.

Each envelope is run-scoped and contains producer/task provenance, media/schema metadata, SHA-256, byte length, sensitivity, and source-artifact references. Bodies are deduplicated by SHA-256 without merging their provenance envelopes. IDs and idempotency keys reject conflicting reuse. Source references and links cannot cross runs.

Raw source snapshots and cleaned passage text are stored byte-for-byte by default. Structured model/tool payloads use explicit redaction by default; callers can opt into text redaction for governed text artifacts. Hidden chain-of-thought is never an accepted persistence payload.

The SQLite adapter provides migrations, WAL durability, foreign keys, immutable-row triggers, stable cursor pagination, checksummed envelopes/links/bodies, concurrent-writer serialization, integrity audits, and verified online backup/restore. Artifact bodies do not belong in LangGraph state, the event store, or the evidence graph.

Typical use:

```python
from deep_researcher.artifacts import SQLiteArtifactStore
from deep_researcher.contracts import ArtifactKind

with SQLiteArtifactStore("artifact_data/artifacts.sqlite3") as store:
    snapshot = store.put_text(
        exact_source_text,
        kind=ArtifactKind.SOURCE_SNAPSHOT,
        producer_id="agent_researcher",
        run_id="run_example",
    )
    store.integrity_check()
```
