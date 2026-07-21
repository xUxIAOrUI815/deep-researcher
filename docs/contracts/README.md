# Background001 contract rules

The public contract surface is `deep_researcher.contracts`. Contracts are
strict and immutable. They serialize with Pydantic `model_dump(mode="json")`
and are transported with an explicit contract type and semantic schema version.

## Schema compatibility

- Patch versions may clarify validation without changing valid serialized data.
- Minor versions may add optional fields with deterministic defaults.
- Major versions may remove, rename, or change the meaning of fields.
- Readers never guess a migration. `SchemaMigrationRegistry` must contain a
  complete, acyclic path to the current model version.
- Canonical JSON is UTF-8, sorted by key, compactly encoded, and fingerprinted
  with SHA-256 when stable identity is required.
- Unknown fields are rejected. Forward-compatible information belongs in an
  explicit `metadata` or `extensions` map defined by the owning contract.

## Identity and time

IDs are namespaced (`task_...`, `event_...`, `artifact_...`) so accidental
cross-entity references are detectable. Timestamps are timezone-aware. Stores
must preserve their original offset or normalize to UTC without dropping
timezone information.

## Content ownership

Large or immutable bodies are stored as artifacts. Contracts carry artifact
IDs, provenance, causation, component versions, and content hashes. Runtime
state may carry only IDs and compact scheduling snapshots.
