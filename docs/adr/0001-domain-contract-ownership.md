# ADR 0001: Framework-independent domain contracts

- Status: Accepted
- Date: 2026-07-22

## Context

The draft system mixes LangGraph state, Pydantic API schemas, SQLite records,
and agent payloads. That makes a framework or storage change indistinguishable
from a change to the research domain.

## Decision

`deep_researcher.contracts` owns the canonical Task, Command, Observation,
Event, Artifact, Evidence, Verification, AgentSpec, Budget, version, dataset,
and evaluation contracts. The package may depend on Python and Pydantic only.
It may not import LangGraph, FastAPI, a database driver, an HTTP/model SDK, or
legacy application schemas.

Contracts are strict, immutable, reject unknown fields, use namespaced IDs,
timezone-aware timestamps, and semantic `schema_version` values. Cross-layer
content is referenced by immutable artifact IDs rather than embedded mutable
runtime objects. Extensions are permitted only through an explicit extension
or metadata field.

## Consequences

Adapters must translate legacy state and provider responses at the boundary.
Database and API representations may add indexes or transport envelopes but
cannot silently redefine domain meaning. Breaking schema changes require a new
semantic version and an explicit migration registered with
`SchemaMigrationRegistry`.
