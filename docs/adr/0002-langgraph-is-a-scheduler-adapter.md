# ADR 0002: LangGraph remains a replaceable scheduler adapter

- Status: Accepted
- Date: 2026-07-22

## Context

The draft graph currently coordinates planning, research, distillation, and
writing while also carrying large mutable business objects. Background001
requires a durable task DAG and independent agent, evidence, event, and
artifact lifecycles.

## Decision

LangGraph is retained as one durable scheduling/checkpoint adapter. It does not
own domain contracts, event truth, artifact bodies, evidence semantics,
evaluation results, or Studio projections. Canonical runtime messages are
`TaskEnvelope`, `TaskResult`, `Command`, `Observation`, `RunEvent`, and artifact
references. Scheduler state contains identifiers and compact scheduling
snapshots only.

The orchestration branch will define a storage-neutral runtime interface and a
LangGraph adapter behind it. No domain package may import LangGraph.

## Consequences

Existing graph behavior remains unchanged in the foundation branch. Later
branches can migrate node-by-node and compare against Frozen Replay. A future
scheduler can replace LangGraph without rewriting agent roles, evidence data,
evaluation, or Studio.
