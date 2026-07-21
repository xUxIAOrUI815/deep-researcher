# ADR 0005: Evaluation dataset splits are one-way separated

- Status: Accepted
- Date: 2026-07-22

## Context

Using the same cases to tune, select, and report quality creates leakage and
invalid release decisions. Production badcases also must not self-promote code,
prompts, skills, policies, or model versions.

## Decision

Datasets are immutable, versioned manifests partitioned as `train`, `dev`,
`selection`, `test`, and `hidden-test`. Access is purpose-bound:

| Purpose | Allowed split |
| --- | --- |
| Training | train |
| Development | train, dev |
| Candidate selection | dev, selection |
| Final evaluation | test |
| Release gate | test, hidden-test |
| Audit | all, read-only |

Each evaluation records its dataset ID, split, declared purpose, subject and
component versions, evaluator, metrics, gates, artifacts, and usage. The domain
contract rejects an invalid purpose/split pair before any evaluator runs.

Production traces may enter a quarantined candidate dataset after redaction and
review. They cannot mutate or promote a global version online. Promotion uses
offline experiments, selection data, untouched test/hidden-test gates, approval,
and a rollback record.

## Consequences

Optimization can iterate on train/dev without consuming release evidence.
Hidden-test contents stay unavailable to candidate generation and selection.
