# ADR 0004: Do not store or expose hidden chain-of-thought

- Status: Accepted
- Date: 2026-07-22

## Context

Trace and evolution features need inspectable decisions, but persisting hidden
reasoning is unnecessary, unsafe, and an unstable interface.

## Decision

The system records structured decision summaries only: observed facts,
selected command IDs, bounded alternatives, policy checks, command arguments,
observations, verification feedback, errors, stop reasons, and artifact
references. Contracts, event payloads, Studio APIs, exports, and datasets must
not define or accept fields named `chain_of_thought`, `cot`, `hidden_reasoning`,
or equivalent free-form private reasoning.

Redaction middleware removes provider-specific accidental reasoning fields
before local persistence. Model raw outputs may be retained only as governed,
access-controlled artifacts after the same redaction policy is applied.

## Consequences

Debugging focuses on reproducible inputs, versions, actions, observations, and
verification results. Offline evolution learns from scored structured traces,
not hidden reasoning transcripts.
