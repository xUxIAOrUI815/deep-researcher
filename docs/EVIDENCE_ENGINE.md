# Evidence Engine and Independent Verifier

Branch 07 implements the Background001 evidence layer. It is independent from
search scheduling, report writing, LangGraph, provider SDKs, and offline
optimization.

## Ownership and graph

The authoritative graph is:

```text
Source
  -> SourceSnapshot (immutable artifact and content hash)
  -> Passage (immutable text artifact, locator, offsets, extraction metadata)
  -> Evidence (candidate relation and exact EvidenceQuote records)
  -> AtomicFact
  -> Claim
  -> Citation (claim/evidence/passage/snapshot/source path)
  -> Section (required claims and coverage result)
  -> Report
```

`EvidenceGraphResolver` reconstructs this graph from revisioned knowledge
storage and rejects cross-run relationships. Its strict mode also checks every
citation path before verified knowledge is exposed.

Source snapshots are inputs to verification. The verifier never mutates or
re-fetches them.

## Candidate and verified boundaries

`CandidateKnowledgeView` exposes only proposed evidence/facts/citations and
draft or contested claims.

`VerifiedKnowledgeView` exposes:

- evidence with an independent verification identity and timestamp;
- verified facts;
- verified citations with exact quote offsets;
- supported claims whose supporting evidence, facts, citations, and complete
  source path all remain verified.

A high-impact claim that is not `supported` is returned by
`blocked_high_impact_claims` and is unavailable through
`claims_for_writing`. Replaying an unchanged distiller candidate preserves its
verified revision. A changed candidate artifact invalidates that shortcut and
returns to the candidate path for re-verification.

## Independent verifier

`build_evidence_verifier_spec` creates the Evidence Verifier AgentSpec. It:

- permits only structured `review` and `stop` commands;
- grants no tools and cannot delegate;
- explicitly forbids search, source-snapshot mutation, and report writing;
- uses the complete kernel middleware pipeline and bounded budgets;
- records short structured decisions, never hidden chain-of-thought.

`AgentSpecSemanticVerificationAdapter` uses the shared model adapter and the
verifier AgentSpec. It accepts only `SemanticJudgment`, performs at most one
schema-repair call, and accounts for both the invalid and repaired model calls.
`DeterministicSemanticVerificationAdapter` provides a stable offline verifier
for tests and Frozen Replay without pretending to be a live model.

## Verification procedure

`EvidenceVerificationEngine.verify_claim` performs these checks:

1. Reconstruct the complete claim graph and compute a deterministic input
   fingerprint that includes the verification policy version.
2. Verify the passage artifact hash and its owning snapshot artifact hash.
3. Verify each persisted quote against its passage ID, content hash, character
   offsets, and exact text.
4. Independently judge each evidence-to-claim and evidence-to-fact relation.
5. Verify citation claim/evidence/passage/snapshot/source path, locator, quote,
   and offsets.
6. Calculate support and contradiction scores.
7. Calculate source access, publisher/domain independence, minimum authority,
   primary-source requirements for high-impact claims, and freshness.
8. Detect overreach and unresolved high/critical conflicts.
9. Assign exactly one claim result and persist every entity revision,
   verification result, repair feedback, and domain event.

The result states are:

| State | Meaning |
| --- | --- |
| `supported` | Support, citations, authority, independence, freshness, and conflict gates all pass. |
| `partially_supported` | Some grounded support exists, but one or more completeness gates fail. |
| `contradicted` | Strong verified refutation exists without meaningful support. |
| `conflicted` | Strong support and refutation coexist, or a severe unresolved conflict applies. |
| `unsupported` | No grounded support reaches the partial-support threshold. |
| `stale` | Grounded support exists, but every supporting source is older than policy. |

High-impact status is explicit on a claim or derived from the policy importance
threshold. It raises authority and source-independence requirements, requires a
primary source, and turns non-supported results into writing blockers.

## Conflicts and section coverage

Open conflicts are assigned low, medium, high, or critical severity and remain
visible as revisioned entities. Definitive resolution requires at least one
verified evidence record plus a structured resolution kind. Accepting a
conflict as unresolved is a separate, explicit state and does not turn the
affected claim into supported knowledge. Both decisions produce immutable
feedback artifacts and evidence-change events.

Section coverage weights high-impact required claims twice. It records:

- supported, unsupported, conflicted, stale, and uncited claim IDs;
- weighted claim coverage and citation coverage;
- `insufficient`, `partial`, `complete`, or `blocked` status.

Any contradicted/conflicted claim blocks the section. Any unsupported
high-impact claim also blocks it.

## Bounded repair and recovery

Every repairable issue maps to a structured `RepairRequest`, such as re-extract
passage, re-fetch source, replace citation, revise claim, research, or resolve
conflict. `VerificationPolicy` bounds both repair rounds and requests per
verification. At the final round, issues remain recorded but are marked
non-repairable and no further request is emitted.

Verification and coverage IDs are content-derived. Repeating an identical
operation does not add artifacts, events, or knowledge revisions. If a process
crashes after the verification result artifact commits but before its repair
artifact or knowledge projection commits, the next invocation reconstructs the
missing feedback and entity revisions from the immutable result artifact.
Snapshot revisions are never reconstructed or changed by this process.
An existing result is reused only when its normalized evidence-graph fingerprint
still matches the current source, snapshot, passage, evidence, claim, citation,
and conflict revisions; a changed authority, hash, locator, quote, or conflict
therefore forces a new independent verification.

`RecordingEvidenceEventSink` is an explicit offline/test sink.
`EventRecorderEvidenceSink` appends idempotent `EVIDENCE_CHANGED` and
`VERIFICATION_COMPLETED` events to the durable Background001 event store and
retains full artifact, usage, trace, task, and version correlation.

## Runtime construction

Production composition must provide all three semantic dependencies:

```python
runtime = build_evidence_runtime(
    root,
    semantic_adapter=semantic_adapter,
    event_sink=event_sink,
    policy=verification_policy,
)
```

There is no implicit no-op verifier or event sink. `EvidenceRuntime` owns the
knowledge runtime, engine, candidate view, and verified view. Its underlying
artifact and knowledge databases use the existing manifest-verified backup,
restore, restart, and corruption checks.

## Explicit branch boundary

This layer does not schedule searches, choose research tasks, write final
report prose, mutate source snapshots, publish versions, optimize skills or
prompts, or perform RL. Those capabilities belong to later Background001
branches.
