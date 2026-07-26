# Background001 Semantic Evaluation and Release Gates

Branch 11 extends the deterministic Evaluation Lab with semantic scoring,
human-calibrated blind judge panels, release policy enforcement, and an
immutable Version Registry. The gate may select, promote, reject, keep, or roll
back a version. It never creates or edits component content.

## Semantic evaluation

`SemanticEvaluationEngine` joins four sealed inputs:

1. an authorized Dataset Registry access record;
2. a durable `EvaluationSnapshot`;
3. explicit claim, retrieval, and report-requirement observations;
4. a fixed-version judge panel result covered by an accepted human calibration.

The access actor must be the evaluated subject version, and bundle, dataset,
split, and purpose must match exactly.

### Metric definitions

| Family | Metrics |
| --- | --- |
| Claims | mean claim support, supported rate, contradicted rate, unsupported rate, uncited factual-claim rate |
| Retrieval | macro precision, macro recall, mean authority, freshness rate, domain/type diversity |
| Report | completeness, depth, instruction following, organization, readability, safety |
| Deterministic | URL validity, schema validity, citation-position validity, cost, source freshness |

Claim state and citation presence come from explicit observations backed by the
Evidence Engine. Precision and recall require labeled relevant source IDs.
Report completeness combines required-topic coverage with the calibrated judge
score. Depth combines word/section targets with the judge score. Instruction
following combines explicit instruction checks with the judge score.

URL, schema, citation position, cost, and source time remain deterministic.
Citation position requires a used citation marker to occur exactly once and
immediately after non-whitespace report content.

Every semantic metric records its direction, threshold, and threshold result.
The immutable result links the deterministic report, judge panel, calibration,
dataset access, and subject version.

## Blind fixed-version multi-judge evaluation

`BlindMultiJudgePanel` requires an odd number of at least three distinct judge
versions. Every judge uses the same immutable rubric version.

For each panel:

- candidate identities are replaced by panel-scoped pseudonymous labels;
- each judge receives a separately seeded random candidate order;
- the request contains only the blinded report, instruction, requirement
  summary, verified-evidence summary, and rubric;
- system/baseline/component identities are not disclosed;
- search, tools, and rewriting are forbidden;
- the response contains bounded scores, a pass vote, violations, and a concise
  decision summary rather than hidden reasoning.

`ModelSemanticJudgeAdapter` is the concrete shared-`ModelAdapter` integration.
It supplies a strict structured schema and bounded repair path. Model and rubric
versions are fixed `VersionRef` values and are recorded on every ballot.

Consensus uses the median score per dimension and majority pass voting.
Per-candidate, per-dimension disagreement records persist judge scores,
population variance, score range, and mixed pass votes.

## Human calibration

`JudgeCalibrator` requires at least three human-rated candidates evaluated by
one fixed judge-version set and rubric. Multiple human ratings for a candidate
are averaged by dimension and majority-voted for pass/fail.

The calibration record contains:

- Pearson correlation by rubric dimension;
- tie-aware Spearman correlation by dimension;
- mean absolute error by dimension;
- aggregate human correlation;
- judge/human pass agreement;
- the judge, rubric, panel, and human-rating identities;
- explicit acceptance thresholds and the resulting accepted/rejected state.

A semantic evaluation using the same fixed judge versions fails if calibration
is rejected. A release gate additionally enforces its own minimum correlation
and pass-agreement thresholds.

## Release policy

`ReleaseGatePolicy` evaluates all of these independent families:

- required directional improvements;
- non-regression tolerances;
- absolute and relative cost;
- repeated-run population variance;
- safety minimums;
- protocol-compliance minimum;
- semantic evaluation pass state;
- human correlation and pass agreement.

Missing required metrics fail closed. Metric directions must agree between the
baseline and candidate. Each check is persisted with observed value, required
value, category, result, and explanation.

### Dataset boundary by stage

| Stage | Permitted access | Passing outcome | Failing outcome |
| --- | --- | --- | --- |
| selection | exactly `selection / candidate_selection` | advance to final | reject |
| final promotion | exactly `test / final_evaluation` plus `hidden-test / release_gate` | promote | reject |
| post-release | the same test and hidden-test final access | keep | rollback |

The access actor must equal the candidate version. Semantic results must cover
exactly the supplied access records. Test and hidden-test records cannot enter a
selection decision; selection results cannot publish a version.

The immutable gate decision is committed before its action. An independent
`GateApplicationRecord` is written after the idempotent Version Registry
transition. If the process fails between those commits, replay finds the
decision, completes or recognizes the transition, and writes the missing
application record without evaluating a different policy.

## Version Registry

The Version Registry accepts exactly these component kinds:

- AgentSpec
- Skill
- Prompt
- Tool Policy
- Stop Policy
- Rubric

Every version requires an immutable content artifact and verified content hash.
Kind, component name, semantic version, version ID, parent, content identity,
and manifest are immutable.

Lifecycle state is reconstructed from an append-only transition journal:

```text
candidate -> promoted
candidate -> rejected
promoted -> superseded
promoted -> rolled_back
superseded -> promoted
```

Promoting a candidate atomically supersedes the current active version for the
same kind/name. Rollback atomically marks the active version rolled back and
restores an explicitly selected superseded version. Rejected and rolled-back
versions are terminal. Every transition requires an immutable release-gate
decision artifact and records actor, reason, sequence, timestamp, and its own
artifact.

The SQLite store uses WAL, immediate transactions, immutable checksummed
manifests/events, rebuildable state and active-version projections, concurrent
connection fencing, integrity verification, and online backup. Projection
rebuild never rewrites the transition journal.

## Explicit boundary

Judges do not search, invoke tools, rewrite reports, alter evidence, or generate
component changes. Gates do not generate patches. This branch contains no
automatic Skill optimization, badcase optimizer, online self-modification,
test/hidden-test tuning, production entry-point migration, Studio UI, or
reinforcement learning. Offline candidate generation and strict selection
optimization belong to Branch 14.
