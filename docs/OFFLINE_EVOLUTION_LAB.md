# Offline Evolution Lab

## Purpose

The Evolution Lab implements the Background001 cross-task optimization loop
without reinforcement learning and without online self-modification. It turns
reviewed, scored historical evidence into bounded candidate component versions,
then delegates selection, final promotion, and rollback to the existing
Evaluation Lab and Version Registry.

Candidate generation is never part of a production research request. Submitting
or reviewing a production trace does not invoke the optimizer, run an
evaluation, create a component version, or publish anything.

## Owned boundary

The package `deep_researcher.evolution_lab` owns:

- the reviewed candidate pool for scored success/failure traces, Studio
  badcases, and evaluation artifacts;
- immutable cross-task experience records;
- optimizer input sealing against the current promoted component version;
- structured text patches and per-round edit budgets;
- candidate component artifacts and Version Registry candidate registration;
- rejected-edit memory;
- the explicit human release gate;
- orchestration of selection, final test/hidden-test promotion, post-release
  evaluation, publication, and rollback; and
- static, versioned `best_skill` manifests for promoted Skill components.

It does not own runtime task memory, model weights, source/evidence mutation,
online inference, evaluation semantics, release policy evaluation, Version
Registry state transitions, production entry points, or Studio history.

## Optimization targets

All eight Background001 targets have explicit component boundaries:

| Target | Permitted component kinds |
| --- | --- |
| Planning | Skill or Prompt |
| Query generation | Skill or Prompt |
| Source selection | Skill, Prompt, or Tool Policy |
| Extraction/citation | Skill or Prompt |
| Report writing | Skill or Prompt |
| Tool routing | Tool Policy |
| Stop policy | Stop Policy |
| Grader rubric | Rubric |

A campaign cannot change its component kind or name. Every candidate has the
current promoted version as its immutable parent.

## Candidate-pool and data boundary

Candidate-pool submission writes an immutable `CandidatePoolEntry` and
`EVOLUTION_POOL_ENTRY` artifact with
`invokes_optimizer=false` and `publishes_versions=false`. Production traces are
explicitly labeled and cannot self-promote.

Every source needs a separately persisted human review before it may enter a
campaign. Reviews can assign only `train` or `dev`. Dataset splits discovered in
source/evaluation/badcase sample artifacts cannot be relabeled, and
`selection`, `test`, or `hidden-test` material cannot be approved for candidate
generation.

A campaign input snapshot is accepted only when it contains all of:

1. at least one scored success trace;
2. at least one scored failure trace;
3. at least one provenance-complete Studio badcase;
4. evaluation artifacts and metrics;
5. reviewed cross-task experiences; and
6. the exact content/hash/manifest of the currently promoted component.

The immutable snapshot also records every previously rejected patch fingerprint
known for the same target and base version.

## Memory separation

The implementation keeps three distinct layers:

- Runtime memory belongs to a live run and is forbidden in an evolution input
  snapshot.
- Cross-task experience contains curated problem statements, recommendations,
  structured edit suggestions, and rejected-edit history. It is explicitly not
  deployable.
- The formal Skill Registry consists of immutable Skill artifacts and Version
  Registry manifests. Only this layer can become active after release gates.

`best_skill` is a static version pointer with release-decision and human-decision
provenance. It performs no inference. Promotion creates a new immutable
`BEST_SKILL` snapshot; rollback creates another snapshot pointing to the
restored Skill, preserving the complete history.

## Structured patches and text learning rate

The only patch vocabulary is:

- `add` over an empty line range;
- `delete` over exact old lines; and
- `replace` over exact old and new lines.

Patches are applied to the immutable base content, not to mutable working
memory. Stale old text, ambiguous anchors, overlapping ranges, duplicate
insertion points, empty edits, and out-of-range lines fail closed.

Each campaign defines:

- maximum candidate rounds;
- maximum operations per round;
- maximum added and deleted lines;
- maximum changed characters; and
- maximum changed-character fraction of the base text.

That fraction is the explicit text learning rate. The patch artifact records
the actual edit metrics, source experience IDs, operation fingerprints,
consulted rejected history, and the exact base content hash.

The built-in `TraceSignalPatchGenerator` is deterministic and network-free. It
orders reviewed experiences by confidence and impact, resolves exact line
operations, excludes operations from rejected-edit memory, and admits only
operations that remain within the edit budget. A custom generator must return
an `OfflineGeneratorResult`; any reported network access or online inference
aborts the attempt before a candidate version is registered.

## Durable campaign lifecycle

The checksummed SQLite store uses WAL with full synchronization. Campaign
requests and journal events are immutable; campaign state is a rebuildable
projection.

```text
draft
  -> ready
  -> generating
       -> ready                 (abandoned/recovered attempt)
       -> candidate_ready
            -> ready            (selection rejected)
            -> awaiting_human   (strict selection passed)
                 -> ready       (human rejected)
                 -> ready_for_final
                      -> ready   (final gate rejected)
                      -> promoted
                           -> promoted   (post-release keep)
                           -> rolled_back
  -> exhausted                  (round budget reached)
```

Generation claims are atomic across store connections. A worker restart marks
the running attempt abandoned, preserves all already-registered immutable
artifacts as non-active candidates, and returns the campaign to `ready` for a
new attempt ID. Projection deletion is recoverable from the journal. Checksums,
SQLite integrity checks, immutable-table triggers, online backup, and restart
reopen are part of the store contract.

## Selection, human gate, publication, and rollback

Selection uses only the Evaluation Lab `selection` split and requires positive
improvement thresholds plus an actual strict improvement in every required
metric. The complete release policy still enforces non-regression, cost,
variance, safety, protocol, semantic, and human-calibration checks.

Failed selection, human rejection, and failed final evaluation each create
immutable `REJECTED_EDIT_MEMORY`. Its operation fingerprints are excluded from
later generation, so rejection history materially changes subsequent
candidates.

Selection success does not change the Version Registry. A separate human
decision is required. Human approval only authorizes final evaluation; it does
not trigger evaluation or publication. Human rejection is an auditable
Version Registry rejection backed by an `EVOLUTION_HUMAN_DECISION` artifact.

Final promotion requires exactly `test` and `hidden-test` access through the
existing release gate. Only a passing final decision promotes the candidate.
Post-release regression uses the same gate service to roll back to the sealed
base version. No code path in the Evolution Lab can bypass these transitions.

## Explicit exclusions

- no RL, reward model, PPO/GRPO, Agent Lightning, verl, or GPU training;
- no model-weight or source-code patching;
- no online Skill/Prompt/Policy modification;
- no runtime-memory promotion;
- no automatic action from a production trace or Studio badcase;
- no test/hidden-test candidate tuning;
- no extra inference in a production research request; and
- no hidden chain-of-thought storage.
