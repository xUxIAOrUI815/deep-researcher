# Background001 Evaluation Lab Core

Branch 10 owns the deterministic evaluation substrate described by
Background001. It evaluates immutable runtime outputs; it does not generate
patches, make release decisions, or mutate a running system.

## Runtime composition

`build_evaluation_lab_runtime()` composes four durable services:

1. `DatasetRegistry` seals versioned five-split datasets and records every
   granted access.
2. `EvaluationSnapshotBuilder` projects the Event Store, Evidence Engine, and
   Writer/Reviewer stores into a storage-neutral evaluation snapshot.
3. `DeterministicEvaluatorSuite` calculates fixed-version structural, source,
   resource, failure, protocol, and trace metrics.
4. `ExperimentRegistry` records reproducible definitions, results, and aligned
   legacy/fixed-workflow/new-runtime comparisons.

Evaluation records are immutable, checksummed rows in
`evaluation_lab.sqlite3`. Detailed reports are immutable artifacts. The SQLite
adapter uses WAL, immediate transactions, serialized connection access,
integrity checks, and online backup.

## Dataset isolation

Every registered bundle contains five non-empty, independently identified
splits:

| Split | Permitted normal use |
| --- | --- |
| `train` | training and development |
| `dev` | development and candidate development |
| `selection` | candidate selection |
| `test` | final evaluation and release-gate evaluation |
| `hidden-test` | release-gate evaluation only |

The shared contract also permits explicit audit access to every split. An
access request is validated before sample records are returned. The immutable
audit record includes the actor, purpose, split, sealed definition, disclosed
sample IDs, run identity when present, and audit artifact.

Split leakage is checked by input-content hash rather than artifact ID or
expected label. Consequently, copying or relabeling an input cannot move it
between splits. A dataset name/version is immutable. A new version may point to
a parent bundle; each split then points to the corresponding parent
definition. Manifest recovery repairs a registry write interrupted after the
artifact commit. Same-version retries, including concurrent retries through
different SQLite connections, resolve to the same sealed bundle.

Live Web evaluation and experiment definitions require an existing access
record. The access actor must equal the evaluated subject version, and the
record must authorize the exact dataset and sample. A hidden-test sample
therefore cannot be smuggled into candidate selection or an unaudited live
evaluation.

## Evaluation modes

### Frozen Replay

Frozen Replay verifies the fixture fingerprint before execution, requires all
fixture and expected artifacts, and performs between 2 and 100 repetitions
with one fixed seed. Any reported network call is a hard violation. Each
execution must return persisted output and event artifacts.

The result separately reports:

- equality with the expected output;
- equality with the expected event stream;
- equality of output, event, and deterministic-metric fingerprints across
  repetitions;
- confirmation that no network call occurred.

JSON comparisons use canonical encoding, so map ordering and insignificant
serialization whitespace do not create false differences.

### Live Web

Live Web performs between 2 and 100 executions. Repetition `n` receives
`base_seed + n`. Every snapshot is evaluated independently. The result records
the arithmetic mean and population variance for every metric present in every
repetition.

Source change is calculated over the union of observed source IDs. A source is
changed when its canonical URL/content hash changes or when it appears in only
some repetitions. The result stores changed IDs, presence counts, observed
versions, and the changed/union ratio.

## Deterministic metrics

The deterministic evaluator version is persisted with every result. It makes
no model call.

| Family | Metrics |
| --- | --- |
| URL and citations | URL validity, URL uniqueness, citation-reference integrity, exact normalized quote grounding, citation completeness |
| Structure | schema validity, required-section claim coverage |
| Sources | type count/diversity, mean authority, freshness, publisher diversity, domain diversity, primary-source share |
| Resources | total tokens, USD cost, latency |
| Reliability | task failure rate, retry recovery rate, idempotency compliance, protocol compliance |
| Trace efficiency | evidence per tool call, redundant-search rate, trace recovery rate, convergence turns, invalid-tool-call rate, budget-violation count |

Freshness uses the configured day window and the evaluation timestamp. Quote
grounding requires normalized exact containment in the persisted passage text.
Coverage only counts supported required claims. Resource and count metrics
retain their natural units and directions; the bounded aggregate uses only
rate metrics plus a zero-budget-violation indicator.

The snapshot builder derives source versions, passage text, verified citation
state, citation-map usage, required claim state, report revision, artifacts,
usage, latency, tool calls, retries, protocol signals, idempotency signals,
convergence cycles, and budget violations from durable stores. Explicit
counter overrides are validated and exist for imported traces whose original
event vocabulary cannot carry a required count.

## Experiment provenance and comparison

An experiment definition seals:

- the evaluated subject and baseline class;
- mode, seed, repetition count, dataset bundle/split/purpose/fingerprint, and
  audited access identity;
- all component `VersionRef` values;
- platform, Python implementation/version, dependency manifest names, sizes,
  hashes, configuration fingerprint, and network mode;
- input and configuration artifact IDs.

Configuration values are never copied into the environment descriptor. Callers
that need reproducible configuration bodies provide redacted immutable
configuration artifacts.

A successful experiment run requires metrics plus evaluation and output
artifacts. Comparison accepts exactly one successful run for each of
`legacy`, `fixed_workflow`, and `new_runtime`. The sealed dataset identity and
split must match. Mode, purpose, seed, repetitions, inputs, and configuration
artifacts must also match. Metric directions must agree. The comparison records
new-versus-legacy and new-versus-fixed deltas and explicitly lists metrics
missing from each baseline.

## Failure and recovery behavior

- Missing artifacts, unauthorized splits, wrong access actors, sample leakage,
  immutable identity reuse, and misaligned comparisons fail closed.
- Dataset access IDs and all registry/result IDs are idempotency identities.
  Reusing one for different content is rejected.
- Store payload checksums detect direct database corruption.
- `integrity_check()` checks SQLite and every evaluation payload; `backup_to()`
  creates a consistent online SQLite backup.
- Artifact-store integrity remains part of
  `EvaluationLabRuntime.integrity_check()`.

## Explicit boundary

This branch contains no LLM judge, semantic judge voting, release gate,
promotion/rollback state, Skill/Prompt/Policy patch generation, online
self-modification, test/hidden-test tuning, or reinforcement learning. Semantic
evaluation and release decisions belong to Branch 11; offline candidate
generation belongs to Branch 14.
