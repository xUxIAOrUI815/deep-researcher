# Background001 Synthesis Writer and Report Reviewer

## Scope

This module implements branch 09 of `BACKGROUND001_EXECUTION_PLAN.md`:

- a Synthesis Writer AgentSpec and AgentKernel role;
- a Report Reviewer AgentSpec and AgentKernel role;
- the verified-evidence writing boundary;
- section synthesis, traceable multi-source citations, explicit evidence gaps,
  conflict presentation, uncertainty, and cross-section consistency;
- the eight-dimension report rubric;
- bounded targeted-research, citation-repair, local-rewrite,
  structural-rewrite, accept, and reject decisions;
- immutable report revisions, report artifacts, review artifacts, revision
  history, terminal outcomes, and citation maps.

This branch does not switch `run_research.py`, Console, or Studio to the new
runtime. Production entry-point migration remains branch 15. It also does not
implement release gates, online optimization, skill evolution, or RL.

## Trust boundaries

### Writer input

`VerifiedWriterPacketBuilder` is the sole Writer knowledge input. It obtains
claims through `EvidenceRuntime.verified.claims_for_writing`, then resolves
their strict citation graphs. It includes only:

- supported claims whose facts and supporting evidence are verified;
- verified citations tied to accepted passages, snapshots, and sources;
- report sections and their required claim IDs;
- explicit gap records for required claims unavailable through the
  verified-only view;
- persisted conflict records touching section claims.

Candidate or unsupported claims never enter `packet.claims`. They can only
enter `packet.gaps`, where the rendered report labels them as unconfirmed.

The Writer AgentSpec grants no tools and allows no search command. The model
adapter exposes only the persisted packet, report structure, revision number,
and typed repair feedback.

### Writer output

The model cannot persist free-form prose directly. It must return a
`WriterDraftProposal`:

- every factual statement declares one or more verified claim IDs;
- every factual statement declares citations for every claim;
- high-impact claims cite at least the configured number of independent
  sources;
- every verified required claim is represented;
- every gap and conflict is explicitly disclosed;
- statements touching an unresolved conflict use conflicted certainty;
- the same claim cannot be rendered inconsistently across sections.

Before persistence, the action executor independently checks all identifier
relationships and asks the configured evidence semantic adapter whether each
statement is supported by its declared verified claims. It rejects overreach,
unknown identifiers, insufficient source independence, missing disclosures,
wrong section ownership, and inconsistent cross-section renderings.

Markdown rendering is deterministic. Citation markers are placed immediately
after factual statements. Evidence gaps and conflict summaries are rendered
from persisted packet records rather than unconstrained model text.

### Reviewer input and authority

The Reviewer reads the immutable `ReportRevision`, structured Writer proposal,
verified packet, and citation map. Its AgentSpec grants no tools and no search.
It cannot write claim, fact, evidence, citation, or conflict entities.

The Reviewer model must score every dimension exactly once:

1. completeness;
2. support;
3. citation;
4. conflicts;
5. instruction following;
6. depth;
7. organization;
8. readability.

`DeterministicReportAuditor` recomputes structural, support-reference,
citation-placement, citation-map, gap, conflict, depth, ordering, and basic
readability invariants from persisted artifacts. Final scores use the lower of
the model score and deterministic score. A model `accept` cannot override a
deterministic defect or configured threshold.

## Report loop

`ReportLoopCoordinator` is a bounded state machine:

```text
verified packet
    -> Writer revision
    -> Reviewer decision
       -> accept
       -> reject
       -> citation repair -> Writer
       -> local rewrite -> Writer
       -> structural rewrite -> Writer
       -> targeted research -> Scheduler/Worker Pool
                              -> Evidence verification
                              -> rebuilt verified packet
                              -> Writer
```

Bounds are explicit in `ReportLoopPolicy`:

- aggregate token, cost, wall-time, model, tool, search, retry, and error
  budget;
- maximum report revisions;
- maximum targeted-research rounds;
- global and support/citation-specific acceptance thresholds;
- normal and high-impact source-count requirements.

Cancellation, aggregate budget exhaustion, revision exhaustion, rejection,
acceptance, and a targeted-research approval boundary all produce one immutable
`ReportLoopOutcome`.

`SchedulerTargetedResearchDispatcher` is the concrete research repair path. It
submits typed `GAP` or `SECTION_SUPPORT` tasks into the real Scheduler, drains
the configured Research Worker Pool, runs evidence verification, and rebuilds
the verified packet. Research output remains candidate-only until the Evidence
Engine promotes it. If the scheduler run is already terminal, the loop returns
`approval_required`; it does not silently reopen or fork a durable run.

## Persistence

`SQLiteReportingStore` uses WAL mode, immediate transactions, checksummed JSON
payloads, and immutable identities. It stores:

- `ReportRevision` rows with parent revision linkage;
- one `CitationMap` per report revision;
- one final `ReviewerDecision` per report revision;
- one terminal `ReportLoopOutcome` per run/report.

Every report revision also has immutable artifacts for:

- the structured Writer proposal;
- the citation map;
- rendered Markdown;
- each rendered section.

Every review and terminal report-loop decision is an artifact linked to its
inputs. The store supports restart reads, integrity checks, concurrent
idempotent saves, corruption detection, and SQLite online backup.

## Composition

Use `build_reporting_runtime` with:

- an existing `EvidenceRuntime`;
- Writer and Reviewer model adapters;
- a durable Kernel event sink;
- a `ReportLoopPolicy`.

To enable targeted research, also pass the same run's Scheduler, Research
Worker Pool, and an explicit per-task research budget. The builder refuses
partial configuration and does not install fake providers, no-op event sinks,
or implicit research fallbacks.

The caller owns the Evidence Runtime. Closing `ReportingRuntime` closes only
its reporting journal.
