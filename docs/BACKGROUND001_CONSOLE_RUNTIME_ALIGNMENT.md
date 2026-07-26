# Background001 Console Runtime Alignment

Branch: `codex/bg001-16-console-runtime-alignment`

This document is the durable execution contract for replacing the draft root
Console after the Background001 production runtime migration. It extends
`docs/BACKGROUND001_EXECUTION_PLAN.md`; it does not alter the Background001
domain architecture, role boundaries, or the explicit exclusion of
reinforcement learning.

## Problem statement

The root routes (`/`, `/console/{research_id}`, and `/report/{research_id}`)
still use the pre-migration Console shell. The shell assumes the removed
Planner/Researcher/Distiller/Writer workflow and guesses fields from raw runtime
objects. The production backend now owns a native scheduler, Research
Supervisor, bounded Research Worker Pool, independent Evidence Verifier,
Synthesis Writer, Report Reviewer, human approval, cancellation, verified-only
writing, report repair, and Studio V1-V4 projections.

This mismatch is functional, not cosmetic:

- polling omits `queued`, `waiting_approval`, `reporting`, and `cancelled`;
- the task tree reads removed `id`, `depth`, `node_type`, and `rationale`
  properties instead of TaskEnvelope and scheduler projection fields;
- coverage reads draft average/sufficiency fields instead of required,
  complete, gap, blocker, and conflict sets;
- gaps, conflicts, evidence packets, sources, and report revisions are rendered
  using shapes that no longer match their contracts;
- approval and cancellation endpoints exist without user-facing controls;
- the Console presents legacy role names and hides the verifier/reviewer loops;
- Studio is disconnected from the operational Console rather than exposed as
  the provenance and debugging surface for the same run.

The old root shell is draft code and is not a compatibility contract.

## Non-negotiable outcomes

1. Replace the root Console as a complete operational workspace, not a label or
   styling patch.
2. Introduce a versioned, typed Console read contract assembled by the
   projection-only Console service. Browser code must not infer domain meaning
   from arbitrary raw dictionaries.
3. Represent every ApplicationRunStatus and every production stage, including
   human approval, cancellation, failure, recovery, reporting, and completion.
4. Represent all five Background001 roles and the scheduler/verification/report
   loops without implying free-form agent chat or exposing hidden reasoning.
5. Provide complete task-DAG operations visibility: dependencies, parentage,
   status, attempts, ownership, leases, budgets, outputs, errors, and approval.
6. Provide complete evidence readiness visibility: section coverage, citation
   coverage, blocked high-impact claims, severe conflicts, open gaps, verified
   packets, and governed source links.
7. Provide complete report lifecycle visibility: revisions, reviewer decision,
   rubric scores, findings, repair actions, terminal outcome, citation map
   metadata, and final artifact readiness.
8. Provide working approve and cancel flows with explicit confirmation,
   required actor/note/reason fields, disabled-state fencing, pending feedback,
   and server error recovery.
9. Preserve Studio V1-V4 as the advanced trace/evidence/replay/comparison
   surface and link to the exact run rather than duplicating Studio internals.
10. Pass contract, API, interaction, accessibility, responsive, syntax,
    failure-state, and full integration tests.

## Information architecture

### Landing workspace

- Product framing reflects evidence-first Background001 research.
- Research creation captures question, bounded instructions, and depth.
- A runtime architecture strip names Supervisor, Worker Pool, Evidence
  Verifier, Synthesis Writer, and Report Reviewer.
- Recent runs support status and depth filtering, query search, refresh,
  terminal/report indicators, timestamp ordering, and direct Console/Report
  navigation.
- Empty, loading, validation, server failure, and provider-not-configured
  guidance are explicit.

### Operational Console

- Stable top navigation: Overview, Tasks, Evidence, Report, Timeline.
- Run identity, current stage, elapsed time, round, recovery flag, scheduler
  revision, and terminal state are visible without opening debug tools.
- Runtime progress models queued, research, independent verification,
  synthesis/review, approval interruption, and terminal outcomes.
- Role rail shows all five roles and highlights the current owner.
- Overview summarizes scheduler state, knowledge counts, blockers, coverage,
  active work, last structured decision, and next required operator action.
- Tasks shows the full parent/dependency DAG and a detailed selected-task
  inspector with budgets, attempts, leases, artifacts, and approval data.
- Evidence shows section readiness, gaps, conflicts, verified writer packets,
  sources, authority, provenance, and blockers.
- Report shows current revision and reviewer state while work is running, and
  links to the complete report viewer when an artifact exists.
- Timeline supports search, event-type filters, pagination, trace export, error
  visibility, usage, versions, artifacts, and permissions.
- Advanced Studio opens the exact underlying run.

### Report workspace

- Final Markdown is rendered with safe structural formatting and external links
  that remain governed by the persisted citation/source data.
- Report outline and section readiness remain visible beside the document.
- Revision identity, parent revision, reviewer decision, rubric scores,
  findings, repair actions, terminal outcome, usage, and evidence/citation
  counts are visible.
- Incomplete, waiting-approval, failed, cancelled, and report-not-yet-generated
  states provide a path back to the operational Console.

## Versioned Console projection

The Console service owns a `ConsoleWorkspace@2` read projection composed only
from public ApplicationRuntime, Scheduler, Knowledge Repository, Reporting
Store, Artifact Store, and Studio projection APIs.

Required projection groups:

- `identity`: research/thread/session/run/trace/report IDs, question,
  instructions, depth, timestamps, resume and report flags.
- `runtime`: application status, current stage, normalized progress steps,
  active role, structured decision summary, elapsed time, round and error.
- `actions`: terminal flag, approve/cancel availability, waiting approval task
  IDs, approval summaries, and action requirements.
- `scheduler`: scheduler status/revision/concurrency/cancellation reason,
  status counts, active/ready/waiting task IDs, and typed task views.
- `evidence`: knowledge counts, typed section readiness, aggregate required
  coverage, gaps, conflicts, writer evidence packets, and sources.
- `reporting`: outline, revision count/latest revision, reviewer decision,
  scores/findings/repair actions, loop outcome, artifact readiness, and usage.
- `timeline`: the bounded recent timeline plus exact Studio and export URLs.

The projection schema version is explicit. Unknown enum values remain
renderable as safe labels, but missing required contract groups fail tests.
Arbitrary internal metadata and hidden-reasoning fields are not projected.

## Frontend implementation boundaries

- Use the existing no-build, same-origin FastAPI static delivery model.
- Use semantic HTML, keyboard-reachable controls, visible focus, labels,
  dialogs, live status announcements, reduced-motion support, and responsive
  layouts at desktop, tablet, and mobile widths.
- Keep all text interpolation escaped. Render Markdown through an allowlisted
  parser implemented in the application bundle; never inject raw report HTML.
- Keep state, data normalization, polling, actions, and render components
  separated in testable exported modules.
- Poll `queued`, `running`, and `waiting_approval`; stop on `completed`,
  `failed`, or `cancelled`. Use visibility-aware scheduling and immediate
  refresh after approve/cancel.
- Preserve the current view, selected task, filters, and scroll-oriented
  interaction across polling updates.
- Do not add a JavaScript framework, Node production server, CDN dependency,
  tracking, external font, or network-only UI dependency.

## Explicit non-goals

- No change to research decomposition, evidence verification conclusions,
  scheduler transition rules, Writer prose, Reviewer scoring, evaluation
  release gates, replay semantics, offline evolution, or provider policy.
- No duplicate implementation of Studio task/evidence graphs, replay, A/B,
  version diffs, badcases, or release controls inside the operational Console.
- No backward-compatibility facade for the draft Console response.
- No persistence of client-only preferences beyond the current page.
- No deployment, `main` merge, version tag, GitHub Release, or RL work in this
  branch.

## Failure and state matrix

| State | Console behavior | Allowed operator action |
| --- | --- | --- |
| `queued` | show queued progress and scheduled owner | cancel |
| `running/researching` | poll and show Supervisor/Worker/Verifier activity | cancel |
| `running/reporting` | poll and show Writer/Reviewer lifecycle | cancel |
| `waiting_approval` | show blocking tasks and approval reasons | approve or cancel |
| `completed` | freeze polling and expose report/Studio/export | inspect |
| `failed` | show typed error, trace, and retained evidence/report state | inspect |
| `cancelled` | show cancellation reason and retained state | inspect |
| missing run | render recoverable not-found page | return to run list |
| request failure | preserve last good projection and show retry banner | retry |

## Test and merge gates

### Projection/API

- typed schema serialization for empty, active, approval, completed, failed,
  cancelled, and recovered runs;
- task parent/dependency ordering, approval, lease, budget, outputs, and errors;
- exact coverage/blocker/conflict/gap/source/evidence-packet mapping;
- report revision, review, repair, outcome, and citation metadata;
- no hidden-reasoning/internal dictionary leakage;
- action availability and invalid approve/cancel state rejection;
- route, static asset, not-found, and structured HTTP error behavior.

### Frontend

- JavaScript syntax and module-boundary tests;
- pure formatter/normalizer tests for every run/task/coverage/report state;
- DOM interaction tests for navigation, filtering, task selection, dialogs,
  approve, cancel, retry, polling, and report rendering;
- no unescaped application payloads or unsafe Markdown HTML;
- keyboard/focus/accessibility assertions and reduced-motion CSS;
- desktop, tablet, and 390 px mobile layout inspection.

### Integration

- deterministic complete run from create through verified report;
- approval pause/resume and cancellation flows;
- failed run with retained trace;
- long task/source/timeline/report payload rendering;
- Console/Studio exact-run navigation;
- existing Studio, application runtime, security/redaction, recovery, and full
  suite tests in normal and reverse order.

Before completion, reread this document and
`docs/BACKGROUND001_EXECUTION_PLAN.md`, append the branch execution record,
merge with `--no-ff` into `codex/bg001-integration`, rerun the integration
suite, and push both the feature and integration branches.
