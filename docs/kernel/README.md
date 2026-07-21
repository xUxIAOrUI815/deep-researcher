# Agent Kernel

Branch 04 implements the complete task-local `Observe -> Decide -> Act -> Verify`
loop required by Background001. The implementation is framework-independent:
it does not import LangGraph, FastAPI, a model SDK, an MCP/A2A transport, or an
evidence repository.

## Ownership and boundary

The kernel owns:

- immutable `AgentSpec` version registration and lookup;
- context construction, deterministic trimming, recursive sensitive-data
  redaction, and component-version injection;
- typed model requests/responses and bounded provider retries;
- structured command parsing, schema validation and bounded model repair;
- command identity normalization and policy enforcement;
- action execution and normalization of every observation status;
- independent verification feedback and repair feedback for the next round;
- task-local token, cost, wall-time, model-call, tool-call, search-call, retry,
  error, and deadline enforcement;
- semantic and operational stop decisions; and
- structured, persistable lifecycle events without hidden chain-of-thought.

The kernel does not own task persistence or DAG scheduling, provider-specific
Function Calling/MCP/A2A behavior, role-specific research logic, evidence
storage or judgment semantics, report writing, evaluation, evolution, or RL.
Those capabilities are delivered by later branches through the typed kernel
interfaces.

## Runtime interfaces

`AgentKernel` requires five concrete collaborators:

| Collaborator | Responsibility |
| --- | --- |
| `AgentSpecRegistry` | Resolve an enabled, immutable AgentSpec version. |
| `ModelAdapter` | Produce or repair a `ModelResponse` for a typed `ModelRequest`. |
| `ActionExecutor` | Execute a normalized `Command` and return a raw or normalized observation. |
| `KernelVerifier` | Independently evaluate every action observation and return repair/convergence feedback. |
| `KernelEventSink` | Durably consume every structured kernel lifecycle event. |

There is intentionally no production no-op event sink. Tests can use an
in-memory collector; real runs can use `EventRecorderKernelSink`, which converts
kernel events into the append-only event store's balanced Agent, Model, and Tool
spans. The sink creates a run root only when one does not already exist and
never terminates the run, because run ownership belongs to orchestration.

`component_versions_for_agent()` builds the complete event provenance set from
the AgentSpec plus explicit runtime and scheduler versions.

## Middleware contract

Every runnable AgentSpec must enable each stage exactly once and in this order:

1. `context_trimming`
2. `redaction`
3. `version_injection`
4. `budget_check`
5. `schema_validation`
6. `schema_repair`
7. `command_normalization`
8. `policy_check`

The kernel rejects incomplete or reordered pipelines rather than silently
running without a safety stage. AgentSpec's default budget is a safety envelope;
the task budget is intersected with it dimension by dimension, so a task can
narrow but cannot broaden an AgentSpec limit. All eight dimensions must be
available after intersection. The earliest AgentSpec, task-budget, or task
deadline is used.

The model never controls command, run, task, or actor identity. Normalization
creates a deterministic command ID and idempotency key from trusted task/spec
identity and redacted command content. High/critical risk commands are forced
to require approval. Tool grants enforce operation allowlists, required and
allowed arguments, JSON-like type/value/range/pattern constraints, and
per-task call ceilings.

## Loop and accounting

For each round the kernel:

```text
check model/time/cost/token budget
  -> build bounded redacted context
  -> call model (bounded retry)
  -> validate/repair/normalize commands
  -> record structured decision summary
  -> check command policy/approval
  -> check tool/search/time/cost budget
  -> execute and normalize observation
  -> independently verify observation
  -> stop or feed structured repair feedback into the next round
```

Model, action, verification, schema-repair, and retry usage is accumulated in a
single immutable `BudgetUsage`. Wall time uses a monotonic clock and active
model/action/verification calls are interrupted when cancellation, task
deadline, or wall-time expiration occurs. Started event spans are closed on
success, failure, timeout, schema-repair failure, and cancellation.

The kernel returns `KernelRunResult`, containing the canonical `TaskResult`,
the exact `StopDecision`, effective budget, normalized commands and
observations, structured decision summaries, and verifier feedback. It never
returns or persists raw model reasoning.

## Stop semantics

| Reason | Task result | Meaning |
| --- | --- | --- |
| `success` | succeeded | Verifier or explicit stop command confirms success. |
| `semantic_complete` | succeeded | Further work is unnecessary even if the latest action did not add data. |
| `low_information_gain` | partial | The configured consecutive no-gain threshold was reached. |
| `budget_exhausted` | partial | A required operation cannot run within one or more hard limits. |
| `deadline_reached` | partial | The earliest effective task deadline was reached. |
| `repeated_error` | failed | The same failure repeated or bounded repair/retry was exhausted. |
| `user_cancelled` | cancelled | Cancellation was observed before or during an operation. |
| `approval_required` | deferred | A side-effect/risk policy requires an external approval decision. |
| `policy_denied` | rejected | The command violates the AgentSpec/tool policy. |
| `verification_failed` | failed | Action or independent verification has no permitted recovery path. |
| `no_action_available` | partial/failed | Empty valid output is partial; provider failure carries a failed result. |

Budget equality is a hard ceiling: a call that reaches a dimension may finish,
but no subsequent operation that consumes that dimension may begin. A zero
retry/error ceiling is valid and does not stop a clean task before its first
operation; it stops as soon as a retry is requested or an error occurs.

## Failure normalization

Provider, schema, protocol, policy, cancellation, timeout, and verification
failures become typed `ErrorRecord` values. An executor exception becomes a
failed observation, and an invalid executor return becomes a protocol-failure
observation, so the Act stage never disappears from the trace. All failure
observations have forced command/run/task/actor identity, attempt count, at
least one tool-call usage, one search-call usage for search, and one error usage.

Model and action retries require both their operation budget and retry budget.
The verifier receives prior observations but is not permitted to mutate them.
Its `information_gain`, `success`, `semantic_complete`, and `repair_feedback`
fields are the only inputs to convergence and the next decision round.

## Tests

`tests/test_bg001_agent_kernel.py` uses deterministic adapters and covers:

- multiple logical roles sharing one kernel;
- complete/reordered/missing middleware pipelines;
- context trimming, redaction, and version injection;
- deterministic identity normalization and injection resistance;
- schema repair success, provider failure, and bounded exhaustion;
- command policies, tool grants, expiry, call limits, denial, and approval;
- all observation statuses and executor exceptions;
- model, action, and verifier retries;
- all eight budget dimensions plus task deadline;
- success, semantic completion, no gain, repeated error, cancellation,
  approval, policy, verification, and no-action stops; and
- persistent balanced event spans on both success and failure.
