# Protocol and Tool Gateway

This document defines the branch-05 runtime boundary from Background001. The
gateway translates provider and inter-agent protocols into the versioned domain
contracts introduced in branch 00. It does not plan research, schedule tasks,
own task status, or mutate evidence.

## Pinned protocols and SDKs

| Protocol | Pinned wire version | SDK | Transport coverage |
| --- | --- | --- | --- |
| Model Context Protocol | `2025-11-25` | `mcp==1.27.2` | stdio and Streamable HTTP |
| Agent2Agent | `1.0` | `a2a-sdk==1.1.1` | JSON-RPC 1.0 binding |

Protocol versions are checked during client initialization/discovery. A server
that does not offer the pinned version and binding is rejected; no silent
downgrade is permitted.

Authoritative specifications:

- MCP specification: <https://modelcontextprotocol.io/specification/2025-11-25>
- MCP Python SDK: <https://github.com/modelcontextprotocol/python-sdk>
- A2A specification: <https://a2a-protocol.org/latest/specification/>
- A2A Python SDK: <https://github.com/a2aproject/a2a-python>

## Internal normalization

All Function Calling shapes are normalized into a domain `Command`. The
normalizer accepts generic function calls, OpenAI nested tool calls, Anthropic
`tool_use` blocks, and provider choice envelopes. Tool identity and version are
resolved from the immutable `ToolRegistry`; command and idempotency identifiers
are deterministic for the same provider call.

`ProtocolToolGateway` is the only execution policy boundary. Before an adapter
runs, it enforces the active tool version, allowed operation, permission scopes,
approval, JSON Schema input, safety scanning, durable idempotency, cache policy,
rate limits, and circuit state. Each attempt then has a timeout and active
cancellation path. Retry and fallback remain gateway-owned so provider adapters
perform exactly one attempt. Outputs pass JSON Schema and safety validation
before they become a normalized `Observation`.

The SQLite tool-state store persists idempotency leases/results, cache entries,
rate-limit windows, circuit-breaker state, and checksummed audit events. It is
restart-safe and deliberately separate from task, artifact, event, and evidence
stores.

## MCP lifecycle

`MCPProtocolClient` uses the official SDK for initialization, pagination-aware
tool/resource/prompt discovery, calls, reads, prompt rendering, ping, timeout,
cancellation, and close. `MCPRemoteToolAdapter` maps discovered remote tools to
normal registry entries. `GatewayMCPHost` exposes governed registry tools plus
explicit resource and prompt registrations through the official low-level MCP
server over stdio or Streamable HTTP.

MCP errors remain structured. A remote `isError` result does not become a
successful observation. Client cancellation maps to the domain cancelled error;
timeout and transport failures are retryable, while protocol and validation
failures are permanent.

## A2A lifecycle

`A2AProtocolClient` resolves an Agent Card, requires the configured A2A 1.0
interface, and submits a redacted `TaskEnvelope` using the official protobuf
models. Local artifacts are handed off as typed A2A parts. Correlation and trace
identifiers are carried in message metadata, request metadata, and call context
headers. The client supports task submission, task status, task cancellation,
stream responses, timeouts, active cancellation, health state, and typed failure
classification. Remote tasks and artifacts are returned as immutable snapshots;
the client does not assume ownership of local task state.

## Research adapters

The former fake `MCPGateway` dispatcher has been removed. Tavily, Exa, and the
configured scraper are ordinary `ToolAdapter` implementations registered as:

- `web.search.tavily@1.0.0`
- `web.search.exa@1.0.0`
- `web.scrape@1.0.0`

Tavily requires `TAVILY_API_KEY`; Exa requires `EXA_API_KEY`. A missing key is a
permanent configuration failure, never a production mock response. Offline test
behavior remains explicit through `RESEARCHER_SEARCH_MODE=mock` and
`SCRAPER_MODE=mock`. The default durable gateway database is
`knowledge_data/tool_gateway.sqlite3` and can be overridden with
`TOOL_GATEWAY_DB_PATH`.

The Tavily definition may fall back to Exa only through the declared registry
fallback chain. Provider authentication values are not placed in tool arguments,
command metadata, or audit payloads.

## Ownership boundary

The gateway owns protocol negotiation, tool catalog/version resolution,
invocation governance, transport execution, normalization, and invocation
health. It must not create a research DAG, select research questions, transition
task state, verify claims, create citations, or write evidence. Those concerns
are delivered by later Background001 branches.
