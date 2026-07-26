from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from deep_researcher.application import (
    ApplicationRuntime,
    ApplicationRuntimeDependencies,
    build_application_runtime,
)
from deep_researcher.contracts import BudgetUsage
from deep_researcher.gateway import (
    ToolAdapterResult,
    ToolHealthStatus,
)
from deep_researcher.kernel import ModelResponse
from deep_researcher.providers import (
    GovernedResearchToolRuntime,
    build_governed_research_tools,
)


STATEMENT = "Verified benchmark is 42."
SOURCE_URL = "https://example.com/verified-source"


class DeterministicSearchAdapter:
    def __init__(self, *, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls = 0
        self.active = 0
        self.peak = 0

    async def execute(self, arguments, context):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            return ToolAdapterResult(
                success=True,
                data={
                    "items": [
                        {
                            "url": SOURCE_URL,
                            "title": "Official source",
                            "snippet": STATEMENT,
                            "score": 0.95,
                            "raw_content": STATEMENT,
                        }
                    ]
                },
                usage=BudgetUsage(search_calls=1),
            )
        finally:
            self.active -= 1

    async def health(self):
        return ToolHealthStatus.HEALTHY


class DeterministicSupervisorModel:
    async def complete(self, request):
        return ModelResponse(
            structured={
                "action": "decompose",
                "tasks": [
                    {
                        "proposal_key": "verified_evidence",
                        "kind": "research",
                        "title": "Acquire and verify evidence",
                        "goal": (
                            "Search an authoritative source, read its exact "
                            "statement, and extract citation-ready candidates."
                        ),
                        "expected_output_schema": "ResearchWorkerResult@1",
                        "budget": {
                            "max_tokens": 24_000,
                            "max_model_calls": 12,
                            "max_tool_calls": 16,
                            "max_search_calls": 6,
                            "max_retries": 4,
                            "max_errors": 4,
                        },
                    }
                ],
                "decision_summary": "Acquire citation-complete verified evidence.",
            },
            usage=BudgetUsage(
                input_tokens=20,
                output_tokens=20,
                model_calls=1,
            ),
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


class DeterministicWorkerModel:
    def __init__(self) -> None:
        self.calls: dict[str, int] = {}

    async def complete(self, request):
        turn = self.calls.get(request.task_id, 0)
        self.calls[request.task_id] = turn + 1
        task = next(
            message["content"]
            for message in request.messages
            if message.get("role") == "user"
            and isinstance(message.get("content"), dict)
            and "constraints" in message["content"]
        )
        section_id = task["constraints"]["required_section_id"]
        if turn == 0:
            payload = {
                "summary": "Search the governed authoritative source.",
                "commands": [
                    {
                        "kind": "search",
                        "name": "research.search",
                        "arguments": {
                            "operation": "search",
                            "query": "verified benchmark",
                            "max_results": 3,
                        },
                    }
                ],
            }
        else:
            payload = {
                "summary": "Extract an exact quote and candidate claim graph.",
                "commands": [
                    {
                        "kind": "extract",
                        "name": "research.extract",
                        "arguments": {
                            "operation": "extract",
                            "section_id": section_id,
                            "semantic_complete": True,
                            "evidence": [
                                {
                                    "id": "evidence_verified_benchmark",
                                    "source_id": "source_result_0",
                                    "quote": STATEMENT,
                                    "summary": "Exact official statement.",
                                    "confidence": 0.95,
                                    "quality_score": 0.95,
                                }
                            ],
                            "atomic_facts": [
                                {
                                    "id": "fact_verified_benchmark",
                                    "text": STATEMENT,
                                    "source_id": "source_result_0",
                                    "confidence": 0.95,
                                    "section_id": section_id,
                                }
                            ],
                            "claims": [
                                {
                                    "id": "claim_verified_benchmark",
                                    "text": STATEMENT,
                                    "fact_ids": ["fact_verified_benchmark"],
                                    "evidence_ids": [
                                        "evidence_verified_benchmark"
                                    ],
                                    "confidence": 0.95,
                                }
                            ],
                        },
                    }
                ],
            }
        return ModelResponse(
            structured=payload,
            usage=BudgetUsage(
                input_tokens=20,
                output_tokens=20,
                model_calls=1,
            ),
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


class DeterministicVerifierModel:
    async def complete(self, request):
        return ModelResponse(
            structured={
                "label": "supports",
                "score": 0.99,
                "overreach_fragments": [],
                "contradiction_fragments": [],
                "decision_summary": "The exact persisted quote supports the statement.",
            },
            usage=BudgetUsage(model_calls=1),
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


class DeterministicWriterModel:
    async def complete(self, request):
        message = next(
            item["content"]
            for item in request.messages
            if isinstance(item.get("content"), dict)
            and item["content"].get("schema") == "WriterEvidencePacket@1"
        )
        packet = message["packet"]
        claims = {item["claim_id"]: item for item in packet["claims"]}
        gaps = {item["claim_id"]: item for item in packet["gaps"]}
        conflicts = {
            item["conflict_id"]: item for item in packet["conflicts"]
        }
        sections: list[dict[str, Any]] = []
        for section in packet["sections"]:
            statements = []
            for index, claim_id in enumerate(section["verified_claim_ids"]):
                claim = claims[claim_id]
                statements.append(
                    {
                        "statement_id": (
                            f"statement_{section['order']}_{index}_verified"
                        ),
                        "text": claim["statement"],
                        "claim_ids": [claim_id],
                        "citation_ids": claim["citation_ids"],
                        "certainty": "definitive",
                    }
                )
            sections.append(
                {
                    "section_id": section["section_id"],
                    "title": section["title"],
                    "statements": statements,
                    "gap_disclosures": [
                        {
                            "claim_id": claim_id,
                            "text": gaps[claim_id]["reason"],
                        }
                        for claim_id in section["gap_claim_ids"]
                    ],
                    "conflict_disclosures": [
                        {
                            "conflict_id": conflict_id,
                            "text": conflicts[conflict_id]["summary"],
                            "citation_ids": [],
                        }
                        for conflict_id in section["conflict_ids"]
                    ],
                }
            )
        return ModelResponse(
            structured={
                "title": message["report_title"],
                "sections": sections,
                "decision_summary": "Synthesized only from verified evidence.",
            },
            usage=BudgetUsage(model_calls=1),
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


class DeterministicReviewerModel:
    async def complete(self, request):
        dimensions = (
            "completeness",
            "support",
            "citation",
            "conflicts",
            "instruction_following",
            "depth",
            "organization",
            "readability",
        )
        return ModelResponse(
            structured={
                "decision": "accept",
                "scores": [
                    {
                        "dimension": item,
                        "score": 1.0,
                        "rationale": "All deterministic requirements pass.",
                    }
                    for item in dimensions
                ],
                "findings": [],
                "repair_actions": [],
                "decision_summary": "Report passes all deterministic gates.",
            },
            usage=BudgetUsage(model_calls=1),
        )

    async def repair(self, request, invalid_response, errors):
        return await self.complete(request)


def build_deterministic_application(
    root: Path,
    *,
    search_delay: float = 0.0,
    supervisor_model: Any | None = None,
    worker_model: Any | None = None,
    verifier_model: Any | None = None,
    writer_model: Any | None = None,
    reviewer_model: Any | None = None,
) -> tuple[
    ApplicationRuntime,
    GovernedResearchToolRuntime,
    DeterministicSearchAdapter,
]:
    search = DeterministicSearchAdapter(delay=search_delay)
    tools = build_governed_research_tools(
        root / "tools",
        tavily=search,
        exa=search,
        retry_base_seconds=0,
    )
    runtime = build_application_runtime(
        root,
        dependencies=ApplicationRuntimeDependencies(
            supervisor_model=(
                supervisor_model or DeterministicSupervisorModel()
            ),
            worker_model=worker_model or DeterministicWorkerModel(),
            verifier_model=verifier_model or DeterministicVerifierModel(),
            writer_model=writer_model or DeterministicWriterModel(),
            reviewer_model=reviewer_model or DeterministicReviewerModel(),
        ),
        tool_runtime=tools,
    )
    return runtime, tools, search
