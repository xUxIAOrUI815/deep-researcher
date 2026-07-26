from __future__ import annotations

import hashlib
import json
from typing import Any
from urllib.parse import urlsplit

from deep_researcher.contracts import (
    BudgetUsage,
    CitationStatus,
    ClaimStatus,
    EventType,
    RunEvent,
    combine_usage,
    utc_now,
)
from deep_researcher.evidence import EvidenceRuntime
from deep_researcher.events import EventQuery, EventStore
from deep_researcher.reporting import SQLiteReportingStore

from .models import (
    CitationObservation,
    EvaluationSnapshot,
    ExecutionCounters,
    SectionObservation,
    SourceObservation,
    ToolCallObservation,
)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


class EvaluationSnapshotBuilder:
    """Projects durable evidence/report/event stores into evaluator input."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        reporting_store: SQLiteReportingStore,
        event_store: EventStore | None = None,
    ) -> None:
        self.evidence = evidence
        self.reporting_store = reporting_store
        self.event_store = event_store

    def build(
        self,
        *,
        run_id: str,
        report_id: str,
        counter_overrides: dict[str, int] | None = None,
        schema_errors: tuple[str, ...] = (),
    ) -> EvaluationSnapshot:
        repository = self.evidence.knowledge.repository
        report = repository.reports.require(report_id)
        if report.run_id != run_id:
            raise ValueError("report does not belong to the evaluation run")
        revisions = self.reporting_store.revisions(run_id, report_id)
        revision = revisions[-1] if revisions else None
        citation_map = (
            self.reporting_store.citation_map(
                run_id,
                report_id,
                revision.revision,
            )
            if revision is not None
            else None
        )
        map_by_citation = (
            {
                item.citation_id: item
                for item in citation_map.entries
            }
            if citation_map is not None
            else {}
        )
        section_entities = tuple(
            repository.sections.require(item) for item in report.section_ids
        )
        used_domain_citations = {
            citation_id
            for section in section_entities
            for citation_id in section.citation_ids
        }
        domain_citations = repository.citations.list(run_id)
        citations: list[CitationObservation] = []
        for citation in domain_citations:
            source = repository.sources.require(citation.source_id)
            passage = repository.passages.require(citation.passage_id)
            try:
                passage_text = self.evidence.knowledge.artifacts.read_bytes(
                    passage.text_artifact_id
                ).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    f"passage text is not UTF-8: {passage.passage_id}"
                ) from exc
            map_entry = map_by_citation.get(citation.citation_id)
            citations.append(
                CitationObservation(
                    citation_id=citation.citation_id,
                    claim_id=citation.claim_id,
                    evidence_id=citation.evidence_id,
                    source_id=citation.source_id,
                    marker=(
                        map_entry.marker
                        if map_entry is not None
                        else None
                    ),
                    canonical_url=source.canonical_url,
                    quote=citation.quote,
                    passage_text=passage_text,
                    locator=citation.locator,
                    verified=citation.status == CitationStatus.VERIFIED,
                    used_in_report=(
                        citation.citation_id in map_by_citation
                        if citation_map is not None
                        else citation.citation_id in used_domain_citations
                    ),
                )
            )
        sources: list[SourceObservation] = []
        for source in repository.sources.list(run_id):
            snapshots = repository.source_snapshots(source.source_id)
            latest = max(
                snapshots,
                key=lambda item: (item.source_version, item.fetched_at),
                default=None,
            )
            domain = (urlsplit(source.canonical_url).hostname or "").casefold()
            sources.append(
                SourceObservation(
                    source_id=source.source_id,
                    canonical_url=source.canonical_url,
                    source_type=source.source_type.value,
                    source_level=source.source_level.value,
                    authority_score=source.authority_score,
                    publisher=source.publisher,
                    domain=domain,
                    published_at=source.published_at,
                    fetched_at=(
                        latest.fetched_at if latest is not None else None
                    ),
                    content_hash=(
                        latest.content_hash if latest is not None else None
                    ),
                )
            )
        sections: list[SectionObservation] = []
        for section in section_entities:
            required = section.required_claim_ids or section.claim_ids
            claims = tuple(
                repository.claims.require(item) for item in required
            )
            supported = tuple(
                item.claim_id
                for item in claims
                if item.status == ClaimStatus.SUPPORTED
            )
            unsupported = tuple(
                item.claim_id
                for item in claims
                if item.status != ClaimStatus.SUPPORTED
            )
            sections.append(
                SectionObservation(
                    section_id=section.section_id,
                    required_claim_ids=required,
                    supported_claim_ids=supported,
                    unsupported_claim_ids=unsupported,
                    citation_ids=section.citation_ids,
                )
            )
        events = self._events(run_id)
        tool_calls = self._tool_calls(events)
        counters = self._counters(events, tool_calls)
        if counter_overrides:
            values = counters.model_dump(mode="python")
            unknown = set(counter_overrides) - set(values)
            if unknown:
                raise ValueError(
                    f"unknown execution counter overrides: {sorted(unknown)}"
                )
            values.update(counter_overrides)
            counters = ExecutionCounters.model_validate(values)
        outcome = self.reporting_store.outcome(run_id, report_id)
        usage = (
            outcome.usage
            if outcome is not None
            else self._usage(events)
        )
        latency_ms = (
            usage.wall_time_seconds * 1000.0
            if usage.wall_time_seconds
            else sum(item.latency_ms for item in events)
        )
        source_fingerprint = hashlib.sha256(
            json.dumps(
                [
                    (
                        item.source_id,
                        item.canonical_url,
                        item.content_hash,
                    )
                    for item in sorted(
                        sources,
                        key=lambda value: value.source_id,
                    )
                ],
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        material = {
            "run_id": run_id,
            "report_id": report_id,
            "report_revision_id": (
                revision.revision_id if revision is not None else None
            ),
            "sources": [item.model_dump(mode="json") for item in sources],
            "citations": [
                item.model_dump(mode="json") for item in citations
            ],
            "sections": [item.model_dump(mode="json") for item in sections],
            "tool_calls": [
                item.model_dump(mode="json") for item in tool_calls
            ],
            "counters": counters.model_dump(mode="json"),
            "schema_errors": list(schema_errors),
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        output_artifacts = tuple(
            item
            for item in dict.fromkeys(
                (
                    *(
                        (
                            revision.report_artifact_id,
                            revision.citation_map_artifact_id,
                        )
                        if revision is not None
                        else ()
                    ),
                    *(
                        (
                            outcome.final_report_artifact_id,
                            outcome.citation_map_artifact_id,
                        )
                        if outcome is not None
                        else ()
                    ),
                )
            )
            if item is not None
        )
        created_at = (
            outcome.completed_at
            if outcome is not None
            else (
                revision.created_at
                if revision is not None
                else (events[-1].occurred_at if events else utc_now())
            )
        )
        return EvaluationSnapshot(
            snapshot_id=_stable_id(
                "evaluation_snapshot",
                run_id,
                report_id,
                fingerprint,
            ),
            run_id=run_id,
            report_id=report_id,
            report_revision_id=(
                revision.revision_id if revision is not None else None
            ),
            report_markdown=(
                revision.markdown if revision is not None else ""
            ),
            sources=tuple(
                sorted(sources, key=lambda item: item.source_id)
            ),
            citations=tuple(
                sorted(citations, key=lambda item: item.citation_id)
            ),
            sections=tuple(
                sorted(sections, key=lambda item: item.section_id)
            ),
            tool_calls=tool_calls,
            counters=counters,
            usage=usage,
            latency_ms=latency_ms,
            schema_errors=schema_errors,
            output_artifact_ids=output_artifacts,
            source_fingerprint=source_fingerprint,
            created_at=created_at,
            metadata={
                "projection": "durable_stores",
                "event_count": len(events),
            },
        )

    def _events(self, run_id: str) -> tuple[RunEvent, ...]:
        if self.event_store is None:
            return ()
        output: list[RunEvent] = []
        after = 0
        while True:
            page = self.event_store.list(
                EventQuery(run_id=run_id, after_sequence=after, limit=1000)
            )
            output.extend(page.items)
            if page.next_after_sequence is None:
                break
            after = page.next_after_sequence
        return tuple(output)

    @staticmethod
    def _tool_calls(
        events: tuple[RunEvent, ...],
    ) -> tuple[ToolCallObservation, ...]:
        output: list[ToolCallObservation] = []
        seen: set[str] = set()
        for event in events:
            if event.event_type not in {
                EventType.TOOL_COMPLETED,
                EventType.TOOL_FAILED,
            }:
                continue
            payload = event.payload
            call_id = str(
                payload.get("command_id")
                or payload.get("call_id")
                or event.span_id
            )
            if not call_id.startswith("call_"):
                call_id = _stable_id("call", event.run_id, call_id)
            if call_id in seen:
                continue
            seen.add(call_id)
            operation = str(
                payload.get("operation")
                or payload.get("name")
                or "unknown"
            )
            tool_name = str(
                payload.get("tool_name")
                or payload.get("name")
                or event.actor_id
            )
            request_key = str(
                payload.get("query")
                or payload.get("request_fingerprint")
                or payload.get("idempotency_key")
                or call_id
            ).strip().casefold()
            evidence_count = int(
                payload.get("evidence_count")
                or len(event.output_artifact_ids)
            )
            output.append(
                ToolCallObservation(
                    call_id=call_id,
                    tool_name=tool_name,
                    operation=operation,
                    request_key=request_key,
                    is_search=(
                        "search" in operation.casefold()
                        or "search" in tool_name.casefold()
                    ),
                    valid=not bool(
                        payload.get("invalid")
                        or payload.get("protocol_error")
                    ),
                    succeeded=event.event_type == EventType.TOOL_COMPLETED,
                    evidence_count=max(0, evidence_count),
                    recovered_after_retry=bool(
                        payload.get("recovered_after_retry")
                        or event.attempt > 1
                    ),
                )
            )
        return tuple(output)

    @staticmethod
    def _counters(
        events: tuple[RunEvent, ...],
        tool_calls: tuple[ToolCallObservation, ...],
    ) -> ExecutionCounters:
        task_ids = {
            event.task_id
            for event in events
            if event.task_id is not None
            and event.event_type
            in {EventType.TASK_CREATED, EventType.TASK_STATE_CHANGED}
        }
        failed_task_ids = {
            event.task_id
            for event in events
            if event.task_id is not None
            and (
                event.event_type == EventType.TASK_STATE_CHANGED
                and str(event.payload.get("status", "")).casefold()
                == "failed"
            )
        }
        retries = sum(
            1
            for event in events
            if event.event_type == EventType.RETRY_SCHEDULED
        )
        recovered = sum(
            1 for item in tool_calls if item.recovered_after_retry
        )
        idempotent = sum(
            1
            for event in events
            if "idempotency" in event.payload
            or "idempotency_key" in event.payload
        )
        idempotency_violations = sum(
            1
            for event in events
            if bool(event.payload.get("idempotency_violation"))
        )
        protocol_events = [
            event
            for event in events
            if any(
                key in event.payload
                for key in (
                    "protocol",
                    "protocol_version",
                    "protocol_valid",
                    "protocol_error",
                )
            )
        ]
        invalid_protocol = sum(
            1
            for event in protocol_events
            if event.payload.get("protocol_valid") is False
            or bool(event.payload.get("protocol_error"))
        )
        cycles = [
            int(event.payload["cycle"])
            for event in events
            if isinstance(event.payload.get("cycle"), int)
        ]
        budget_violations = sum(
            1
            for event in events
            if event.event_type == EventType.BUDGET_CHANGED
            and bool(
                event.payload.get("violated")
                or event.payload.get("exceeded_dimensions")
            )
        )
        return ExecutionCounters(
            task_count=len(task_ids),
            failed_task_count=len(failed_task_ids),
            retried_operation_count=retries,
            recovered_operation_count=min(retries, recovered),
            idempotent_operation_count=max(
                idempotent,
                idempotency_violations,
            ),
            idempotency_violation_count=idempotency_violations,
            protocol_operation_count=len(protocol_events),
            invalid_protocol_operation_count=invalid_protocol,
            convergence_turns=(max(cycles) + 1 if cycles else 0),
            budget_violation_count=budget_violations,
        )

    @staticmethod
    def _usage(events: tuple[RunEvent, ...]) -> BudgetUsage:
        terminal = [
            event
            for event in events
            if event.event_type
            in {
                EventType.RUN_COMPLETED,
                EventType.RUN_FAILED,
                EventType.RUN_CANCELLED,
            }
        ]
        if terminal:
            return terminal[-1].usage
        completed = [
            event.usage
            for event in events
            if event.event_type
            in {
                EventType.MODEL_COMPLETED,
                EventType.TOOL_COMPLETED,
                EventType.TOOL_FAILED,
            }
        ]
        return combine_usage(completed)
