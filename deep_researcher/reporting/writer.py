from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import hashlib
import json
import re
from copy import deepcopy
from typing import Any
from collections.abc import Awaitable, Callable

from pydantic import ValidationError

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    AgentSpec,
    ArtifactKind,
    Budget,
    BudgetUsage,
    Command,
    CommandKind,
    EvidenceRelation,
    Observation,
    ObservationStatus,
    ReportStatus,
    SectionStatus,
    TaskEnvelope,
    TaskKind,
    TaskResultStatus,
    TaskStatus,
    utc_now,
)
from deep_researcher.evidence import EvidenceRuntime, SemanticLabel
from deep_researcher.kernel import (
    AgentKernel,
    CancellationToken,
    ModelAdapter,
    ModelInvocationError,
    ModelRequest,
    ModelResponse,
    RawObservation,
    VerificationFeedback,
)

from .evidence import read_json_artifact
from .models import (
    CitationMap,
    CitationMapEntry,
    ConflictPacket,
    DraftStatement,
    ReportLoopPolicy,
    ReportRevision,
    StatementCertainty,
    WriterDraftProposal,
    WriterEvidencePacket,
)
from .store import SQLiteReportingStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _usage_sum(items: list[BudgetUsage]) -> BudgetUsage:
    usage = BudgetUsage()
    for item in items:
        usage = usage.plus(
            input_tokens=item.input_tokens,
            output_tokens=item.output_tokens,
            cost_usd=item.cost_usd,
            wall_time_seconds=item.wall_time_seconds,
            model_calls=item.model_calls,
            tool_calls=item.tool_calls,
            search_calls=item.search_calls,
            retries=item.retries,
            errors=item.errors,
        )
    return usage


def _constraints(request: ModelRequest) -> dict[str, Any]:
    for message in request.messages:
        content = message.get("content")
        if isinstance(content, dict):
            value = content.get("constraints")
            if isinstance(value, dict):
                return value
    return {}


_EMPTY_VERIFIED_SECTION_NOTICE = (
    "本节在当前已验证证据包中没有可引用的事实性陈述。"
)


def _payload(response: ModelResponse, key: str) -> dict[str, Any]:
    value: Any = response.structured
    if isinstance(value, dict) and key in value:
        value = value[key]
    if value is None:
        text = response.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        value = json.loads(text)
        if isinstance(value, dict) and key in value:
            value = value[key]
    if not isinstance(value, dict):
        raise ValueError(f"model response must contain a structured {key}")
    return value


class SynthesisWriterModelAdapter(ModelAdapter):
    """Makes the verified packet visible and repairs typed Writer proposals."""

    def __init__(
        self,
        model: ModelAdapter,
        *,
        artifact_store: ArtifactStore,
        max_proposal_repairs: int = 2,
        proposal_validator: Callable[
            [WriterDraftProposal, WriterEvidencePacket, str],
            Awaitable[BudgetUsage],
        ]
        | None = None,
    ) -> None:
        if max_proposal_repairs < 0:
            raise ValueError("max_proposal_repairs cannot be negative")
        self.model = model
        self.artifact_store = artifact_store
        self.max_proposal_repairs = max_proposal_repairs
        self.proposal_validator = proposal_validator

    async def complete(self, request: ModelRequest) -> ModelResponse:
        specialized = self._request(request)
        response = await self.model.complete(specialized)
        return await self._validated(
            request=specialized,
            response=response,
            usage_items=[],
        )

    async def repair(
        self,
        request: ModelRequest,
        invalid_response: ModelResponse,
        errors: tuple[str, ...],
    ) -> ModelResponse:
        specialized = self._request(request)
        response = await self.model.repair(
            specialized,
            invalid_response,
            errors,
        )
        return await self._validated(
            request=specialized,
            response=response,
            usage_items=[],
        )

    def _request(self, request: ModelRequest) -> ModelRequest:
        constraints = _constraints(request)
        artifact_id = str(
            constraints.get("evidence_packet_artifact_id") or ""
        )
        if not artifact_id:
            raise ModelInvocationError(
                "Writer task is missing evidence_packet_artifact_id",
                retryable=False,
            )
        payload = read_json_artifact(self.artifact_store, artifact_id)
        packet = WriterEvidencePacket.model_validate(
            payload.get("packet"),
            strict=False,
        )
        if packet.run_id != request.run_id:
            raise ModelInvocationError(
                "Writer evidence packet belongs to another run",
                retryable=False,
            )
        evidence_message = {
            "role": "user",
            "content": {
                "schema": "WriterEvidencePacket@1",
                "verified_only": True,
                "packet": packet.model_dump(mode="json"),
                "revision": int(constraints.get("revision", 1)),
                "report_title": str(
                    constraints.get("report_title") or "Research report"
                ),
                "repair_feedback": constraints.get("repair_feedback", []),
                "rules": {
                    "every_statement_requires_claim_ids": True,
                    "every_statement_requires_citation_ids": True,
                    "disclose_every_gap": True,
                    "disclose_every_conflict": True,
                    "do_not_add_external_facts": True,
                    "required_gap_ids_by_section": {
                        section.section_id: list(section.gap_claim_ids)
                        for section in packet.sections
                    },
                    "required_conflict_ids_by_section": {
                        section.section_id: list(section.conflict_ids)
                        for section in packet.sections
                    },
                    "empty_id_list_means_disclose_none": True,
                    "never_invent_claim_or_conflict_ids": True,
                },
            },
        }
        return ModelRequest(
            run_id=request.run_id,
            task_id=request.task_id,
            actor_id=request.actor_id,
            system=(
                "Act as the Synthesis Writer. Use only the supplied verified "
                "evidence packet. Return one WriterDraftProposal. Every factual "
                "statement must enumerate the verified claim IDs and citation "
                "IDs that support it. Include every required verified claim, "
                "every evidence gap, and every conflict. Do not search, invent "
                "facts, hide uncertainty, or reveal hidden reasoning."
            ),
            messages=(*request.messages, evidence_message),
            command_schema=WriterDraftProposal.model_json_schema(),
            model_version=request.model_version,
            prompt_version=request.prompt_version,
            max_output_tokens=request.max_output_tokens,
            metadata={
                **request.metadata,
                "structured_output": "WriterDraftProposal@1",
                "verified_packet_id": packet.packet_id,
            },
        )

    async def _validated(
        self,
        *,
        request: ModelRequest,
        response: ModelResponse,
        usage_items: list[BudgetUsage],
    ) -> ModelResponse:
        current = response
        usage = [*usage_items, response.usage]
        packet_payload = read_json_artifact(
            self.artifact_store,
            str(_constraints(request)["evidence_packet_artifact_id"]),
        )
        packet = WriterEvidencePacket.model_validate(
            packet_payload.get("packet"),
            strict=False,
        )
        for attempt in range(self.max_proposal_repairs + 1):
            try:
                proposal = self._parse(request, current, packet)
                if self.proposal_validator is not None:
                    await self.proposal_validator(
                        proposal,
                        packet,
                        request.task_id,
                    )
            except (
                ValidationError,
                WriterTraceabilityError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as exc:
                if attempt >= self.max_proposal_repairs:
                    raise ModelInvocationError(
                        "Writer proposal remained invalid after bounded repair: "
                        f"{exc}",
                        retryable=False,
                    ) from exc
                current = await self.model.repair(
                    request,
                    current,
                    (str(exc),),
                )
                usage.append(
                    current.usage.model_copy(
                        update={"retries": current.usage.retries + 1}
                    )
                )
                continue
            artifact_id = str(
                _constraints(request)["evidence_packet_artifact_id"]
            )
            return ModelResponse(
                content=proposal.decision_summary,
                structured={
                    "summary": proposal.decision_summary,
                    "commands": [
                        {
                            "kind": CommandKind.SYNTHESIZE.value,
                            "name": "writer.persist_revision",
                            "arguments": {
                                "proposal": proposal.model_dump(mode="json"),
                                "evidence_packet_artifact_id": artifact_id,
                            },
                            "input_artifact_ids": [artifact_id],
                            "expected_output_schema": "ReportRevision@1",
                        }
                    ],
                    "writer_proposal": proposal.model_dump(mode="json"),
                },
                usage=_usage_sum(usage),
                latency_ms=current.latency_ms,
                finish_reason=current.finish_reason,
                response_id=current.response_id,
            )
        raise AssertionError("unreachable Writer repair loop")

    @staticmethod
    def _parse(
        request: ModelRequest,
        response: ModelResponse,
        packet: WriterEvidencePacket,
    ) -> WriterDraftProposal:
        constraints = _constraints(request)
        value = deepcopy(_payload(response, "proposal"))
        SynthesisWriterModelAdapter._remove_untraceable_empty_section_narrative(
            value,
            packet,
        )
        value.update(
            run_id=request.run_id,
            report_id=str(constraints["report_id"]),
            revision=int(constraints["revision"]),
        )
        return WriterDraftProposal.model_validate(value, strict=False)

    @staticmethod
    def _remove_untraceable_empty_section_narrative(
        value: dict[str, Any],
        packet: WriterEvidencePacket,
    ) -> None:
        """Discard model prose where the evidence packet permits no citation.

        Report plans intentionally contain boundary sections such as the
        executive summary and methodology.  When such a section has no
        verified claims, a model cannot legally attach claim or citation IDs
        to prose in that section.  Some providers nevertheless emit a
        sentence with both ID lists empty.  That sentence must not be repaired
        by borrowing unrelated evidence.  Remove only that exact, provably
        untraceable shape; partially linked statements and statements in a
        section with verified claims still fail normal validation.
        """

        empty_section_ids = {
            section.section_id
            for section in packet.sections
            if not section.verified_claim_ids
        }
        sections = value.get("sections")
        if not isinstance(sections, list):
            return
        for section in sections:
            if (
                not isinstance(section, dict)
                or section.get("section_id") not in empty_section_ids
            ):
                continue
            statements = section.get("statements")
            if not isinstance(statements, list):
                continue
            section["statements"] = [
                statement
                for statement in statements
                if not (
                    isinstance(statement, dict)
                    and not statement.get("claim_ids")
                    and not statement.get("citation_ids")
                )
            ]


class WriterTraceabilityError(ValueError):
    pass


class SynthesisWriterActionExecutor:
    """Validates traceability, renders Markdown, and persists one revision."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        artifact_store: ArtifactStore,
        reporting_store: SQLiteReportingStore,
        policy: ReportLoopPolicy,
        actor_id: str,
        clock=utc_now,
    ) -> None:
        self.evidence = evidence
        self.artifact_store = artifact_store
        self.reporting_store = reporting_store
        self.policy = policy
        self.actor_id = actor_id
        self.clock = clock
        self._validated_usage: dict[str, BudgetUsage] = {}

    async def execute(self, command: Command) -> RawObservation:
        if (
            command.kind != CommandKind.SYNTHESIZE
            or command.name != "writer.persist_revision"
        ):
            raise ValueError(
                "Writer executor accepts only writer.persist_revision"
            )
        started = self.clock()
        proposal = WriterDraftProposal.model_validate(
            command.arguments.get("proposal"),
            strict=False,
        )
        artifact_id = str(
            command.arguments.get("evidence_packet_artifact_id") or ""
        )
        payload = read_json_artifact(self.artifact_store, artifact_id)
        packet = WriterEvidencePacket.model_validate(
            payload.get("packet"),
            strict=False,
        )
        if (
            proposal.run_id != command.run_id
            or proposal.run_id != packet.run_id
            or proposal.report_id != packet.report_id
        ):
            raise WriterTraceabilityError(
                "Writer proposal, packet, and command identities differ"
            )
        prior = self.reporting_store.revisions(
            proposal.run_id,
            proposal.report_id,
        )
        validation_key = self._validation_key(proposal, packet, command.task_id)
        semantic_usage = self._validated_usage.pop(validation_key, None)
        if semantic_usage is None:
            semantic_usage = await self.validate_proposal(
                proposal,
                packet,
                command.task_id,
            )
            self._validated_usage.pop(validation_key, None)
        markdown, entries, used_citation_ids = self._render(proposal, packet)

        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "packet_id": packet.packet_id,
                    "markdown": markdown,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        revision_id = _stable_id(
            "report_revision",
            proposal.report_id,
            str(proposal.revision),
            fingerprint,
        )
        draft_artifact_id = _stable_id("artifact", revision_id, "draft")
        citation_artifact_id = _stable_id(
            "artifact",
            revision_id,
            "citations",
        )
        report_artifact_id = _stable_id("artifact", revision_id, "report")
        parent = prior[-1] if prior else None
        recorded_at = (
            parent.created_at + timedelta(microseconds=1)
            if parent is not None
            else packet.created_at
        )
        source_ids = tuple(
            dict.fromkeys(
                (
                    artifact_id,
                    *(
                        (parent.report_artifact_id,)
                        if parent is not None
                        else ()
                    ),
                )
            )
        )
        self.artifact_store.put_json(
            {
                "schema": "WriterDraftProposal@1",
                "proposal": proposal.model_dump(mode="json"),
                "packet_id": packet.packet_id,
                "traceability_validated": True,
                "semantic_support_validated": True,
            },
            redact=False,
            kind=ArtifactKind.REPORT,
            producer_id=self.actor_id,
            run_id=proposal.run_id,
            task_id=command.task_id,
            content_schema="WriterDraftProposal@1",
            source_artifact_ids=source_ids,
            artifact_id=draft_artifact_id,
            idempotency_key=f"writer-draft:{revision_id}",
        )
        citation_map = CitationMap(
            citation_map_id=_stable_id(
                "citation_map",
                revision_id,
            ),
            run_id=proposal.run_id,
            report_id=proposal.report_id,
            revision=proposal.revision,
            entries=entries,
            artifact_id=citation_artifact_id,
            created_at=recorded_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "CitationMap@1",
                "citation_map": citation_map.model_dump(mode="json"),
                "used_citation_ids": list(used_citation_ids),
            },
            redact=False,
            kind=ArtifactKind.CITATION_MAP,
            producer_id=self.actor_id,
            run_id=proposal.run_id,
            task_id=command.task_id,
            content_schema="CitationMap@1",
            source_artifact_ids=(artifact_id, draft_artifact_id),
            artifact_id=citation_artifact_id,
            idempotency_key=f"citation-map:{revision_id}",
        )
        self.artifact_store.put_text(
            markdown,
            redact=False,
            kind=ArtifactKind.REPORT,
            producer_id=self.actor_id,
            run_id=proposal.run_id,
            task_id=command.task_id,
            content_schema="ReportMarkdown@1",
            source_artifact_ids=(
                artifact_id,
                draft_artifact_id,
                citation_artifact_id,
            ),
            artifact_id=report_artifact_id,
            idempotency_key=f"report-revision:{revision_id}",
        )
        revision = ReportRevision(
            revision_id=revision_id,
            run_id=proposal.run_id,
            report_id=proposal.report_id,
            revision=proposal.revision,
            title=proposal.title,
            markdown=markdown,
            section_ids=tuple(item.section_id for item in proposal.sections),
            statement_ids=tuple(
                statement.statement_id
                for section in proposal.sections
                for statement in section.statements
            ),
            report_artifact_id=report_artifact_id,
            draft_artifact_id=draft_artifact_id,
            citation_map_artifact_id=citation_artifact_id,
            evidence_packet_artifact_id=artifact_id,
            parent_revision_id=(
                parent.revision_id if parent is not None else None
            ),
            usage=semantic_usage,
            created_at=recorded_at,
        )
        self.reporting_store.save_citation_map(citation_map)
        self.reporting_store.save_revision(revision)
        self._update_domain(
            proposal=proposal,
            packet=packet,
            revision=revision,
            entries=entries,
        )
        return RawObservation(
            status=ObservationStatus.SUCCEEDED.value,
            data={
                "semantic_complete": True,
                "revision_id": revision.revision_id,
                "revision": revision.revision,
                "report_artifact_id": report_artifact_id,
                "draft_artifact_id": draft_artifact_id,
                "citation_map_artifact_id": citation_artifact_id,
                "statement_count": len(revision.statement_ids),
                "citation_count": len(entries),
                "verified_only": True,
            },
            output_artifact_ids=(
                draft_artifact_id,
                citation_artifact_id,
                report_artifact_id,
            ),
            usage=semantic_usage,
            started_at=started,
            completed_at=self.clock(),
        )

    async def validate_proposal(
        self,
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
        task_id: str,
    ) -> BudgetUsage:
        if (
            proposal.run_id != packet.run_id
            or proposal.report_id != packet.report_id
        ):
            raise WriterTraceabilityError(
                "Writer proposal and verified packet identities differ"
            )
        prior = self.reporting_store.revisions(
            proposal.run_id,
            proposal.report_id,
        )
        expected_revision = len(prior) + 1
        if proposal.revision != expected_revision:
            raise WriterTraceabilityError(
                f"Writer revision must be {expected_revision}"
            )
        usage = await self._validate(proposal, packet, task_id)
        self._validated_usage[
            self._validation_key(proposal, packet, task_id)
        ] = usage
        return usage

    @staticmethod
    def _validation_key(
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
        task_id: str,
    ) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "proposal": proposal.model_dump(mode="json"),
                    "packet_id": packet.packet_id,
                    "task_id": task_id,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()

    async def _validate(
        self,
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
        task_id: str,
    ) -> BudgetUsage:
        packet_sections = {item.section_id: item for item in packet.sections}
        proposal_sections = {
            item.section_id: item for item in proposal.sections
        }
        if tuple(item.section_id for item in proposal.sections) != tuple(
            item.section_id
            for item in sorted(
                packet.sections,
                key=lambda value: (value.order, value.section_id),
            )
        ):
            raise WriterTraceabilityError(
                "Writer sections must exactly follow the report section order"
            )
        if set(proposal_sections) != set(packet_sections):
            raise WriterTraceabilityError(
                "Writer sections differ from the verified packet"
            )
        claims = {item.claim_id: item for item in packet.claims}
        citations = {item.citation_id: item for item in packet.citations}
        gaps = {item.claim_id: item for item in packet.gaps}
        conflicts = {item.conflict_id: item for item in packet.conflicts}
        statement_ids: set[str] = set()
        claim_renderings: dict[str, tuple[str, StatementCertainty]] = {}
        usage_items: list[BudgetUsage] = []
        for section_id, section in proposal_sections.items():
            packet_section = packet_sections[section_id]
            if section.title != packet_section.title:
                raise WriterTraceabilityError(
                    f"section title changed outside the report plan: {section_id}"
                )
            disclosed_gaps = {
                item.claim_id for item in section.gap_disclosures
            }
            if disclosed_gaps != set(packet_section.gap_claim_ids):
                raise WriterTraceabilityError(
                    f"section {section_id} must disclose every and only its gaps"
                )
            disclosed_conflicts = {
                item.conflict_id for item in section.conflict_disclosures
            }
            if disclosed_conflicts != set(packet_section.conflict_ids):
                raise WriterTraceabilityError(
                    f"section {section_id} must disclose every and only its conflicts"
                )
            covered_claims: set[str] = set()
            for statement in section.statements:
                if statement.statement_id in statement_ids:
                    raise WriterTraceabilityError(
                        "statement IDs must be unique across the report"
                    )
                statement_ids.add(statement.statement_id)
                allowed_claims = set(packet_section.verified_claim_ids)
                if not set(statement.claim_ids).issubset(allowed_claims):
                    raise WriterTraceabilityError(
                        f"statement {statement.statement_id} references a claim "
                        "outside its verified section boundary"
                    )
                self._validate_citations(statement, claims, citations)
                covered_claims.update(statement.claim_ids)
                for claim_id in statement.claim_ids:
                    rendering = (
                        " ".join(statement.text.split()).casefold(),
                        statement.certainty,
                    )
                    existing_rendering = claim_renderings.get(claim_id)
                    if (
                        existing_rendering is not None
                        and existing_rendering != rendering
                    ):
                        raise WriterTraceabilityError(
                            f"claim {claim_id} is rendered inconsistently across "
                            "report sections"
                        )
                    claim_renderings[claim_id] = rendering
                unresolved_conflict_claims = {
                    claim_id
                    for conflict_id in packet_section.conflict_ids
                    for claim_id in conflicts[conflict_id].claim_ids
                    if conflicts[conflict_id].resolution is None
                }
                if (
                    unresolved_conflict_claims.intersection(statement.claim_ids)
                    and statement.certainty != StatementCertainty.CONFLICTED
                ):
                    raise WriterTraceabilityError(
                        "statements touching unresolved conflicts must use "
                        "conflicted certainty"
                    )
                judgment = await self.evidence.engine.semantic_adapter.judge(
                    run_id=packet.run_id,
                    task_id=task_id,
                    subject_id=statement.statement_id,
                    statement=statement.text,
                    evidence_id=statement.claim_ids[0],
                    relation=EvidenceRelation.SUPPORTS,
                    passages=tuple(
                        claims[claim_id].statement
                        for claim_id in statement.claim_ids
                    ),
                )
                usage_items.append(judgment.usage)
                if (
                    judgment.label != SemanticLabel.SUPPORTS
                    or judgment.score
                    < self.evidence.engine.policy.support_threshold
                    or judgment.overreach_fragments
                ):
                    raise WriterTraceabilityError(
                        f"statement {statement.statement_id} is not semantically "
                        "supported by its declared verified claims"
                    )
            if covered_claims != set(packet_section.verified_claim_ids):
                missing = sorted(
                    set(packet_section.verified_claim_ids) - covered_claims
                )
                raise WriterTraceabilityError(
                    f"section {section_id} omits verified required claims: {missing}"
                )
            for disclosure in section.gap_disclosures:
                if disclosure.claim_id not in gaps:
                    raise WriterTraceabilityError(
                        "gap disclosure references an unknown gap"
                    )
            for disclosure in section.conflict_disclosures:
                conflict = conflicts.get(disclosure.conflict_id)
                if conflict is None:
                    raise WriterTraceabilityError(
                        "conflict disclosure references an unknown conflict"
                    )
                allowed_citations = {
                    item.citation_id
                    for item in packet.citations
                    if item.claim_id in conflict.claim_ids
                }
                if not set(disclosure.citation_ids).issubset(
                    allowed_citations
                ):
                    raise WriterTraceabilityError(
                        "conflict disclosure uses unrelated citations"
                    )
                verified_conflict_claims = set(conflict.claim_ids) & set(
                    claims
                )
                cited_conflict_claims = {
                    citations[citation_id].claim_id
                    for citation_id in disclosure.citation_ids
                }
                if not verified_conflict_claims.issubset(
                    cited_conflict_claims
                ):
                    raise WriterTraceabilityError(
                        "conflict disclosure must cite each verified claim "
                        "participating in the conflict"
                    )
        return _usage_sum(usage_items)

    def _validate_citations(
        self,
        statement: DraftStatement,
        claims: dict[str, Any],
        citations: dict[str, Any],
    ) -> None:
        unknown_claims = set(statement.claim_ids) - set(claims)
        unknown_citations = set(statement.citation_ids) - set(citations)
        if unknown_claims or unknown_citations:
            raise WriterTraceabilityError(
                "statement references unknown or unverified claims/citations"
            )
        for claim_id in statement.claim_ids:
            selected = [
                citations[citation_id]
                for citation_id in statement.citation_ids
                if citations[citation_id].claim_id == claim_id
            ]
            if not selected:
                raise WriterTraceabilityError(
                    f"statement {statement.statement_id} lacks a citation for "
                    f"claim {claim_id}"
                )
            source_count = len({item.source_id for item in selected})
            claim = claims[claim_id]
            required = (
                self.policy.high_impact_minimum_sources
                if claim.high_impact
                else self.policy.minimum_sources_per_statement
            )
            if source_count < required:
                raise WriterTraceabilityError(
                    f"claim {claim_id} requires {required} independently cited "
                    f"sources, got {source_count}"
                )
        if any(
            citations[citation_id].claim_id not in set(statement.claim_ids)
            for citation_id in statement.citation_ids
        ):
            raise WriterTraceabilityError(
                "statement cites evidence for an undeclared claim"
            )

    @staticmethod
    def _render(
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
    ) -> tuple[str, tuple[CitationMapEntry, ...], tuple[str, ...]]:
        citations = {item.citation_id: item for item in packet.citations}
        gaps = {item.claim_id: item for item in packet.gaps}
        conflicts = {item.conflict_id: item for item in packet.conflicts}
        marker_by_id: dict[str, str] = {}
        used: list[str] = []

        def markers(citation_ids: tuple[str, ...]) -> str:
            values: list[str] = []
            for citation_id in citation_ids:
                marker = marker_by_id.get(citation_id)
                if marker is None:
                    marker = f"[{len(marker_by_id) + 1}]"
                    marker_by_id[citation_id] = marker
                    used.append(citation_id)
                values.append(marker)
            return "".join(values)

        lines = [f"# {proposal.title}", "", f"> 研究问题：{packet.research_question}"]
        for section in proposal.sections:
            lines.extend(("", f"## {section.title}", ""))
            if (
                not section.statements
                and not section.gap_disclosures
                and not section.conflict_disclosures
            ):
                lines.extend((_EMPTY_VERIFIED_SECTION_NOTICE, ""))
            for statement in section.statements:
                text = statement.text.rstrip()
                punctuation = (
                    ""
                    if text.endswith((".", "。", "!", "！", "?", "？"))
                    else "。"
                )
                lines.append(
                    f"{text}{punctuation}{markers(statement.citation_ids)}"
                )
                lines.append("")
            if section.gap_disclosures:
                lines.append("### 证据缺口与不确定性")
                lines.append("")
                for disclosure in section.gap_disclosures:
                    gap = gaps[disclosure.claim_id]
                    lines.append(
                        f"- 尚无法确认“{gap.statement}”。{gap.reason}"
                    )
                lines.append("")
            if section.conflict_disclosures:
                lines.append("### 证据冲突")
                lines.append("")
                for disclosure in section.conflict_disclosures:
                    conflict: ConflictPacket = conflicts[
                        disclosure.conflict_id
                    ]
                    resolution = (
                        f" 当前处置：{conflict.resolution}"
                        if conflict.resolution
                        else " 当前证据不足以消解该冲突。"
                    )
                    suffix = markers(disclosure.citation_ids)
                    lines.append(
                        f"- {conflict.summary}{resolution}{suffix}"
                    )
                lines.append("")
        lines.extend(("## 引用", ""))
        entries: list[CitationMapEntry] = []
        for citation_id in used:
            citation = citations[citation_id]
            marker = marker_by_id[citation_id]
            lines.append(
                f"{marker} {citation.source_title}. "
                f"{citation.canonical_url}（{citation.locator}）"
            )
            entries.append(
                CitationMapEntry(
                    marker=marker,
                    citation_id=citation.citation_id,
                    claim_id=citation.claim_id,
                    evidence_id=citation.evidence_id,
                    source_id=citation.source_id,
                    canonical_url=citation.canonical_url,
                    locator=citation.locator,
                    quote=citation.quote,
                )
            )
        return "\n".join(lines).rstrip() + "\n", tuple(entries), tuple(used)

    def _update_domain(
        self,
        *,
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
        revision: ReportRevision,
        entries: tuple[CitationMapEntry, ...],
    ) -> None:
        repository = self.evidence.knowledge.repository
        report = repository.reports.require(proposal.report_id)
        if report.status == ReportStatus.PUBLISHED:
            raise WriterTraceabilityError("published reports are immutable")
        if report.status == ReportStatus.APPROVED:
            report = report.transition(ReportStatus.REVISION_REQUIRED)
        if report.status == ReportStatus.VERIFYING:
            report = report.transition(ReportStatus.REVISION_REQUIRED)
        if report.status == ReportStatus.FAILED:
            report = report.transition(ReportStatus.DRAFT)
        if report.status == ReportStatus.REVISION_REQUIRED:
            report = report.transition(ReportStatus.DRAFT)
        citation_ids_by_section = {
            section.section_id: tuple(
                dict.fromkeys(
                    citation_id
                    for statement in section.statements
                    for citation_id in statement.citation_ids
                )
            )
            for section in proposal.sections
        }
        packet_sections = {item.section_id: item for item in packet.sections}
        proposal_sections = {
            item.section_id: item for item in proposal.sections
        }
        for section_id in report.section_ids:
            section = repository.sections.require(section_id)
            if section.status in {
                SectionStatus.APPROVED,
                SectionStatus.VERIFIED,
            }:
                section = section.transition(SectionStatus.NEEDS_REPAIR)
            if section.status in {
                SectionStatus.PLANNED,
                SectionStatus.NEEDS_REPAIR,
            }:
                section = section.transition(SectionStatus.DRAFTING)
            packet_section = packet_sections[section_id]
            section_proposal = proposal_sections[section_id]
            section_markdown = "\n".join(
                [
                    f"## {section_proposal.title}",
                    *(
                        [
                            statement.text
                            for statement in section_proposal.statements
                        ]
                        or [_EMPTY_VERIFIED_SECTION_NOTICE]
                    ),
                ]
            )
            section_artifact_id = _stable_id(
                "artifact",
                revision.revision_id,
                section_id,
            )
            self.artifact_store.put_text(
                section_markdown,
                redact=False,
                kind=ArtifactKind.REPORT,
                producer_id=self.actor_id,
                run_id=proposal.run_id,
                content_schema="ReportSectionMarkdown@1",
                source_artifact_ids=(
                    revision.evidence_packet_artifact_id,
                    revision.draft_artifact_id,
                ),
                artifact_id=section_artifact_id,
                idempotency_key=(
                    f"report-section:{revision.revision_id}:{section_id}"
                ),
            )
            required_count = len(packet_section.required_claim_ids)
            supported_count = len(packet_section.verified_claim_ids)
            section = section.model_copy(
                update={
                    "claim_ids": packet_section.required_claim_ids,
                    "required_claim_ids": packet_section.required_claim_ids,
                    "citation_ids": citation_ids_by_section[section_id],
                    "content_artifact_id": section_artifact_id,
                    "coverage_score": (
                        supported_count / required_count
                        if required_count
                        else 1.0
                    ),
                    "citation_score": (
                        1.0
                        if all(
                            statement.citation_ids
                            for statement in section_proposal.statements
                        )
                        else 0.0
                    ),
                    "unsupported_claim_ids": packet_section.gap_claim_ids,
                    "conflicted_claim_ids": tuple(
                        dict.fromkeys(
                            claim_id
                            for conflict_id in packet_section.conflict_ids
                            for claim_id in next(
                                item
                                for item in packet.conflicts
                                if item.conflict_id == conflict_id
                            ).claim_ids
                            if claim_id in packet_section.required_claim_ids
                        )
                    ),
                    "updated_at": self.clock(),
                }
            )
            repository.sections.save(section)
        report = report.model_copy(
            update={
                "title": proposal.title,
                "content_artifact_id": revision.report_artifact_id,
                "version": proposal.revision,
                "updated_at": self.clock(),
            }
        )
        repository.reports.save(report)


class WriterRevisionVerifier:
    def __init__(
        self,
        *,
        reporting_store: SQLiteReportingStore,
        artifact_store: ArtifactStore,
    ) -> None:
        self.reporting_store = reporting_store
        self.artifact_store = artifact_store

    async def verify(
        self,
        *,
        spec: AgentSpec,
        task: TaskEnvelope,
        command: Command,
        observation: Observation,
        prior_observations: tuple[Observation, ...],
    ) -> VerificationFeedback:
        del spec, command, prior_observations
        if observation.status != ObservationStatus.SUCCEEDED:
            return VerificationFeedback(
                passed=False,
                summary="Writer revision execution failed.",
                repair_feedback=("Return a fully traceable Writer proposal.",),
            )
        revision_id = str(
            observation.normalized_data.get("revision_id") or ""
        )
        revisions = self.reporting_store.revisions(
            task.run_id,
            str(task.constraints["report_id"]),
        )
        revision = next(
            (item for item in revisions if item.revision_id == revision_id),
            None,
        )
        if revision is None:
            return VerificationFeedback(
                passed=False,
                summary="Writer revision was not durably journaled.",
                repair_feedback=("Persist the report revision atomically.",),
            )
        required = (
            revision.report_artifact_id,
            revision.draft_artifact_id,
            revision.citation_map_artifact_id,
            revision.evidence_packet_artifact_id,
        )
        if any(self.artifact_store.get(item) is None for item in required):
            return VerificationFeedback(
                passed=False,
                summary="Writer revision references a missing artifact.",
                repair_feedback=("Repair the report artifact graph.",),
            )
        citation_map = self.reporting_store.citation_map(
            revision.run_id,
            revision.report_id,
            revision.revision,
        )
        if citation_map is None:
            return VerificationFeedback(
                passed=False,
                summary="Writer revision is missing its citation map.",
                repair_feedback=("Persist a matching citation map.",),
            )
        return VerificationFeedback(
            passed=True,
            success=True,
            semantic_complete=True,
            information_gain=1.0,
            summary=(
                "Writer revision is verified-only, traceable, and durably "
                "persisted with a matching citation map."
            ),
        )


@dataclass(frozen=True)
class WriterRun:
    revision: ReportRevision
    usage: BudgetUsage


class SynthesisWriterRunner:
    def __init__(
        self,
        *,
        agent_spec_id: str,
        kernel: AgentKernel,
        reporting_store: SQLiteReportingStore,
    ) -> None:
        self.agent_spec_id = agent_spec_id
        self.kernel = kernel
        self.reporting_store = reporting_store

    async def write(
        self,
        *,
        packet: WriterEvidencePacket,
        title: str,
        revision: int,
        budget: Budget,
        repair_feedback: tuple[dict[str, Any], ...] = (),
        cancellation: CancellationToken | None = None,
    ) -> WriterRun:
        if packet.packet_artifact_id is None:
            raise ValueError("Writer packet must be persisted before writing")
        task = TaskEnvelope(
            task_id=_stable_id(
                "task",
                packet.report_id,
                str(revision),
                "synthesis",
            ),
            run_id=packet.run_id,
            kind=TaskKind.SYNTHESIS,
            status=TaskStatus.READY,
            title=f"Synthesize report revision {revision}",
            goal=(
                "Produce a complete report revision from the verified-only "
                "evidence packet."
            ),
            constraints={
                "report_id": packet.report_id,
                "report_title": title,
                "revision": revision,
                "evidence_packet_artifact_id": packet.packet_artifact_id,
                "repair_feedback": list(repair_feedback),
            },
            input_artifact_ids=(packet.packet_artifact_id,),
            expected_output_schema="ReportRevision@1",
            budget=budget,
            priority=0.9,
            max_attempts=3,
            created_by=self.agent_spec_id,
            assigned_actor_id=self.agent_spec_id,
            tags=("reporting", "writer", f"revision:{revision}"),
        )
        result = await self.kernel.run(
            agent_spec_id=self.agent_spec_id,
            task=task,
            cancellation=cancellation,
        )
        if result.task_result.status != TaskResultStatus.SUCCEEDED:
            raise RuntimeError(
                f"Writer failed: {result.task_result.summary}"
            )
        revisions = self.reporting_store.revisions(
            packet.run_id,
            packet.report_id,
        )
        value = next(
            (item for item in revisions if item.revision == revision),
            None,
        )
        if value is None:
            raise RuntimeError("Writer completed without a persisted revision")
        return WriterRun(revision=value, usage=result.task_result.usage)
