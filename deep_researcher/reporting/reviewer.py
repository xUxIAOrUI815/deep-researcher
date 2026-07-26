from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from pydantic import ValidationError

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    AgentSpec,
    ArtifactKind,
    Budget,
    BudgetUsage,
    Command,
    CommandKind,
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
from deep_researcher.evidence import EvidenceRuntime
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
    FindingSeverity,
    ReportLoopPolicy,
    ReportRepairAction,
    ReportRevision,
    ReviewActionKind,
    ReviewDimension,
    ReviewFinding,
    ReviewerDecision,
    RubricScore,
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


def _payload(response: ModelResponse) -> dict[str, Any]:
    value: Any = response.structured
    if isinstance(value, dict) and "decision" in value and isinstance(
        value["decision"],
        dict,
    ):
        value = value["decision"]
    if value is None:
        text = response.content.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("decision"), dict):
            value = value["decision"]
    if not isinstance(value, dict):
        raise ValueError("Reviewer must return a structured decision")
    return value


class ReportReviewerModelAdapter(ModelAdapter):
    """Injects immutable report context and repairs the full rubric output."""

    def __init__(
        self,
        model: ModelAdapter,
        *,
        artifact_store: ArtifactStore,
        reporting_store: SQLiteReportingStore,
        max_decision_repairs: int = 2,
    ) -> None:
        if max_decision_repairs < 0:
            raise ValueError("max_decision_repairs cannot be negative")
        self.model = model
        self.artifact_store = artifact_store
        self.reporting_store = reporting_store
        self.max_decision_repairs = max_decision_repairs

    async def complete(self, request: ModelRequest) -> ModelResponse:
        specialized = self._request(request)
        response = await self.model.complete(specialized)
        return await self._validated(specialized, response)

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
        return await self._validated(specialized, response)

    def _request(self, request: ModelRequest) -> ModelRequest:
        constraints = _constraints(request)
        revision_id = str(constraints.get("revision_id") or "")
        revision = self.reporting_store.revision(revision_id)
        if revision is None or revision.run_id != request.run_id:
            raise ModelInvocationError(
                "Reviewer task references an unknown report revision",
                retryable=False,
            )
        draft_payload = read_json_artifact(
            self.artifact_store,
            revision.draft_artifact_id,
        )
        proposal = WriterDraftProposal.model_validate(
            draft_payload.get("proposal"),
            strict=False,
        )
        packet_payload = read_json_artifact(
            self.artifact_store,
            revision.evidence_packet_artifact_id,
        )
        packet = WriterEvidencePacket.model_validate(
            packet_payload.get("packet"),
            strict=False,
        )
        citation_map = self.reporting_store.citation_map(
            revision.run_id,
            revision.report_id,
            revision.revision,
        )
        if citation_map is None:
            raise ModelInvocationError(
                "Reviewer task is missing the citation map",
                retryable=False,
            )
        review_message = {
            "role": "user",
            "content": {
                "schema": "ReportReviewContext@1",
                "report_revision": revision.model_dump(mode="json"),
                "writer_proposal": proposal.model_dump(mode="json"),
                "evidence_packet": packet.model_dump(mode="json"),
                "citation_map": citation_map.model_dump(mode="json"),
                "rubric_dimensions": [
                    item.value for item in ReviewDimension
                ],
                "allowed_decisions": [
                    item.value for item in ReviewActionKind
                ],
                "rules": {
                    "score_every_dimension_once": True,
                    "repairs_must_be_structured_and_bounded": True,
                    "do_not_change_evidence_status": True,
                    "accept_requires_full_traceability": True,
                },
            },
        }
        return ModelRequest(
            run_id=request.run_id,
            task_id=request.task_id,
            actor_id=request.actor_id,
            system=(
                "Act as the Report Reviewer. Review the immutable revision "
                "against all eight rubric dimensions: completeness, support, "
                "citation, conflicts, instruction following, depth, "
                "organization, and readability. Return one ReviewerDecision "
                "using only bounded targeted_research, citation_repair, "
                "local_rewrite, structural_rewrite, accept, or reject. Never "
                "change claim, evidence, citation, or conflict verification "
                "conclusions and never reveal hidden reasoning."
            ),
            messages=(*request.messages, review_message),
            command_schema=ReviewerDecision.model_json_schema(),
            model_version=request.model_version,
            prompt_version=request.prompt_version,
            max_output_tokens=request.max_output_tokens,
            metadata={
                **request.metadata,
                "structured_output": "ReviewerDecision@1",
                "revision_id": revision.revision_id,
            },
        )

    async def _validated(
        self,
        request: ModelRequest,
        response: ModelResponse,
    ) -> ModelResponse:
        current = response
        usage_items = [response.usage]
        for attempt in range(self.max_decision_repairs + 1):
            try:
                decision = self._parse(request, current)
            except (ValidationError, TypeError, ValueError, json.JSONDecodeError) as exc:
                if attempt >= self.max_decision_repairs:
                    raise ModelInvocationError(
                        "Reviewer decision remained invalid after bounded "
                        f"repair: {exc}",
                        retryable=False,
                    ) from exc
                current = await self.model.repair(
                    request,
                    current,
                    (str(exc),),
                )
                usage_items.append(
                    current.usage.model_copy(
                        update={"retries": current.usage.retries + 1}
                    )
                )
                continue
            return ModelResponse(
                content=decision.decision_summary,
                structured={
                    "summary": decision.decision_summary,
                    "commands": [
                        {
                            "kind": CommandKind.REVIEW.value,
                            "name": "reviewer.persist_decision",
                            "arguments": {
                                "decision": decision.model_dump(mode="json"),
                            },
                            "input_artifact_ids": list(
                                _constraints(request).get(
                                    "input_artifact_ids",
                                    (),
                                )
                            ),
                            "expected_output_schema": "ReviewerDecision@1",
                        }
                    ],
                    "reviewer_decision": decision.model_dump(mode="json"),
                },
                usage=_usage_sum(usage_items),
                latency_ms=current.latency_ms,
                finish_reason=current.finish_reason,
                response_id=current.response_id,
            )
        raise AssertionError("unreachable Reviewer repair loop")

    def _parse(
        self,
        request: ModelRequest,
        response: ModelResponse,
    ) -> ReviewerDecision:
        constraints = _constraints(request)
        value = _payload(response)
        value.update(
            run_id=request.run_id,
            report_id=str(constraints["report_id"]),
            revision_id=str(constraints["revision_id"]),
            review_artifact_id=None,
        )
        value.pop("review_id", None)
        value.pop("created_at", None)
        revision = self.reporting_store.revision(
            str(constraints["revision_id"])
        )
        if revision is None:
            raise ValueError("Reviewer revision disappeared during validation")
        fingerprint = hashlib.sha256(
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        value["review_id"] = _stable_id(
            "report_review",
            str(constraints["revision_id"]),
            fingerprint,
        )
        value["created_at"] = revision.created_at
        for index, finding in enumerate(value.get("findings", ())):
            if isinstance(finding, dict):
                finding["finding_id"] = _stable_id(
                    "finding",
                    value["review_id"],
                    str(index),
                    json.dumps(
                        finding,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
        for index, action in enumerate(value.get("repair_actions", ())):
            if isinstance(action, dict):
                action["action_id"] = _stable_id(
                    "repair_action",
                    value["review_id"],
                    str(index),
                    json.dumps(
                        action,
                        ensure_ascii=False,
                        sort_keys=True,
                        default=str,
                    ),
                )
        return ReviewerDecision.model_validate(value, strict=False)


@dataclass(frozen=True)
class DeterministicAudit:
    scores: tuple[RubricScore, ...]
    findings: tuple[ReviewFinding, ...]


class DeterministicReportAuditor:
    """Recomputes traceability and report invariants independently of the LLM."""

    def audit(
        self,
        *,
        revision: ReportRevision,
        proposal: WriterDraftProposal,
        packet: WriterEvidencePacket,
        citation_map: CitationMap,
        policy: ReportLoopPolicy,
    ) -> DeterministicAudit:
        findings: list[ReviewFinding] = []
        packet_sections = {item.section_id: item for item in packet.sections}
        proposal_sections = {
            item.section_id: item for item in proposal.sections
        }
        claims = {item.claim_id: item for item in packet.claims}
        citations = {item.citation_id: item for item in packet.citations}
        map_by_citation = {
            item.citation_id: item for item in citation_map.entries
        }
        map_by_marker = {item.marker: item for item in citation_map.entries}

        expected_order = tuple(
            item.section_id
            for item in sorted(
                packet.sections,
                key=lambda value: (value.order, value.section_id),
            )
        )
        actual_order = tuple(item.section_id for item in proposal.sections)
        if expected_order != actual_order or set(packet_sections) != set(
            proposal_sections
        ):
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.ORGANIZATION,
                    FindingSeverity.CRITICAL,
                    "Report section structure differs from the approved plan.",
                )
            )

        required_claims = {
            claim_id
            for section in packet.sections
            for claim_id in section.required_claim_ids
        }
        verified_required = {
            claim_id
            for section in packet.sections
            for claim_id in section.verified_claim_ids
        }
        covered_claims: set[str] = set()
        used_citations: list[str] = []
        citation_defects = 0
        support_defects = 0
        readability_penalties = 0
        high_impact_total = 0
        high_impact_deep = 0
        for section in proposal.sections:
            packet_section = packet_sections.get(section.section_id)
            if packet_section is None:
                continue
            if section.title != packet_section.title:
                findings.append(
                    self._finding(
                        revision,
                        ReviewDimension.INSTRUCTION_FOLLOWING,
                        FindingSeverity.HIGH,
                        "A section title changed outside the report plan.",
                        section_id=section.section_id,
                    )
                )
            for statement in section.statements:
                covered_claims.update(statement.claim_ids)
                if len(statement.text.strip()) < 12 or len(statement.text) > 1800:
                    readability_penalties += 1
                for claim_id in statement.claim_ids:
                    claim = claims.get(claim_id)
                    if claim is None:
                        support_defects += 1
                        continue
                    selected = [
                        citations[citation_id]
                        for citation_id in statement.citation_ids
                        if citation_id in citations
                        and citations[citation_id].claim_id == claim_id
                    ]
                    required_sources = (
                        policy.high_impact_minimum_sources
                        if claim.high_impact
                        else policy.minimum_sources_per_statement
                    )
                    source_count = len(
                        {item.source_id for item in selected}
                    )
                    if claim.high_impact:
                        high_impact_total += 1
                        if source_count >= required_sources:
                            high_impact_deep += 1
                    if not selected or source_count < required_sources:
                        support_defects += 1
                        findings.append(
                            self._finding(
                                revision,
                                ReviewDimension.SUPPORT,
                                FindingSeverity.CRITICAL,
                                "A factual statement lacks enough independently "
                                "verified support.",
                                section_id=section.section_id,
                                claim_ids=(claim_id,),
                                citation_ids=statement.citation_ids,
                            )
                        )
                for citation_id in statement.citation_ids:
                    used_citations.append(citation_id)
                    citation = citations.get(citation_id)
                    entry = map_by_citation.get(citation_id)
                    if (
                        citation is None
                        or entry is None
                        or entry.claim_id != citation.claim_id
                        or entry.evidence_id != citation.evidence_id
                        or entry.source_id != citation.source_id
                    ):
                        citation_defects += 1
                markers = "".join(
                    map_by_citation[item].marker
                    for item in statement.citation_ids
                    if item in map_by_citation
                )
                text = statement.text.rstrip()
                punctuation = (
                    ""
                    if text.endswith((".", "。", "!", "！", "?", "？"))
                    else "。"
                )
                rendered = f"{text}{punctuation}{markers}"
                if rendered not in revision.markdown:
                    citation_defects += 1
                    findings.append(
                        self._finding(
                            revision,
                            ReviewDimension.CITATION,
                            FindingSeverity.CRITICAL,
                            "Citation markers are not placed immediately after "
                            "their factual statement.",
                            section_id=section.section_id,
                            claim_ids=statement.claim_ids,
                            citation_ids=statement.citation_ids,
                        )
                    )

        omitted = verified_required - covered_claims
        if omitted:
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.COMPLETENESS,
                    FindingSeverity.CRITICAL,
                    "Verified required claims are missing from the report.",
                    claim_ids=tuple(sorted(omitted)),
                )
            )
        declared_gaps = {
            item.claim_id
            for section in proposal.sections
            for item in section.gap_disclosures
        }
        packet_gaps = {item.claim_id for item in packet.gaps}
        missing_gaps = packet_gaps - declared_gaps
        if missing_gaps:
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.COMPLETENESS,
                    FindingSeverity.CRITICAL,
                    "Evidence gaps are not explicitly disclosed.",
                    claim_ids=tuple(sorted(missing_gaps)),
                )
            )
        high_impact_gaps = {
            item.claim_id for item in packet.gaps if item.high_impact
        }
        if high_impact_gaps:
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.COMPLETENESS,
                    FindingSeverity.HIGH,
                    "High-impact evidence gaps require bounded targeted research.",
                    claim_ids=tuple(sorted(high_impact_gaps)),
                )
            )
        declared_conflicts = {
            item.conflict_id
            for section in proposal.sections
            for item in section.conflict_disclosures
        }
        packet_conflicts = {item.conflict_id for item in packet.conflicts}
        missing_conflicts = packet_conflicts - declared_conflicts
        if missing_conflicts:
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.CONFLICTS,
                    FindingSeverity.CRITICAL,
                    "Evidence conflicts are not explicitly presented.",
                )
            )

        used_unique = tuple(dict.fromkeys(used_citations))
        map_unique = tuple(item.citation_id for item in citation_map.entries)
        expected_markers = tuple(
            f"[{index}]" for index in range(1, len(map_unique) + 1)
        )
        if (
            used_unique != map_unique
            or tuple(item.marker for item in citation_map.entries)
            != expected_markers
            or len(map_by_marker) != len(citation_map.entries)
        ):
            citation_defects += 1
            findings.append(
                self._finding(
                    revision,
                    ReviewDimension.CITATION,
                    FindingSeverity.CRITICAL,
                    "Report citation usage and the citation map disagree.",
                    citation_ids=tuple(used_unique),
                )
            )

        total_required = len(required_claims)
        supported_count = len(verified_required & covered_claims)
        completeness = (
            supported_count / total_required if total_required else 1.0
        )
        scores = {
            ReviewDimension.COMPLETENESS: completeness,
            ReviewDimension.SUPPORT: 0.0 if support_defects else 1.0,
            ReviewDimension.CITATION: 0.0 if citation_defects else 1.0,
            ReviewDimension.CONFLICTS: 0.0 if missing_conflicts else 1.0,
            ReviewDimension.INSTRUCTION_FOLLOWING: (
                0.0
                if any(
                    item.dimension == ReviewDimension.INSTRUCTION_FOLLOWING
                    for item in findings
                )
                else 1.0
            ),
            ReviewDimension.DEPTH: (
                high_impact_deep / high_impact_total
                if high_impact_total
                else 1.0
            ),
            ReviewDimension.ORGANIZATION: (
                0.0 if expected_order != actual_order else 1.0
            ),
            ReviewDimension.READABILITY: (
                max(
                    0.0,
                    1.0
                    - readability_penalties
                    / max(
                        1,
                        sum(
                            len(item.statements)
                            for item in proposal.sections
                        ),
                    ),
                )
            ),
        }
        return DeterministicAudit(
            scores=tuple(
                RubricScore(
                    dimension=dimension,
                    score=scores[dimension],
                    rationale=(
                        "Deterministic audit recomputed this dimension from "
                        "the persisted report, proposal, packet, and citation map."
                    ),
                )
                for dimension in ReviewDimension
            ),
            findings=tuple(findings),
        )

    @staticmethod
    def _finding(
        revision: ReportRevision,
        dimension: ReviewDimension,
        severity: FindingSeverity,
        message: str,
        *,
        section_id: str | None = None,
        claim_ids: tuple[str, ...] = (),
        citation_ids: tuple[str, ...] = (),
    ) -> ReviewFinding:
        return ReviewFinding(
            finding_id=_stable_id(
                "finding",
                revision.revision_id,
                dimension.value,
                severity.value,
                message,
                section_id or "",
                *claim_ids,
                *citation_ids,
            ),
            dimension=dimension,
            severity=severity,
            message=message,
            section_id=section_id,
            claim_ids=claim_ids,
            citation_ids=citation_ids,
        )


class ReportReviewerActionExecutor:
    """Combines model review with non-overridable deterministic checks."""

    def __init__(
        self,
        *,
        evidence: EvidenceRuntime,
        artifact_store: ArtifactStore,
        reporting_store: SQLiteReportingStore,
        policy: ReportLoopPolicy,
        actor_id: str,
        auditor: DeterministicReportAuditor | None = None,
        clock=utc_now,
    ) -> None:
        self.evidence = evidence
        self.artifact_store = artifact_store
        self.reporting_store = reporting_store
        self.policy = policy
        self.actor_id = actor_id
        self.auditor = auditor or DeterministicReportAuditor()
        self.clock = clock

    async def execute(self, command: Command) -> RawObservation:
        if (
            command.kind != CommandKind.REVIEW
            or command.name != "reviewer.persist_decision"
        ):
            raise ValueError(
                "Reviewer executor accepts only reviewer.persist_decision"
            )
        started = self.clock()
        model_decision = ReviewerDecision.model_validate(
            command.arguments.get("decision"),
            strict=False,
        )
        revision = self.reporting_store.revision(
            model_decision.revision_id
        )
        if (
            revision is None
            or revision.run_id != command.run_id
            or revision.report_id != model_decision.report_id
        ):
            raise ValueError("Reviewer decision references another revision")
        draft_payload = read_json_artifact(
            self.artifact_store,
            revision.draft_artifact_id,
        )
        proposal = WriterDraftProposal.model_validate(
            draft_payload.get("proposal"),
            strict=False,
        )
        packet_payload = read_json_artifact(
            self.artifact_store,
            revision.evidence_packet_artifact_id,
        )
        packet = WriterEvidencePacket.model_validate(
            packet_payload.get("packet"),
            strict=False,
        )
        citation_map = self.reporting_store.citation_map(
            revision.run_id,
            revision.report_id,
            revision.revision,
        )
        if citation_map is None:
            raise ValueError("Reviewer cannot evaluate a missing citation map")
        self._validate_model_references(
            model_decision,
            packet,
        )
        audit = self.auditor.audit(
            revision=revision,
            proposal=proposal,
            packet=packet,
            citation_map=citation_map,
            policy=self.policy,
        )
        decision = self._final_decision(
            model_decision=model_decision,
            audit=audit,
            revision=revision,
            packet=packet,
        )
        artifact_id = _stable_id(
            "artifact",
            decision.review_id,
        )
        decision = decision.model_copy(
            update={
                "review_artifact_id": artifact_id,
                "created_at": revision.created_at,
            }
        )
        self.artifact_store.put_json(
            {
                "schema": "ReviewerDecision@1",
                "decision": decision.model_dump(mode="json"),
                "model_decision": model_decision.model_dump(mode="json"),
                "deterministic_audit": {
                    "scores": [
                        item.model_dump(mode="json")
                        for item in audit.scores
                    ],
                    "findings": [
                        item.model_dump(mode="json")
                        for item in audit.findings
                    ],
                    "model_cannot_override_defects": True,
                },
                "evidence_mutated": False,
            },
            redact=False,
            kind=ArtifactKind.REPORT_REVIEW,
            producer_id=self.actor_id,
            run_id=revision.run_id,
            task_id=command.task_id,
            content_schema="ReviewerDecision@1",
            source_artifact_ids=(
                revision.report_artifact_id,
                revision.draft_artifact_id,
                revision.citation_map_artifact_id,
                revision.evidence_packet_artifact_id,
            ),
            artifact_id=artifact_id,
            idempotency_key=f"report-review:{decision.review_id}",
        )
        self.reporting_store.save_review(decision)
        self._update_domain(decision)
        return RawObservation(
            status=ObservationStatus.SUCCEEDED.value,
            data={
                "semantic_complete": True,
                "review_id": decision.review_id,
                "review_artifact_id": artifact_id,
                "revision_id": revision.revision_id,
                "decision": decision.decision.value,
                "repair_action_count": len(decision.repair_actions),
                "deterministic_findings": len(audit.findings),
                "evidence_mutated": False,
            },
            output_artifact_ids=(artifact_id,),
            usage=decision.usage,
            started_at=started,
            completed_at=self.clock(),
        )

    @staticmethod
    def _validate_model_references(
        decision: ReviewerDecision,
        packet: WriterEvidencePacket,
    ) -> None:
        section_ids = {item.section_id for item in packet.sections}
        claim_ids = {
            *[item.claim_id for item in packet.claims],
            *[item.claim_id for item in packet.gaps],
        }
        citation_ids = {item.citation_id for item in packet.citations}
        for finding in decision.findings:
            if finding.section_id and finding.section_id not in section_ids:
                raise ValueError("Reviewer finding references an unknown section")
            if not set(finding.claim_ids).issubset(claim_ids):
                raise ValueError("Reviewer finding references an unknown claim")
            if not set(finding.citation_ids).issubset(citation_ids):
                raise ValueError(
                    "Reviewer finding references an unknown citation"
                )
        for action in decision.repair_actions:
            if not set(action.section_ids).issubset(section_ids):
                raise ValueError("Reviewer repair references an unknown section")
            if not set(action.claim_ids).issubset(claim_ids):
                raise ValueError("Reviewer repair references an unknown claim")
            if not set(action.citation_ids).issubset(citation_ids):
                raise ValueError("Reviewer repair references an unknown citation")

    def _final_decision(
        self,
        *,
        model_decision: ReviewerDecision,
        audit: DeterministicAudit,
        revision: ReportRevision,
        packet: WriterEvidencePacket,
    ) -> ReviewerDecision:
        model_scores = {
            item.dimension: item for item in model_decision.scores
        }
        audit_scores = {item.dimension: item for item in audit.scores}
        scores = tuple(
            RubricScore(
                dimension=dimension,
                score=min(
                    model_scores[dimension].score,
                    audit_scores[dimension].score,
                ),
                rationale=(
                    f"Model: {model_scores[dimension].rationale} "
                    f"Deterministic: {audit_scores[dimension].rationale}"
                )[:2000],
            )
            for dimension in ReviewDimension
        )
        findings_by_id = {
            item.finding_id: item
            for item in (*model_decision.findings, *audit.findings)
        }
        findings = tuple(findings_by_id.values())
        score_by_dimension = {item.dimension: item.score for item in scores}
        critical = any(
            item.severity == FindingSeverity.CRITICAL for item in findings
        )
        thresholds_pass = (
            all(
                score >= self.policy.minimum_score
                for score in score_by_dimension.values()
            )
            and score_by_dimension[ReviewDimension.SUPPORT]
            >= self.policy.minimum_support_score
            and score_by_dimension[ReviewDimension.CITATION]
            >= self.policy.minimum_citation_score
            and not critical
        )
        if model_decision.decision == ReviewActionKind.REJECT:
            return model_decision.model_copy(
                update={
                    "scores": scores,
                    "findings": findings,
                    "repair_actions": (),
                    "decision_summary": (
                        "Reviewer rejected the revision after the full rubric "
                        "and deterministic audit."
                    ),
                }
            )
        if model_decision.decision == ReviewActionKind.ACCEPT and thresholds_pass:
            return model_decision.model_copy(
                update={
                    "scores": scores,
                    "findings": findings,
                    "repair_actions": (),
                    "decision_summary": (
                        "Revision passed every acceptance threshold and all "
                        "deterministic traceability checks."
                    ),
                }
            )

        action_kind = self._required_repair(
            scores=score_by_dimension,
            findings=findings,
            model_decision=model_decision,
            packet=packet,
        )
        compatible = tuple(
            item
            for item in model_decision.repair_actions
            if item.kind == action_kind
        )
        if compatible:
            repairs = compatible
        else:
            sections = tuple(
                dict.fromkeys(
                    item.section_id
                    for item in findings
                    if item.section_id is not None
                )
            )
            claims = tuple(
                dict.fromkeys(
                    claim_id
                    for item in findings
                    for claim_id in item.claim_ids
                )
            )
            citations = tuple(
                dict.fromkeys(
                    citation_id
                    for item in findings
                    for citation_id in item.citation_ids
                )
            )
            repairs = (
                ReportRepairAction(
                    action_id=_stable_id(
                        "repair_action",
                        revision.revision_id,
                        action_kind.value,
                    ),
                    kind=action_kind,
                    reason=(
                        "Deterministic review defects or rubric thresholds "
                        "require this bounded repair."
                    ),
                    section_ids=sections,
                    claim_ids=claims,
                    citation_ids=citations,
                    constraints={
                        "source_revision_id": revision.revision_id,
                        "bounded": True,
                    },
                ),
            )
        return model_decision.model_copy(
            update={
                "decision": action_kind,
                "scores": scores,
                "findings": findings,
                "repair_actions": repairs,
                "decision_summary": (
                    f"Revision requires bounded {action_kind.value} before it "
                    "can pass the report gate."
                ),
            }
        )

    def _required_repair(
        self,
        *,
        scores: dict[ReviewDimension, float],
        findings: tuple[ReviewFinding, ...],
        model_decision: ReviewerDecision,
        packet: WriterEvidencePacket,
    ) -> ReviewActionKind:
        high_impact_gap_ids = {
            item.claim_id for item in packet.gaps if item.high_impact
        }
        if high_impact_gap_ids or scores[ReviewDimension.COMPLETENESS] < (
            self.policy.minimum_score
        ):
            return ReviewActionKind.TARGETED_RESEARCH
        if scores[ReviewDimension.CITATION] < (
            self.policy.minimum_citation_score
        ):
            return ReviewActionKind.CITATION_REPAIR
        if (
            scores[ReviewDimension.ORGANIZATION] < self.policy.minimum_score
            or any(
                item.dimension == ReviewDimension.ORGANIZATION
                and item.severity
                in {FindingSeverity.HIGH, FindingSeverity.CRITICAL}
                for item in findings
            )
        ):
            return ReviewActionKind.STRUCTURAL_REWRITE
        if model_decision.decision in {
            ReviewActionKind.TARGETED_RESEARCH,
            ReviewActionKind.CITATION_REPAIR,
            ReviewActionKind.LOCAL_REWRITE,
            ReviewActionKind.STRUCTURAL_REWRITE,
        }:
            return model_decision.decision
        return ReviewActionKind.LOCAL_REWRITE

    def _update_domain(self, decision: ReviewerDecision) -> None:
        repository = self.evidence.knowledge.repository
        report = repository.reports.require(decision.report_id)
        if report.status == ReportStatus.DRAFT:
            report = report.transition(ReportStatus.VERIFYING)
        if report.status != ReportStatus.VERIFYING:
            raise ValueError(
                "Reviewer requires a draft report in verifying state"
            )
        quality_scores = {
            item.dimension.value: item.score for item in decision.scores
        }
        if decision.decision == ReviewActionKind.ACCEPT:
            report = report.transition(ReportStatus.APPROVED)
        elif decision.decision == ReviewActionKind.REJECT:
            report = report.transition(ReportStatus.FAILED)
        else:
            report = report.transition(ReportStatus.REVISION_REQUIRED)
        report = report.model_copy(
            update={
                "quality_scores": quality_scores,
                "updated_at": self.clock(),
            }
        )
        repository.reports.save(report)
        for section_id in report.section_ids:
            section = repository.sections.require(section_id)
            if decision.decision == ReviewActionKind.ACCEPT:
                if section.status == SectionStatus.NEEDS_REPAIR:
                    section = section.transition(SectionStatus.DRAFTING)
                if section.status == SectionStatus.DRAFTING:
                    section = section.transition(SectionStatus.VERIFIED)
                if section.status == SectionStatus.VERIFIED:
                    section = section.transition(SectionStatus.APPROVED)
            else:
                if section.status in {
                    SectionStatus.APPROVED,
                    SectionStatus.VERIFIED,
                }:
                    section = section.transition(SectionStatus.NEEDS_REPAIR)
                elif section.status == SectionStatus.DRAFTING:
                    section = section.transition(SectionStatus.NEEDS_REPAIR)
            repository.sections.save(
                section.model_copy(update={"updated_at": self.clock()})
            )


class ReviewerDecisionVerifier:
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
                summary="Report review execution failed.",
                repair_feedback=("Return a valid full-rubric decision.",),
            )
        review_id = str(observation.normalized_data.get("review_id") or "")
        reviews = self.reporting_store.reviews(
            task.run_id,
            str(task.constraints["report_id"]),
        )
        decision = next(
            (item for item in reviews if item.review_id == review_id),
            None,
        )
        if (
            decision is None
            or decision.review_artifact_id is None
            or self.artifact_store.get(decision.review_artifact_id) is None
        ):
            return VerificationFeedback(
                passed=False,
                summary="Reviewer decision was not durably persisted.",
                repair_feedback=("Persist the immutable review artifact.",),
            )
        return VerificationFeedback(
            passed=True,
            success=True,
            semantic_complete=True,
            information_gain=1.0,
            summary=(
                "Reviewer completed all eight rubric dimensions and persisted "
                "a bounded decision without mutating evidence."
            ),
        )


@dataclass(frozen=True)
class ReviewerRun:
    decision: ReviewerDecision
    usage: BudgetUsage


class ReportReviewerRunner:
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

    async def review(
        self,
        *,
        revision: ReportRevision,
        budget: Budget,
        cancellation: CancellationToken | None = None,
    ) -> ReviewerRun:
        task = TaskEnvelope(
            task_id=_stable_id(
                "task",
                revision.revision_id,
                "review",
            ),
            run_id=revision.run_id,
            kind=TaskKind.REVIEW,
            status=TaskStatus.READY,
            title=f"Review report revision {revision.revision}",
            goal=(
                "Evaluate every rubric dimension and issue one bounded report "
                "loop decision."
            ),
            constraints={
                "report_id": revision.report_id,
                "revision_id": revision.revision_id,
                "input_artifact_ids": [
                    revision.report_artifact_id,
                    revision.draft_artifact_id,
                    revision.citation_map_artifact_id,
                    revision.evidence_packet_artifact_id,
                ],
            },
            input_artifact_ids=(
                revision.report_artifact_id,
                revision.draft_artifact_id,
                revision.citation_map_artifact_id,
                revision.evidence_packet_artifact_id,
            ),
            expected_output_schema="ReviewerDecision@1",
            budget=budget,
            priority=0.95,
            max_attempts=3,
            created_by=self.agent_spec_id,
            assigned_actor_id=self.agent_spec_id,
            tags=("reporting", "reviewer", f"revision:{revision.revision}"),
        )
        result = await self.kernel.run(
            agent_spec_id=self.agent_spec_id,
            task=task,
            cancellation=cancellation,
        )
        if result.task_result.status != TaskResultStatus.SUCCEEDED:
            raise RuntimeError(
                f"Reviewer failed: {result.task_result.summary}"
            )
        reviews = self.reporting_store.reviews(
            revision.run_id,
            revision.report_id,
        )
        decision = next(
            (
                item
                for item in reviews
                if item.revision_id == revision.revision_id
            ),
            None,
        )
        if decision is None:
            raise RuntimeError(
                "Reviewer completed without a persisted decision"
            )
        return ReviewerRun(
            decision=decision,
            usage=result.task_result.usage,
        )
