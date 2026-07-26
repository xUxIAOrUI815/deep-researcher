from __future__ import annotations

from datetime import timedelta
import hashlib
import json
from statistics import fmean
from typing import Any
from urllib.parse import urlsplit

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    EvaluationMetric,
    MetricDirection,
)

from .models import (
    DeterministicEvaluationReport,
    EvaluationSnapshot,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _valid_url(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and bool(parsed.hostname)
        and parsed.username is None
        and parsed.password is None
    )


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


class DeterministicEvaluatorSuite:
    """Complete non-LLM metric suite for Evaluation Lab core."""

    VERSION = "1.0.0"

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        freshness_days: int = 730,
        evaluator_id: str = "evaluator_deterministic_core_1_0_0",
    ) -> None:
        if freshness_days < 1:
            raise ValueError("freshness_days must be positive")
        self.artifact_store = artifact_store
        self.store = store
        self.freshness_days = freshness_days
        self.evaluator_id = evaluator_id

    def evaluate(
        self,
        snapshot: EvaluationSnapshot,
    ) -> DeterministicEvaluationReport:
        metrics = self._metrics(snapshot)
        quality = self._quality_values(metrics)
        aggregate = fmean(quality) if quality else 0.0
        snapshot_fingerprint = hashlib.sha256(
            json.dumps(
                snapshot.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        report_id = _stable_id(
            "deterministic_evaluation",
            snapshot.snapshot_id,
            self.VERSION,
            snapshot_fingerprint,
        )
        artifact_id = _stable_id("artifact", report_id)
        report = DeterministicEvaluationReport(
            report_id=report_id,
            snapshot_id=snapshot.snapshot_id,
            evaluator_version=self.VERSION,
            metrics=metrics,
            aggregate_score=max(0.0, min(1.0, aggregate)),
            details_artifact_id=artifact_id,
            created_at=snapshot.created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "DeterministicEvaluationReport@1",
                "snapshot": snapshot.model_dump(mode="json"),
                "report": report.model_dump(mode="json"),
                "metric_methodology": self.methodology(),
                "llm_judge_used": False,
            },
            redact=False,
            kind=ArtifactKind.EVALUATION_RESULT,
            producer_id=self.evaluator_id,
            run_id=snapshot.run_id,
            content_schema="DeterministicEvaluationReport@1",
            source_artifact_ids=snapshot.output_artifact_ids,
            artifact_id=artifact_id,
            idempotency_key=f"deterministic-evaluation:{report_id}",
        )
        self.store.save_evaluation_report(report, run_id=snapshot.run_id)
        return report

    def _metrics(
        self,
        snapshot: EvaluationSnapshot,
    ) -> tuple[EvaluationMetric, ...]:
        sources = snapshot.sources
        citations = snapshot.citations
        sections = snapshot.sections
        source_ids = {item.source_id for item in sources}
        valid_source_urls = sum(
            1 for item in sources if _valid_url(item.canonical_url)
        )
        valid_citation_urls = sum(
            1 for item in citations if _valid_url(item.canonical_url)
        )
        total_urls = len(sources) + len(citations)
        valid_urls = valid_source_urls + valid_citation_urls
        canonical_urls = [
            item.canonical_url for item in sources if item.canonical_url
        ]
        citation_integrity = sum(
            1
            for item in citations
            if item.verified
            and item.source_id in source_ids
            and _valid_url(item.canonical_url)
            and bool(item.claim_id)
            and bool(item.evidence_id)
        )
        quote_grounded = sum(
            1
            for item in citations
            if _normalize_text(item.quote)
            and _normalize_text(item.quote)
            in _normalize_text(item.passage_text)
        )
        supported_claim_ids = {
            claim_id
            for section in sections
            for claim_id in section.supported_claim_ids
        }
        cited_claim_ids = {
            item.claim_id
            for item in citations
            if item.verified and item.used_in_report
        }
        required_claim_ids = {
            claim_id
            for section in sections
            for claim_id in section.required_claim_ids
        }
        unsupported_claim_ids = {
            claim_id
            for section in sections
            for claim_id in section.unsupported_claim_ids
        }
        covered_claim_ids = supported_claim_ids - unsupported_claim_ids
        used_citations = [
            item for item in citations if item.used_in_report
        ]
        positioned_citations = sum(
            1
            for item in used_citations
            if (
                item.marker is not None
                and snapshot.report_markdown.count(item.marker) == 1
                and snapshot.report_markdown.find(item.marker) > 0
                and not snapshot.report_markdown[
                    snapshot.report_markdown.find(item.marker) - 1
                ].isspace()
            )
        )
        reference_time = snapshot.created_at
        fresh_sources = sum(
            1
            for item in sources
            if (
                item.published_at or item.fetched_at
            ) is not None
            and reference_time
            - (item.published_at or item.fetched_at)
            <= timedelta(days=self.freshness_days)
        )
        publishers = {
            (item.publisher or item.domain).strip().casefold()
            for item in sources
            if (item.publisher or item.domain).strip()
        }
        domains = {
            item.domain.strip().casefold()
            for item in sources
            if item.domain.strip()
        }
        source_types = {
            item.source_type.strip().casefold()
            for item in sources
            if item.source_type.strip()
        }
        primary_sources = sum(
            1
            for item in sources
            if item.source_level.casefold() == "primary"
        )
        tool_calls = snapshot.tool_calls
        search_calls = [item for item in tool_calls if item.is_search]
        search_keys = [item.request_key for item in search_calls]
        redundant_searches = len(search_keys) - len(set(search_keys))
        invalid_tool_calls = sum(1 for item in tool_calls if not item.valid)
        evidence_count = sum(item.evidence_count for item in tool_calls)
        counters = snapshot.counters

        values: list[tuple[str, float, MetricDirection, str]] = [
            (
                "url_validity_rate",
                _ratio(valid_urls, total_urls),
                MetricDirection.HIGHER_IS_BETTER,
                "url_integrity_v1",
            ),
            (
                "url_uniqueness_rate",
                _ratio(len(set(canonical_urls)), len(canonical_urls)),
                MetricDirection.HIGHER_IS_BETTER,
                "url_integrity_v1",
            ),
            (
                "citation_reference_integrity_rate",
                _ratio(citation_integrity, len(citations)),
                MetricDirection.HIGHER_IS_BETTER,
                "citation_integrity_v1",
            ),
            (
                "citation_quote_grounding_rate",
                _ratio(quote_grounded, len(citations)),
                MetricDirection.HIGHER_IS_BETTER,
                "quote_grounding_v1",
            ),
            (
                "citation_completeness_rate",
                _ratio(
                    len(supported_claim_ids & cited_claim_ids),
                    len(supported_claim_ids),
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "citation_completeness_v1",
            ),
            (
                "citation_position_validity_rate",
                _ratio(positioned_citations, len(used_citations)),
                MetricDirection.HIGHER_IS_BETTER,
                "citation_position_v1",
            ),
            (
                "schema_validity_rate",
                1.0 if not snapshot.schema_errors else 0.0,
                MetricDirection.HIGHER_IS_BETTER,
                "schema_validation_v1",
            ),
            (
                "section_coverage_rate",
                _ratio(
                    len(required_claim_ids & covered_claim_ids),
                    len(required_claim_ids),
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "section_coverage_v1",
            ),
            (
                "source_type_count",
                float(len(source_types)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_profile_v1",
            ),
            (
                "source_type_diversity_rate",
                _ratio(len(source_types), len(sources)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_profile_v1",
            ),
            (
                "source_authority_mean",
                fmean(item.authority_score for item in sources)
                if sources
                else 0.0,
                MetricDirection.HIGHER_IS_BETTER,
                "source_authority_v1",
            ),
            (
                "source_freshness_rate",
                _ratio(fresh_sources, len(sources)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_freshness_v1",
            ),
            (
                "source_publisher_diversity_rate",
                _ratio(len(publishers), len(sources)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_diversity_v1",
            ),
            (
                "source_domain_diversity_rate",
                _ratio(len(domains), len(sources)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_diversity_v1",
            ),
            (
                "primary_source_share",
                _ratio(primary_sources, len(sources)),
                MetricDirection.HIGHER_IS_BETTER,
                "source_primary_share_v1",
            ),
            (
                "total_tokens",
                float(snapshot.usage.total_tokens),
                MetricDirection.LOWER_IS_BETTER,
                "resource_usage_v1",
            ),
            (
                "cost_usd",
                snapshot.usage.cost_usd,
                MetricDirection.LOWER_IS_BETTER,
                "resource_usage_v1",
            ),
            (
                "latency_ms",
                snapshot.latency_ms,
                MetricDirection.LOWER_IS_BETTER,
                "resource_usage_v1",
            ),
            (
                "failure_rate",
                _ratio(counters.failed_task_count, counters.task_count),
                MetricDirection.LOWER_IS_BETTER,
                "failure_recovery_v1",
            ),
            (
                "recovery_rate",
                _ratio(
                    counters.recovered_operation_count,
                    counters.retried_operation_count,
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "failure_recovery_v1",
            ),
            (
                "idempotency_compliance_rate",
                1.0
                - _ratio(
                    counters.idempotency_violation_count,
                    counters.idempotent_operation_count,
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "idempotency_v1",
            ),
            (
                "protocol_compliance_rate",
                1.0
                - _ratio(
                    counters.invalid_protocol_operation_count,
                    counters.protocol_operation_count,
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "protocol_compliance_v1",
            ),
            (
                "evidence_per_tool_call",
                _ratio(evidence_count, len(tool_calls)),
                MetricDirection.HIGHER_IS_BETTER,
                "trace_efficiency_v1",
            ),
            (
                "redundant_search_rate",
                _ratio(redundant_searches, len(search_calls)),
                MetricDirection.LOWER_IS_BETTER,
                "trace_efficiency_v1",
            ),
            (
                "trace_recovery_rate",
                _ratio(
                    sum(
                        1
                        for item in tool_calls
                        if item.recovered_after_retry
                    ),
                    counters.retried_operation_count,
                ),
                MetricDirection.HIGHER_IS_BETTER,
                "trace_recovery_v1",
            ),
            (
                "convergence_turns",
                float(counters.convergence_turns),
                MetricDirection.LOWER_IS_BETTER,
                "trace_convergence_v1",
            ),
            (
                "invalid_tool_call_rate",
                _ratio(invalid_tool_calls, len(tool_calls)),
                MetricDirection.LOWER_IS_BETTER,
                "trace_protocol_v1",
            ),
            (
                "budget_violation_count",
                float(counters.budget_violation_count),
                MetricDirection.LOWER_IS_BETTER,
                "trace_budget_v1",
            ),
        ]
        return tuple(
            EvaluationMetric(
                name=name,
                value=float(value),
                direction=direction,
                evaluator=evaluator,
            )
            for name, value, direction, evaluator in values
        )

    @staticmethod
    def _quality_values(
        metrics: tuple[EvaluationMetric, ...],
    ) -> tuple[float, ...]:
        raw_names = {
            "source_type_count",
            "total_tokens",
            "cost_usd",
            "latency_ms",
            "evidence_per_tool_call",
            "convergence_turns",
            "budget_violation_count",
        }
        output: list[float] = []
        for metric in metrics:
            if metric.name in raw_names:
                continue
            value = max(0.0, min(1.0, metric.value))
            output.append(
                value
                if metric.direction == MetricDirection.HIGHER_IS_BETTER
                else 1.0 - value
            )
        budget = next(
            item.value
            for item in metrics
            if item.name == "budget_violation_count"
        )
        output.append(1.0 if budget == 0 else 0.0)
        return tuple(output)

    def methodology(self) -> dict[str, Any]:
        return {
            "version": self.VERSION,
            "freshness_days": self.freshness_days,
            "url": "RFC-style http/https host and credential exclusion",
            "quote": "normalized exact containment in persisted passage text",
            "coverage": "supported required claims divided by required claims",
            "source_diversity": "unique publishers/domains divided by sources",
            "variance": "population variance for repeated Live Web metrics",
            "llm_judge": False,
        }
