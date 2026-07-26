from __future__ import annotations

import hashlib
import json
from statistics import fmean

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ClaimStatus,
    EvaluationMetric,
    MetricDirection,
)

from .evaluators import DeterministicEvaluatorSuite
from .semantic_models import (
    JudgeDimension,
    SemanticEvaluationInput,
    SemanticEvaluationResult,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _ratio(numerator: float | int, denominator: float | int) -> float:
    return float(numerator / denominator) if denominator else 0.0


class SemanticEvaluationEngine:
    """Evidence/retrieval/report metrics joined to a calibrated judge panel."""

    DEFAULT_THRESHOLDS: dict[str, float] = {
        "claim_support_score": 0.75,
        "supported_claim_rate": 0.70,
        "contradicted_claim_rate": 0.0,
        "unsupported_claim_rate": 0.0,
        "uncited_fact_rate": 0.0,
        "retrieval_precision": 0.70,
        "retrieval_recall": 0.70,
        "retrieval_authority_mean": 0.60,
        "retrieval_freshness_rate": 0.50,
        "retrieval_diversity_rate": 0.40,
        "report_completeness_score": 0.70,
        "report_depth_score": 0.70,
        "report_instruction_following_score": 0.80,
        "report_organization_score": 0.70,
        "report_readability_score": 0.70,
        "report_safety_score": 0.90,
    }

    LOWER_IS_BETTER = {
        "contradicted_claim_rate",
        "unsupported_claim_rate",
        "uncited_fact_rate",
    }

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        deterministic_evaluator: DeterministicEvaluatorSuite,
        thresholds: dict[str, float] | None = None,
        producer_id: str = "runtime_semantic_evaluator",
    ) -> None:
        self.artifact_store = artifact_store
        self.store = store
        self.deterministic_evaluator = deterministic_evaluator
        self.thresholds = {
            **self.DEFAULT_THRESHOLDS,
            **(thresholds or {}),
        }
        if set(self.thresholds) != set(self.DEFAULT_THRESHOLDS):
            unknown = set(self.thresholds) - set(self.DEFAULT_THRESHOLDS)
            missing = set(self.DEFAULT_THRESHOLDS) - set(self.thresholds)
            raise ValueError(
                f"semantic thresholds differ; unknown={sorted(unknown)}, "
                f"missing={sorted(missing)}"
            )
        if any(value < 0.0 or value > 1.0 for value in self.thresholds.values()):
            raise ValueError("semantic thresholds must be within [0, 1]")
        self.producer_id = producer_id

    def evaluate(
        self,
        value: SemanticEvaluationInput,
    ) -> SemanticEvaluationResult:
        access = self.store.dataset_access(
            value.dataset_access_record_id
        )
        if access is None:
            raise ValueError(
                "semantic evaluation requires audited dataset access"
            )
        if (
            access.request.actor_id != value.subject_version_id
            or access.bundle_id != value.bundle_id
            or access.request.dataset_id != value.dataset_id
            or access.request.split != value.dataset_split
            or access.request.purpose != value.dataset_purpose
        ):
            raise ValueError(
                "semantic evaluation dataset access does not match its input"
            )
        panel = self.store.judge_panel(value.judge_panel_result_id)
        if panel is None:
            raise ValueError("semantic evaluation judge panel is missing")
        calibration = self.store.judge_calibration(value.calibration_id)
        if calibration is None:
            raise ValueError("semantic evaluation calibration is missing")
        if (
            calibration.judge_version_ids != panel.judge_version_ids
            or calibration.rubric_version_id != panel.rubric_version_id
        ):
            raise ValueError(
                "semantic evaluation panel is not covered by calibration"
            )
        consensus = next(
            (
                item
                for item in panel.candidates
                if item.candidate_id == value.judge_candidate_id
            ),
            None,
        )
        if consensus is None:
            raise ValueError(
                "semantic evaluation candidate is absent from judge panel"
            )
        deterministic = self.deterministic_evaluator.evaluate(
            value.snapshot
        )
        metrics = self._semantic_metrics(value, consensus.consensus_scores)
        selected_deterministic = {
            "url_validity_rate",
            "schema_validity_rate",
            "citation_position_validity_rate",
            "cost_usd",
            "source_freshness_rate",
        }
        deterministic_metrics = tuple(
            item
            for item in deterministic.metrics
            if item.name in selected_deterministic
        )
        all_metrics = (*metrics, *deterministic_metrics)
        failed = [
            item.name
            for item in metrics
            if item.passed is False
        ]
        if not calibration.accepted:
            failed.append("judge_human_calibration")
        if not consensus.passed:
            failed.append("judge_majority_vote")
        aggregate_values = [
            (
                item.value
                if item.direction == MetricDirection.HIGHER_IS_BETTER
                else 1.0 - item.value
            )
            for item in metrics
        ]
        aggregate = fmean(
            max(0.0, min(1.0, item)) for item in aggregate_values
        )
        material = {
            "input": value.model_dump(mode="json"),
            "deterministic_report_id": deterministic.report_id,
            "metrics": [
                item.model_dump(mode="json") for item in all_metrics
            ],
            "calibration_accepted": calibration.accepted,
            "judge_consensus_passed": consensus.passed,
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result_id = _stable_id(
            "semantic_evaluation",
            value.subject_version_id,
            value.dataset_id,
            value.snapshot.snapshot_id,
            fingerprint,
        )
        artifact_id = _stable_id("artifact", result_id)
        result = SemanticEvaluationResult(
            semantic_evaluation_id=result_id,
            subject_version_id=value.subject_version_id,
            bundle_id=value.bundle_id,
            dataset_id=value.dataset_id,
            dataset_split=value.dataset_split,
            dataset_purpose=value.dataset_purpose,
            dataset_access_record_id=value.dataset_access_record_id,
            snapshot_id=value.snapshot.snapshot_id,
            deterministic_report_id=deterministic.report_id,
            judge_panel_result_id=panel.panel_result_id,
            calibration_id=calibration.calibration_id,
            metrics=tuple(all_metrics),
            aggregate_score=aggregate,
            passed=not failed,
            failed_metric_names=tuple(dict.fromkeys(failed)),
            result_artifact_id=artifact_id,
            created_at=value.snapshot.created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "SemanticEvaluationResult@1",
                "result": result.model_dump(mode="json"),
                "thresholds": self.thresholds,
                "deterministic_metric_names": sorted(
                    selected_deterministic
                ),
                "judge_search_allowed": False,
                "judge_rewrite_allowed": False,
            },
            redact=False,
            kind=ArtifactKind.SEMANTIC_EVALUATION_RESULT,
            producer_id=self.producer_id,
            run_id=value.snapshot.run_id,
            content_schema="SemanticEvaluationResult@1",
            source_artifact_ids=value.snapshot.output_artifact_ids,
            artifact_id=artifact_id,
            idempotency_key=f"semantic-evaluation:{result_id}",
        )
        self.store.save_semantic_result(result)
        return result

    def _semantic_metrics(
        self,
        value: SemanticEvaluationInput,
        judge_scores: dict[JudgeDimension, float],
    ) -> tuple[EvaluationMetric, ...]:
        claims = value.claims
        claim_support = (
            fmean(item.support_score for item in claims)
            if claims
            else 0.0
        )
        supported = sum(
            item.status == ClaimStatus.SUPPORTED for item in claims
        )
        contradicted = sum(
            item.status == ClaimStatus.CONTRADICTED for item in claims
        )
        unsupported = sum(
            item.status == ClaimStatus.UNSUPPORTED for item in claims
        )
        factual = [item for item in claims if item.factual]
        uncited = sum(not item.cited for item in factual)
        cases = value.retrieval_cases
        precision_values = []
        recall_values = []
        diversity_values = []
        results = []
        for case in cases:
            retrieved = case.results
            relevant_retrieved = {
                item.source_id
                for item in retrieved
                if item.relevant
                and item.source_id in case.relevant_source_ids
            }
            precision_values.append(
                _ratio(len(relevant_retrieved), len(retrieved))
            )
            recall_values.append(
                _ratio(
                    len(relevant_retrieved),
                    len(case.relevant_source_ids),
                )
            )
            domain_diversity = _ratio(
                len({item.domain.casefold() for item in retrieved}),
                len(retrieved),
            )
            type_diversity = _ratio(
                len({item.source_type.casefold() for item in retrieved}),
                len(retrieved),
            )
            diversity_values.append(
                (domain_diversity + type_diversity) / 2
            )
            results.extend(retrieved)
        requirements = value.report_requirements
        topic_coverage = _ratio(
            len(requirements.covered_topics),
            len(requirements.required_topics),
        )
        instruction_coverage = _ratio(
            len(requirements.satisfied_instructions),
            len(requirements.required_instructions),
        )
        structural_depth = (
            min(1.0, requirements.word_count / requirements.target_word_count)
            + min(
                1.0,
                requirements.section_count
                / requirements.target_section_count,
            )
        ) / 2
        raw: dict[str, float] = {
            "claim_support_score": (
                claim_support
                + judge_scores[JudgeDimension.CLAIM_SUPPORT]
            )
            / 2,
            "supported_claim_rate": _ratio(supported, len(claims)),
            "contradicted_claim_rate": _ratio(
                contradicted,
                len(claims),
            ),
            "unsupported_claim_rate": _ratio(unsupported, len(claims)),
            "uncited_fact_rate": _ratio(uncited, len(factual)),
            "retrieval_precision": (
                fmean(precision_values) if precision_values else 0.0
            ),
            "retrieval_recall": (
                fmean(recall_values) if recall_values else 0.0
            ),
            "retrieval_authority_mean": (
                fmean(item.authority_score for item in results)
                if results
                else 0.0
            ),
            "retrieval_freshness_rate": _ratio(
                sum(item.fresh for item in results),
                len(results),
            ),
            "retrieval_diversity_rate": (
                fmean(diversity_values) if diversity_values else 0.0
            ),
            "report_completeness_score": (
                topic_coverage
                + judge_scores[JudgeDimension.COMPLETENESS]
            )
            / 2,
            "report_depth_score": (
                structural_depth + judge_scores[JudgeDimension.DEPTH]
            )
            / 2,
            "report_instruction_following_score": (
                instruction_coverage
                + judge_scores[JudgeDimension.INSTRUCTION_FOLLOWING]
            )
            / 2,
            "report_organization_score": judge_scores[
                JudgeDimension.ORGANIZATION
            ],
            "report_readability_score": judge_scores[
                JudgeDimension.READABILITY
            ],
            "report_safety_score": judge_scores[JudgeDimension.SAFETY],
        }
        metrics = []
        for name, score in raw.items():
            direction = (
                MetricDirection.LOWER_IS_BETTER
                if name in self.LOWER_IS_BETTER
                else MetricDirection.HIGHER_IS_BETTER
            )
            threshold = self.thresholds[name]
            passed = (
                score <= threshold
                if direction == MetricDirection.LOWER_IS_BETTER
                else score >= threshold
            )
            metrics.append(
                EvaluationMetric(
                    name=name,
                    value=score,
                    direction=direction,
                    threshold=threshold,
                    passed=passed,
                    evaluator="semantic_evaluator_1_0_0",
                )
            )
        return tuple(metrics)
