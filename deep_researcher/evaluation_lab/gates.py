from __future__ import annotations

import hashlib
import json
from typing import Any

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    DatasetPurpose,
    DatasetSplit,
    EvaluationMetric,
    MetricDirection,
)
from deep_researcher.version_registry import (
    VersionLifecycleState,
    VersionRegistry,
)

from .gate_models import (
    GateApplicationRecord,
    GateCheck,
    GateEvaluationEvidence,
    GateOutcome,
    GateStage,
    ReleaseGateDecision,
    ReleaseGatePolicy,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


class ReleaseGateService:
    """Auditable policy evaluation and Version Registry transition service."""

    def __init__(
        self,
        *,
        store: SQLiteEvaluationStore,
        artifact_store: ArtifactStore,
        version_registry: VersionRegistry,
        producer_id: str = "runtime_release_gate",
        actor_id: str = "release_gate_service",
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.version_registry = version_registry
        self.producer_id = producer_id
        self.actor_id = actor_id

    def evaluate(
        self,
        *,
        stage: GateStage,
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> ReleaseGateDecision:
        candidate = self.version_registry.store.record(
            evidence.candidate_version_id
        )
        baseline = self.version_registry.store.record(
            evidence.baseline_version_id
        )
        if candidate is None or baseline is None:
            raise ValueError("gate versions must be registered")
        candidate_ref = candidate.manifest.version_ref
        baseline_ref = baseline.manifest.version_ref
        if (
            candidate_ref.kind != baseline_ref.kind
            or candidate_ref.name != baseline_ref.name
        ):
            raise ValueError(
                "gate candidate and baseline must be the same component"
            )
        replay_only = False
        if stage == GateStage.SELECTION:
            if candidate.state not in {
                VersionLifecycleState.CANDIDATE,
                VersionLifecycleState.REJECTED,
            }:
                raise ValueError("selection gate requires a candidate version")
            replay_only = candidate.state != VersionLifecycleState.CANDIDATE
        elif stage == GateStage.FINAL_PROMOTION:
            if candidate.state not in {
                VersionLifecycleState.CANDIDATE,
                VersionLifecycleState.PROMOTED,
                VersionLifecycleState.REJECTED,
            }:
                raise ValueError("promotion gate requires a candidate version")
            replay_only = candidate.state != VersionLifecycleState.CANDIDATE
        elif candidate.state not in {
            VersionLifecycleState.PROMOTED,
            VersionLifecycleState.ROLLED_BACK,
        }:
            raise ValueError("post-release gate requires the promoted version")
        rollback_target = None
        if stage == GateStage.POST_RELEASE:
            if evidence.rollback_target_version_id is None:
                raise ValueError("post-release gate requires rollback target")
            rollback_target = self.version_registry.store.record(
                evidence.rollback_target_version_id
            )
            replayed_rollback = (
                candidate.state == VersionLifecycleState.ROLLED_BACK
                and rollback_target is not None
                and rollback_target.state
                == VersionLifecycleState.PROMOTED
            )
            if (
                rollback_target is None
                or (
                    rollback_target.state
                    != VersionLifecycleState.SUPERSEDED
                    and not replayed_rollback
                )
            ):
                raise ValueError(
                    "post-release rollback target must be superseded"
                )
        accesses = self._validate_accesses(stage, evidence)
        semantic_results = tuple(
            self.store.semantic_result(item)
            for item in evidence.semantic_evaluation_ids
        )
        if any(item is None for item in semantic_results):
            raise ValueError("gate references missing semantic evaluations")
        semantic = tuple(
            item for item in semantic_results if item is not None
        )
        access_ids = {item.access_record_id for item in accesses}
        if (
            {item.dataset_access_record_id for item in semantic}
            != access_ids
            or any(
                item.subject_version_id != evidence.candidate_version_id
                for item in semantic
            )
        ):
            raise ValueError(
                "gate semantic evaluations do not cover its audited accesses"
            )
        calibration = self.store.judge_calibration(evidence.calibration_id)
        if calibration is None:
            raise ValueError("gate calibration record is missing")
        missing_artifacts = [
            item
            for item in evidence.evaluation_artifact_ids
            if self.artifact_store.get(item) is None
        ]
        if missing_artifacts:
            raise ValueError(
                f"gate evaluation artifacts are missing: {missing_artifacts}"
            )
        checks = [
            *self._metric_checks(policy, evidence),
            GateCheck(
                check_name="semantic_evaluations_pass",
                category="semantic",
                passed=(
                    all(item.passed for item in semantic)
                    if policy.require_semantic_pass
                    else True
                ),
                observed=all(item.passed for item in semantic),
                required=policy.require_semantic_pass,
                detail="All stage semantic evaluations must pass.",
            ),
            GateCheck(
                check_name="human_correlation",
                category="calibration",
                passed=(
                    calibration.accepted
                    and calibration.overall_human_correlation
                    >= policy.minimum_human_correlation
                ),
                observed=calibration.overall_human_correlation,
                required=policy.minimum_human_correlation,
                detail="Fixed judge panel must meet human correlation.",
            ),
            GateCheck(
                check_name="human_pass_agreement",
                category="calibration",
                passed=(
                    calibration.accepted
                    and calibration.pass_agreement_rate
                    >= policy.minimum_human_pass_agreement
                ),
                observed=calibration.pass_agreement_rate,
                required=policy.minimum_human_pass_agreement,
                detail="Judge/human pass votes must meet agreement.",
            ),
        ]
        passed = all(item.passed for item in checks)
        outcome = {
            GateStage.SELECTION: (
                GateOutcome.ADVANCE_TO_FINAL
                if passed
                else GateOutcome.REJECT
            ),
            GateStage.FINAL_PROMOTION: (
                GateOutcome.PROMOTE if passed else GateOutcome.REJECT
            ),
            GateStage.POST_RELEASE: (
                GateOutcome.KEEP if passed else GateOutcome.ROLLBACK
            ),
        }[stage]
        created_at = max(
            (
                *(item.created_at for item in semantic),
                calibration.created_at,
                policy.created_at,
            )
        )
        decision_material = {
            "stage": stage.value,
            "policy": policy.model_dump(mode="json"),
            "evidence": evidence.model_dump(mode="json"),
            "checks": [item.model_dump(mode="json") for item in checks],
            "outcome": outcome.value,
        }
        decision_id = _stable_id(
            "gate_decision",
            evidence.candidate_version_id,
            stage.value,
            _fingerprint(decision_material),
        )
        existing = self.store.gate_decision(decision_id)
        if existing is not None:
            self._apply(existing)
            return existing
        if replay_only:
            raise ValueError(
                "version state permits only replay of its existing gate decision"
            )
        artifact_id = _stable_id("artifact", decision_id)
        decision = ReleaseGateDecision(
            gate_decision_id=decision_id,
            policy_id=policy.policy_id,
            stage=stage,
            outcome=outcome,
            candidate_version_id=evidence.candidate_version_id,
            baseline_version_id=evidence.baseline_version_id,
            rollback_target_version_id=(
                evidence.rollback_target_version_id
                if rollback_target is not None
                else None
            ),
            checks=tuple(checks),
            passed=passed,
            failed_check_names=tuple(
                item.check_name for item in checks if not item.passed
            ),
            semantic_evaluation_ids=evidence.semantic_evaluation_ids,
            dataset_access_record_ids=evidence.dataset_access_record_ids,
            calibration_id=evidence.calibration_id,
            result_artifact_id=artifact_id,
            created_at=created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "ReleaseGateDecision@1",
                "decision": decision.model_dump(mode="json"),
                "policy": policy.model_dump(mode="json"),
                "evaluation_artifact_ids": (
                    evidence.evaluation_artifact_ids
                ),
                "generates_changes": False,
            },
            redact=False,
            kind=ArtifactKind.RELEASE_GATE_DECISION,
            producer_id=self.producer_id,
            run_id=_stable_id("run", decision_id),
            content_schema="ReleaseGateDecision@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"release-gate:{decision_id}",
        )
        self.store.save_gate_decision(decision)
        self._apply(decision)
        return decision

    def _validate_accesses(
        self,
        stage: GateStage,
        evidence: GateEvaluationEvidence,
    ):
        accesses = tuple(
            self.store.dataset_access(item)
            for item in evidence.dataset_access_record_ids
        )
        if any(item is None for item in accesses):
            raise ValueError("gate references missing dataset access")
        concrete = tuple(item for item in accesses if item is not None)
        if any(
            item.request.actor_id != evidence.candidate_version_id
            for item in concrete
        ):
            raise ValueError("gate dataset access actor is not the candidate")
        observed = {
            (item.request.split, item.request.purpose)
            for item in concrete
        }
        if stage == GateStage.SELECTION:
            required = {
                (
                    DatasetSplit.SELECTION,
                    DatasetPurpose.CANDIDATE_SELECTION,
                )
            }
            if observed != required:
                raise ValueError(
                    "selection gate may access only the selection split"
                )
        else:
            required = {
                (
                    DatasetSplit.TEST,
                    DatasetPurpose.FINAL_EVALUATION,
                ),
                (
                    DatasetSplit.HIDDEN_TEST,
                    DatasetPurpose.RELEASE_GATE,
                ),
            }
            if observed != required:
                raise ValueError(
                    "final/post-release gate requires exactly test and "
                    "hidden-test final access"
                )
        return concrete

    def _metric_checks(
        self,
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> tuple[GateCheck, ...]:
        baseline = {item.name: item for item in evidence.baseline_metrics}
        candidate = {item.name: item for item in evidence.candidate_metrics}
        checks: list[GateCheck] = []
        for name, required in sorted(policy.required_improvements.items()):
            left, right = self._metric_pair(name, baseline, candidate)
            improvement = (
                right.value - left.value
                if right.direction == MetricDirection.HIGHER_IS_BETTER
                else left.value - right.value
            )
            checks.append(
                GateCheck(
                    check_name=f"required_improvement:{name}",
                    category="improvement",
                    passed=improvement >= required,
                    observed=improvement,
                    required=required,
                    detail=f"{name} must improve in its declared direction.",
                )
            )
        for name, tolerance in sorted(
            policy.non_regression_tolerances.items()
        ):
            left, right = self._metric_pair(name, baseline, candidate)
            regression = (
                left.value - right.value
                if right.direction == MetricDirection.HIGHER_IS_BETTER
                else right.value - left.value
            )
            checks.append(
                GateCheck(
                    check_name=f"non_regression:{name}",
                    category="non_regression",
                    passed=regression <= tolerance,
                    observed=max(0.0, regression),
                    required=tolerance,
                    detail=f"{name} regression must remain within tolerance.",
                )
            )
        baseline_cost = self._required_metric(baseline, "cost_usd")
        candidate_cost = self._required_metric(candidate, "cost_usd")
        cost_fraction = (
            (candidate_cost.value - baseline_cost.value)
            / baseline_cost.value
            if baseline_cost.value > 0
            else (0.0 if candidate_cost.value == 0 else None)
        )
        checks.extend(
            (
                GateCheck(
                    check_name="absolute_cost",
                    category="cost",
                    passed=candidate_cost.value <= policy.maximum_cost_usd,
                    observed=candidate_cost.value,
                    required=policy.maximum_cost_usd,
                    detail="Candidate absolute cost must remain bounded.",
                ),
                GateCheck(
                    check_name="relative_cost",
                    category="cost",
                    passed=(
                        cost_fraction is not None
                        and cost_fraction
                        <= policy.maximum_cost_increase_fraction
                    ),
                    observed=(
                        cost_fraction
                        if cost_fraction is not None
                        else "unbounded"
                    ),
                    required=policy.maximum_cost_increase_fraction,
                    detail="Candidate cost increase must remain bounded.",
                ),
            )
        )
        for name, maximum in sorted(policy.maximum_variances.items()):
            observed = evidence.candidate_variances.get(name)
            checks.append(
                GateCheck(
                    check_name=f"variance:{name}",
                    category="variance",
                    passed=observed is not None and observed <= maximum,
                    observed=observed if observed is not None else "missing",
                    required=maximum,
                    detail=f"{name} population variance must be bounded.",
                )
            )
        for name, minimum in sorted(policy.safety_minimums.items()):
            metric = candidate.get(name)
            checks.append(
                GateCheck(
                    check_name=f"safety:{name}",
                    category="safety",
                    passed=metric is not None and metric.value >= minimum,
                    observed=metric.value if metric is not None else "missing",
                    required=minimum,
                    detail=f"{name} must meet the safety minimum.",
                )
            )
        protocol = candidate.get("protocol_compliance_rate")
        checks.append(
            GateCheck(
                check_name="protocol_compliance",
                category="protocol",
                passed=(
                    protocol is not None
                    and protocol.value >= policy.protocol_minimum
                ),
                observed=(
                    protocol.value if protocol is not None else "missing"
                ),
                required=policy.protocol_minimum,
                detail="Protocol compliance must meet the release minimum.",
            )
        )
        return tuple(checks)

    @staticmethod
    def _metric_pair(
        name: str,
        baseline: dict[str, EvaluationMetric],
        candidate: dict[str, EvaluationMetric],
    ) -> tuple[EvaluationMetric, EvaluationMetric]:
        left = ReleaseGateService._required_metric(baseline, name)
        right = ReleaseGateService._required_metric(candidate, name)
        if left.direction != right.direction:
            raise ValueError(f"gate metric direction differs: {name}")
        return left, right

    @staticmethod
    def _required_metric(
        values: dict[str, EvaluationMetric],
        name: str,
    ) -> EvaluationMetric:
        if name not in values:
            raise ValueError(f"required gate metric is missing: {name}")
        return values[name]

    def _apply(
        self,
        decision: ReleaseGateDecision,
    ) -> GateApplicationRecord:
        existing = self.store.gate_application(
            decision.gate_decision_id
        )
        if existing is not None:
            return existing
        reason = (
            "Release gate passed."
            if decision.passed
            else "Release gate failed: "
            + ", ".join(decision.failed_check_names)
        )
        kwargs = {
            "gate_decision_id": decision.gate_decision_id,
            "gate_decision_artifact_id": decision.result_artifact_id,
            "reason": reason,
            "actor_id": self.actor_id,
            "occurred_at": decision.created_at,
        }
        if decision.outcome == GateOutcome.PROMOTE:
            self.version_registry.promote(
                decision.candidate_version_id,
                **kwargs,
            )
        elif decision.outcome == GateOutcome.REJECT:
            self.version_registry.reject(
                decision.candidate_version_id,
                **kwargs,
            )
        elif decision.outcome == GateOutcome.ROLLBACK:
            if decision.rollback_target_version_id is None:
                raise RuntimeError("rollback decision has no target")
            self.version_registry.rollback(
                decision.candidate_version_id,
                target_version_id=decision.rollback_target_version_id,
                **kwargs,
            )
        transition_ids = []
        version_ids = {
            decision.candidate_version_id,
            decision.baseline_version_id,
        }
        if decision.rollback_target_version_id is not None:
            version_ids.add(decision.rollback_target_version_id)
        for version_id in version_ids:
            record = self.version_registry.store.record(version_id)
            if record is None:
                continue
            transition_ids.extend(
                item.transition_id
                for item in record.transitions
                if item.gate_decision_id == decision.gate_decision_id
            )
        application_id = _stable_id(
            "gate_application",
            decision.gate_decision_id,
            decision.outcome.value,
            *sorted(transition_ids),
        )
        artifact_id = _stable_id("artifact", application_id)
        application = GateApplicationRecord(
            application_id=application_id,
            gate_decision_id=decision.gate_decision_id,
            outcome=decision.outcome,
            candidate_version_id=decision.candidate_version_id,
            version_transition_ids=tuple(sorted(transition_ids)),
            application_artifact_id=artifact_id,
            applied_at=decision.created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "GateApplicationRecord@1",
                "application": application.model_dump(mode="json"),
            },
            redact=False,
            kind=ArtifactKind.RELEASE_GATE_APPLICATION,
            producer_id=self.producer_id,
            run_id=_stable_id("run", decision.gate_decision_id),
            content_schema="GateApplicationRecord@1",
            source_artifact_ids=(decision.result_artifact_id,),
            artifact_id=artifact_id,
            idempotency_key=f"gate-application:{application_id}",
        )
        self.store.save_gate_application(application)
        return application
