from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
import random
from statistics import fmean, median, pvariance
from typing import Any, Protocol

from pydantic import ValidationError

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentKind,
    VersionRef,
    combine_usage,
    utc_now,
)
from deep_researcher.kernel import ModelAdapter, ModelRequest, ModelResponse

from .semantic_models import (
    BlindJudgeCandidate,
    BlindJudgeRequest,
    HumanRating,
    JudgeBallot,
    JudgeCalibrationRecord,
    JudgeCandidateConsensus,
    JudgeDimension,
    JudgeDisagreementRecord,
    JudgePanelResult,
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


class SemanticJudgeAdapter(Protocol):
    @property
    def judge_version_id(self) -> str: ...

    @property
    def rubric_version_id(self) -> str: ...

    async def judge(self, request: BlindJudgeRequest) -> JudgeBallot: ...


class ModelSemanticJudgeAdapter:
    """Fixed-version, no-tool semantic judge backed by ModelAdapter."""

    def __init__(
        self,
        *,
        model: ModelAdapter,
        judge_version: VersionRef,
        rubric_version: VersionRef,
        max_repairs: int = 2,
        clock=utc_now,
    ) -> None:
        if judge_version.kind != ComponentKind.MODEL:
            raise ValueError("judge_version must be a model VersionRef")
        if rubric_version.kind != ComponentKind.RUBRIC:
            raise ValueError("rubric_version must be a rubric VersionRef")
        if max_repairs < 0 or max_repairs > 5:
            raise ValueError("max_repairs must be between zero and five")
        self.model = model
        self.judge_version = judge_version
        self.rubric_version = rubric_version
        self.max_repairs = max_repairs
        self.clock = clock

    @property
    def judge_version_id(self) -> str:
        return self.judge_version.version_id

    @property
    def rubric_version_id(self) -> str:
        return self.rubric_version.version_id

    async def judge(self, request: BlindJudgeRequest) -> JudgeBallot:
        model_request = ModelRequest(
            run_id=_stable_id("run", request.panel_id),
            task_id=_stable_id(
                "task",
                request.panel_id,
                self.judge_version_id,
                request.blind_label,
            ),
            actor_id=self.judge_version_id,
            system=(
                "You are a blind report-quality evaluator. Evaluate only the "
                "provided candidate against the fixed rubric. Do not search, "
                "call tools, rewrite the report, infer system identity, or "
                "provide hidden reasoning. Return bounded scores, a pass vote, "
                "short violations, and a concise decision summary."
            ),
            messages=(
                {
                    "role": "user",
                    "content": {
                        "blind_label": request.blind_label,
                        "instruction": request.instruction,
                        "requirements": request.requirement_summary,
                        "verified_evidence_summary": request.evidence_summary,
                        "report": request.report_markdown,
                        "dimensions": [
                            item.value
                            for item in request.rubric_dimensions
                        ],
                    },
                },
            ),
            command_schema=self._schema(),
            model_version=self.judge_version.version,
            prompt_version=self.rubric_version.version,
            max_output_tokens=2000,
            metadata={
                "blind": True,
                "search_allowed": False,
                "rewrite_allowed": False,
                "judge_version_id": self.judge_version_id,
                "rubric_version_id": self.rubric_version_id,
            },
        )
        response = await self.model.complete(model_request)
        errors: tuple[str, ...] = ()
        for attempt in range(self.max_repairs + 1):
            try:
                return self._ballot(request, response)
            except (ValidationError, TypeError, ValueError, KeyError) as exc:
                errors = (str(exc),)
                if attempt >= self.max_repairs:
                    raise ValueError(
                        "semantic judge exhausted structured-output repairs"
                    ) from exc
                response = await self.model.repair(
                    model_request,
                    response,
                    errors,
                )
        raise AssertionError("unreachable judge repair path")

    def _ballot(
        self,
        request: BlindJudgeRequest,
        response: ModelResponse,
    ) -> JudgeBallot:
        if not isinstance(response.structured, dict):
            raise TypeError("judge response must be a structured object")
        raw = response.structured.get("ballot", response.structured)
        if not isinstance(raw, dict):
            raise TypeError("judge ballot must be an object")
        if not isinstance(raw["scores"], dict):
            raise TypeError("judge scores must be an object")
        scores = {
            JudgeDimension(key): float(value)
            for key, value in raw["scores"].items()
        }
        material = {
            "panel_id": request.panel_id,
            "blind_label": request.blind_label,
            "judge_version_id": self.judge_version_id,
            "rubric_version_id": self.rubric_version_id,
            "scores": scores,
            "passed": raw["passed"],
            "violations": raw.get("violations", ()),
            "rationale_summary": raw["rationale_summary"],
        }
        return JudgeBallot(
            ballot_id=_stable_id(
                "judge_ballot",
                request.panel_id,
                request.blind_label,
                self.judge_version_id,
                _fingerprint(material),
            ),
            panel_id=request.panel_id,
            blind_label=request.blind_label,
            judge_version_id=self.judge_version_id,
            rubric_version_id=self.rubric_version_id,
            scores=scores,
            passed=bool(raw["passed"]),
            violations=tuple(raw.get("violations", ())),
            rationale_summary=str(raw["rationale_summary"]),
            usage=response.usage,
            created_at=self.clock(),
        )

    @staticmethod
    def _schema() -> dict[str, Any]:
        score_properties = {
            item.value: {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
            }
            for item in JudgeDimension
        }
        return {
            "type": "object",
            "required": ["ballot"],
            "properties": {
                "ballot": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "scores",
                        "passed",
                        "violations",
                        "rationale_summary",
                    ],
                    "properties": {
                        "scores": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": list(score_properties),
                            "properties": score_properties,
                        },
                        "passed": {"type": "boolean"},
                        "violations": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "rationale_summary": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": 2000,
                        },
                    },
                }
            },
        }


class BlindMultiJudgePanel:
    """Blind randomized panel with majority voting and disagreement audit."""

    def __init__(
        self,
        *,
        judges: tuple[SemanticJudgeAdapter, ...],
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        disagreement_threshold: float = 0.25,
        producer_id: str = "runtime_blind_multi_judge_panel",
    ) -> None:
        if len(judges) < 3 or len(judges) % 2 == 0:
            raise ValueError(
                "judge panels require an odd count of at least three"
            )
        judge_ids = [item.judge_version_id for item in judges]
        if len(judge_ids) != len(set(judge_ids)):
            raise ValueError("judge panel versions must be unique")
        rubric_ids = {item.rubric_version_id for item in judges}
        if len(rubric_ids) != 1:
            raise ValueError("all judges must use one fixed rubric version")
        if disagreement_threshold < 0 or disagreement_threshold > 1:
            raise ValueError("disagreement threshold must be within [0, 1]")
        self.judges = judges
        self.artifact_store = artifact_store
        self.store = store
        self.disagreement_threshold = disagreement_threshold
        self.producer_id = producer_id

    async def evaluate(
        self,
        candidates: tuple[BlindJudgeCandidate, ...],
        *,
        randomization_seed: int,
    ) -> JudgePanelResult:
        if not candidates:
            raise ValueError("judge panel requires at least one candidate")
        candidate_ids = [item.candidate_id for item in candidates]
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError("judge panel candidate IDs must be unique")
        rubric_version_id = self.judges[0].rubric_version_id
        judge_ids = tuple(item.judge_version_id for item in self.judges)
        panel_id = _stable_id(
            "judge_panel",
            str(randomization_seed),
            rubric_version_id,
            *sorted(judge_ids),
            *sorted(candidate_ids),
        )
        blind_label_map = {
            self._blind_label(panel_id, candidate.candidate_id):
            candidate.candidate_id
            for candidate in candidates
        }
        candidate_by_label = {
            label: next(
                item
                for item in candidates
                if item.candidate_id == candidate_id
            )
            for label, candidate_id in blind_label_map.items()
        }
        orders: dict[str, tuple[str, ...]] = {}
        ballots: list[JudgeBallot] = []
        for judge in self.judges:
            labels = list(candidate_by_label)
            shuffle_seed = int.from_bytes(
                hashlib.sha256(
                    (
                        f"{randomization_seed}\0{panel_id}\0"
                        f"{judge.judge_version_id}"
                    ).encode("utf-8")
                ).digest()[:8],
                "big",
            )
            random.Random(shuffle_seed).shuffle(labels)
            orders[judge.judge_version_id] = tuple(labels)
            for label in labels:
                candidate = candidate_by_label[label]
                ballot = await judge.judge(
                    BlindJudgeRequest(
                        panel_id=panel_id,
                        blind_label=label,
                        report_markdown=candidate.report_markdown,
                        instruction=candidate.instruction,
                        evidence_summary=candidate.evidence_summary,
                        requirement_summary=candidate.requirement_summary,
                        rubric_dimensions=tuple(JudgeDimension),
                        randomization_seed=shuffle_seed,
                    )
                )
                if (
                    ballot.panel_id != panel_id
                    or ballot.blind_label != label
                    or ballot.judge_version_id
                    != judge.judge_version_id
                    or ballot.rubric_version_id != rubric_version_id
                ):
                    raise ValueError(
                        "judge ballot identity does not match its blind request"
                    )
                ballots.append(ballot)

        consensus: list[JudgeCandidateConsensus] = []
        disagreements: list[JudgeDisagreementRecord] = []
        for candidate in candidates:
            labels = tuple(
                label
                for label, candidate_id in blind_label_map.items()
                if candidate_id == candidate.candidate_id
            )
            candidate_ballots = tuple(
                item for item in ballots if item.blind_label in labels
            )
            pass_votes = sum(item.passed for item in candidate_ballots)
            scores = {
                dimension: float(
                    median(
                        item.scores[dimension]
                        for item in candidate_ballots
                    )
                )
                for dimension in JudgeDimension
            }
            consensus.append(
                JudgeCandidateConsensus(
                    candidate_id=candidate.candidate_id,
                    blind_labels=labels,
                    ballot_ids=tuple(
                        item.ballot_id for item in candidate_ballots
                    ),
                    consensus_scores=scores,
                    pass_votes=pass_votes,
                    total_votes=len(candidate_ballots),
                    passed=pass_votes > len(candidate_ballots) / 2,
                )
            )
            mixed = 0 < pass_votes < len(candidate_ballots)
            for dimension in JudgeDimension:
                judge_scores = {
                    item.judge_version_id: item.scores[dimension]
                    for item in candidate_ballots
                }
                score_range = max(judge_scores.values()) - min(
                    judge_scores.values()
                )
                if score_range >= self.disagreement_threshold or mixed:
                    disagreements.append(
                        JudgeDisagreementRecord(
                            disagreement_id=_stable_id(
                                "judge_disagreement",
                                panel_id,
                                candidate.candidate_id,
                                dimension.value,
                            ),
                            candidate_id=candidate.candidate_id,
                            dimension=dimension,
                            judge_scores=judge_scores,
                            score_range=score_range,
                            population_variance=pvariance(
                                judge_scores.values()
                            ),
                            mixed_pass_votes=mixed,
                        )
                    )
        material = {
            "panel_id": panel_id,
            "judge_version_ids": judge_ids,
            "rubric_version_id": rubric_version_id,
            "orders": orders,
            "blind_label_map": blind_label_map,
            "ballots": [
                item.model_dump(mode="json") for item in ballots
            ],
            "consensus": [
                item.model_dump(mode="json") for item in consensus
            ],
        }
        result_id = _stable_id(
            "judge_panel_result",
            panel_id,
            _fingerprint(material),
        )
        artifact_id = _stable_id("artifact", result_id)
        result = JudgePanelResult(
            panel_result_id=result_id,
            panel_id=panel_id,
            judge_version_ids=judge_ids,
            rubric_version_id=rubric_version_id,
            candidate_order_by_judge=orders,
            blind_label_map=blind_label_map,
            ballots=tuple(ballots),
            candidates=tuple(consensus),
            disagreements=tuple(disagreements),
            randomization_seed=randomization_seed,
            result_artifact_id=artifact_id,
            usage=combine_usage(item.usage for item in ballots),
            created_at=max(item.created_at for item in ballots),
        )
        self.artifact_store.put_json(
            {
                "schema": "JudgePanelResult@1",
                "result": result.model_dump(mode="json"),
                "candidate_identity_disclosed_to_judges": False,
                "candidate_order_randomized_per_judge": True,
                "search_allowed": False,
                "rewrite_allowed": False,
            },
            redact=False,
            kind=ArtifactKind.JUDGE_EVALUATION,
            producer_id=self.producer_id,
            run_id=_stable_id("run", panel_id),
            content_schema="JudgePanelResult@1",
            artifact_id=artifact_id,
            idempotency_key=f"judge-panel:{result_id}",
        )
        self.store.save_judge_panel(result)
        return result

    @staticmethod
    def _blind_label(panel_id: str, candidate_id: str) -> str:
        digest = hashlib.sha256(
            f"{panel_id}\0{candidate_id}".encode("utf-8")
        ).hexdigest()
        return f"candidate_{digest[:12]}"


class JudgeCalibrator:
    """Records panel-to-human Pearson/Spearman calibration."""

    def __init__(
        self,
        *,
        artifact_store: ArtifactStore,
        store: SQLiteEvaluationStore,
        producer_id: str = "runtime_judge_calibrator",
        clock=utc_now,
    ) -> None:
        self.artifact_store = artifact_store
        self.store = store
        self.producer_id = producer_id
        self.clock = clock

    def calibrate(
        self,
        *,
        panel_result_ids: tuple[str, ...],
        human_ratings: tuple[HumanRating, ...],
        minimum_correlation: float,
        minimum_pass_agreement: float,
    ) -> JudgeCalibrationRecord:
        panels = tuple(
            self.store.judge_panel(item) for item in panel_result_ids
        )
        if any(item is None for item in panels):
            raise ValueError("calibration references an unknown judge panel")
        concrete = tuple(item for item in panels if item is not None)
        versions = {item.judge_version_ids for item in concrete}
        rubrics = {item.rubric_version_id for item in concrete}
        if len(versions) != 1 or len(rubrics) != 1:
            raise ValueError(
                "calibration requires fixed judge and rubric versions"
            )
        consensus_by_candidate = {
            candidate.candidate_id: candidate
            for panel in concrete
            for candidate in panel.candidates
        }
        grouped_human: dict[str, list[HumanRating]] = defaultdict(list)
        for rating in human_ratings:
            if rating.candidate_id not in consensus_by_candidate:
                raise ValueError(
                    "human rating does not match a calibrated candidate"
                )
            grouped_human[rating.candidate_id].append(rating)
        if len(grouped_human) < 3:
            raise ValueError(
                "human calibration requires at least three rated candidates"
            )
        candidate_ids = tuple(sorted(grouped_human))
        human_scores = {
            candidate_id: {
                dimension: fmean(
                    item.scores[dimension]
                    for item in grouped_human[candidate_id]
                )
                for dimension in JudgeDimension
            }
            for candidate_id in candidate_ids
        }
        human_pass = {
            candidate_id: (
                sum(item.passed for item in grouped_human[candidate_id])
                > len(grouped_human[candidate_id]) / 2
            )
            for candidate_id in candidate_ids
        }
        pearson = {}
        spearman = {}
        mae = {}
        for dimension in JudgeDimension:
            judged = [
                consensus_by_candidate[item].consensus_scores[dimension]
                for item in candidate_ids
            ]
            human = [
                human_scores[item][dimension] for item in candidate_ids
            ]
            pearson[dimension] = self._pearson(judged, human)
            spearman[dimension] = self._spearman(judged, human)
            mae[dimension] = fmean(
                abs(left - right)
                for left, right in zip(judged, human, strict=True)
            )
        dimension_correlations = [
            (pearson[item] + spearman[item]) / 2
            for item in JudgeDimension
        ]
        overall = fmean(dimension_correlations)
        agreement = fmean(
            consensus_by_candidate[item].passed == human_pass[item]
            for item in candidate_ids
        )
        accepted = (
            overall >= minimum_correlation
            and agreement >= minimum_pass_agreement
        )
        material = {
            "judge_version_ids": next(iter(versions)),
            "rubric_version_id": next(iter(rubrics)),
            "panel_result_ids": sorted(panel_result_ids),
            "human_ratings": [
                item.model_dump(mode="json")
                for item in sorted(
                    human_ratings,
                    key=lambda value: value.rating_id,
                )
            ],
            "minimum_correlation": minimum_correlation,
            "minimum_pass_agreement": minimum_pass_agreement,
        }
        calibration_id = _stable_id(
            "judge_calibration",
            _fingerprint(material),
        )
        artifact_id = _stable_id("artifact", calibration_id)
        record = JudgeCalibrationRecord(
            calibration_id=calibration_id,
            judge_version_ids=next(iter(versions)),
            rubric_version_id=next(iter(rubrics)),
            panel_result_ids=tuple(sorted(set(panel_result_ids))),
            human_rating_ids=tuple(
                sorted(item.rating_id for item in human_ratings)
            ),
            sample_count=len(candidate_ids),
            pearson_by_dimension=pearson,
            spearman_by_dimension=spearman,
            mean_absolute_error_by_dimension=mae,
            overall_human_correlation=overall,
            pass_agreement_rate=agreement,
            minimum_correlation=minimum_correlation,
            minimum_pass_agreement=minimum_pass_agreement,
            accepted=accepted,
            result_artifact_id=artifact_id,
            created_at=self.clock(),
        )
        self.artifact_store.put_json(
            {
                "schema": "JudgeCalibrationRecord@1",
                "record": record.model_dump(mode="json"),
                "human_ratings": [
                    item.model_dump(mode="json") for item in human_ratings
                ],
            },
            redact=True,
            kind=ArtifactKind.JUDGE_CALIBRATION,
            producer_id=self.producer_id,
            run_id=_stable_id("run", calibration_id),
            content_schema="JudgeCalibrationRecord@1",
            # Calibration aggregates panel runs. Their immutable IDs remain in
            # the payload; ArtifactStore provenance edges are run-local.
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"judge-calibration:{calibration_id}",
        )
        self.store.save_judge_calibration(record)
        return record

    @staticmethod
    def _pearson(left: list[float], right: list[float]) -> float:
        left_mean = fmean(left)
        right_mean = fmean(right)
        numerator = sum(
            (x - left_mean) * (y - right_mean)
            for x, y in zip(left, right, strict=True)
        )
        denominator = math.sqrt(
            sum((x - left_mean) ** 2 for x in left)
            * sum((y - right_mean) ** 2 for y in right)
        )
        if denominator == 0:
            return 1.0 if left == right else 0.0
        return max(-1.0, min(1.0, numerator / denominator))

    @classmethod
    def _spearman(
        cls,
        left: list[float],
        right: list[float],
    ) -> float:
        return cls._pearson(cls._ranks(left), cls._ranks(right))

    @staticmethod
    def _ranks(values: list[float]) -> list[float]:
        output = [0.0] * len(values)
        ordered = sorted(range(len(values)), key=values.__getitem__)
        index = 0
        while index < len(ordered):
            end = index + 1
            while (
                end < len(ordered)
                and values[ordered[end]] == values[ordered[index]]
            ):
                end += 1
            rank = (index + 1 + end) / 2
            for position in ordered[index:end]:
                output[position] = rank
            index = end
        return output
