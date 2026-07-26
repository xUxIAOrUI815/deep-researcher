from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentKind,
    DatasetSplit,
    EvaluationMetric,
    MetricDirection,
    VersionRef,
    utc_now,
)
from deep_researcher.evaluation_lab import (
    GateEvaluationEvidence,
    GateOutcome,
    GateStage,
    ReleaseGateDecision,
    ReleaseGatePolicy,
    ReleaseGateService,
)
from deep_researcher.studio import StudioBadcase
from deep_researcher.version_registry import (
    VersionLifecycleState,
    VersionRegistry,
)

from .models import (
    BestSkillSnapshot,
    CandidatePoolEntry,
    CandidatePoolReview,
    CandidateStatus,
    CrossTaskExperience,
    EvolutionCampaignRecord,
    EvolutionCampaignRequest,
    EvolutionCampaignStatus,
    EvolutionCandidate,
    EvolutionHumanDecision,
    EvolutionInputSnapshot,
    EvolutionPatch,
    ExperienceEditSuggestion,
    HumanGateOutcome,
    OptimizationTarget,
    PoolEntryStatus,
    PoolSourceKind,
    RejectedEditMemory,
    ReviewedPoolEntry,
    TextEditBudget,
)
from .patching import (
    OfflinePatchGenerator,
    TextPatchApplier,
    TraceSignalPatchGenerator,
    canonical_hash,
    operation_fingerprint,
    patch_fingerprint,
)
from .store import EvolutionConflict, SQLiteEvolutionStore


_TRACE_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.TRACE_EXPORT,
        ArtifactKind.STUDIO_REPLAY_RESULT,
        ArtifactKind.FROZEN_REPLAY_RESULT,
        ArtifactKind.LIVE_WEB_RESULT,
    }
)
_EVALUATION_ARTIFACT_KINDS = frozenset(
    {
        ArtifactKind.EVALUATION_RESULT,
        ArtifactKind.SEMANTIC_EVALUATION_RESULT,
        ArtifactKind.JUDGE_EVALUATION,
        ArtifactKind.JUDGE_CALIBRATION,
        ArtifactKind.EXPERIMENT_RESULT,
        ArtifactKind.FROZEN_REPLAY_RESULT,
        ArtifactKind.LIVE_WEB_RESULT,
        ArtifactKind.RELEASE_GATE_DECISION,
    }
)
_COMPONENT_ARTIFACT_KINDS = {
    ComponentKind.SKILL: ArtifactKind.SKILL,
    ComponentKind.PROMPT: ArtifactKind.PROMPT,
    ComponentKind.TOOL_POLICY: ArtifactKind.POLICY,
    ComponentKind.STOP_POLICY: ArtifactKind.POLICY,
    ComponentKind.RUBRIC: ArtifactKind.RUBRIC,
}


def stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()
    return f"{prefix}_{digest[:32]}"


class OfflineEvolutionLab:
    """Bounded offline candidate generation and governed release workflow."""

    def __init__(
        self,
        *,
        store: SQLiteEvolutionStore,
        artifact_store: ArtifactStore,
        version_registry: VersionRegistry,
        release_gate: ReleaseGateService,
        generator: OfflinePatchGenerator | None = None,
        producer_id: str = "runtime_offline_evolution",
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.version_registry = version_registry
        self.release_gate = release_gate
        self.generator = generator or TraceSignalPatchGenerator()
        self.applier = TextPatchApplier()
        self.producer_id = producer_id

    def submit_candidate_source(
        self,
        *,
        source_kind: PoolSourceKind,
        source_artifact_id: str,
        submitted_by: str,
        source_run_id: str | None = None,
        source_span_id: str | None = None,
        evaluation_artifact_ids: tuple[str, ...] = (),
        score_metrics: tuple[EvaluationMetric, ...] = (),
        production_trace: bool = False,
    ) -> ReviewedPoolEntry:
        source = self._require_artifact(source_artifact_id)
        split_artifact_ids = [source_artifact_id]
        trace_kinds = {
            PoolSourceKind.SCORED_SUCCESS_TRACE,
            PoolSourceKind.SCORED_FAILURE_TRACE,
        }
        if source_kind in trace_kinds:
            if source.kind not in _TRACE_ARTIFACT_KINDS:
                raise ValueError(
                    "scored trace source has an unsupported artifact kind"
                )
            source_run_id = source_run_id or source.run_id
            if source_run_id != source.run_id:
                raise ValueError(
                    "scored trace source run differs from its artifact"
                )
        elif source_kind == PoolSourceKind.BADCASE:
            if source.kind != ArtifactKind.BADCASE:
                raise ValueError("badcase source must be a BADCASE artifact")
            badcase = self._read_badcase(source_artifact_id)
            source_run_id = source_run_id or badcase.source_run_id
            source_span_id = source_span_id or badcase.source_span_id
            if (
                source_run_id != badcase.source_run_id
                or source_span_id != badcase.source_span_id
            ):
                raise ValueError(
                    "badcase run/span differs from sealed provenance"
                )
            evaluation_artifact_ids = tuple(
                dict.fromkeys(
                    (
                        *evaluation_artifact_ids,
                        *badcase.evaluation_artifact_ids,
                    )
                )
            )
            split_artifact_ids.append(
                badcase.dataset_sample_artifact_id
            )
        elif source_kind == PoolSourceKind.EVALUATION:
            if source.kind not in _EVALUATION_ARTIFACT_KINDS:
                raise ValueError(
                    "evaluation source has an unsupported artifact kind"
                )
            if evaluation_artifact_ids:
                raise ValueError(
                    "evaluation source uses source_artifact_id directly"
                )
        else:
            raise ValueError(
                f"unsupported candidate source kind: {source_kind.value}"
            )
        for artifact_id in evaluation_artifact_ids:
            evaluation = self._require_artifact(artifact_id)
            if evaluation.kind not in _EVALUATION_ARTIFACT_KINDS:
                raise ValueError(
                    f"unsupported evaluation artifact: {artifact_id}"
                )
            split_artifact_ids.append(artifact_id)
        observed_splits = self._observed_dataset_splits(
            tuple(dict.fromkeys(split_artifact_ids))
        )
        material = {
            "source_kind": source_kind.value,
            "source_artifact_id": source_artifact_id,
            "source_run_id": source_run_id,
            "source_span_id": source_span_id,
            "evaluation_artifact_ids": evaluation_artifact_ids,
            "score_metrics": [
                item.model_dump(mode="json") for item in score_metrics
            ],
            "observed_dataset_splits": [
                item.value for item in observed_splits
            ],
            "production_trace": production_trace,
            "submitted_by": submitted_by,
        }
        entry_id = stable_id(
            "evolution_pool",
            canonical_hash(material),
        )
        existing = self.store.pool_entry(entry_id)
        if existing is not None:
            return existing
        artifact_id = stable_id("artifact", entry_id)
        entry = CandidatePoolEntry(
            pool_entry_id=entry_id,
            source_kind=source_kind,
            source_artifact_id=source_artifact_id,
            entry_artifact_id=artifact_id,
            source_run_id=source_run_id,
            source_span_id=source_span_id,
            evaluation_artifact_ids=evaluation_artifact_ids,
            score_metrics=score_metrics,
            observed_dataset_splits=observed_splits,
            production_trace=production_trace,
            submitted_by=submitted_by,
            submitted_at=source.created_at,
            invokes_optimizer=False,
            publishes_versions=False,
        )
        self.artifact_store.put_json(
            {
                "schema": "CandidatePoolEntry@1",
                "entry": entry.model_dump(mode="json"),
                "review_required": True,
                "invokes_optimizer": False,
                "publishes_versions": False,
                "production_trace_cannot_self_promote": True,
            },
            redact=False,
            kind=ArtifactKind.EVOLUTION_POOL_ENTRY,
            producer_id=self.producer_id,
            run_id=stable_id("run", "evolution_candidate_pool"),
            content_schema="CandidatePoolEntry@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"evolution-pool:{entry_id}",
            metadata={
                "source_artifact_id": source_artifact_id,
                "production_trace": production_trace,
            },
        )
        return self.store.submit_pool_entry(entry)

    def review_candidate_source(
        self,
        pool_entry_id: str,
        *,
        approved: bool,
        assigned_split: DatasetSplit | None,
        reviewer_id: str,
        review_note: str,
        reviewed_at: datetime | None = None,
    ) -> ReviewedPoolEntry:
        item = self.store.pool_entry(pool_entry_id)
        if item is None:
            raise KeyError(f"unknown candidate-pool entry: {pool_entry_id}")
        if item.review is not None:
            return item
        if approved and any(
            split not in {DatasetSplit.TRAIN, DatasetSplit.DEV}
            for split in item.entry.observed_dataset_splits
        ):
            raise ValueError(
                "selection/test/hidden-test source cannot be approved for "
                "offline candidate generation"
            )
        if (
            approved
            and item.entry.observed_dataset_splits
            and (
                len(item.entry.observed_dataset_splits) != 1
                or assigned_split
                != item.entry.observed_dataset_splits[0]
            )
        ):
            raise ValueError(
                "candidate source review cannot relabel its observed "
                "dataset split"
            )
        material = {
            "pool_entry_id": pool_entry_id,
            "approved": approved,
            "assigned_split": (
                assigned_split.value
                if assigned_split is not None
                else None
            ),
            "reviewer_id": reviewer_id,
            "review_note": review_note,
        }
        review_id = stable_id(
            "evolution_pool_review",
            canonical_hash(material),
        )
        artifact_id = stable_id("artifact", review_id)
        existing_artifact = self.artifact_store.get(artifact_id)
        if existing_artifact is not None:
            if existing_artifact.kind != ArtifactKind.EVOLUTION_POOL_REVIEW:
                raise ValueError(
                    "pool-review artifact ID belongs to another kind"
                )
            review = CandidatePoolReview.model_validate(
                self._read_json_payload(artifact_id)["review"],
                strict=False,
            )
            if (
                review.review_id != review_id
                or review.pool_entry_id != pool_entry_id
                or review.approved != approved
                or review.assigned_split != assigned_split
                or review.reviewer_id != reviewer_id
                or review.review_note != review_note
            ):
                raise ValueError(
                    "persisted pool review differs from retry request"
                )
        else:
            review = CandidatePoolReview(
                review_id=review_id,
                pool_entry_id=pool_entry_id,
                review_artifact_id=artifact_id,
                approved=approved,
                assigned_split=assigned_split,
                reviewer_id=reviewer_id,
                review_note=review_note,
                reviewed_at=reviewed_at or utc_now(),
                triggers_generation=False,
            )
            self.artifact_store.put_json(
                {
                    "schema": "CandidatePoolReview@1",
                    "review": review.model_dump(mode="json"),
                    "triggers_generation": False,
                    "triggers_publication": False,
                },
                redact=False,
                kind=ArtifactKind.EVOLUTION_POOL_REVIEW,
                producer_id=self.producer_id,
                run_id=stable_id("run", "evolution_candidate_pool"),
                content_schema="CandidatePoolReview@1",
                source_artifact_ids=(item.entry.entry_artifact_id,),
                artifact_id=artifact_id,
                idempotency_key=f"evolution-pool-review:{pool_entry_id}",
            )
        return self.store.review_pool_entry(review)

    def create_cross_task_experience(
        self,
        *,
        target: OptimizationTarget,
        summary: str,
        recommendation: str,
        suggestion: ExperienceEditSuggestion,
        confidence: float,
        impact: float,
        pool_entry_ids: tuple[str, ...],
        created_by: str,
    ) -> CrossTaskExperience:
        reviewed = tuple(
            self._require_approved_pool_entry(item)
            for item in pool_entry_ids
        )
        source_ids = tuple(
            dict.fromkeys(
                artifact_id
                for item in reviewed
                for artifact_id in (
                    item.entry.source_artifact_id,
                    *item.entry.evaluation_artifact_ids,
                )
            )
        )
        material = {
            "target": target.value,
            "summary": summary,
            "recommendation": recommendation,
            "suggestion": suggestion.model_dump(mode="json"),
            "confidence": confidence,
            "impact": impact,
            "pool_entry_ids": pool_entry_ids,
            "source_artifact_ids": source_ids,
            "created_by": created_by,
        }
        experience_id = stable_id(
            "evolution_experience",
            canonical_hash(material),
        )
        existing = self.store.experience(experience_id)
        if existing is not None:
            return existing
        artifact_id = stable_id("artifact", experience_id)
        created_at = max(
            item.review.reviewed_at
            for item in reviewed
            if item.review is not None
        )
        experience = CrossTaskExperience(
            experience_id=experience_id,
            target=target,
            summary=summary,
            recommendation=recommendation,
            suggestion=suggestion,
            confidence=confidence,
            impact=impact,
            pool_entry_ids=pool_entry_ids,
            source_artifact_ids=source_ids,
            created_by=created_by,
            artifact_id=artifact_id,
            created_at=created_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "CrossTaskExperience@1",
                "experience": experience.model_dump(mode="json"),
                "memory_layer": "cross_task_experience",
                "runtime_memory": False,
                "directly_deployable": False,
            },
            redact=False,
            kind=ArtifactKind.EVOLUTION_EXPERIENCE,
            producer_id=self.producer_id,
            run_id=stable_id("run", "evolution_experience_memory"),
            content_schema="CrossTaskExperience@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"evolution-experience:{experience_id}",
            metadata={"source_artifact_ids": list(source_ids)},
        )
        return self.store.save_experience(experience)

    def create_campaign(
        self,
        *,
        target: OptimizationTarget,
        component_kind: ComponentKind,
        component_name: str,
        base_version_id: str,
        pool_entry_ids: tuple[str, ...],
        experience_ids: tuple[str, ...],
        edit_budget: TextEditBudget,
        offline_environment_id: str,
        created_by: str,
    ) -> EvolutionCampaignRecord:
        base = self.version_registry.store.record(base_version_id)
        if base is None:
            raise KeyError(f"unknown base version: {base_version_id}")
        if base.state != VersionLifecycleState.PROMOTED:
            raise ValueError(
                "offline evolution must start from a promoted version"
            )
        base_ref = base.manifest.version_ref
        if (
            base_ref.kind != component_kind
            or base_ref.name != component_name
        ):
            raise ValueError(
                "campaign component differs from its base version"
            )
        active = self.version_registry.store.active(
            component_kind.value,
            component_name,
        )
        if (
            active is None
            or active.manifest.version_ref.version_id != base_version_id
        ):
            raise ValueError(
                "offline evolution must seal the currently active version"
            )
        reviewed = tuple(
            self._require_approved_pool_entry(item)
            for item in pool_entry_ids
        )
        observed_kinds = {item.entry.source_kind for item in reviewed}
        required_kinds = set(PoolSourceKind)
        if not required_kinds.issubset(observed_kinds):
            missing = sorted(
                item.value for item in required_kinds - observed_kinds
            )
            raise ValueError(
                "campaign requires scored success/failure traces, badcases, "
                f"and evaluations; missing {missing}"
            )
        experiences = tuple(
            self._require_experience(item) for item in experience_ids
        )
        if any(item.target != target for item in experiences):
            raise ValueError(
                "campaign experiences must match its optimization target"
            )
        allowed_pool_ids = {item.entry.pool_entry_id for item in reviewed}
        if any(
            not set(item.pool_entry_ids).issubset(allowed_pool_ids)
            for item in experiences
        ):
            raise ValueError(
                "campaign experience references an unsealed pool source"
            )
        material = {
            "target": target.value,
            "component_kind": component_kind.value,
            "component_name": component_name,
            "base_version_id": base_version_id,
            "pool_entry_ids": pool_entry_ids,
            "experience_ids": experience_ids,
            "edit_budget": edit_budget.model_dump(mode="json"),
            "offline_environment_id": offline_environment_id,
            "created_by": created_by,
        }
        campaign_id = stable_id(
            "evolution_campaign",
            canonical_hash(material),
        )
        existing = self.store.campaign(campaign_id)
        if (
            existing is not None
            and existing.status != EvolutionCampaignStatus.DRAFT
        ):
            return existing
        campaign_artifact_id = stable_id("artifact", campaign_id)
        created_at = max(
            (
                *(item.entry.submitted_at for item in reviewed),
                *(item.created_at for item in experiences),
            )
        )
        request = (
            existing.request
            if existing is not None
            else EvolutionCampaignRequest(
                campaign_id=campaign_id,
                campaign_artifact_id=campaign_artifact_id,
                target=target,
                component_kind=component_kind,
                component_name=component_name,
                base_version_id=base_version_id,
                pool_entry_ids=pool_entry_ids,
                experience_ids=experience_ids,
                edit_budget=edit_budget,
                offline_environment_id=offline_environment_id,
                created_by=created_by,
                created_at=created_at,
                allow_online_inference=False,
                allow_runtime_memory=False,
                allow_automatic_publish=False,
            )
        )
        campaign_run_id = stable_id("run", campaign_id)
        self.artifact_store.put_json(
            {
                "schema": "EvolutionCampaignRequest@1",
                "request": request.model_dump(mode="json"),
                "explicit_start": True,
                "online_inference_allowed": False,
                "automatic_publish_allowed": False,
            },
            redact=False,
            kind=ArtifactKind.EVOLUTION_CAMPAIGN,
            producer_id=self.producer_id,
            run_id=campaign_run_id,
            content_schema="EvolutionCampaignRequest@1",
            source_artifact_ids=(),
            artifact_id=campaign_artifact_id,
            idempotency_key=f"evolution-campaign:{campaign_id}",
        )
        if existing is None:
            self.store.create_campaign(request)
        snapshot = self._build_input_snapshot(
            request=request,
            base_ref=base_ref,
            reviewed=reviewed,
            experiences=experiences,
        )
        self.artifact_store.put_json(
            {
                "schema": "EvolutionInputSnapshot@1",
                "snapshot": snapshot.model_dump(mode="json"),
                "immutable": True,
                "all_source_categories_present": True,
                "runtime_memory_used": False,
                "online_inference_count": 0,
            },
            redact=False,
            kind=ArtifactKind.EVOLUTION_INPUT,
            producer_id=self.producer_id,
            run_id=campaign_run_id,
            content_schema="EvolutionInputSnapshot@1",
            source_artifact_ids=(campaign_artifact_id,),
            artifact_id=snapshot.snapshot_artifact_id,
            idempotency_key=f"evolution-input:{snapshot.snapshot_id}",
            metadata={
                "external_source_artifact_ids": [
                    *snapshot.success_trace_artifact_ids,
                    *snapshot.failure_trace_artifact_ids,
                    *snapshot.badcase_artifact_ids,
                    *snapshot.evaluation_artifact_ids,
                ]
            },
        )
        return self.store.seal_inputs(snapshot)

    def generate_candidate(
        self,
        campaign_id: str,
        *,
        worker_id: str,
    ) -> EvolutionCampaignRecord:
        attempt = self.store.claim_generation(
            campaign_id,
            worker_id=worker_id,
        )
        try:
            record = self._require_campaign(campaign_id)
            if record.input_snapshot is None:
                raise RuntimeError("campaign input snapshot is missing")
            request = record.request
            snapshot = record.input_snapshot
            active = self.version_registry.store.active(
                request.component_kind.value,
                request.component_name,
            )
            if (
                active is None
                or active.manifest.version_ref.version_id
                != snapshot.active_version_id_at_seal
            ):
                raise ValueError(
                    "active component changed after input sealing"
                )
            base_text = self._read_component_text(
                snapshot.base_content_artifact_id,
                request.component_kind,
            )
            experiences = tuple(
                self._require_experience(item)
                for item in request.experience_ids
            )
            rejection_memory = self.store.rejection_memory(
                target=request.target,
                base_version_id=request.base_version_id,
            )
            rejected_operations = frozenset(
                fingerprint
                for item in rejection_memory
                for fingerprint in item.operation_fingerprints
            )
            generated = self.generator.generate(
                target=request.target,
                base_text=base_text,
                experiences=experiences,
                budget=request.edit_budget,
                rejected_operation_fingerprints=rejected_operations,
            )
            if generated.online_inference_count or generated.network_accessed:
                raise RuntimeError(
                    "offline generator reported online inference/network use"
                )
            updated_text, edit_metrics = self.applier.apply(
                base_text,
                generated.operations,
                request.edit_budget,
            )
            patch_hash = patch_fingerprint(
                base_content_hash=snapshot.base_content_hash,
                target=request.target,
                operations=generated.operations,
            )
            rejected_patches = {
                item.patch_fingerprint for item in rejection_memory
            }
            if patch_hash in rejected_patches:
                raise ValueError(
                    "generator reproduced a patch from rejected-edit memory"
                )
            round_no = attempt.round_no
            created_at = request.created_at + timedelta(
                microseconds=round_no
            )
            patch_id = stable_id(
                "evolution_patch",
                campaign_id,
                str(round_no),
                patch_hash,
            )
            patch_artifact_id = stable_id("artifact", patch_id)
            operation_hashes = tuple(
                operation_fingerprint(item)
                for item in generated.operations
            )
            patch = EvolutionPatch(
                patch_id=patch_id,
                campaign_id=campaign_id,
                round_no=round_no,
                target=request.target,
                base_version_id=request.base_version_id,
                base_artifact_id=snapshot.base_content_artifact_id,
                base_content_hash=snapshot.base_content_hash,
                operations=generated.operations,
                rationale=generated.rationale,
                experience_ids=generated.consulted_experience_ids,
                rejected_patch_fingerprints_consulted=tuple(
                    sorted(rejected_patches)
                ),
                rejected_operation_fingerprints_skipped=(
                    generated.skipped_rejected_operation_fingerprints
                ),
                operation_fingerprints=operation_hashes,
                edit_metrics=edit_metrics,
                patch_fingerprint=patch_hash,
                artifact_id=patch_artifact_id,
                created_at=created_at,
            )
            campaign_run_id = stable_id("run", campaign_id)
            self.artifact_store.put_json(
                {
                    "schema": "EvolutionPatch@1",
                    "patch": patch.model_dump(mode="json"),
                    "allowed_operations": [
                        "add",
                        "delete",
                        "replace",
                    ],
                    "text_learning_rate_enforced": True,
                    "online_inference_count": 0,
                },
                redact=False,
                kind=ArtifactKind.EVOLUTION_PATCH,
                producer_id=self.producer_id,
                run_id=campaign_run_id,
                content_schema="EvolutionPatch@1",
                source_artifact_ids=(
                    snapshot.snapshot_artifact_id,
                ),
                artifact_id=patch_artifact_id,
                idempotency_key=f"evolution-patch:{patch_id}",
            )
            candidate_id = stable_id(
                "evolution_candidate",
                patch_id,
            )
            content_artifact_id = stable_id(
                "artifact",
                candidate_id,
                "content",
            )
            content_envelope = self.artifact_store.put_text(
                updated_text,
                redact=False,
                kind=_COMPONENT_ARTIFACT_KINDS[
                    request.component_kind
                ],
                producer_id=self.producer_id,
                run_id=campaign_run_id,
                content_schema=(
                    f"EvolutionCandidateContent:"
                    f"{request.component_kind.value}@1"
                ),
                source_artifact_ids=(
                    snapshot.snapshot_artifact_id,
                    patch_artifact_id,
                ),
                artifact_id=content_artifact_id,
                idempotency_key=(
                    f"evolution-candidate-content:{candidate_id}"
                ),
                metadata={
                    "base_artifact_id": snapshot.base_content_artifact_id,
                    "base_content_hash": snapshot.base_content_hash,
                },
            )
            version_ref = VersionRef(
                version_id=stable_id("version", candidate_id),
                kind=request.component_kind,
                name=request.component_name,
                version=self._candidate_semver(
                    snapshot.base_version.version,
                    round_no=round_no,
                    patch_hash=patch_hash,
                ),
                artifact_id=content_artifact_id,
                content_hash=content_envelope.content_hash,
                created_at=created_at,
            )
            self.version_registry.register(
                version_ref,
                parent_version_id=request.base_version_id,
            )
            candidate_artifact_id = stable_id(
                "artifact",
                candidate_id,
            )
            candidate = EvolutionCandidate(
                candidate_id=candidate_id,
                campaign_id=campaign_id,
                round_no=round_no,
                patch_id=patch_id,
                patch_artifact_id=patch_artifact_id,
                base_version_id=request.base_version_id,
                version_ref=version_ref,
                content_artifact_id=content_artifact_id,
                candidate_artifact_id=candidate_artifact_id,
                created_at=created_at,
                online_inference_count=0,
                automatically_published=False,
            )
            self.artifact_store.put_json(
                {
                    "schema": "EvolutionCandidate@1",
                    "candidate": candidate.model_dump(mode="json"),
                    "candidate_state": "registered_not_published",
                    "requires_selection_gate": True,
                    "requires_human_gate": True,
                    "requires_test_hidden_gate": True,
                },
                redact=False,
                kind=ArtifactKind.EVOLUTION_CANDIDATE,
                producer_id=self.producer_id,
                run_id=campaign_run_id,
                content_schema="EvolutionCandidate@1",
                source_artifact_ids=(
                    patch_artifact_id,
                    content_artifact_id,
                ),
                artifact_id=candidate_artifact_id,
                idempotency_key=f"evolution-candidate:{candidate_id}",
            )
            return self.store.finish_generation(
                attempt_id=attempt.attempt_id,
                patch=patch,
                candidate=candidate,
                at=created_at,
            )
        except Exception as exc:
            try:
                self.store.abandon_generation(
                    campaign_id,
                    attempt_id=attempt.attempt_id,
                    error=str(exc)[:8000] or type(exc).__name__,
                )
            except EvolutionConflict:
                pass
            raise

    def evaluate_selection(
        self,
        campaign_id: str,
        *,
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> EvolutionCampaignRecord:
        record = self._require_campaign(campaign_id)
        if record.status != EvolutionCampaignStatus.CANDIDATE_READY:
            raise ValueError(
                "selection requires a newly generated candidate"
            )
        candidate = record.candidates[-1]
        self._validate_gate_identity(record, candidate, evidence)
        self._require_strict_selection(policy, evidence)
        decision = self.release_gate.evaluate(
            stage=GateStage.SELECTION,
            policy=policy,
            evidence=evidence,
        )
        rejection = (
            None
            if decision.passed
            else self._build_rejection(
                record=record,
                candidate=candidate,
                failed_check_names=decision.failed_check_names,
                reasons=tuple(
                    item.detail
                    for item in decision.checks
                    if not item.passed
                ),
                gate_decision=decision,
            )
        )
        return self.store.record_selection(
            campaign_id,
            candidate_id=candidate.candidate.candidate_id,
            passed=decision.passed,
            gate_decision_id=decision.gate_decision_id,
            gate_artifact_id=decision.result_artifact_id,
            failed_check_names=decision.failed_check_names,
            rejection=rejection,
            at=decision.created_at,
        )

    def decide_human_gate(
        self,
        campaign_id: str,
        *,
        approved: bool,
        reviewer_id: str,
        note: str,
        decided_at: datetime | None = None,
    ) -> EvolutionCampaignRecord:
        record = self._require_campaign(campaign_id)
        if record.status != EvolutionCampaignStatus.AWAITING_HUMAN:
            raise ValueError(
                "human gate requires selection-passed candidate"
            )
        candidate = record.candidates[-1]
        if (
            candidate.status != CandidateStatus.SELECTION_PASSED
            or candidate.selection_gate_decision_id is None
            or candidate.selection_gate_artifact_id is None
        ):
            raise RuntimeError(
                "selection-passed candidate provenance is incomplete"
            )
        outcome = (
            HumanGateOutcome.APPROVED
            if approved
            else HumanGateOutcome.REJECTED
        )
        material = {
            "campaign_id": campaign_id,
            "candidate_id": candidate.candidate.candidate_id,
            "selection_gate_decision_id": (
                candidate.selection_gate_decision_id
            ),
            "outcome": outcome.value,
            "reviewer_id": reviewer_id,
            "note": note,
        }
        decision_id = stable_id(
            "evolution_human",
            canonical_hash(material),
        )
        artifact_id = stable_id("artifact", decision_id)
        campaign_run_id = stable_id("run", campaign_id)
        existing_artifact = self.artifact_store.get(artifact_id)
        if existing_artifact is not None:
            if (
                existing_artifact.kind
                != ArtifactKind.EVOLUTION_HUMAN_DECISION
            ):
                raise ValueError(
                    "human-decision artifact ID belongs to another kind"
                )
            decision = EvolutionHumanDecision.model_validate(
                self._read_json_payload(artifact_id)["decision"],
                strict=False,
            )
            if (
                decision.campaign_id != campaign_id
                or decision.candidate_id
                != candidate.candidate.candidate_id
                or decision.outcome != outcome
                or decision.reviewer_id != reviewer_id
                or decision.note != note
            ):
                raise ValueError(
                    "persisted human decision differs from retry request"
                )
        else:
            decision = EvolutionHumanDecision(
                human_decision_id=decision_id,
                campaign_id=campaign_id,
                candidate_id=candidate.candidate.candidate_id,
                selection_gate_decision_id=(
                    candidate.selection_gate_decision_id
                ),
                selection_gate_artifact_id=(
                    candidate.selection_gate_artifact_id
                ),
                outcome=outcome,
                reviewer_id=reviewer_id,
                note=note,
                decision_artifact_id=artifact_id,
                decided_at=decided_at or utc_now(),
                triggers_final_evaluation=False,
                publishes_version=False,
            )
            self.artifact_store.put_json(
                {
                    "schema": "EvolutionHumanDecision@1",
                    "decision": decision.model_dump(mode="json"),
                    "records_approval_only": True,
                    "triggers_final_evaluation": False,
                    "publishes_version": False,
                },
                redact=False,
                kind=ArtifactKind.EVOLUTION_HUMAN_DECISION,
                producer_id=self.producer_id,
                run_id=campaign_run_id,
                content_schema="EvolutionHumanDecision@1",
                source_artifact_ids=(
                    candidate.candidate.candidate_artifact_id,
                ),
                artifact_id=artifact_id,
                idempotency_key=f"evolution-human:{decision_id}",
            )
        rejection = None
        if not approved:
            version_id = candidate.candidate.version_ref.version_id
            self.version_registry.reject(
                version_id,
                gate_decision_id=decision.human_decision_id,
                gate_decision_artifact_id=artifact_id,
                reason=f"Human gate rejected candidate: {note}",
                actor_id=reviewer_id,
                occurred_at=decision.decided_at,
            )
            rejection = self._build_rejection(
                record=record,
                candidate=candidate,
                failed_check_names=("human_gate_rejected",),
                reasons=(note,),
                human_decision=decision,
            )
        return self.store.record_human_decision(
            decision,
            rejection=rejection,
        )

    def evaluate_final_promotion(
        self,
        campaign_id: str,
        *,
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> EvolutionCampaignRecord:
        record = self._require_campaign(campaign_id)
        if record.status != EvolutionCampaignStatus.READY_FOR_FINAL:
            raise ValueError(
                "final promotion requires explicit human approval"
            )
        candidate = record.candidates[-1]
        if (
            candidate.human_decision is None
            or candidate.human_decision.outcome
            != HumanGateOutcome.APPROVED
        ):
            raise ValueError("candidate lacks an approving human gate")
        self._validate_gate_identity(record, candidate, evidence)
        decision = self.release_gate.evaluate(
            stage=GateStage.FINAL_PROMOTION,
            policy=policy,
            evidence=evidence,
        )
        rejection = (
            None
            if decision.passed
            else self._build_rejection(
                record=record,
                candidate=candidate,
                failed_check_names=decision.failed_check_names,
                reasons=tuple(
                    item.detail
                    for item in decision.checks
                    if not item.passed
                ),
                gate_decision=decision,
            )
        )
        best = (
            self._build_best_skill(
                record=record,
                candidate=candidate,
                decision=decision,
                restored_version_id=None,
            )
            if decision.passed
            and record.request.component_kind == ComponentKind.SKILL
            else None
        )
        return self.store.record_final_gate(
            campaign_id,
            candidate_id=candidate.candidate.candidate_id,
            passed=decision.passed,
            gate_decision_id=decision.gate_decision_id,
            gate_artifact_id=decision.result_artifact_id,
            failed_check_names=decision.failed_check_names,
            rejection=rejection,
            best_skill=best,
            at=decision.created_at,
        )

    def evaluate_post_release(
        self,
        campaign_id: str,
        *,
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> EvolutionCampaignRecord:
        record = self._require_campaign(campaign_id)
        if record.status != EvolutionCampaignStatus.PROMOTED:
            raise ValueError(
                "post-release evaluation requires promoted campaign"
            )
        candidate = record.candidates[-1]
        self._validate_gate_identity(record, candidate, evidence)
        if evidence.rollback_target_version_id != record.request.base_version_id:
            raise ValueError(
                "post-release rollback target must be the sealed base version"
            )
        decision = self.release_gate.evaluate(
            stage=GateStage.POST_RELEASE,
            policy=policy,
            evidence=evidence,
        )
        kept = decision.outcome == GateOutcome.KEEP
        if decision.outcome not in {
            GateOutcome.KEEP,
            GateOutcome.ROLLBACK,
        }:
            raise RuntimeError("unexpected post-release gate outcome")
        best = (
            self._build_best_skill(
                record=record,
                candidate=candidate,
                decision=decision,
                restored_version_id=record.request.base_version_id,
            )
            if not kept
            and record.request.component_kind == ComponentKind.SKILL
            else None
        )
        return self.store.record_post_release(
            campaign_id,
            candidate_id=candidate.candidate.candidate_id,
            kept=kept,
            gate_decision_id=decision.gate_decision_id,
            gate_artifact_id=decision.result_artifact_id,
            best_skill=best,
            at=decision.created_at,
        )

    def _build_input_snapshot(
        self,
        *,
        request: EvolutionCampaignRequest,
        base_ref: VersionRef,
        reviewed: tuple[ReviewedPoolEntry, ...],
        experiences: tuple[CrossTaskExperience, ...],
    ) -> EvolutionInputSnapshot:
        if base_ref.artifact_id is None:
            raise ValueError("base version has no content artifact")
        base_artifact = self._require_artifact(base_ref.artifact_id)
        expected_kind = _COMPONENT_ARTIFACT_KINDS.get(
            request.component_kind
        )
        if expected_kind is None or base_artifact.kind != expected_kind:
            raise ValueError(
                "base component artifact kind is unsupported or mismatched"
            )
        success = tuple(
            item.entry.source_artifact_id
            for item in reviewed
            if item.entry.source_kind
            == PoolSourceKind.SCORED_SUCCESS_TRACE
        )
        failures = tuple(
            item.entry.source_artifact_id
            for item in reviewed
            if item.entry.source_kind
            == PoolSourceKind.SCORED_FAILURE_TRACE
        )
        badcases = tuple(
            item.entry.source_artifact_id
            for item in reviewed
            if item.entry.source_kind == PoolSourceKind.BADCASE
        )
        evaluations = tuple(
            dict.fromkeys(
                artifact_id
                for item in reviewed
                for artifact_id in (
                    (
                        item.entry.source_artifact_id,
                    )
                    if item.entry.source_kind
                    == PoolSourceKind.EVALUATION
                    else item.entry.evaluation_artifact_ids
                )
            )
        )
        rejections = self.store.rejection_memory(
            target=request.target,
            base_version_id=request.base_version_id,
        )
        snapshot_id = stable_id(
            "evolution_input",
            request.campaign_id,
            canonical_hash(
                {
                    "pool_entry_ids": request.pool_entry_ids,
                    "experience_ids": request.experience_ids,
                    "base_content_hash": base_artifact.content_hash,
                    "rejected_patch_fingerprints": [
                        item.patch_fingerprint for item in rejections
                    ],
                }
            ),
        )
        return EvolutionInputSnapshot(
            snapshot_id=snapshot_id,
            campaign_id=request.campaign_id,
            target=request.target,
            base_version=base_ref.model_copy(
                update={"content_hash": base_artifact.content_hash}
            ),
            base_content_artifact_id=base_artifact.artifact_id,
            base_content_hash=base_artifact.content_hash,
            active_version_id_at_seal=request.base_version_id,
            pool_entries=reviewed,
            experience_ids=tuple(
                item.experience_id for item in experiences
            ),
            success_trace_artifact_ids=success,
            failure_trace_artifact_ids=failures,
            badcase_artifact_ids=badcases,
            evaluation_artifact_ids=evaluations,
            rejected_patch_fingerprints=tuple(
                item.patch_fingerprint for item in rejections
            ),
            snapshot_artifact_id=stable_id(
                "artifact",
                snapshot_id,
            ),
            sealed_at=request.created_at + timedelta(microseconds=1),
            runtime_memory_used=False,
            online_inference_count=0,
        )

    def _build_rejection(
        self,
        *,
        record: EvolutionCampaignRecord,
        candidate,
        failed_check_names: tuple[str, ...],
        reasons: tuple[str, ...],
        gate_decision: ReleaseGateDecision | None = None,
        human_decision: EvolutionHumanDecision | None = None,
    ) -> RejectedEditMemory:
        patch = candidate.patch
        material = {
            "campaign_id": record.request.campaign_id,
            "candidate_id": candidate.candidate.candidate_id,
            "patch_fingerprint": patch.patch_fingerprint,
            "failed_check_names": failed_check_names,
            "reasons": reasons,
            "gate_decision_id": (
                gate_decision.gate_decision_id
                if gate_decision is not None
                else (
                    human_decision.human_decision_id
                    if human_decision is not None
                    else None
                )
            ),
        }
        rejection_id = stable_id(
            "rejected_edit",
            canonical_hash(material),
        )
        artifact_id = stable_id("artifact", rejection_id)
        rejected_at = (
            gate_decision.created_at
            if gate_decision is not None
            else (
                human_decision.decided_at
                if human_decision is not None
                else utc_now()
            )
        )
        rejection = RejectedEditMemory(
            rejection_id=rejection_id,
            campaign_id=record.request.campaign_id,
            candidate_id=candidate.candidate.candidate_id,
            target=record.request.target,
            base_version_id=record.request.base_version_id,
            patch_id=patch.patch_id,
            patch_fingerprint=patch.patch_fingerprint,
            operation_fingerprints=patch.operation_fingerprints,
            failed_check_names=failed_check_names,
            reasons=reasons,
            gate_decision_id=(
                gate_decision.gate_decision_id
                if gate_decision is not None
                else None
            ),
            gate_decision_artifact_id=(
                gate_decision.result_artifact_id
                if gate_decision is not None
                else None
            ),
            source_experience_ids=patch.experience_ids,
            artifact_id=artifact_id,
            rejected_at=rejected_at,
        )
        sources = [
            candidate.candidate.patch_artifact_id,
            candidate.candidate.candidate_artifact_id,
        ]
        if (
            human_decision is not None
            and self._require_artifact(
                human_decision.decision_artifact_id
            ).run_id
            == stable_id("run", record.request.campaign_id)
        ):
            sources.append(human_decision.decision_artifact_id)
        self.artifact_store.put_json(
            {
                "schema": "RejectedEditMemory@1",
                "rejection": rejection.model_dump(mode="json"),
                "affects_future_generation": True,
                "directly_deployable": False,
                "memory_layer": "cross_task_experience",
            },
            redact=False,
            kind=ArtifactKind.REJECTED_EDIT_MEMORY,
            producer_id=self.producer_id,
            run_id=stable_id("run", record.request.campaign_id),
            content_schema="RejectedEditMemory@1",
            source_artifact_ids=tuple(sources),
            artifact_id=artifact_id,
            idempotency_key=f"rejected-edit:{rejection_id}",
        )
        return rejection

    def _build_best_skill(
        self,
        *,
        record: EvolutionCampaignRecord,
        candidate,
        decision: ReleaseGateDecision,
        restored_version_id: str | None,
    ) -> BestSkillSnapshot:
        human = candidate.human_decision
        if human is None or human.outcome != HumanGateOutcome.APPROVED:
            raise RuntimeError("best_skill publication lacks human approval")
        version_id = (
            restored_version_id
            or candidate.candidate.version_ref.version_id
        )
        version = self.version_registry.store.record(version_id)
        if version is None:
            raise RuntimeError("best_skill version is not registered")
        ref = version.manifest.version_ref
        if ref.kind != ComponentKind.SKILL or ref.artifact_id is None:
            raise ValueError("best_skill requires a registered Skill version")
        previous = self.store.best_skill(
            target=record.request.target,
            component_name=record.request.component_name,
        )
        material = {
            "target": record.request.target.value,
            "component_name": record.request.component_name,
            "version_id": version_id,
            "previous_best_skill_id": (
                previous.best_skill_id if previous is not None else None
            ),
            "release_gate_decision_id": decision.gate_decision_id,
            "human_decision_id": human.human_decision_id,
        }
        best_id = stable_id("best_skill", canonical_hash(material))
        artifact_id = stable_id("artifact", best_id)
        best = BestSkillSnapshot(
            best_skill_id=best_id,
            target=record.request.target,
            component_name=record.request.component_name,
            skill_version_id=version_id,
            skill_content_artifact_id=ref.artifact_id,
            previous_best_skill_id=(
                previous.best_skill_id if previous is not None else None
            ),
            release_gate_decision_id=decision.gate_decision_id,
            release_gate_artifact_id=decision.result_artifact_id,
            human_decision_id=human.human_decision_id,
            artifact_id=artifact_id,
            published_at=decision.created_at,
        )
        source_ids = [candidate.candidate.candidate_artifact_id]
        if ref.artifact_id == candidate.candidate.content_artifact_id:
            source_ids.append(ref.artifact_id)
        self.artifact_store.put_json(
            {
                "schema": "BestSkillSnapshot@1",
                "best_skill": best.model_dump(mode="json"),
                "static_versioned": True,
                "runtime_memory": False,
                "online_inference_count": 0,
            },
            redact=False,
            kind=ArtifactKind.BEST_SKILL,
            producer_id=self.producer_id,
            run_id=stable_id("run", record.request.campaign_id),
            content_schema="BestSkillSnapshot@1",
            source_artifact_ids=tuple(dict.fromkeys(source_ids)),
            artifact_id=artifact_id,
            idempotency_key=f"best-skill:{best_id}",
        )
        return best

    def _validate_gate_identity(
        self,
        record: EvolutionCampaignRecord,
        candidate,
        evidence: GateEvaluationEvidence,
    ) -> None:
        if (
            evidence.candidate_version_id
            != candidate.candidate.version_ref.version_id
            or evidence.baseline_version_id
            != record.request.base_version_id
        ):
            raise ValueError(
                "release-gate evidence does not match campaign versions"
            )

    @staticmethod
    def _require_strict_selection(
        policy: ReleaseGatePolicy,
        evidence: GateEvaluationEvidence,
    ) -> None:
        if not policy.required_improvements or any(
            value <= 0
            for value in policy.required_improvements.values()
        ):
            raise ValueError(
                "offline selection requires positive improvement thresholds"
            )
        baseline = {item.name: item for item in evidence.baseline_metrics}
        candidate = {item.name: item for item in evidence.candidate_metrics}
        improvements = []
        for name in policy.required_improvements:
            if name not in baseline or name not in candidate:
                raise ValueError(
                    f"strict selection metric is missing: {name}"
                )
            left = baseline[name]
            right = candidate[name]
            if left.direction != right.direction:
                raise ValueError(
                    f"strict selection metric direction differs: {name}"
                )
            improvements.append(
                right.value - left.value
                if right.direction == MetricDirection.HIGHER_IS_BETTER
                else left.value - right.value
            )
        if not all(value > 0 for value in improvements):
            raise ValueError(
                "candidate must strictly improve every required metric"
            )

    def _require_approved_pool_entry(
        self,
        pool_entry_id: str,
    ) -> ReviewedPoolEntry:
        item = self.store.pool_entry(pool_entry_id)
        if item is None:
            raise KeyError(f"unknown candidate-pool entry: {pool_entry_id}")
        if (
            item.status != PoolEntryStatus.APPROVED
            or item.review is None
            or item.review.assigned_split
            not in {DatasetSplit.TRAIN, DatasetSplit.DEV}
        ):
            raise ValueError(
                f"candidate-pool entry is not approved: {pool_entry_id}"
            )
        return item

    def _require_experience(
        self,
        experience_id: str,
    ) -> CrossTaskExperience:
        item = self.store.experience(experience_id)
        if item is None:
            raise KeyError(f"unknown cross-task experience: {experience_id}")
        return item

    def _require_campaign(
        self,
        campaign_id: str,
    ) -> EvolutionCampaignRecord:
        item = self.store.campaign(campaign_id)
        if item is None:
            raise KeyError(f"unknown evolution campaign: {campaign_id}")
        return item

    def _require_artifact(self, artifact_id: str):
        artifact = self.artifact_store.get(artifact_id)
        if artifact is None:
            raise ValueError(f"artifact is missing: {artifact_id}")
        return artifact

    def _read_badcase(self, artifact_id: str) -> StudioBadcase:
        try:
            body = json.loads(
                self.artifact_store.read_bytes(artifact_id).decode("utf-8")
            )
            badcase = StudioBadcase.model_validate(
                body["badcase"],
                strict=False,
            )
        except (
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "badcase artifact does not contain sealed StudioBadcase "
                "provenance"
            ) from exc
        if badcase.triggers_change:
            raise ValueError("badcase cannot trigger evolution directly")
        return badcase

    def _read_json_payload(self, artifact_id: str) -> dict:
        try:
            value = json.loads(
                self.artifact_store.read_bytes(artifact_id).decode("utf-8")
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"artifact does not contain JSON: {artifact_id}"
            ) from exc
        if not isinstance(value, dict):
            raise ValueError(
                f"artifact JSON must be an object: {artifact_id}"
            )
        return value

    def _read_component_text(
        self,
        artifact_id: str,
        component_kind: ComponentKind,
    ) -> str:
        artifact = self._require_artifact(artifact_id)
        expected = _COMPONENT_ARTIFACT_KINDS.get(component_kind)
        if expected is None or artifact.kind != expected:
            raise ValueError("component content artifact kind differs")
        try:
            text = self.artifact_store.read_bytes(artifact_id).decode(
                "utf-8"
            )
        except UnicodeDecodeError as exc:
            raise ValueError(
                "offline structured patching requires UTF-8 component text"
            ) from exc
        if not text:
            raise ValueError("component content cannot be empty")
        return text

    def _observed_dataset_splits(
        self,
        artifact_ids: tuple[str, ...],
    ) -> tuple[DatasetSplit, ...]:
        observed: list[DatasetSplit] = []

        def visit(value) -> None:
            if isinstance(value, dict):
                for key, nested in value.items():
                    normalized = str(key).lower().replace("-", "_")
                    if normalized in {
                        "dataset_split",
                        "split",
                    } and isinstance(nested, str):
                        try:
                            observed.append(DatasetSplit(nested))
                        except ValueError:
                            pass
                    visit(nested)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    visit(nested)

        for artifact_id in artifact_ids:
            self._require_artifact(artifact_id)
            try:
                value = json.loads(
                    self.artifact_store.read_bytes(artifact_id).decode(
                        "utf-8"
                    )
                )
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            visit(value)
        return tuple(dict.fromkeys(observed))

    @staticmethod
    def _candidate_semver(
        base_version: str,
        *,
        round_no: int,
        patch_hash: str,
    ) -> str:
        core = base_version.split("-", 1)[0].split("+", 1)[0]
        major, minor, patch = (int(item) for item in core.split("."))
        return (
            f"{major}.{minor}.{patch + 1}-"
            f"evo.r{round_no}.{patch_hash[:8]}"
        )
