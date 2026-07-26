from __future__ import annotations

from dataclasses import dataclass
import difflib
import hashlib
import json
from typing import Any, Iterable

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    ComponentKind,
    ComponentVersionSet,
    ErrorCategory,
    ErrorRecord,
    EventType,
    RunEvent,
    VersionRef,
    canonical_contract_json,
    utc_now,
)
from deep_researcher.events import EventStore
from deep_researcher.version_registry.store import (
    SQLiteVersionRegistryStore,
)

from .advanced_models import (
    ComparisonValue,
    ComponentComparison,
    ComponentDiffLine,
    GraphComparison,
    ReplayApprovalGrant,
    ReplayAttemptStatus,
    ReplayCapsule,
    ReplayExecutionOutcome,
    ReplayMode,
    ReplayRecord,
    ReplayRequest,
    StudioABComparison,
    StudioBadcase,
    StudioComponentDiff,
)
from .advanced_replay import (
    KernelReplayBackend,
    ReplayCapsuleRepository,
    all_run_events,
    source_event_fingerprint,
    span_tree_events,
    stable_id,
)
from .advanced_store import SQLiteStudioAdvancedStore
from .v2 import StudioV2Service


_EVALUATION_ARTIFACT_KINDS = {
    ArtifactKind.EVALUATION_RESULT,
    ArtifactKind.SEMANTIC_EVALUATION_RESULT,
    ArtifactKind.JUDGE_EVALUATION,
    ArtifactKind.JUDGE_CALIBRATION,
    ArtifactKind.EXPERIMENT_RESULT,
    ArtifactKind.FROZEN_REPLAY_RESULT,
    ArtifactKind.LIVE_WEB_RESULT,
}


def _component_refs(
    versions: ComponentVersionSet,
) -> tuple[VersionRef, ...]:
    return tuple(
        item
        for item in (
            versions.runtime,
            versions.scheduler,
            versions.model,
            versions.agent_spec,
            versions.prompt,
            versions.skill,
            versions.tool_policy,
            versions.stop_policy,
            versions.verification_policy,
            versions.rubric,
            *versions.tools,
        )
        if item is not None
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _evaluation_ids(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold()
            if (
                normalized == "evaluation_id"
                or normalized.endswith("_evaluation_id")
            ) and isinstance(item, str):
                found.add(item)
            found.update(_evaluation_ids(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.update(_evaluation_ids(item))
    return found


@dataclass(frozen=True)
class SourceRunSeal:
    event_count: int
    fingerprint: str


class StudioAdvancedService:
    """Studio V3/V4 replay, A/B, component-diff and badcase facade."""

    def __init__(
        self,
        *,
        store: SQLiteStudioAdvancedStore,
        event_store: EventStore,
        artifact_store: ArtifactStore,
        version_store: SQLiteVersionRegistryStore,
        studio_v2: StudioV2Service,
        replay_backend: KernelReplayBackend | None = None,
    ) -> None:
        self.store = store
        self.event_store = event_store
        self.artifact_store = artifact_store
        self.version_store = version_store
        self.studio_v2 = studio_v2
        self.capsules = ReplayCapsuleRepository(
            event_store=event_store,
            artifact_store=artifact_store,
        )
        self.replay_backend = replay_backend

    def create_capsule(self, capsule: ReplayCapsule) -> dict[str, Any]:
        envelope = self.capsules.create(capsule)
        return envelope.model_dump(mode="json")

    def replay_eligibility(
        self,
        source_run_id: str,
        source_span_id: str,
    ) -> dict[str, Any]:
        return self.capsules.eligibility(
            source_run_id,
            source_span_id,
        ).model_dump(mode="json")

    def prepare_replay(
        self,
        *,
        source_run_id: str,
        source_span_id: str,
        mode: ReplayMode,
        selected_component_versions: ComponentVersionSet,
        requested_by: str,
        reason: str,
        restart_failed_span: bool = False,
        environment_label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ReplayRecord:
        eligibility = self.capsules.eligibility(
            source_run_id,
            source_span_id,
        )
        if not eligibility.eligible:
            raise ValueError(
                "span is not replay-eligible: "
                + "; ".join(eligibility.reasons)
            )
        if restart_failed_span and not eligibility.terminal_failed:
            raise ValueError(
                "failed-span restart requires a failed terminal span"
            )
        if not eligibility.capsule_artifact_id:
            raise RuntimeError(
                "eligible replay span has no capsule artifact"
            )
        capsule = self.capsules.load(
            eligibility.capsule_artifact_id
        )
        span_events = span_tree_events(
            self.event_store,
            source_run_id,
            source_span_id,
        )
        required_approvals = tuple(
            dict.fromkeys(
                item.command_fingerprint
                for item in capsule.tool_exchanges
                if item.side_effecting or item.requires_approval
            )
        )
        label = (
            "sealed-network-free"
            if mode == ReplayMode.SAVED_TOOL_RESULTS
            else str(environment_label or "").strip()
        )
        if mode == ReplayMode.LIVE_ENVIRONMENT and not label:
            raise ValueError(
                "live replay requires an explicit environment label"
            )
        request = ReplayRequest(
            source_run_id=source_run_id,
            source_span_id=source_span_id,
            source_terminal_event_id=(
                eligibility.terminal_event_id or ""
            ),
            capsule_artifact_id=eligibility.capsule_artifact_id,
            mode=mode,
            selected_component_versions=selected_component_versions,
            restart_failed_span=restart_failed_span,
            requested_by=requested_by,
            reason=reason,
            dataset_sample_artifact_id=(
                capsule.dataset_sample_artifact_id
            ),
            source_event_count=len(span_events),
            source_event_fingerprint=source_event_fingerprint(
                span_events
            ),
            required_approval_fingerprints=required_approvals,
            environment_label=label,
            metadata=metadata or {},
        )
        return self.store.create_request(request)

    def approve_replay(
        self,
        *,
        replay_request_id: str,
        command_fingerprint: str,
        approved_by: str,
        reason: str,
    ) -> ReplayRecord:
        grant = ReplayApprovalGrant(
            replay_request_id=replay_request_id,
            command_fingerprint=command_fingerprint,
            approved_by=approved_by,
            reason=reason,
        )
        return self.store.grant_approval(grant)

    async def execute_replay(
        self,
        replay_request_id: str,
    ) -> ReplayRecord:
        if self.replay_backend is None:
            raise RuntimeError(
                "Studio replay execution backend is not configured"
            )
        before = self._seal_source_run(
            self._require_record(replay_request_id).request.source_run_id
        )
        attempt = self.store.claim(replay_request_id)
        record = self._require_record(replay_request_id)
        capsule = self.capsules.load(
            record.request.capsule_artifact_id
        )
        self._assert_source_span_sealed(record.request)
        approved = frozenset(
            item.command_fingerprint for item in record.approvals
        )
        try:
            outcome = await self.replay_backend.execute(
                request=record.request,
                attempt=attempt,
                capsule=capsule,
                approved_fingerprints=approved,
            )
        except Exception as exc:
            outcome = ReplayExecutionOutcome(
                replay_request_id=replay_request_id,
                attempt_id=attempt.attempt_id,
                target_run_id=attempt.target_run_id,
                status=ReplayAttemptStatus.FAILED,
                network_calls=0,
                network_accounting_complete=(
                    record.request.mode
                    == ReplayMode.SAVED_TOOL_RESULTS
                ),
                environment_label=record.request.environment_label,
                error=ErrorRecord(
                    category=ErrorCategory.INTERNAL,
                    code="studio_replay_execution_failed",
                    message=f"{type(exc).__name__}: {exc}"[:2000],
                    retryable=False,
                    fatal=True,
                    attempt=attempt.attempt_no,
                    actor_id=record.request.requested_by,
                ),
                completed_at=utc_now(),
            )
        after = self._seal_source_run(record.request.source_run_id)
        if after != before:
            raise RuntimeError(
                "source run changed during replay; target result was not "
                "committed to the replay journal"
            )
        if (
            record.request.mode == ReplayMode.SAVED_TOOL_RESULTS
            and outcome.network_calls != 0
        ):
            raise RuntimeError(
                "saved-tool-result replay reported network access"
            )
        if outcome.environment_label != record.request.environment_label:
            raise RuntimeError(
                "replay outcome environment label changed"
            )
        return self.store.finish(outcome)

    def recover_replays(self) -> tuple[ReplayRecord, ...]:
        return self.store.recover_incomplete()

    def compare_runs(
        self,
        *,
        left_run_id: str,
        right_run_id: str,
        dataset_sample_artifact_id: str,
        left_span_id: str | None = None,
        right_span_id: str | None = None,
    ) -> StudioABComparison:
        if left_run_id == right_run_id:
            raise ValueError("A/B comparison requires distinct runs")
        if (left_span_id is None) != (right_span_id is None):
            raise ValueError("span comparison requires both span IDs")
        sample = self.artifact_store.get(dataset_sample_artifact_id)
        if sample is None or sample.kind != ArtifactKind.DATASET_SAMPLE:
            raise ValueError(
                "A/B comparison requires a dataset-sample artifact"
            )
        for run_id in (left_run_id, right_run_id):
            if self.event_store.get_run(run_id) is None:
                raise KeyError(f"unknown comparison run: {run_id}")
            if dataset_sample_artifact_id not in (
                self._run_dataset_samples(run_id)
            ):
                raise ValueError(
                    "A/B runs are not aligned to the same dataset sample: "
                    f"{run_id}"
                )
        left_events = self._span_scope_events(
            left_run_id,
            left_span_id,
        )
        right_events = self._span_scope_events(
            right_run_id,
            right_span_id,
        )
        comparison_key = _canonical_hash(
            {
                "left_run_id": left_run_id,
                "right_run_id": right_run_id,
                "left_span_id": left_span_id,
                "right_span_id": right_span_id,
                "dataset_sample_artifact_id": (
                    dataset_sample_artifact_id
                ),
                "left_event_fingerprint": source_event_fingerprint(
                    left_events
                ),
                "right_event_fingerprint": source_event_fingerprint(
                    right_events
                ),
            }
        )
        left_tasks = self._full_graph(
            self.studio_v2.task_graph,
            left_run_id,
        )
        right_tasks = self._full_graph(
            self.studio_v2.task_graph,
            right_run_id,
        )
        left_evidence = self._full_graph(
            self.studio_v2.evidence_graph,
            left_run_id,
        )
        right_evidence = self._full_graph(
            self.studio_v2.evidence_graph,
            right_run_id,
        )
        left_metrics = self.studio_v2.metrics(left_run_id)
        right_metrics = self.studio_v2.metrics(right_run_id)
        metric_names = sorted(
            set(left_metrics.totals) | set(right_metrics.totals)
        )
        metric_comparison = {
            name: ComparisonValue(
                left=float(left_metrics.totals.get(name, 0)),
                right=float(right_metrics.totals.get(name, 0)),
                delta=float(right_metrics.totals.get(name, 0))
                - float(left_metrics.totals.get(name, 0)),
            )
            for name in metric_names
        }
        components = self._compare_components(
            left_metrics.component_versions,
            right_metrics.component_versions,
        )
        comparison_key = _canonical_hash(
            {
                "identity": comparison_key,
                "left_task_graph": left_tasks,
                "right_task_graph": right_tasks,
                "left_evidence_graph": left_evidence,
                "right_evidence_graph": right_evidence,
                "left_metrics": left_metrics.totals,
                "right_metrics": right_metrics.totals,
                "components": [
                    item.model_dump(mode="json")
                    for item in components
                ],
            }
        )
        comparison_id = f"studio_comparison_{comparison_key[:24]}"
        existing = self.store.comparison(comparison_id)
        if existing is not None:
            return existing
        artifact_id = stable_id("artifact", comparison_id)
        created_at = utc_now()
        comparison = StudioABComparison(
            comparison_id=comparison_id,
            left_run_id=left_run_id,
            right_run_id=right_run_id,
            left_span_id=left_span_id,
            right_span_id=right_span_id,
            dataset_sample_artifact_id=dataset_sample_artifact_id,
            left_event_fingerprint=source_event_fingerprint(left_events),
            right_event_fingerprint=source_event_fingerprint(
                right_events
            ),
            run_status={
                "left": self.event_store.get_run(
                    left_run_id
                ).status.value,
                "right": self.event_store.get_run(
                    right_run_id
                ).status.value,
            },
            span_summary=self._compare_span_graphs(
                left_events,
                right_events,
            ),
            task_graph=self._compare_graphs(
                left_tasks,
                right_tasks,
            ),
            evidence_graph=self._compare_graphs(
                left_evidence,
                right_evidence,
            ),
            components=components,
            metrics=metric_comparison,
            convergence={
                "left": self._convergence(left_events),
                "right": self._convergence(right_events),
            },
            result_artifact_id=artifact_id,
            created_at=created_at,
            publishes_versions=False,
        )
        self.artifact_store.put_json(
            {
                "schema": "StudioABComparison@1",
                "comparison": comparison.model_dump(mode="json"),
                "alignment": {
                    "dataset_sample_artifact_id": (
                        dataset_sample_artifact_id
                    ),
                    "verified_for_both_runs": True,
                },
                "event_fingerprints": {
                    "left": source_event_fingerprint(left_events),
                    "right": source_event_fingerprint(right_events),
                },
                "publishes_versions": False,
                "optimizer_invoked": False,
            },
            redact=False,
            kind=ArtifactKind.STUDIO_COMPARISON,
            producer_id="runtime_studio_comparison",
            run_id=stable_id("run", comparison_id),
            content_schema="StudioABComparison@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"studio-comparison:{comparison_id}",
        )
        return self.store.save_comparison(comparison)

    def component_diff(
        self,
        left_version_id: str,
        right_version_id: str,
    ) -> StudioComponentDiff:
        left = self.version_store.manifest(left_version_id)
        right = self.version_store.manifest(right_version_id)
        if left is None or right is None:
            raise KeyError("component diff references an unknown version")
        left_ref = left.version_ref
        right_ref = right.version_ref
        if (
            left_ref.kind != right_ref.kind
            or left_ref.name != right_ref.name
        ):
            raise ValueError(
                "component diff requires the same component kind and name"
            )
        if left_ref.kind not in {
            ComponentKind.PROMPT,
            ComponentKind.SKILL,
            ComponentKind.TOOL_POLICY,
            ComponentKind.STOP_POLICY,
            ComponentKind.VERIFICATION_POLICY,
        }:
            raise ValueError(
                "component diff supports Prompt, Skill, and Policy only"
            )
        left_text = self._artifact_text(left.content_artifact_id)
        right_text = self._artifact_text(right.content_artifact_id)
        left_lines = left_text.splitlines()
        right_lines = right_text.splitlines()
        matcher = difflib.SequenceMatcher(
            a=left_lines,
            b=right_lines,
            autojunk=False,
        )
        changes = tuple(
            ComponentDiffLine(
                operation=tag,
                left_start=i1,
                left_end=i2,
                right_start=j1,
                right_end=j2,
                left_lines=tuple(left_lines[i1:i2]),
                right_lines=tuple(right_lines[j1:j2]),
            )
            for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        )
        unified = "\n".join(
            difflib.unified_diff(
                left_lines,
                right_lines,
                fromfile=(
                    f"{left_ref.name}@{left_ref.version}"
                ),
                tofile=(
                    f"{right_ref.name}@{right_ref.version}"
                ),
                lineterm="",
            )
        )
        return StudioComponentDiff(
            diff_id=stable_id(
                "component_diff",
                left_version_id,
                right_version_id,
                left.content_hash,
                right.content_hash,
            ),
            component_kind=left_ref.kind,
            component_name=left_ref.name,
            left_version_id=left_version_id,
            right_version_id=right_version_id,
            left_artifact_id=left.content_artifact_id,
            right_artifact_id=right.content_artifact_id,
            left_content_hash=left.content_hash,
            right_content_hash=right.content_hash,
            changes=changes,
            unified_diff=unified,
        )

    def create_badcase(
        self,
        *,
        source_run_id: str,
        source_span_id: str,
        dataset_sample_artifact_id: str,
        evaluation_ids: tuple[str, ...],
        evaluation_artifact_ids: tuple[str, ...],
        human_note: str,
        created_by: str,
        additional_input_artifact_ids: tuple[str, ...] = (),
    ) -> StudioBadcase:
        events = span_tree_events(
            self.event_store,
            source_run_id,
            source_span_id,
        )
        if not events:
            raise KeyError("badcase source span has no durable events")
        if not any(
            item.span_id == source_span_id
            and item.event_type
            in {
                EventType.RUN_COMPLETED,
                EventType.RUN_FAILED,
                EventType.RUN_CANCELLED,
                EventType.SPAN_COMPLETED,
                EventType.SPAN_FAILED,
                EventType.MODEL_COMPLETED,
                EventType.MODEL_FAILED,
                EventType.TOOL_COMPLETED,
                EventType.TOOL_FAILED,
            }
            for item in events
        ):
            raise ValueError("badcase source span must be terminal")
        versions = {
            canonical_contract_json(item.component_versions): (
                item.component_versions
            )
            for item in events
        }
        if len(versions) != 1:
            raise ValueError(
                "badcase source span changed component versions"
            )
        component_versions = next(iter(versions.values()))
        sample = self.artifact_store.get(dataset_sample_artifact_id)
        if sample is None or sample.kind != ArtifactKind.DATASET_SAMPLE:
            raise ValueError(
                "badcase requires the original dataset-sample artifact"
            )
        if dataset_sample_artifact_id not in (
            self._run_dataset_samples(source_run_id)
        ):
            raise ValueError(
                "badcase dataset sample is not source-run provenance"
            )
        inputs = list(additional_input_artifact_ids)
        for event in events:
            inputs.extend(event.input_artifact_ids)
            payload_inputs = event.payload.get("input_artifact_ids", ())
            if isinstance(payload_inputs, (list, tuple)):
                inputs.extend(str(item) for item in payload_inputs)
        found = self.capsules.find(source_run_id, source_span_id)
        if found is not None:
            inputs.extend(found[1].task.input_artifact_ids)
            inputs.append(found[0])
        input_ids = tuple(dict.fromkeys(inputs))
        if not input_ids:
            raise ValueError(
                "badcase source span has no input artifact provenance"
            )
        for artifact_id in input_ids:
            if self.artifact_store.get(artifact_id) is None:
                raise ValueError(
                    f"badcase input artifact is missing: {artifact_id}"
                )
        if not evaluation_ids or not evaluation_artifact_ids:
            raise ValueError(
                "badcase requires evaluation IDs and evaluation artifacts"
            )
        observed_evaluation_ids: set[str] = set()
        for artifact_id in evaluation_artifact_ids:
            envelope = self.artifact_store.get(artifact_id)
            if (
                envelope is None
                or envelope.kind not in _EVALUATION_ARTIFACT_KINDS
            ):
                raise ValueError(
                    "badcase evaluation artifact is missing or has an "
                    f"unsupported kind: {artifact_id}"
                )
            try:
                evaluation_payload = json.loads(
                    self.artifact_store.read_bytes(artifact_id).decode()
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    "badcase evaluation artifacts must contain JSON "
                    f"provenance: {artifact_id}"
                ) from exc
            observed_evaluation_ids.update(
                _evaluation_ids(evaluation_payload)
            )
        missing_evaluation_ids = set(evaluation_ids).difference(
            observed_evaluation_ids
        )
        if missing_evaluation_ids:
            raise ValueError(
                "badcase evaluation IDs are not present in the supplied "
                f"evaluation artifacts: {sorted(missing_evaluation_ids)}"
            )
        component_ids = tuple(
            item.version_id
            for item in _component_refs(component_versions)
        )
        material = {
            "source_run_id": source_run_id,
            "source_span_id": source_span_id,
            "source_event_ids": [item.event_id for item in events],
            "input_artifact_ids": input_ids,
            "component_version_ids": component_ids,
            "evaluation_ids": evaluation_ids,
            "evaluation_artifact_ids": evaluation_artifact_ids,
            "dataset_sample_artifact_id": dataset_sample_artifact_id,
            "human_note": human_note,
            "created_by": created_by,
        }
        badcase_id = f"badcase_{_canonical_hash(material)[:24]}"
        existing = self.store.badcase(badcase_id)
        if existing is not None:
            return existing
        artifact_id = stable_id("artifact", badcase_id)
        created_at = utc_now()
        badcase = StudioBadcase(
            badcase_id=badcase_id,
            source_run_id=source_run_id,
            source_span_id=source_span_id,
            source_event_ids=tuple(
                item.event_id for item in events
            ),
            input_artifact_ids=input_ids,
            component_versions=component_versions,
            component_version_ids=component_ids,
            evaluation_ids=evaluation_ids,
            evaluation_artifact_ids=evaluation_artifact_ids,
            dataset_sample_artifact_id=dataset_sample_artifact_id,
            human_note=human_note,
            created_by=created_by,
            artifact_id=artifact_id,
            created_at=created_at,
            triggers_change=False,
        )
        same_run_sources = tuple(
            dict.fromkeys(
                artifact_id
                for artifact_id in (
                    *input_ids,
                    *evaluation_artifact_ids,
                    dataset_sample_artifact_id,
                )
                if (
                    self.artifact_store.get(artifact_id) is not None
                    and self.artifact_store.get(artifact_id).run_id
                    == source_run_id
                )
            )
        )
        self.artifact_store.put_json(
            {
                "schema": "StudioBadcase@1",
                "badcase": badcase.model_dump(mode="json"),
                "original_provenance_preserved": True,
                "triggers_change": False,
                "optimizer_invoked": False,
            },
            redact=False,
            kind=ArtifactKind.BADCASE,
            producer_id="runtime_studio_badcase",
            run_id=source_run_id,
            content_schema="StudioBadcase@1",
            source_artifact_ids=same_run_sources,
            artifact_id=artifact_id,
            idempotency_key=f"studio-badcase:{badcase_id}",
        )
        return self.store.save_badcase(badcase)

    def _require_record(self, request_id: str) -> ReplayRecord:
        record = self.store.get(request_id)
        if record is None:
            raise KeyError(f"unknown replay request: {request_id}")
        return record

    def _assert_source_span_sealed(
        self,
        request: ReplayRequest,
    ) -> None:
        events = span_tree_events(
            self.event_store,
            request.source_run_id,
            request.source_span_id,
        )
        if (
            len(events) != request.source_event_count
            or source_event_fingerprint(events)
            != request.source_event_fingerprint
        ):
            raise RuntimeError(
                "source span differs from the sealed replay request"
            )

    def _seal_source_run(self, run_id: str) -> SourceRunSeal:
        events = all_run_events(self.event_store, run_id)
        return SourceRunSeal(
            event_count=len(events),
            fingerprint=source_event_fingerprint(events),
        )

    def _run_dataset_samples(self, run_id: str) -> frozenset[str]:
        result: set[str] = set()
        events = all_run_events(self.event_store, run_id)
        for event in events:
            for artifact_id in (
                *event.input_artifact_ids,
                *event.output_artifact_ids,
                *((event.state_artifact_id,) if event.state_artifact_id else ()),
            ):
                envelope = self.artifact_store.get(artifact_id)
                if (
                    envelope is not None
                    and envelope.kind == ArtifactKind.DATASET_SAMPLE
                ):
                    result.add(artifact_id)
                if (
                    envelope is not None
                    and envelope.kind == ArtifactKind.STUDIO_REPLAY_RESULT
                ):
                    payload = json.loads(
                        self.artifact_store.read_bytes(
                            artifact_id
                        ).decode()
                    )
                    candidate = (
                        payload.get("replay_request", {}).get(
                            "dataset_sample_artifact_id"
                        )
                    )
                    if candidate:
                        result.add(str(candidate))
        found_cursor = None
        while True:
            from deep_researcher.artifacts.store import ArtifactQuery

            page = self.artifact_store.list(
                ArtifactQuery(
                    run_id=run_id,
                    kinds=(
                        ArtifactKind.DATASET_SAMPLE,
                        ArtifactKind.REPLAY_CAPSULE,
                        ArtifactKind.STUDIO_REPLAY_RESULT,
                    ),
                    after_created_at=(
                        found_cursor[0] if found_cursor else None
                    ),
                    after_artifact_id=(
                        found_cursor[1] if found_cursor else None
                    ),
                    limit=1000,
                )
            )
            for envelope in page.items:
                if envelope.kind == ArtifactKind.DATASET_SAMPLE:
                    result.add(envelope.artifact_id)
                elif envelope.kind == ArtifactKind.REPLAY_CAPSULE:
                    candidate = envelope.metadata.get(
                        "dataset_sample_artifact_id"
                    )
                    if candidate:
                        result.add(str(candidate))
                elif envelope.kind == ArtifactKind.STUDIO_REPLAY_RESULT:
                    payload = json.loads(
                        self.artifact_store.read_bytes(
                            envelope.artifact_id
                        ).decode()
                    )
                    candidate = (
                        payload.get("replay_request", {}).get(
                            "dataset_sample_artifact_id"
                        )
                    )
                    if candidate:
                        result.add(str(candidate))
            if page.next_cursor is None:
                break
            found_cursor = page.next_cursor
        return frozenset(result)

    @staticmethod
    def _full_graph(method, run_id: str) -> dict[str, dict[str, Any]]:
        cursor = None
        nodes: dict[str, Any] = {}
        edges: dict[str, Any] = {}
        while True:
            page = method(run_id, cursor=cursor, limit=1000)
            nodes.update(
                {
                    item.node_id: item.model_dump(mode="json")
                    for item in page.nodes
                }
            )
            edges.update(
                {
                    item.edge_id: item.model_dump(mode="json")
                    for item in page.edges
                }
            )
            if page.next_cursor is None:
                break
            cursor = page.next_cursor
        return {"nodes": nodes, "edges": edges}

    @staticmethod
    def _compare_graphs(
        left: dict[str, dict[str, Any]],
        right: dict[str, dict[str, Any]],
    ) -> GraphComparison:
        left_nodes = left["nodes"]
        right_nodes = right["nodes"]
        common = set(left_nodes) & set(right_nodes)
        return GraphComparison(
            left_node_count=len(left_nodes),
            right_node_count=len(right_nodes),
            added_node_ids=tuple(
                sorted(set(right_nodes) - set(left_nodes))
            ),
            removed_node_ids=tuple(
                sorted(set(left_nodes) - set(right_nodes))
            ),
            changed_node_ids=tuple(
                sorted(
                    item
                    for item in common
                    if left_nodes[item] != right_nodes[item]
                )
            ),
            left_edge_count=len(left["edges"]),
            right_edge_count=len(right["edges"]),
            added_edge_ids=tuple(
                sorted(set(right["edges"]) - set(left["edges"]))
            ),
            removed_edge_ids=tuple(
                sorted(set(left["edges"]) - set(right["edges"]))
            ),
        )

    @classmethod
    def _compare_span_graphs(
        cls,
        left_events: tuple[RunEvent, ...],
        right_events: tuple[RunEvent, ...],
    ) -> GraphComparison:
        def graph(events: tuple[RunEvent, ...]):
            nodes: dict[str, dict[str, Any]] = {}
            edges: dict[str, dict[str, Any]] = {}
            for event in events:
                node = nodes.setdefault(
                    event.span_id,
                    {
                        "span_id": event.span_id,
                        "span_kind": event.span_kind.value,
                        "event_types": [],
                        "status": event.status.value,
                    },
                )
                node["event_types"].append(event.event_type.value)
                node["status"] = event.status.value
                if event.parent_span_id:
                    edge_id = (
                        f"{event.parent_span_id}->{event.span_id}"
                    )
                    edges[edge_id] = {
                        "source": event.parent_span_id,
                        "target": event.span_id,
                    }
            return {"nodes": nodes, "edges": edges}

        return cls._compare_graphs(graph(left_events), graph(right_events))

    def _span_scope_events(
        self,
        run_id: str,
        span_id: str | None,
    ) -> tuple[RunEvent, ...]:
        events = all_run_events(self.event_store, run_id)
        if span_id is None:
            return events
        parents: dict[str, str | None] = {}
        for event in events:
            parents[event.span_id] = event.parent_span_id
        if span_id not in parents:
            raise KeyError(f"unknown span: {run_id}/{span_id}")

        def belongs(candidate: str) -> bool:
            seen: set[str] = set()
            while candidate not in seen:
                if candidate == span_id:
                    return True
                seen.add(candidate)
                parent = parents.get(candidate)
                if parent is None:
                    return False
                candidate = parent
            return False

        return tuple(
            item for item in events if belongs(item.span_id)
        )

    @staticmethod
    def _compare_components(
        left: Iterable[Any],
        right: Iterable[Any],
    ) -> tuple[ComponentComparison, ...]:
        left_map = {
            (item.kind, item.name): item for item in left
        }
        right_map = {
            (item.kind, item.name): item for item in right
        }
        return tuple(
            ComponentComparison(
                component_kind=kind,
                component_name=name,
                left_version_id=(
                    left_map[(kind, name)].version_id
                    if (kind, name) in left_map
                    else None
                ),
                right_version_id=(
                    right_map[(kind, name)].version_id
                    if (kind, name) in right_map
                    else None
                ),
                changed=(
                    (kind, name) not in left_map
                    or (kind, name) not in right_map
                    or left_map[(kind, name)].version_id
                    != right_map[(kind, name)].version_id
                ),
            )
            for kind, name in sorted(
                set(left_map) | set(right_map)
            )
        )

    @staticmethod
    def _convergence(events: tuple[RunEvent, ...]) -> dict[str, Any]:
        candidates: list[dict[str, Any]] = []
        for event in events:
            payload = event.payload
            if (
                "convergence" in payload
                or "convergence_action" in payload
                or payload.get("kernel_event_type")
                in {"kernel.stopped", "decision.recorded"}
                or event.event_type == EventType.DECISION_RECORDED
            ):
                candidates.append(
                    {
                        "event_id": event.event_id,
                        "sequence_no": event.sequence_no,
                        "event_type": event.event_type.value,
                        "action": payload.get(
                            "convergence_action",
                            payload.get("action"),
                        ),
                        "reason": payload.get(
                            "reason",
                            payload.get("summary"),
                        ),
                        "payload": payload,
                    }
                )
        return {
            "decision_count": len(candidates),
            "last_decision": candidates[-1] if candidates else None,
            "retry_count": sum(
                item.event_type == EventType.RETRY_SCHEDULED
                for item in events
            ),
            "event_count": len(events),
        }

    def _artifact_text(self, artifact_id: str) -> str:
        envelope = self.artifact_store.get(artifact_id)
        if envelope is None:
            raise KeyError(f"unknown component artifact: {artifact_id}")
        content = self.artifact_store.read_bytes(artifact_id)
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(
                "component diff requires UTF-8 text content"
            ) from exc
        if envelope.media_type.startswith("application/json"):
            try:
                return json.dumps(
                    json.loads(text),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
            except json.JSONDecodeError:
                pass
        return text
