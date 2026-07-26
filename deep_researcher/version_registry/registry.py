from __future__ import annotations

from datetime import datetime
import hashlib
import json

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import ArtifactKind, VersionRef

from .models import (
    VersionLifecycleState,
    VersionManifest,
    VersionRecord,
    VersionTransition,
)
from .store import SQLiteVersionRegistryStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


class VersionRegistry:
    """Immutable component versions with auditable release transitions."""

    def __init__(
        self,
        *,
        store: SQLiteVersionRegistryStore,
        artifact_store: ArtifactStore,
        producer_id: str = "runtime_version_registry",
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.producer_id = producer_id

    def register(
        self,
        version_ref: VersionRef,
        *,
        parent_version_id: str | None = None,
    ) -> VersionRecord:
        if version_ref.artifact_id is None:
            raise ValueError(
                "registered component versions require a content artifact"
            )
        content = self.artifact_store.get(version_ref.artifact_id)
        if content is None:
            raise ValueError("version content artifact is missing")
        if (
            version_ref.content_hash is not None
            and version_ref.content_hash != content.content_hash
        ):
            raise ValueError("version content hash does not match its artifact")
        parent = (
            self.store.record(parent_version_id)
            if parent_version_id is not None
            else None
        )
        if parent_version_id is not None and parent is None:
            raise ValueError("version parent is not registered")
        if parent is not None and (
            parent.manifest.version_ref.kind != version_ref.kind
            or parent.manifest.version_ref.name != version_ref.name
        ):
            raise ValueError(
                "version lineage cannot change component kind or name"
            )
        manifest_id = _stable_id(
            "version_manifest",
            version_ref.version_id,
            content.content_hash,
            parent_version_id or "",
        )
        artifact_id = _stable_id("artifact", manifest_id)
        manifest = VersionManifest(
            manifest_id=manifest_id,
            version_ref=version_ref.model_copy(
                update={"content_hash": content.content_hash}
            ),
            parent_version_id=parent_version_id,
            content_artifact_id=content.artifact_id,
            content_hash=content.content_hash,
            manifest_artifact_id=artifact_id,
            registered_at=version_ref.created_at,
        )
        existing = self.store.record(version_ref.version_id)
        if existing is not None:
            if existing.manifest != manifest:
                raise ValueError(
                    "version ID is already registered with other content"
                )
            return existing
        existing_artifact = self.artifact_store.get(artifact_id)
        if existing_artifact is not None:
            payload = json.loads(
                self.artifact_store.read_bytes(artifact_id).decode("utf-8")
            )
            recovered = VersionManifest.model_validate(
                payload["manifest"],
                strict=False,
            )
            if recovered != manifest:
                raise ValueError(
                    "version manifest artifact contains other content"
                )
            self.store.save_manifest(recovered)
            record = self.store.record(version_ref.version_id)
            if record is None:
                raise RuntimeError("version manifest recovery failed")
            return record
        self.artifact_store.put_json(
            {
                "schema": "VersionManifest@1",
                "manifest": manifest.model_dump(mode="json"),
                "immutable": True,
            },
            redact=False,
            kind=ArtifactKind.VERSION_MANIFEST,
            producer_id=self.producer_id,
            run_id=_stable_id("run", manifest_id),
            content_schema="VersionManifest@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"version-manifest:{manifest_id}",
        )
        self.store.save_manifest(manifest)
        record = self.store.record(version_ref.version_id)
        if record is None:
            raise RuntimeError("version registration did not persist")
        return record

    def promote(
        self,
        version_id: str,
        *,
        gate_decision_id: str,
        gate_decision_artifact_id: str,
        reason: str,
        actor_id: str,
        occurred_at: datetime,
    ) -> VersionRecord:
        candidate = self._require(version_id)
        if candidate.state == VersionLifecycleState.PROMOTED:
            return candidate
        if candidate.state not in {
            VersionLifecycleState.CANDIDATE,
            VersionLifecycleState.SUPERSEDED,
        }:
            raise ValueError(
                f"cannot promote version in {candidate.state.value}"
            )
        ref = candidate.manifest.version_ref
        active = self.store.active(ref.kind.value, ref.name)
        transitions: list[VersionTransition] = []
        if active is not None and (
            active.manifest.version_ref.version_id != version_id
        ):
            transitions.append(
                self._transition(
                    active,
                    VersionLifecycleState.SUPERSEDED,
                    gate_decision_id=gate_decision_id,
                    gate_decision_artifact_id=gate_decision_artifact_id,
                    reason=(
                        f"Superseded by {version_id}: {reason}"
                    ),
                    actor_id=actor_id,
                    occurred_at=occurred_at,
                )
            )
        transitions.append(
            self._transition(
                candidate,
                VersionLifecycleState.PROMOTED,
                gate_decision_id=gate_decision_id,
                gate_decision_artifact_id=gate_decision_artifact_id,
                reason=reason,
                actor_id=actor_id,
                occurred_at=occurred_at,
            )
        )
        self.store.append_batch(tuple(transitions))
        return self._require(version_id)

    def reject(
        self,
        version_id: str,
        *,
        gate_decision_id: str,
        gate_decision_artifact_id: str,
        reason: str,
        actor_id: str,
        occurred_at: datetime,
    ) -> VersionRecord:
        candidate = self._require(version_id)
        if candidate.state == VersionLifecycleState.REJECTED:
            return candidate
        if candidate.state != VersionLifecycleState.CANDIDATE:
            raise ValueError(
                f"cannot reject version in {candidate.state.value}"
            )
        transition = self._transition(
            candidate,
            VersionLifecycleState.REJECTED,
            gate_decision_id=gate_decision_id,
            gate_decision_artifact_id=gate_decision_artifact_id,
            reason=reason,
            actor_id=actor_id,
            occurred_at=occurred_at,
        )
        self.store.append_batch((transition,))
        return self._require(version_id)

    def rollback(
        self,
        active_version_id: str,
        *,
        target_version_id: str,
        gate_decision_id: str,
        gate_decision_artifact_id: str,
        reason: str,
        actor_id: str,
        occurred_at: datetime,
    ) -> tuple[VersionRecord, VersionRecord]:
        active = self._require(active_version_id)
        target = self._require(target_version_id)
        if active.state == VersionLifecycleState.ROLLED_BACK:
            current = self.store.active(
                target.manifest.version_ref.kind.value,
                target.manifest.version_ref.name,
            )
            if (
                current is not None
                and current.manifest.version_ref.version_id
                == target_version_id
            ):
                return active, current
        if active.state != VersionLifecycleState.PROMOTED:
            raise ValueError("rollback source must be the promoted version")
        if target.state != VersionLifecycleState.SUPERSEDED:
            raise ValueError("rollback target must be a superseded version")
        active_ref = active.manifest.version_ref
        target_ref = target.manifest.version_ref
        if (
            active_ref.kind != target_ref.kind
            or active_ref.name != target_ref.name
        ):
            raise ValueError(
                "rollback versions must identify the same component"
            )
        transitions = (
            self._transition(
                active,
                VersionLifecycleState.ROLLED_BACK,
                gate_decision_id=gate_decision_id,
                gate_decision_artifact_id=gate_decision_artifact_id,
                reason=reason,
                actor_id=actor_id,
                occurred_at=occurred_at,
            ),
            self._transition(
                target,
                VersionLifecycleState.PROMOTED,
                gate_decision_id=gate_decision_id,
                gate_decision_artifact_id=gate_decision_artifact_id,
                reason=(
                    f"Restored by rollback of {active_version_id}: {reason}"
                ),
                actor_id=actor_id,
                occurred_at=occurred_at,
            ),
        )
        self.store.append_batch(transitions)
        return (
            self._require(active_version_id),
            self._require(target_version_id),
        )

    def _transition(
        self,
        record: VersionRecord,
        to_state: VersionLifecycleState,
        *,
        gate_decision_id: str,
        gate_decision_artifact_id: str,
        reason: str,
        actor_id: str,
        occurred_at: datetime,
    ) -> VersionTransition:
        decision_artifact = self.artifact_store.get(
            gate_decision_artifact_id
        )
        if decision_artifact is None:
            raise ValueError("version transition gate artifact is missing")
        if decision_artifact.kind != ArtifactKind.RELEASE_GATE_DECISION:
            raise ValueError(
                "version transition requires a release-gate decision artifact"
            )
        version_id = record.manifest.version_ref.version_id
        transition_id = _stable_id(
            "version_transition",
            version_id,
            str(record.revision + 1),
            record.state.value,
            to_state.value,
            gate_decision_id,
        )
        artifact_id = _stable_id("artifact", transition_id)
        transition = VersionTransition(
            transition_id=transition_id,
            version_id=version_id,
            sequence=record.revision + 1,
            from_state=record.state,
            to_state=to_state,
            gate_decision_id=gate_decision_id,
            gate_decision_artifact_id=gate_decision_artifact_id,
            reason=reason,
            actor_id=actor_id,
            transition_artifact_id=artifact_id,
            occurred_at=occurred_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "VersionTransition@1",
                "transition": transition.model_dump(mode="json"),
                "changes_component_content": False,
            },
            redact=False,
            kind=ArtifactKind.VERSION_TRANSITION,
            producer_id=self.producer_id,
            run_id=_stable_id("run", gate_decision_id),
            content_schema="VersionTransition@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"version-transition:{transition_id}",
        )
        return transition

    def _require(self, version_id: str) -> VersionRecord:
        record = self.store.record(version_id)
        if record is None:
            raise KeyError(f"unknown registered version: {version_id}")
        return record
