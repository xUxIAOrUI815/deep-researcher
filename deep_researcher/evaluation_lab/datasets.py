from __future__ import annotations

import hashlib
import json
import threading
from typing import Any

from deep_researcher.artifacts.store import ArtifactStore
from deep_researcher.contracts import (
    ArtifactKind,
    DatasetAccessRequest,
    DatasetDefinition,
    DatasetPurpose,
    DatasetSample,
    DatasetSplit,
    utc_now,
)

from .models import (
    DatasetAccessRecord,
    DatasetBundle,
    DatasetSampleSpec,
    RegisteredDataset,
)
from .store import SQLiteEvaluationStore


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:24]}"


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
        separators=(",", ":"),
    )


def _fingerprint(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


class DatasetRegistry:
    """Versioned, sealed five-way dataset registry with access audit."""

    def __init__(
        self,
        *,
        store: SQLiteEvaluationStore,
        artifact_store: ArtifactStore,
        producer_id: str = "runtime_dataset_registry",
        clock=utc_now,
    ) -> None:
        self.store = store
        self.artifact_store = artifact_store
        self.producer_id = producer_id
        self.clock = clock
        self._registration_lock = threading.RLock()

    def register(
        self,
        *,
        name: str,
        version: str,
        sample_schema: str,
        samples_by_split: dict[
            DatasetSplit,
            tuple[DatasetSampleSpec, ...],
        ],
        description: str,
        parent_bundle_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredDataset:
        # Registration writes one immutable manifest plus several normalized
        # registry rows. Serialize the local transaction boundary so
        # same-version concurrent retries observe the first completed result
        # instead of racing with different wall-clock creation timestamps.
        with self._registration_lock:
            return self._register(
                name=name,
                version=version,
                sample_schema=sample_schema,
                samples_by_split=samples_by_split,
                description=description,
                parent_bundle_id=parent_bundle_id,
                metadata=metadata,
            )

    def _register(
        self,
        *,
        name: str,
        version: str,
        sample_schema: str,
        samples_by_split: dict[
            DatasetSplit,
            tuple[DatasetSampleSpec, ...],
        ],
        description: str,
        parent_bundle_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RegisteredDataset:
        if set(samples_by_split) != set(DatasetSplit):
            raise ValueError(
                "dataset registration requires all five explicit splits"
            )
        if any(not values for values in samples_by_split.values()):
            raise ValueError("every dataset split requires at least one sample")
        parent = (
            self.registered(parent_bundle_id)
            if parent_bundle_id is not None
            else None
        )
        if parent is not None and parent.bundle.name != name:
            raise ValueError("dataset lineage cannot change the dataset name")

        sample_material: dict[DatasetSplit, list[dict[str, Any]]] = {}
        sample_content_fingerprints: dict[
            tuple[DatasetSplit, int],
            str,
        ] = {}
        referenced_created_at = []
        for split in DatasetSplit:
            split_items: list[dict[str, Any]] = []
            for index, spec in enumerate(samples_by_split[split]):
                input_envelope = self.artifact_store.get(
                    spec.input_artifact_id
                )
                if input_envelope is None:
                    raise ValueError(
                        f"dataset input artifact is missing: "
                        f"{spec.input_artifact_id}"
                    )
                referenced_created_at.append(input_envelope.created_at)
                expected_hash = None
                if spec.expected_artifact_id is not None:
                    expected_envelope = self.artifact_store.get(
                        spec.expected_artifact_id
                    )
                    if expected_envelope is None:
                        raise ValueError(
                            "dataset expected artifact is missing: "
                            f"{spec.expected_artifact_id}"
                        )
                    referenced_created_at.append(expected_envelope.created_at)
                    expected_hash = expected_envelope.content_hash
                # Split leakage is keyed by input content, not artifact ID or
                # expected label. Relabeling cannot hide duplicated inputs.
                content_fingerprint = input_envelope.content_hash
                sample_content_fingerprints[(split, index)] = (
                    content_fingerprint
                )
                split_items.append(
                    {
                        "input_artifact_id": spec.input_artifact_id,
                        "input_content_hash": input_envelope.content_hash,
                        "expected_artifact_id": spec.expected_artifact_id,
                        "expected_content_hash": expected_hash,
                        "source_run_id": spec.source_run_id,
                        "tags": list(spec.tags),
                        "metadata": spec.metadata,
                    }
                )
            sample_material[split] = split_items
        split_fingerprints = {
            split: _fingerprint(sample_material[split])
            for split in DatasetSplit
        }
        registration_material = {
            "name": name,
            "version": version,
            "sample_schema": sample_schema,
            "description": description,
            "parent_bundle_id": parent_bundle_id,
            "splits": {
                split.value: sample_material[split]
                for split in DatasetSplit
            },
            "metadata": metadata or {},
        }
        bundle_fingerprint = _fingerprint(registration_material)
        existing = self.store.find_dataset_bundle(name, version)
        if existing is not None:
            if existing.fingerprint != bundle_fingerprint:
                raise ValueError(
                    "dataset name/version already exists with different content"
                )
            return self.registered(existing.bundle_id)

        bundle_id = _stable_id(
            "dataset_bundle",
            name,
            version,
            bundle_fingerprint,
        )
        manifest_artifact_id = _stable_id(
            "artifact",
            bundle_id,
            "manifest",
        )
        artifact = self.artifact_store.get(manifest_artifact_id)
        if artifact is not None:
            payload = json.loads(
                self.artifact_store.read_bytes(
                    manifest_artifact_id
                ).decode("utf-8")
            )
            registered = RegisteredDataset.model_validate(
                payload["registered_dataset"],
                strict=False,
            )
            sample_fps = {
                item["sample_id"]: item["sample_fingerprint"]
                for item in payload["sample_fingerprints"]
            }
            self.store.save_registered_dataset(
                registered,
                sample_fingerprints=sample_fps,
            )
            return registered

        split_dataset_ids = {
            split: _stable_id(
                "dataset",
                bundle_id,
                split.value,
                split_fingerprints[split],
            )
            for split in DatasetSplit
        }
        # The immutable version timestamp is the dataset content cutoff. This
        # is stable across process retries and cannot precede any sealed sample.
        created_at = max(referenced_created_at)
        samples: list[DatasetSample] = []
        sample_fps: dict[str, str] = {}
        for split in DatasetSplit:
            dataset_id = split_dataset_ids[split]
            for index, spec in enumerate(samples_by_split[split]):
                content_fingerprint = sample_content_fingerprints[
                    (split, index)
                ]
                sample_id = _stable_id(
                    "sample",
                    dataset_id,
                    str(index),
                    content_fingerprint,
                )
                sample = DatasetSample(
                    sample_id=sample_id,
                    dataset_id=dataset_id,
                    split=split,
                    input_artifact_id=spec.input_artifact_id,
                    expected_artifact_id=spec.expected_artifact_id,
                    source_run_id=spec.source_run_id,
                    tags=spec.tags,
                    metadata=spec.metadata,
                )
                samples.append(sample)
                sample_fps[sample_id] = content_fingerprint
        definitions = tuple(
            DatasetDefinition(
                dataset_id=split_dataset_ids[split],
                name=f"{name}:{split.value}",
                version=version,
                split=split,
                description=description,
                sample_schema=sample_schema,
                manifest_artifact_id=manifest_artifact_id,
                sample_count=len(samples_by_split[split]),
                fingerprint=split_fingerprints[split],
                parent_dataset_id=(
                    parent.bundle.split_dataset_ids[split]
                    if parent is not None
                    else None
                ),
                created_at=created_at,
                metadata={
                    **(metadata or {}),
                    "bundle_id": bundle_id,
                    "sealed": True,
                },
            )
            for split in DatasetSplit
        )
        bundle = DatasetBundle(
            bundle_id=bundle_id,
            name=name,
            version=version,
            sample_schema=sample_schema,
            split_dataset_ids=split_dataset_ids,
            manifest_artifact_id=manifest_artifact_id,
            fingerprint=bundle_fingerprint,
            parent_bundle_id=parent_bundle_id,
            created_at=created_at,
            metadata={**(metadata or {}), "sealed": True},
        )
        registered = RegisteredDataset(
            bundle=bundle,
            definitions=definitions,
            samples=tuple(samples),
        )
        self.artifact_store.put_json(
            {
                "schema": "RegisteredDataset@1",
                "registered_dataset": registered.model_dump(mode="json"),
                "sample_fingerprints": [
                    {
                        "sample_id": sample_id,
                        "sample_fingerprint": fingerprint,
                    }
                    for sample_id, fingerprint in sorted(sample_fps.items())
                ],
                "registration_material_fingerprint": bundle_fingerprint,
                "split_isolation_enforced": True,
            },
            redact=False,
            kind=ArtifactKind.DATASET_MANIFEST,
            producer_id=self.producer_id,
            run_id=_stable_id("run", bundle_id, "registration"),
            content_schema="RegisteredDataset@1",
            # Dataset versions may intentionally aggregate samples captured by
            # different runs. ArtifactStore forbids cross-run provenance links;
            # the sealed manifest body and registry rows retain those IDs.
            source_artifact_ids=(),
            artifact_id=manifest_artifact_id,
            idempotency_key=f"dataset-manifest:{bundle_id}",
        )
        self.store.save_registered_dataset(
            registered,
            sample_fingerprints=sample_fps,
        )
        return registered

    def registered(self, bundle_id: str) -> RegisteredDataset:
        bundle = self.store.dataset_bundle(bundle_id)
        if bundle is None:
            raise KeyError(f"unknown dataset bundle: {bundle_id}")
        split_order = {
            split: index for index, split in enumerate(DatasetSplit)
        }
        definitions = tuple(
            sorted(
                self.store.dataset_definitions(bundle_id),
                key=lambda item: (
                    split_order[item.split],
                    item.dataset_id,
                ),
            )
        )
        samples = tuple(
            sorted(
                self.store.dataset_samples(bundle_id=bundle_id),
                key=lambda item: (
                    split_order[item.split],
                    item.sample_id,
                ),
            )
        )
        return RegisteredDataset(
            bundle=bundle,
            definitions=definitions,
            samples=samples,
        )

    def access(
        self,
        *,
        bundle_id: str,
        split: DatasetSplit,
        purpose: DatasetPurpose,
        actor_id: str,
        run_id: str | None = None,
        request_id: str | None = None,
    ) -> tuple[DatasetAccessRecord, tuple[DatasetSample, ...]]:
        registered = self.registered(bundle_id)
        dataset_id = registered.bundle.split_dataset_ids[split]
        definition = next(
            item
            for item in registered.definitions
            if item.dataset_id == dataset_id
        )
        request_values: dict[str, Any] = {
            "actor_id": actor_id,
            "dataset_id": dataset_id,
            "split": split,
            "purpose": purpose,
            "run_id": run_id,
        }
        if request_id is not None:
            request_values["request_id"] = request_id
            # A caller-supplied request ID is an idempotency identity. Keep
            # the complete request payload deterministic so simultaneous
            # retries cannot create conflicting immutable audit artifacts.
            request_values["requested_at"] = registered.bundle.created_at
        request = DatasetAccessRequest(**request_values)
        samples = tuple(
            item
            for item in registered.samples
            if item.dataset_id == dataset_id
        )
        access_record_id = _stable_id(
            "dataset_access_record",
            request.request_id,
            bundle_id,
        )
        existing_record = self.store.dataset_access(access_record_id)
        if existing_record is not None:
            if (
                existing_record.request.actor_id != actor_id
                or existing_record.request.dataset_id != dataset_id
                or existing_record.request.split != split
                or existing_record.request.purpose != purpose
                or existing_record.request.run_id != run_id
            ):
                raise ValueError(
                    "dataset access request ID was reused for another request"
                )
            return existing_record, samples
        artifact_id = _stable_id(
            "artifact",
            access_record_id,
        )
        existing_artifact = self.artifact_store.get(artifact_id)
        if existing_artifact is not None:
            payload = json.loads(
                self.artifact_store.read_bytes(artifact_id).decode("utf-8")
            )
            recovered = DatasetAccessRecord.model_validate(
                payload["record"],
                strict=False,
            )
            if (
                recovered.request.actor_id != actor_id
                or recovered.request.dataset_id != dataset_id
                or recovered.request.split != split
                or recovered.request.purpose != purpose
                or recovered.request.run_id != run_id
            ):
                raise ValueError(
                    "dataset access artifact belongs to another request"
                )
            self.store.save_dataset_access(recovered)
            return recovered, samples
        record = DatasetAccessRecord(
            access_record_id=access_record_id,
            request=request,
            bundle_id=bundle_id,
            dataset_definition=definition,
            sample_ids=tuple(item.sample_id for item in samples),
            audit_artifact_id=artifact_id,
            granted_at=request.requested_at,
        )
        self.artifact_store.put_json(
            {
                "schema": "DatasetAccessRecord@1",
                "record": record.model_dump(mode="json"),
                "sample_payloads_disclosed": False,
            },
            redact=False,
            kind=ArtifactKind.DATASET_MANIFEST,
            producer_id=self.producer_id,
            run_id=run_id or _stable_id("run", access_record_id),
            content_schema="DatasetAccessRecord@1",
            source_artifact_ids=(),
            artifact_id=artifact_id,
            idempotency_key=f"dataset-access:{access_record_id}",
        )
        self.store.save_dataset_access(record)
        return record, samples
