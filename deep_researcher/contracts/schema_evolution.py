from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
from typing import Any, TypeVar

from pydantic import Field

from ._base import ContractModel, validate_semantic_version


ContractT = TypeVar("ContractT", bound=ContractModel)
MigrationFunction = Callable[[dict[str, Any]], dict[str, Any]]


class SerializedContract(ContractModel):
    contract_type: str = Field(min_length=1, max_length=255)
    schema_version: str
    data: dict[str, Any]


class SchemaMigrationRegistry:
    """Deterministic, in-process schema migration graph for domain contracts."""

    def __init__(self) -> None:
        self._models: dict[str, type[ContractModel]] = {}
        self._migrations: dict[tuple[str, str], tuple[str, MigrationFunction]] = {}

    def register_model(self, model: type[ContractT], *, name: str | None = None) -> None:
        contract_type = name or model.__name__
        existing = self._models.get(contract_type)
        if existing is not None and existing is not model:
            raise ValueError(f"contract type already registered: {contract_type}")
        self._models[contract_type] = model

    def register_migration(
        self,
        contract_type: str,
        from_version: str,
        to_version: str,
        migrate: MigrationFunction,
    ) -> None:
        validate_semantic_version(from_version)
        validate_semantic_version(to_version)
        if from_version == to_version:
            raise ValueError("migration versions must differ")
        key = (contract_type, from_version)
        if key in self._migrations:
            raise ValueError(f"migration already registered: {contract_type}@{from_version}")
        self._migrations[key] = (to_version, migrate)

    def encode(self, value: ContractModel) -> SerializedContract:
        contract_type = type(value).__name__
        registered = self._models.get(contract_type)
        if registered is not type(value):
            raise ValueError(f"unregistered contract type: {contract_type}")
        return SerializedContract(
            contract_type=contract_type,
            schema_version=value.schema_version,
            data=value.model_dump(mode="json"),
        )

    def decode(self, envelope: SerializedContract) -> ContractModel:
        model = self._models.get(envelope.contract_type)
        if model is None:
            raise ValueError(f"unknown contract type: {envelope.contract_type}")
        target_version = str(model.model_fields["schema_version"].default)
        version = envelope.schema_version
        data = dict(envelope.data)
        visited: set[str] = set()
        while version != target_version:
            if version in visited:
                raise ValueError("schema migration cycle detected")
            visited.add(version)
            migration = self._migrations.get((envelope.contract_type, version))
            if migration is None:
                raise ValueError(f"no migration path for {envelope.contract_type} from {version} to {target_version}")
            next_version, migrate = migration
            migrated = migrate(dict(data))
            if not isinstance(migrated, dict):
                raise TypeError("schema migration must return a dictionary")
            data = migrated
            version = next_version
        data["schema_version"] = target_version
        return model.model_validate(data)


def canonical_contract_json(value: ContractModel | SerializedContract) -> str:
    payload = value.model_dump(mode="json")
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def contract_fingerprint(value: ContractModel | SerializedContract) -> str:
    return hashlib.sha256(canonical_contract_json(value).encode("utf-8")).hexdigest()
