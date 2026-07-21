from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import re
import uuid
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


CONTRACT_SCHEMA_VERSION = "1.0.0"
_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*_[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_VERSION_PATTERN = re.compile(r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:[-+][0-9A-Za-z.-]+)?$")
_FORBIDDEN_REASONING_KEYS = frozenset({"chain_of_thought", "cot", "hidden_reasoning", "private_reasoning"})


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    normalized = re.sub(r"[^a-z0-9_]+", "_", prefix.strip().lower()).strip("_")
    if not normalized or not normalized[0].isalpha():
        raise ValueError("identifier prefix must begin with a letter")
    return f"{normalized}_{uuid.uuid4().hex}"


def validate_identifier(value: str) -> str:
    value = value.strip()
    if not _ID_PATTERN.fullmatch(value):
        raise ValueError("identifier must be namespaced, for example task_<value>")
    return value


def validate_semantic_version(value: str) -> str:
    value = value.strip()
    if not _VERSION_PATTERN.fullmatch(value):
        raise ValueError("version must use semantic version syntax")
    return value


def reject_hidden_reasoning_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_REASONING_KEYS:
                raise ValueError(f"hidden reasoning field is not permitted: {key}")
            reject_hidden_reasoning_keys(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            reject_hidden_reasoning_keys(nested)
    return value


class ContractModel(BaseModel):
    """Strict, immutable base for data exchanged across architecture layers."""

    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        frozen=True,
        validate_default=True,
        use_enum_values=False,
    )

    schema_version: str = CONTRACT_SCHEMA_VERSION

    @field_validator("schema_version")
    @classmethod
    def _valid_schema_version(cls, value: str) -> str:
        return validate_semantic_version(value)

    @field_validator("*", mode="after")
    @classmethod
    def _timezone_aware_datetimes(cls, value: Any) -> Any:
        if isinstance(value, datetime) and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("contract timestamps must be timezone-aware")
        return reject_hidden_reasoning_keys(value)


class ExtensibleContract(ContractModel):
    """Contract with an explicit extension bag for forward-compatible metadata."""

    extensions: dict[str, Any] = Field(default_factory=dict)


class VersionedEntity(ExtensibleContract):
    id_field: ClassVar[str] = "id"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("timestamps must be timezone-aware")
        return value
