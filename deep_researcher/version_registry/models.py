from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from deep_researcher.contracts import ComponentKind, ContractModel, VersionRef
from deep_researcher.contracts._base import (
    new_id,
    validate_identifier,
)


VERSION_REGISTRY_KINDS = frozenset(
    {
        ComponentKind.AGENT_SPEC,
        ComponentKind.SKILL,
        ComponentKind.PROMPT,
        ComponentKind.TOOL_POLICY,
        ComponentKind.STOP_POLICY,
        ComponentKind.RUBRIC,
    }
)


class VersionLifecycleState(str, Enum):
    CANDIDATE = "candidate"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    ROLLED_BACK = "rolled_back"


_ALLOWED_TRANSITIONS = {
    VersionLifecycleState.CANDIDATE: frozenset(
        {
            VersionLifecycleState.PROMOTED,
            VersionLifecycleState.REJECTED,
        }
    ),
    VersionLifecycleState.PROMOTED: frozenset(
        {
            VersionLifecycleState.SUPERSEDED,
            VersionLifecycleState.ROLLED_BACK,
        }
    ),
    VersionLifecycleState.SUPERSEDED: frozenset(
        {VersionLifecycleState.PROMOTED}
    ),
    VersionLifecycleState.REJECTED: frozenset(),
    VersionLifecycleState.ROLLED_BACK: frozenset(),
}


class VersionManifest(ContractModel):
    manifest_id: str = Field(
        default_factory=lambda: new_id("version_manifest")
    )
    version_ref: VersionRef
    parent_version_id: str | None = None
    content_artifact_id: str
    content_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    manifest_artifact_id: str
    registered_at: datetime

    @field_validator(
        "manifest_id",
        "parent_version_id",
        "content_artifact_id",
        "manifest_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @model_validator(mode="after")
    def _kind_and_content(self) -> "VersionManifest":
        if self.version_ref.kind not in VERSION_REGISTRY_KINDS:
            raise ValueError(
                "Version Registry accepts AgentSpec, Skill, Prompt, Tool "
                "Policy, Stop Policy, and Rubric versions only"
            )
        if self.version_ref.artifact_id != self.content_artifact_id:
            raise ValueError(
                "version ref and manifest content artifacts differ"
            )
        if (
            self.version_ref.content_hash is not None
            and self.version_ref.content_hash != self.content_hash
        ):
            raise ValueError("version ref and manifest content hashes differ")
        return self


class VersionTransition(ContractModel):
    transition_id: str = Field(
        default_factory=lambda: new_id("version_transition")
    )
    version_id: str
    sequence: int = Field(ge=1)
    from_state: VersionLifecycleState
    to_state: VersionLifecycleState
    gate_decision_id: str
    gate_decision_artifact_id: str
    reason: str = Field(min_length=1, max_length=4000)
    actor_id: str
    transition_artifact_id: str
    occurred_at: datetime

    @field_validator(
        "transition_id",
        "version_id",
        "gate_decision_id",
        "gate_decision_artifact_id",
        "actor_id",
        "transition_artifact_id",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @model_validator(mode="after")
    def _allowed(self) -> "VersionTransition":
        if self.to_state not in _ALLOWED_TRANSITIONS[self.from_state]:
            raise ValueError(
                f"invalid version transition: "
                f"{self.from_state.value}->{self.to_state.value}"
            )
        return self


class VersionRecord(ContractModel):
    manifest: VersionManifest
    state: VersionLifecycleState
    revision: int = Field(ge=0)
    transitions: tuple[VersionTransition, ...] = ()

    @model_validator(mode="after")
    def _replay(self) -> "VersionRecord":
        state = VersionLifecycleState.CANDIDATE
        sequence = 0
        for item in self.transitions:
            sequence += 1
            if (
                item.version_id != self.manifest.version_ref.version_id
                or item.sequence != sequence
                or item.from_state != state
            ):
                raise ValueError("version record transition history is invalid")
            state = item.to_state
        if state != self.state or sequence != self.revision:
            raise ValueError("version record projection disagrees with history")
        return self
