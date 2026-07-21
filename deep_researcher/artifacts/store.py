from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from deep_researcher.contracts import ArtifactEnvelope, ArtifactKind, ArtifactLink, ArtifactLinkRelation


class ArtifactStoreError(RuntimeError):
    """Base class for immutable artifact-store failures."""


class ArtifactConflict(ArtifactStoreError):
    """An artifact ID or idempotency key was reused for different content."""


class ArtifactNotFound(ArtifactStoreError):
    """An artifact or content blob does not exist."""


class ArtifactCorruption(ArtifactStoreError):
    """Stored bytes, envelopes, links, or indexes failed integrity checks."""


@dataclass(frozen=True)
class ArtifactQuery:
    run_id: str
    kinds: tuple[ArtifactKind, ...] = ()
    producer_id: str | None = None
    task_id: str | None = None
    after_created_at: datetime | None = None
    after_artifact_id: str | None = None
    limit: int = 100

    def __post_init__(self) -> None:
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        if (self.after_created_at is None) != (self.after_artifact_id is None):
            raise ValueError("artifact cursor fields must be supplied together")
        if self.after_created_at is not None and (
            self.after_created_at.tzinfo is None or self.after_created_at.utcoffset() is None
        ):
            raise ValueError("artifact cursor timestamp must be timezone-aware")


@dataclass(frozen=True)
class ArtifactPage:
    items: tuple[ArtifactEnvelope, ...]
    next_cursor: tuple[datetime, str] | None


class ArtifactStore(Protocol):
    def put_json(self, value: Any, *, redact: bool = True, **kwargs: Any) -> ArtifactEnvelope:
        ...

    def put_text(self, value: str, *, redact: bool = False, **kwargs: Any) -> ArtifactEnvelope:
        ...

    def put_bytes(
        self,
        content: bytes,
        *,
        kind: ArtifactKind,
        media_type: str,
        producer_id: str,
        run_id: str,
        task_id: str | None = None,
        content_schema: str | None = None,
        source_artifact_ids: tuple[str, ...] = (),
        metadata: dict[str, Any] | None = None,
        artifact_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> ArtifactEnvelope:
        ...

    def get(self, artifact_id: str) -> ArtifactEnvelope | None:
        ...

    def read_bytes(self, artifact_id: str) -> bytes:
        ...

    def link(self, link: ArtifactLink) -> None:
        ...

    def links(
        self,
        artifact_id: str,
        *,
        incoming: bool = False,
        relation: ArtifactLinkRelation | None = None,
    ) -> tuple[ArtifactLink, ...]:
        ...

    def list(self, query: ArtifactQuery) -> ArtifactPage:
        ...

    def integrity_check(self) -> None:
        ...

    def close(self) -> None:
        ...
