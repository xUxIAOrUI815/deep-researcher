from __future__ import annotations

from typing import Any

from pydantic import Field

from deep_researcher.contracts import ContractModel


class ScrapedDocument(ContractModel):
    """Normalized page snapshot before knowledge-domain ingestion."""

    url: str
    markdown: str
    title: str = ""
    fetch_method: str = "unknown"
    error: str | None = None
    http_status: int = Field(default=200, ge=0, le=599)
    metadata: dict[str, Any] = Field(default_factory=dict)
