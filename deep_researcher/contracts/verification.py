from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from ._base import ContractModel, new_id, utc_now, validate_identifier
from .budgets import BudgetUsage


class VerificationCategory(str, Enum):
    SOURCE_ACCESS = "source_access"
    SOURCE_AUTHORITY = "source_authority"
    PASSAGE_INTEGRITY = "passage_integrity"
    EVIDENCE_SUPPORT = "evidence_support"
    CLAIM_SUPPORT = "claim_support"
    CITATION_ACCURACY = "citation_accuracy"
    CITATION_COMPLETENESS = "citation_completeness"
    COVERAGE = "coverage"
    CONFLICT = "conflict"
    CONSISTENCY = "consistency"
    REPORT_STRUCTURE = "report_structure"
    POLICY = "policy"


class VerificationSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class VerificationIssue(ContractModel):
    issue_id: str = Field(default_factory=lambda: new_id("verification_issue"))
    category: VerificationCategory
    severity: VerificationSeverity
    code: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=4000)
    subject_id: str
    evidence_ids: tuple[str, ...] = ()
    repairable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("issue_id", "subject_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("evidence_ids")
    @classmethod
    def _evidence_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return value


class RepairAction(str, Enum):
    RESEARCH = "research"
    REFETCH_SOURCE = "refetch_source"
    REEXTRACT_PASSAGE = "reextract_passage"
    REVISE_CLAIM = "revise_claim"
    REPLACE_CITATION = "replace_citation"
    RESOLVE_CONFLICT = "resolve_conflict"
    REWRITE_SECTION = "rewrite_section"
    REQUEST_APPROVAL = "request_approval"


class RepairRequest(ContractModel):
    repair_id: str = Field(default_factory=lambda: new_id("repair"))
    verification_id: str
    issue_ids: tuple[str, ...]
    action: RepairAction
    target_id: str
    instructions: str = Field(min_length=1, max_length=5000)
    priority: float = Field(default=0.5, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("repair_id", "verification_id", "target_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return validate_identifier(value)

    @field_validator("issue_ids")
    @classmethod
    def _issues(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("repair requests require at least one issue")
        for item in value:
            validate_identifier(item)
        return value


class VerificationResult(ContractModel):
    verification_id: str = Field(default_factory=lambda: new_id("verification"))
    run_id: str
    task_id: str | None = None
    verifier_id: str
    subject_id: str
    subject_type: str = Field(min_length=1, max_length=120)
    passed: bool
    score: float = Field(ge=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    checks_performed: tuple[VerificationCategory, ...]
    issues: tuple[VerificationIssue, ...] = ()
    repair_requests: tuple[RepairRequest, ...] = ()
    usage: BudgetUsage = Field(default_factory=BudgetUsage)
    policy_version_id: str
    input_artifact_ids: tuple[str, ...] = ()
    result_artifact_id: str | None = None
    started_at: datetime
    completed_at: datetime = Field(default_factory=utc_now)

    @field_validator("verification_id", "run_id", "task_id", "verifier_id", "subject_id", "policy_version_id", "result_artifact_id")
    @classmethod
    def _ids(cls, value: str | None) -> str | None:
        return validate_identifier(value) if value is not None else None

    @field_validator("input_artifact_ids")
    @classmethod
    def _artifacts(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            validate_identifier(item)
        return value

    @model_validator(mode="after")
    def _consistent(self) -> "VerificationResult":
        if not self.checks_performed:
            raise ValueError("verification must perform at least one check")
        if self.passed != (self.score >= self.threshold):
            raise ValueError("passed must agree with score and threshold")
        blocking = {VerificationSeverity.ERROR, VerificationSeverity.CRITICAL}
        if self.passed and any(issue.severity in blocking for issue in self.issues):
            raise ValueError("a passed result cannot contain blocking issues")
        issue_ids = {issue.issue_id for issue in self.issues}
        for repair in self.repair_requests:
            if repair.verification_id != self.verification_id:
                raise ValueError("repair request belongs to a different verification")
            if not set(repair.issue_ids).issubset(issue_ids):
                raise ValueError("repair request references an unknown issue")
        if self.completed_at < self.started_at:
            raise ValueError("verification completion cannot precede start")
        return self
