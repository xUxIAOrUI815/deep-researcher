from __future__ import annotations

import hashlib
import json
from typing import Protocol

from deep_researcher.contracts import canonical_contract_json

from .models import (
    AddPlacement,
    CrossTaskExperience,
    OfflineGeneratorResult,
    OptimizationTarget,
    PatchEditMetrics,
    PatchOperationKind,
    TextEditBudget,
    TextPatchOperation,
)


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def operation_fingerprint(operation: TextPatchOperation) -> str:
    return hashlib.sha256(
        canonical_contract_json(operation).encode("utf-8")
    ).hexdigest()


def patch_fingerprint(
    *,
    base_content_hash: str,
    target: OptimizationTarget,
    operations: tuple[TextPatchOperation, ...],
) -> str:
    return canonical_hash(
        {
            "base_content_hash": base_content_hash,
            "target": target.value,
            "operations": [
                item.model_dump(mode="json") for item in operations
            ],
        }
    )


class PatchApplicationError(ValueError):
    pass


class PatchBudgetExceeded(PatchApplicationError):
    pass


class TextPatchApplier:
    """Exact line patching with an enforced per-round text learning rate."""

    def apply(
        self,
        base_text: str,
        operations: tuple[TextPatchOperation, ...],
        budget: TextEditBudget,
    ) -> tuple[str, PatchEditMetrics]:
        if not operations:
            raise PatchApplicationError("structured patch cannot be empty")
        if len(operations) > budget.max_operations_per_round:
            raise PatchBudgetExceeded(
                "patch exceeds max_operations_per_round"
            )
        newline = "\r\n" if "\r\n" in base_text else "\n"
        trailing_newline = base_text.endswith(("\n", "\r"))
        base_lines = base_text.splitlines()
        self._validate_ranges(base_lines, operations)
        added_lines = sum(len(item.new_lines) for item in operations)
        deleted_lines = sum(len(item.old_lines) for item in operations)
        changed_characters = sum(
            sum(len(line) + 1 for line in item.old_lines)
            + sum(len(line) + 1 for line in item.new_lines)
            for item in operations
        )
        edit_fraction = changed_characters / max(len(base_text), 1)
        metrics = PatchEditMetrics(
            operation_count=len(operations),
            added_lines=added_lines,
            deleted_lines=deleted_lines,
            changed_characters=changed_characters,
            edit_fraction=edit_fraction,
        )
        if added_lines > budget.max_added_lines_per_round:
            raise PatchBudgetExceeded(
                "patch exceeds max_added_lines_per_round"
            )
        if deleted_lines > budget.max_deleted_lines_per_round:
            raise PatchBudgetExceeded(
                "patch exceeds max_deleted_lines_per_round"
            )
        if (
            changed_characters
            > budget.max_changed_characters_per_round
        ):
            raise PatchBudgetExceeded(
                "patch exceeds max_changed_characters_per_round"
            )
        if edit_fraction > budget.max_edit_fraction_per_round:
            raise PatchBudgetExceeded(
                "patch exceeds max_edit_fraction_per_round"
            )
        updated = list(base_lines)
        for operation in sorted(
            operations,
            key=lambda item: (item.start_line, item.end_line),
            reverse=True,
        ):
            updated[operation.start_line : operation.end_line] = list(
                operation.new_lines
            )
        result = newline.join(updated)
        if trailing_newline and updated:
            result += newline
        if result == base_text:
            raise PatchApplicationError("structured patch made no change")
        return result, metrics

    @staticmethod
    def _validate_ranges(
        base_lines: list[str],
        operations: tuple[TextPatchOperation, ...],
    ) -> None:
        ordered = sorted(
            operations,
            key=lambda item: (item.start_line, item.end_line),
        )
        occupied: set[int] = set()
        insertion_points: set[int] = set()
        for operation in ordered:
            if operation.end_line > len(base_lines):
                raise PatchApplicationError(
                    "patch line range exceeds base content"
                )
            if (
                tuple(
                    base_lines[
                        operation.start_line : operation.end_line
                    ]
                )
                != operation.old_lines
            ):
                raise PatchApplicationError(
                    "patch old lines do not match immutable base content"
                )
            if operation.operation == PatchOperationKind.ADD:
                point = operation.start_line
                if point in insertion_points or point in occupied:
                    raise PatchApplicationError(
                        "patch contains overlapping insertions"
                    )
                insertion_points.add(point)
                continue
            indexes = set(range(operation.start_line, operation.end_line))
            if occupied.intersection(indexes):
                raise PatchApplicationError(
                    "patch operations overlap each other"
                )
            if any(
                operation.start_line <= point <= operation.end_line
                for point in insertion_points
            ):
                raise PatchApplicationError(
                    "patch insertion overlaps a replacement range"
                )
            occupied.update(indexes)


class OfflinePatchGenerator(Protocol):
    def generate(
        self,
        *,
        target: OptimizationTarget,
        base_text: str,
        experiences: tuple[CrossTaskExperience, ...],
        budget: TextEditBudget,
        rejected_operation_fingerprints: frozenset[str],
    ) -> OfflineGeneratorResult:
        ...


class TraceSignalPatchGenerator:
    """Deterministic optimizer over reviewed, structured trace signals."""

    def __init__(self, *, applier: TextPatchApplier | None = None) -> None:
        self.applier = applier or TextPatchApplier()

    def generate(
        self,
        *,
        target: OptimizationTarget,
        base_text: str,
        experiences: tuple[CrossTaskExperience, ...],
        budget: TextEditBudget,
        rejected_operation_fingerprints: frozenset[str],
    ) -> OfflineGeneratorResult:
        relevant = tuple(
            sorted(
                (item for item in experiences if item.target == target),
                key=lambda item: (
                    -(item.confidence * item.impact),
                    item.experience_id,
                ),
            )
        )
        if not relevant:
            raise ValueError(
                f"no reviewed cross-task experience for {target.value}"
            )
        base_lines = base_text.splitlines()
        selected: list[TextPatchOperation] = []
        rationale: list[str] = []
        skipped: list[str] = []
        for experience in relevant:
            operation = self._resolve(experience, base_lines)
            if operation is None:
                continue
            fingerprint = operation_fingerprint(operation)
            if fingerprint in rejected_operation_fingerprints:
                skipped.append(fingerprint)
                continue
            trial = tuple((*selected, operation))
            try:
                self.applier.apply(base_text, trial, budget)
            except PatchApplicationError:
                continue
            selected.append(operation)
            rationale.append(
                f"{experience.summary}: {experience.recommendation}"
            )
            if len(selected) >= budget.max_operations_per_round:
                break
        if not selected:
            raise ValueError(
                "all applicable structured edits were invalid, over budget, "
                "or present in rejected-edit memory"
            )
        return OfflineGeneratorResult(
            operations=tuple(selected),
            rationale=tuple(rationale),
            consulted_experience_ids=tuple(
                item.experience_id for item in relevant
            ),
            skipped_rejected_operation_fingerprints=tuple(
                dict.fromkeys(skipped)
            ),
            online_inference_count=0,
            network_accessed=False,
        )

    def _resolve(
        self,
        experience: CrossTaskExperience,
        base_lines: list[str],
    ) -> TextPatchOperation | None:
        suggestion = experience.suggestion
        if suggestion.operation == PatchOperationKind.ADD:
            placement = suggestion.add_placement
            if placement == AddPlacement.START:
                index = 0
            elif placement == AddPlacement.END:
                index = len(base_lines)
            else:
                matches = self._matches(
                    base_lines,
                    suggestion.match_lines,
                )
                if len(matches) != 1:
                    return None
                index = matches[0]
                if placement == AddPlacement.AFTER:
                    index += len(suggestion.match_lines)
            return TextPatchOperation(
                operation=PatchOperationKind.ADD,
                start_line=index,
                end_line=index,
                old_lines=(),
                new_lines=suggestion.new_lines,
            )
        matches = self._matches(base_lines, suggestion.match_lines)
        if len(matches) != 1:
            return None
        start = matches[0]
        end = start + len(suggestion.match_lines)
        return TextPatchOperation(
            operation=suggestion.operation,
            start_line=start,
            end_line=end,
            old_lines=suggestion.match_lines,
            new_lines=suggestion.new_lines,
        )

    @staticmethod
    def _matches(
        lines: list[str],
        needle: tuple[str, ...],
    ) -> tuple[int, ...]:
        if not needle or len(needle) > len(lines):
            return ()
        width = len(needle)
        return tuple(
            index
            for index in range(len(lines) - width + 1)
            if tuple(lines[index : index + width]) == needle
        )
