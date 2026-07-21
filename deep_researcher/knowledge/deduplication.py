from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Protocol

from .normalization import content_fingerprint, normalize_text


class VectorSimilarityAdapter(Protocol):
    def similarity(self, left: str, right: str) -> float:
        ...


@dataclass(frozen=True)
class DuplicateDecision:
    duplicate: bool
    score: float
    method: str
    left_fingerprint: str
    right_fingerprint: str


class DeduplicationService:
    def __init__(
        self,
        *,
        lexical_threshold: float = 0.92,
        vector_threshold: float = 0.94,
        vector_adapter: VectorSimilarityAdapter | None = None,
    ) -> None:
        if not 0.0 <= lexical_threshold <= 1.0 or not 0.0 <= vector_threshold <= 1.0:
            raise ValueError("deduplication thresholds must be between zero and one")
        self.lexical_threshold = lexical_threshold
        self.vector_threshold = vector_threshold
        self.vector_adapter = vector_adapter

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return set(re.findall(r"[\w\u4e00-\u9fff]+", normalize_text(value).casefold()))

    def compare(self, left: str, right: str) -> DuplicateDecision:
        left_hash = content_fingerprint(left)
        right_hash = content_fingerprint(right)
        if left_hash == right_hash:
            return DuplicateDecision(True, 1.0, "exact", left_hash, right_hash)
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        union = left_tokens | right_tokens
        lexical = len(left_tokens & right_tokens) / len(union) if union else 0.0
        if lexical >= self.lexical_threshold:
            return DuplicateDecision(True, lexical, "lexical_jaccard", left_hash, right_hash)
        if self.vector_adapter is not None:
            vector = float(self.vector_adapter.similarity(left, right))
            if not math.isfinite(vector) or vector < -1.0 or vector > 1.0:
                raise ValueError("vector adapter returned an invalid similarity")
            if vector >= self.vector_threshold:
                return DuplicateDecision(True, vector, "vector", left_hash, right_hash)
            return DuplicateDecision(False, max(lexical, vector), "lexical_and_vector", left_hash, right_hash)
        return DuplicateDecision(False, lexical, "lexical_jaccard", left_hash, right_hash)
