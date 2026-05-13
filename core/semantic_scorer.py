from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


def _normalize_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _tokenize(value: object) -> list[str]:
    return re.findall(r"[\w\u4e00-\u9fff]+", _normalize_text(value).lower())


def _hash_embedding(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokenize(text):
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:2], "big") % dimensions
        sign = 1.0 if digest[2] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    return sum(a * b for a, b in zip(left, right))


class SemanticScorer(Protocol):
    backend_name: str

    def score(self, query_text: str, row_text: str, *, kind: str = "") -> float:
        ...


class HashSemanticScorer:
    backend_name = "hash"

    def score(self, query_text: str, row_text: str, *, kind: str = "") -> float:
        query_vector = _hash_embedding(query_text)
        row_vector = _hash_embedding(row_text)
        return _cosine_similarity(query_vector, row_vector)
