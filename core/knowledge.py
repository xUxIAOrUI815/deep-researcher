import asyncio
import os
import uuid
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
import httpx

from schemas.state import AtomicFact

SILICON_FLOW_API_KEY = os.getenv("SILICON_FLOW_API_KEY", "")
SILICON_FLOW_EMBEDDING_URL = "https://api.siliconflow.cn/v1/embeddings"


class FactStatus(Enum):
    ACTIVE = "active"
    VERIFIED = "verified"
    CONFLICTING = "conflicting"
    SUPERSEDED = "superseded"


class EmbeddingModel:
    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self.model_name = model_name
        self.api_key = SILICON_FLOW_API_KEY
        self._embedding_lock = asyncio.Lock()
        self._last_request_time = 0
        self._min_request_interval = 0.1

    async def get_embedding(self, text: str) -> List[float]:
        async with self._embedding_lock:
            now = asyncio.get_event_loop().time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = asyncio.get_event_loop().time()

        if not self.api_key:
            raise ValueError("SILICON_FLOW_API_KEY not set in environment")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": text
        }

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                SILICON_FLOW_EMBEDDING_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("data", [{}])[0].get("embedding", [])
            return embedding

    async def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise ValueError("SILICON_FLOW_API_KEY not set in environment")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model_name,
            "input": texts
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                SILICON_FLOW_EMBEDDING_URL,
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            data = response.json()

            embeddings = [item.get("embedding", []) for item in data.get("data", [])]
            return embeddings


@dataclass
class FactConflict:
    fact_id_1: str
    fact_id_2: str
    conflict_description: str
    detected_at: datetime = field(default_factory=datetime.now)


@dataclass
class KnowledgeStats:
    total_facts: int = 0
    verified_facts: int = 0
    conflicting_facts: int = 0
    conflicts_detected: int = 0
    duplicates_merged: int = 0


class StoredFact:
    def __init__(self, fact: AtomicFact, embedding: List[float]):
        self.id = str(uuid.uuid4())
        self.fact = fact
        self.embedding = np.array(embedding)
        self.status = FactStatus.ACTIVE.value
        self.verified_by: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "text": self.fact.text,
            "source_url": self.fact.source_url,
            "confidence": self.fact.confidence,
            "task_id": self.fact.task_id,
            "timestamp": self.fact.timestamp.isoformat(),
            "status": self.status,
            "verified_by": self.verified_by
        }


class KnowledgeManager:
    def __init__(self, storage_path: str = "./knowledge_data"):
        self.storage_path = storage_path
        self.embedding_model = EmbeddingModel()
        self._facts: Dict[str, StoredFact] = {}
        self._semaphore = asyncio.Semaphore(3)
        self._stats = KnowledgeStats()
        self._conflicts: List[FactConflict] = []
        os.makedirs(storage_path, exist_ok=True)
        self._load()

    def _load(self):
        index_file = os.path.join(self.storage_path, "facts_index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._stats = KnowledgeStats(**data.get("stats", {}))
                    facts_data = data.get("facts", {})
                    for fid, fdict in facts_data.items():
                        fact = AtomicFact(
                            text=fdict["text"],
                            source_url=fdict["source_url"],
                            confidence=fdict["confidence"],
                            task_id=fdict.get("task_id")
                        )
                        emb = np.array(fdict["embedding"])
                        sf = StoredFact(fact, emb.tolist())
                        sf.id = fid
                        sf.status = fdict.get("status", "active")
                        sf.verified_by = fdict.get("verified_by")
                        self._facts[fid] = sf
            except Exception as e:
                print(f"[KnowledgeManager] Warning: Could not load existing data: {e}")

    def _save(self):
        index_file = os.path.join(self.storage_path, "facts_index.json")
        data = {
            "stats": {
                "total_facts": self._stats.total_facts,
                "verified_facts": self._stats.verified_facts,
                "conflicting_facts": self._stats.conflicting_facts,
                "conflicts_detected": self._stats.conflicts_detected,
                "duplicates_merged": self._stats.duplicates_merged
            },
            "facts": {fid: sf.to_dict() | {"embedding": sf.embedding.tolist()} for fid, sf in self._facts.items()}
        }
        try:
            with open(index_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[KnowledgeManager] Warning: Could not save data: {e}")

    async def _compute_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if norm_product == 0:
            return 0.0
        return dot_product / norm_product

    async def _find_similar(self, text: str, threshold: float = 0.7) -> List[Tuple[str, float, StoredFact]]:
        embedding = await self.embedding_model.get_embedding(text)
        query_vec = np.array(embedding)
        similar = []
        for fid, sf in self._facts.items():
            score = await self._compute_similarity(query_vec, sf.embedding)
            if score >= threshold:
                similar.append((fid, score, sf))
        similar.sort(key=lambda x: x[1], reverse=True)
        return similar

    async def add_facts(self, facts: List[AtomicFact], task_id: Optional[str] = None) -> Tuple[List[str], List[str]]:
        added_ids = []
        duplicate_ids = []
        for fact in facts:
            async with self._semaphore:
                result = await self._add_single_fact(fact, task_id)
                if result:
                    added_ids.append(result)
                else:
                    duplicate_ids.append(str(uuid.uuid4())[:8])
        self._save()
        return added_ids, duplicate_ids

    async def _add_single_fact(self, fact: AtomicFact, task_id: Optional[str] = None) -> Optional[str]:
        embedding = await self.embedding_model.get_embedding(fact.text)
        similar = await self._find_similar(fact.text, threshold=0.7)

        for existing_id, score, existing_fact in similar:
            if score > 0.85 and existing_fact.fact.source_url == fact.source_url:
                self._stats.duplicates_merged += 1
                return None

            if score > 0.85 and existing_fact.fact.source_url != fact.source_url:
                existing_fact.status = FactStatus.VERIFIED.value
                existing_fact.verified_by = existing_id
                new_fact = AtomicFact(
                    text=fact.text,
                    source_url=fact.source_url,
                    confidence=min(1.0, fact.confidence + 0.1),
                    task_id=task_id or fact.task_id
                )
                new_stored = StoredFact(new_fact, embedding)
                new_stored.status = FactStatus.VERIFIED.value
                self._facts[new_stored.id] = new_stored
                self._stats.verified_facts += 1
                self._stats.total_facts += 1
                return new_stored.id

            if score > 0.7 and existing_fact.fact.source_url != fact.source_url:
                numbers1 = set(re.findall(r'\d+\.?\d*', existing_fact.fact.text))
                numbers2 = set(re.findall(r'\d+\.?\d*', fact.text))
                if numbers1 and numbers2 and numbers1 != numbers2:
                    conflict = FactConflict(
                        fact_id_1=existing_id,
                        fact_id_2="new",
                        conflict_description=f"数值矛盾: '{existing_fact.fact.text[:50]}' vs '{fact.text[:50]}'"
                    )
                    self._conflicts.append(conflict)
                    self._stats.conflicts_detected += 1
                    existing_fact.status = FactStatus.CONFLICTING.value
                    self._stats.conflicting_facts += 1

        new_fact = AtomicFact(
            text=fact.text,
            source_url=fact.source_url,
            confidence=fact.confidence,
            task_id=task_id or fact.task_id
        )
        new_stored = StoredFact(new_fact, embedding)
        self._facts[new_stored.id] = new_stored
        self._stats.total_facts += 1
        return new_stored.id

    async def search_facts(self, query: str, task_id_filter: Optional[str] = None, status_filter: Optional[FactStatus] = None, limit: int = 10) -> List[Dict[str, Any]]:
        similar = await self._find_similar(query, threshold=0.0)
        results = []
        for fid, score, sf in similar:
            if task_id_filter and sf.fact.task_id != task_id_filter:
                continue
            if status_filter and sf.status != status_filter.value:
                continue
            results.append({
                "id": fid,
                "text": sf.fact.text,
                "source_url": sf.fact.source_url,
                "confidence": sf.fact.confidence,
                "status": sf.status,
                "score": float(score)
            })
            if len(results) >= limit:
                break
        return results

    async def get_fact_by_id(self, fact_id: str) -> Optional[Dict[str, Any]]:
        sf = self._facts.get(fact_id)
        if sf:
            return sf.to_dict()
        return None

    async def get_conflicts(self) -> List[Dict[str, Any]]:
        return [{"fact_id_1": c.fact_id_1, "fact_id_2": c.fact_id_2, "description": c.conflict_description, "detected_at": c.detected_at.isoformat()} for c in self._conflicts]

    def get_stats(self) -> KnowledgeStats:
        return self._stats

    async def delete_fact(self, fact_id: str) -> bool:
        if fact_id in self._facts:
            del self._facts[fact_id]
            self._save()
            return True
        return False

    async def clear_collection(self):
        self._facts = {}
        self._stats = KnowledgeStats()
        self._conflicts = []
        self._save()


import re
