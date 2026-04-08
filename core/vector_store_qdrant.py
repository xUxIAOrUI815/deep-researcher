import uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np
from typing import List, Tuple, Optional, Dict, Any


class QdrantVectorStore:
    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334
    ):
        self.client = QdrantClient(
            host=host,
            port=port,
            prefer_grpc=True
        )
        self.vector_size = 1024

    def _ensure_uuid(self, point_id: str) -> str:
        try:
            uuid.UUID(point_id)
            return point_id
        except ValueError:
            return str(uuid.uuid4())

    def create_collection(self, collection_name: str) -> bool:
        try:
            self.client.recreate_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE
                )
            )
            print(f"[QdrantVectorStore] Collection '{collection_name}' created")
            return True
        except Exception as e:
            print(f"[QdrantVectorStore] Create collection failed: {e}")
            return False

    def collection_exists(self, collection_name: str) -> bool:
        try:
            self.client.get_collection(collection_name)
            return True
        except:
            return False

    def upsert_point(
        self,
        collection_name: str,
        point_id: str,
        vector: List[float],
        payload: Dict[str, Any]
    ) -> bool:
        try:
            if not self.collection_exists(collection_name):
                self.create_collection(collection_name)

            point_id = self._ensure_uuid(point_id)

            self.client.upsert(
                collection_name=collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return True
        except Exception as e:
            print(f"[QdrantVectorStore] Upsert failed: {e}")
            return False

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        try:
            if not self.collection_exists(collection_name):
                return []

            results = self.client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                with_payload=True
            )
            return [(r.id, r.score, r.payload) for r in results.points]
        except Exception as e:
            print(f"[QdrantVectorStore] Search failed: {e}")
            return []

    def retrieve(
        self,
        collection_name: str,
        point_id: str
    ) -> Optional[Dict]:
        try:
            if not self.collection_exists(collection_name):
                return None

            point_id = self._ensure_uuid(point_id)
            results = self.client.retrieve(
                collection_name=collection_name,
                ids=[point_id],
                with_payload=True
            )
            return results[0].payload if results else None
        except Exception as e:
            print(f"[QdrantVectorStore] Retrieve failed: {e}")
            return None

    def delete_collection(self, collection_name: str) -> bool:
        try:
            self.client.delete_collection(collection_name=collection_name)
            print(f"[QdrantVectorStore] Collection '{collection_name}' deleted")
            return True
        except Exception as e:
            print(f"[QdrantVectorStore] Delete collection failed: {e}")
            return False

    def get_collection_info(self, collection_name: str) -> Optional[Dict]:
        try:
            info = self.client.get_collection(collection_name)
            return {
                "points_count": info.points_count,
                "status": info.status
            }
        except Exception as e:
            print(f"[QdrantVectorStore] Get collection info failed: {e}")
            return None
