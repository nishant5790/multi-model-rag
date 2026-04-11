from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from agent.config import QDRANT_URL, QDRANT_API_KEY, QUERY_INDEX_COLLECTION
from multimodal_rag.config import EMBEDDING_DIMENSIONALITY
from multimodal_rag.models import get_embeddings


class ReportStore:
    """Manages the query_index Qdrant collection for tracking processed queries."""

    def __init__(
        self,
        collection_name: str = QUERY_INDEX_COLLECTION,
        **_kwargs,
    ) -> None:
        self._embeddings = get_embeddings()
        self._client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        self._ensure_collection(collection_name)
        self._collection = QdrantVectorStore(
            client=self._client,
            collection_name=collection_name,
            embedding=self._embeddings,
        )

    def _ensure_collection(self, name: str) -> None:
        from qdrant_client.models import Distance, VectorParams
        collections = [c.name for c in self._client.get_collections().collections]
        if name not in collections:
            self._client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIMENSIONALITY,
                    distance=Distance.COSINE,
                ),
            )

    def store_query(self, query: str, job_id: str) -> None:
        doc = Document(
            page_content=query,
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
            },
        )
        self._collection.add_documents([doc], ids=[job_id])

    def find_similar(
        self, query: str, threshold: float = 0.85, ttl_days: int = 30
    ) -> str | None:
        results = self._collection.similarity_search_with_score(query, k=1)
        if not results:
            return None
        doc, score = results[0]
        if score < threshold:
            return None
        created_str = doc.metadata.get("created_at", "")
        if created_str:
            created = datetime.fromisoformat(created_str)
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            if created < cutoff:
                return None
        return doc.page_content

    def cleanup_old(self, days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        results = self._collection.similarity_search("", k=1000)
        ids_to_delete: list[str] = []
        for doc in results:
            created_str = doc.metadata.get("created_at", "")
            if not created_str:
                continue
            created = datetime.fromisoformat(created_str)
            if created < cutoff:
                doc_id = doc.metadata.get("job_id")
                if doc_id:
                    ids_to_delete.append(doc_id)
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
