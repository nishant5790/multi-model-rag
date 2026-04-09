from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from agent.config import CHROMA_DIR, QUERY_INDEX_COLLECTION
from multimodal_rag.models import get_embeddings


class ReportStore:
    """Manages the query_index Chroma collection for tracking processed queries."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = QUERY_INDEX_COLLECTION,
    ) -> None:
        chroma_path = Path(chroma_dir).expanduser().resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._embeddings = get_embeddings()
        self._collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(chroma_path),
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
        results = self._collection.similarity_search_with_relevance_scores(
            query, k=1
        )
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
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        for i, meta in enumerate(all_data.get("metadatas") or []):
            created_str = (meta or {}).get("created_at", "")
            if not created_str:
                continue
            created = datetime.fromisoformat(created_str)
            if created < cutoff:
                doc_id = all_data["ids"][i]
                ids_to_delete.append(doc_id)
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
