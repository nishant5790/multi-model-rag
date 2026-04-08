from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from agent.config import CHROMA_DIR, QUERY_INDEX_COLLECTION, REPORTS_DIR
from multimodal_rag.models import get_embeddings


class ReportStore:
    """Manages the query_index Chroma collection and on-disk report JSONs."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = QUERY_INDEX_COLLECTION,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        chroma_path = Path(chroma_dir).expanduser().resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._embeddings = get_embeddings()
        self._collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(chroma_path),
        )

    def store_query(self, query: str, job_id: str, report: dict) -> None:
        report_path = self._reports_dir / f"{job_id}.json"
        report_path.write_text(json.dumps(report, default=str), encoding="utf-8")
        doc = Document(
            page_content=query,
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "report_path": str(report_path),
            },
        )
        self._collection.add_documents([doc], ids=[job_id])

    def find_similar(
        self, query: str, threshold: float = 0.85, ttl_days: int = 30
    ) -> tuple[str, dict] | None:
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
        report_path = doc.metadata.get("report_path", "")
        if report_path and Path(report_path).exists():
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        else:
            report = {}
        return doc.page_content, report

    def get_report(self, job_id: str) -> dict | None:
        report_path = self._reports_dir / f"{job_id}.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return None

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
                report_path = (meta or {}).get("report_path", "")
                if report_path and Path(report_path).exists():
                    Path(report_path).unlink()
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
