"""Chroma vector store helpers."""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from multimodal_rag.models import get_embeddings


def get_vectorstore(
    persist_directory: Path | str,
    *,
    collection_name: str = "multimodal_rag",
    embeddings: Embeddings | None = None,
) -> Chroma:
    path = Path(persist_directory).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    emb = embeddings or get_embeddings()
    return Chroma(
        collection_name=collection_name,
        embedding_function=emb,
        persist_directory=str(path),
    )
