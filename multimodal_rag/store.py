"""Qdrant Cloud vector store helpers."""

import os

from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from multimodal_rag.config import EMBEDDING_DIMENSIONALITY
from multimodal_rag.models import get_embeddings


def _get_qdrant_client() -> QdrantClient:
    url = os.environ["QDRANT_URL"]
    api_key = os.environ["QDRANT_API_KEY"]
    return QdrantClient(url=url, api_key=api_key)


def _ensure_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """Create collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if name not in collections:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )


def get_vectorstore(
    collection_name: str = "multimodal_rag",
    *,
    embeddings: Embeddings | None = None,
    **_kwargs,
) -> QdrantVectorStore:
    emb = embeddings or get_embeddings()
    client = _get_qdrant_client()
    _ensure_collection(client, collection_name, EMBEDDING_DIMENSIONALITY)
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=emb,
    )
