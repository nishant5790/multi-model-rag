"""
vector_store.py
───────────────
Builds a LangChain MultiVectorRetriever backed by:
  • Chroma          — stores *summaries/text* as dense vectors
  • LocalFileByteStore — stores *full raw content* keyed by doc_id (persisted to disk)

Architecture
────────────
┌─────────────────────────────────────────────────────────┐
│                MultiVectorRetriever                     │
│                                                         │
│  ┌──────────────────┐      ┌────────────────────────┐   │
│  │   Chroma (index) │      │  LocalFileByteStore    │   │
│  │                  │      │  (raw content store)   │   │
│  │ text summary  ──────────► TextElement (json)     │   │
│  │ table summary ──────────► TableElement (json)    │   │
│  │ image summary ──────────► ImageElement (json)    │   │
│  └──────────────────┘      └────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

Query flow:
  1. embed(query) → Chroma similarity search → top-k summaries
  2. summaries carry doc_id → fetch raw element from ByteStore
  3. raw element (text / table / image+base64) returned to RAG chain
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import List, Sequence

from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.stores import BaseStore
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from rich.console import Console

from document_processor import TextElement, TableElement, ImageElement
import config

console = Console()

# Key used in Chroma metadata to link summaries → raw docs
DOC_ID_KEY = "doc_id"


# ── Persistent file-backed byte store ─────────────────────────────────────────

class LocalFileByteStore(BaseStore[str, bytes]):
    """
    Drop-in replacement for InMemoryByteStore that persists each
    key-value pair as a file on disk.  Keys become filenames inside
    *root_path*; values are stored as raw bytes.
    """

    def __init__(self, root_path: str | Path) -> None:
        self._root = Path(root_path)
        self._root.mkdir(parents=True, exist_ok=True)

    def _key_path(self, key: str) -> Path:
        return self._root / key

    def mget(self, keys: Sequence[str]) -> list[bytes | None]:
        results: list[bytes | None] = []
        for k in keys:
            p = self._key_path(k)
            results.append(p.read_bytes() if p.exists() else None)
        return results

    def mset(self, key_value_pairs: Sequence[tuple[str, bytes]]) -> None:
        for key, value in key_value_pairs:
            self._key_path(key).write_bytes(value)

    def mdelete(self, keys: Sequence[str]) -> None:
        for k in keys:
            p = self._key_path(k)
            p.unlink(missing_ok=True)

    def yield_keys(self, *, prefix: str | None = None):
        for p in self._root.iterdir():
            if p.is_file() and (prefix is None or p.name.startswith(prefix)):
                yield p.name


# ── Embedding Model ────────────────────────────────────────────────────────────

def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Returns a GoogleGenerativeAIEmbeddings instance using gemini-embedding-2-preview.

    The task_type "RETRIEVAL_DOCUMENT" is used for indexing;
    queries use "RETRIEVAL_QUERY" (set automatically by the retriever).
    """
    return GoogleGenerativeAIEmbeddings(
        model     = config.EMBEDDING_MODEL,
        task_type = "RETRIEVAL_DOCUMENT",
    )


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _element_to_json(el: TextElement | TableElement | ImageElement) -> bytes:
    """Serialise an element to JSON bytes (no heavy image data in Chroma)."""
    d = {
        "type":         el.element_type,
        "doc_id":       el.doc_id,
        "source":       el.source,
        "source_name":  el.source_name,
        "page_number":  el.page_number,
        "doc_link":     el.doc_link,
    }
    if isinstance(el, TextElement):
        d["content"] = el.content

    elif isinstance(el, TableElement):
        d["content"]      = el.content
        d["html_content"] = el.html_content
        d["summary"]      = getattr(el, "summary", "")

    elif isinstance(el, ImageElement):
        d["image_path"]   = el.image_path
        d["image_base64"] = el.image_base64   # full base64 in docstore only
        d["mime_type"]    = el.mime_type
        d["summary"]      = el.summary

    return json.dumps(d, ensure_ascii=False).encode("utf-8")


def json_to_element(
    raw: bytes,
) -> TextElement | TableElement | ImageElement:
    """Deserialise bytes from the docstore back to a typed element."""
    d = json.loads(raw.decode("utf-8"))
    kind = d.get("type", "text")

    base = dict(
        doc_id      = d["doc_id"],
        source      = d["source"],
        source_name = d["source_name"],
        page_number = d["page_number"],
        doc_link    = d["doc_link"],
    )
    if kind == "table":
        return TableElement(
            **base,
            content      = d.get("content", ""),
            html_content = d.get("html_content", ""),
        )
    if kind == "image":
        el = ImageElement(
            **base,
            image_path   = d.get("image_path", ""),
            image_base64 = d.get("image_base64", ""),
            mime_type    = d.get("mime_type", "image/jpeg"),
        )
        el.summary = d.get("summary", "")
        return el

    return TextElement(**base, content=d.get("content", ""))


# ── VectorStore Builder ────────────────────────────────────────────────────────

class MultiModalVectorStore:
    """
    Manages indexing and retrieval of multi-modal document elements.
    """

    def __init__(
        self,
        persist_directory: Path | None = None,
        collection_name:   str  = "multimodal_rag",
    ):
        self.persist_directory = persist_directory or config.VECTOR_STORE_PATH
        self.collection_name   = collection_name
        self._embeddings       = get_embeddings()
        self._byte_store       = LocalFileByteStore(config.DOCSTORE_PATH)
        self._vectorstore: Chroma | None  = None
        self._retriever:   MultiVectorRetriever | None = None

    # ── Initialise / load ──────────────────────────────────────────────────────

    def _get_vectorstore(self) -> Chroma:
        if self._vectorstore is None:
            self._vectorstore = Chroma(
                collection_name    = self.collection_name,
                embedding_function = self._embeddings,
                persist_directory  = str(self.persist_directory),
            )
        return self._vectorstore

    def _get_retriever(self) -> MultiVectorRetriever:
        if self._retriever is None:
            self._retriever = MultiVectorRetriever(
                vectorstore = self._get_vectorstore(),
                byte_store  = self._byte_store,
                id_key      = DOC_ID_KEY,
                search_kwargs = {"k": config.RETRIEVER_K},
            )
        return self._retriever

    # ── Indexing ───────────────────────────────────────────────────────────────

    def add_texts(self, elements: Sequence[TextElement]) -> None:
        """Index text chunks directly (content is both summary and raw)."""
        if not elements:
            return
        console.print(f"  [dim]Indexing {len(elements)} text chunks…[/dim]")

        docs: list[Document] = []
        raw_pairs: list[tuple[str, bytes]] = []

        for el in elements:
            meta = {
                DOC_ID_KEY:   el.doc_id,
                "source":     el.source,
                "source_name":el.source_name,
                "page_number":el.page_number,
                "doc_link":   el.doc_link,
                "type":       "text",
            }
            docs.append(Document(page_content=el.content, metadata=meta))
            raw_pairs.append((el.doc_id, _element_to_json(el)))

        retriever = self._get_retriever()
        retriever.vectorstore.add_documents(docs)
        retriever.byte_store.mset(raw_pairs)

    def add_tables(self, elements: Sequence[TableElement]) -> None:
        """Index table summaries; raw HTML+content stored in docstore."""
        if not elements:
            return
        console.print(f"  [dim]Indexing {len(elements)} tables…[/dim]")

        docs: list[Document] = []
        raw_pairs: list[tuple[str, bytes]] = []

        for el in elements:
            summary = getattr(el, "summary", "") or el.content
            meta = {
                DOC_ID_KEY:   el.doc_id,
                "source":     el.source,
                "source_name":el.source_name,
                "page_number":el.page_number,
                "doc_link":   el.doc_link,
                "type":       "table",
            }
            docs.append(Document(page_content=summary, metadata=meta))
            raw_pairs.append((el.doc_id, _element_to_json(el)))

        retriever = self._get_retriever()
        retriever.vectorstore.add_documents(docs)
        retriever.byte_store.mset(raw_pairs)

    def add_images(self, elements: Sequence[ImageElement]) -> None:
        """Index image summaries; full base64 stored in docstore."""
        if not elements:
            return
        console.print(f"  [dim]Indexing {len(elements)} images…[/dim]")

        docs: list[Document] = []
        raw_pairs: list[tuple[str, bytes]] = []

        for el in elements:
            # Only index if we have a meaningful summary
            summary = el.summary or f"Image from {el.source_name} page {el.page_number}"
            meta = {
                DOC_ID_KEY:   el.doc_id,
                "source":     el.source,
                "source_name":el.source_name,
                "page_number":el.page_number,
                "doc_link":   el.doc_link,
                "type":       "image",
            }
            docs.append(Document(page_content=summary, metadata=meta))
            raw_pairs.append((el.doc_id, _element_to_json(el)))

        retriever = self._get_retriever()
        retriever.vectorstore.add_documents(docs)
        retriever.byte_store.mset(raw_pairs)

    def index_all(
        self,
        texts:  Sequence[TextElement],
        tables: Sequence[TableElement],
        images: Sequence[ImageElement],
    ) -> None:
        """One-shot index for all element types."""
        console.print("[bold green]📥 Indexing documents…[/bold green]")
        self.add_texts(texts)
        self.add_tables(tables)
        self.add_images(images)
        console.print(
            f"[green]✓ Indexed {len(texts)} texts, "
            f"{len(tables)} tables, {len(images)} images[/green]"
        )

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        k: int = config.RETRIEVER_K,
    ) -> list[TextElement | TableElement | ImageElement]:
        """
        Retrieve the top-k most relevant elements for a query.
        Returns typed element objects (with full base64 for images).
        """
        retriever = self._get_retriever()
        retriever.search_kwargs = {"k": k}

        # We query the vectorstore directly since MultiVectorRetriever docstore expects Document objects
        vs_docs = self._get_vectorstore().similarity_search(query, k=k)

        elements = []
        for doc in vs_docs:
            doc_id = doc.metadata.get(DOC_ID_KEY)
            raw_bytes_list = self._byte_store.mget([doc_id]) if doc_id else []
            raw_bytes = raw_bytes_list[0] if raw_bytes_list else None

            if raw_bytes:
                try:
                    el = json_to_element(raw_bytes)
                    elements.append(el)
                except Exception:
                    pass
            else:
                # Fallback: wrap as plain text element
                from document_processor import TextElement
                elements.append(TextElement(
                    doc_id      = doc.metadata.get(DOC_ID_KEY, "unknown"),
                    source      = doc.metadata.get("source", ""),
                    source_name = doc.metadata.get("source_name", ""),
                    page_number = doc.metadata.get("page_number", 0),
                    doc_link    = doc.metadata.get("doc_link", ""),
                    content     = doc.page_content if isinstance(doc.page_content, str)
                                  else str(doc.page_content),
                ))
        return elements

    def retrieve_with_scores(
        self,
        query: str,
        k: int = config.RETRIEVER_K,
    ) -> list[tuple[Document, float]]:
        """Return (Document, similarity_score) pairs from the vector store."""
        vs = self._get_vectorstore()
        query_embed = GoogleGenerativeAIEmbeddings(
            model     = config.EMBEDDING_MODEL,
            task_type = "RETRIEVAL_QUERY",
        )
        q_vec = query_embed.embed_query(query)
        return vs.similarity_search_with_score_by_vector(q_vec, k=k)
