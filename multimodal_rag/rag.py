"""Retrieval-augmented generation with Gemini chat and citation-style context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from multimodal_rag.graph_store import (
    ChunkGraph,
    build_local_graph_from_ordered_docs,
    expand_chunk_ids,
    graph_json_path,
)
from multimodal_rag.ingestion import pdf_elements_to_documents
from multimodal_rag.models import get_chat, get_embeddings
from multimodal_rag.store import get_vectorstore


def _format_context(docs: list[Document]) -> str:
    blocks: list[str] = []
    for i, doc in enumerate(docs, 1):
        m = doc.metadata
        line = (
            f"[Source {i}] type={m.get('content_type', 'unknown')} "
            f"page={m.get('page', '')} doc_link={m.get('doc_link', '')}"
        )
        if m.get("chunk_id"):
            line += f" chunk_id={m['chunk_id']}"
        if m.get("image_path"):
            line += f" image_path={m['image_path']}"
        if m.get("image_link"):
            line += f" image_link={m['image_link']}"
        blocks.append(f"{line}\n{m.get('source', '')}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


def _documents_from_chroma_get(vs: Any, ids: list[str]) -> list[Document]:
    if not ids:
        return []
    data = vs.get(ids=ids, include=["documents", "metadatas"])
    out: list[Document] = []
    raw_ids = data.get("ids") or []
    texts = data.get("documents") or []
    metas = data.get("metadatas") or []
    for i, _doc_id in enumerate(raw_ids):
        if i >= len(texts):
            break
        text = texts[i]
        if text is None:
            continue
        meta_raw = metas[i] if i < len(metas) else None
        meta = dict(meta_raw) if meta_raw else {}
        out.append(Document(page_content=text, metadata=meta))
    return out


RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You answer questions using only the provided context. "
            "Cite sources as [Source N] where N matches the context. "
            "For images or charts, mention the image_link or image_path when relevant. "
            "If the context is insufficient, say so briefly.",
        ),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {question}",
        ),
    ]
)


class MultiModalRAG:
    """Ingest PDFs into Chroma and run retrieval + Gemini chat."""

    def __init__(
        self,
        persist_directory: Path | str,
        *,
        collection_name: str = "multimodal_rag",
        retrieve_k: int = 6,
        graph_expand_k: int = 8,
        max_graph_hops: int = 1,
        use_graph_expand: bool = True,
    ) -> None:
        self._persist_directory = Path(persist_directory).expanduser().resolve()
        self._graph_path = graph_json_path(self._persist_directory)
        self._retrieve_k = retrieve_k
        self._graph_expand_k = graph_expand_k
        self._max_graph_hops = max_graph_hops
        self._use_graph_expand = use_graph_expand
        self._embeddings = get_embeddings()
        self._chat = get_chat()
        self._vs = get_vectorstore(
            self._persist_directory,
            collection_name=collection_name,
            embeddings=self._embeddings,
        )
        self._retriever = self._vs.as_retriever(
            search_kwargs={"k": self._retrieve_k},
        )
        self._chunk_graph = ChunkGraph.load(self._graph_path)
        self._qa = RAG_PROMPT | self._chat | StrOutputParser()

    def ingest_pdf(
        self,
        pdf_path: Path | str,
        *,
        image_output_dir: Path | str | None = None,
        caption_images: bool = True,
    ) -> int:
        """Chunk a PDF and add documents to the vector store. Returns number of chunks added."""
        pdf_path = Path(pdf_path).expanduser().resolve()
        out_dir = (
            Path(image_output_dir).expanduser().resolve()
            if image_output_dir
            else pdf_path.parent / f"{pdf_path.stem}_extracted_images"
        )
        docs, ids = pdf_elements_to_documents(
            pdf_path,
            image_output_dir=out_dir,
            chat=self._chat if caption_images else None,
            caption_images=caption_images,
        )
        if not docs:
            return 0
        local = build_local_graph_from_ordered_docs(docs)
        self._chunk_graph.merge_from_local(local)
        self._chunk_graph.save(self._graph_path)
        self._vs.add_documents(docs, ids=ids)
        return len(docs)

    def query(self, question: str) -> dict[str, Any]:
        """Retrieve, optionally expand via chunk graph, generate an answer, return sources."""
        self._chunk_graph = ChunkGraph.load(self._graph_path)
        retrieved: list[Document] = self._retriever.invoke(question)
        final_docs = list(retrieved)

        if self._use_graph_expand and self._graph_expand_k > 0:
            seed_ids = [
                str(m["chunk_id"])
                for m in (d.metadata for d in retrieved)
                if m.get("chunk_id")
            ]
            if seed_ids:
                extra_ids = expand_chunk_ids(
                    seed_ids,
                    self._chunk_graph,
                    max_hops=self._max_graph_hops,
                    expand_k=self._graph_expand_k,
                )
                have = {str(d.metadata.get("chunk_id")) for d in retrieved if d.metadata.get("chunk_id")}
                extra_ids = [eid for eid in extra_ids if eid not in have]
                expanded_raw = _documents_from_chroma_get(self._vs, extra_ids)
                by_id: dict[str, Document] = {}
                for d in expanded_raw:
                    cid = d.metadata.get("chunk_id")
                    if cid:
                        by_id[str(cid)] = d
                expanded = [by_id[eid] for eid in extra_ids if eid in by_id]
                final_docs = retrieved + expanded

        context = _format_context(final_docs)
        answer = self._qa.invoke({"context": context, "question": question})
        sources: list[dict[str, Any]] = []
        for i, doc in enumerate(final_docs, 1):
            m = doc.metadata
            sources.append(
                {
                    "index": i,
                    "chunk_id": m.get("chunk_id"),
                    "content_type": m.get("content_type"),
                    "page": m.get("page"),
                    "doc_link": m.get("doc_link"),
                    "source_file": m.get("source"),
                    "image_path": m.get("image_path"),
                    "image_link": m.get("image_link"),
                    "graph_expanded": i > len(retrieved),
                    "snippet": doc.page_content[:500]
                    + ("…" if len(doc.page_content) > 500 else ""),
                }
            )
        return {"answer": answer, "sources": sources}

    def retrieve_docs(self, question: str) -> dict[str, Any]:
        """Retrieve documents from Chroma without running the LLM. Returns raw content and metadata."""
        self._chunk_graph = ChunkGraph.load(self._graph_path)
        retrieved: list[Document] = self._retriever.invoke(question)
        final_docs = list(retrieved)

        if self._use_graph_expand and self._graph_expand_k > 0:
            seed_ids = [
                str(m["chunk_id"])
                for m in (d.metadata for d in retrieved)
                if m.get("chunk_id")
            ]
            if seed_ids:
                extra_ids = expand_chunk_ids(
                    seed_ids,
                    self._chunk_graph,
                    max_hops=self._max_graph_hops,
                    expand_k=self._graph_expand_k,
                )
                have = {str(d.metadata.get("chunk_id")) for d in retrieved if d.metadata.get("chunk_id")}
                extra_ids = [eid for eid in extra_ids if eid not in have]
                expanded_raw = _documents_from_chroma_get(self._vs, extra_ids)
                by_id: dict[str, Document] = {}
                for d in expanded_raw:
                    cid = d.metadata.get("chunk_id")
                    if cid:
                        by_id[str(cid)] = d
                expanded = [by_id[eid] for eid in extra_ids if eid in by_id]
                final_docs = retrieved + expanded

        documents = []
        for doc in final_docs:
            documents.append({
                "content": doc.page_content,
                "metadata": dict(doc.metadata),
            })
        return {"documents": documents, "count": len(documents)}
