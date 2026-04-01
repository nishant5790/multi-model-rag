"""Retrieval-augmented generation with Gemini chat and citation-style context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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
        if m.get("image_path"):
            line += f" image_path={m['image_path']}"
        if m.get("image_link"):
            line += f" image_link={m['image_link']}"
        blocks.append(f"{line}\n{m.get('source', '')}\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)


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
    ) -> None:
        self._embeddings = get_embeddings()
        self._chat = get_chat()
        self._vs = get_vectorstore(
            persist_directory,
            collection_name=collection_name,
            embeddings=self._embeddings,
        )
        self._retriever = self._vs.as_retriever(search_kwargs={"k": 6})
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
        docs = pdf_elements_to_documents(
            pdf_path,
            image_output_dir=out_dir,
            chat=self._chat if caption_images else None,
            caption_images=caption_images,
        )
        if not docs:
            return 0
        self._vs.add_documents(docs)
        return len(docs)

    def query(self, question: str) -> dict[str, Any]:
        """Retrieve, generate an answer, and return structured source info."""
        retrieved: list[Document] = self._retriever.invoke(question)
        context = _format_context(retrieved)
        answer = self._qa.invoke({"context": context, "question": question})
        sources: list[dict[str, Any]] = []
        for i, doc in enumerate(retrieved, 1):
            m = doc.metadata
            sources.append(
                {
                    "index": i,
                    "content_type": m.get("content_type"),
                    "page": m.get("page"),
                    "doc_link": m.get("doc_link"),
                    "source_file": m.get("source"),
                    "image_path": m.get("image_path"),
                    "image_link": m.get("image_link"),
                    "snippet": doc.page_content[:500]
                    + ("…" if len(doc.page_content) > 500 else ""),
                }
            )
        return {"answer": answer, "sources": sources}
