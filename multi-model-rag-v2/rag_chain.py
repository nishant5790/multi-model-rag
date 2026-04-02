"""
rag_chain.py
────────────
Multi-modal RAG chain using Gemini 2.5 Flash.

Flow:
  1. Retrieve relevant elements (text / table / image) via MultiVectorRetriever
  2. Build a rich multi-modal prompt:
       - Text & table content  → inserted as text blocks
       - Images                → inserted as base64 image_url blocks
  3. Call Gemini 2.5 Flash with the full context
  4. Return structured RAGResponse with answer + source citations + doc links

RAGResponse fields:
  • answer       — LLM's response
  • sources      — list of SourceRef (type, name, page, link)
  • text_context — retrieved text/table passages used
  • images_used  — list of ImageElement used in context
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from rich.console import Console

from document_processor import TextElement, TableElement, ImageElement
from vector_store import MultiModalVectorStore
import config

console = Console()


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class SourceRef:
    """A citable source reference returned to the user."""
    element_type: str    # "text" | "table" | "image"
    source_name:  str
    page_number:  int
    doc_link:     str

    def __str__(self) -> str:
        label = {"text": "📄", "table": "📊", "image": "🖼"}.get(self.element_type, "📎")
        return (
            f"{label} [{self.element_type.upper()}] "
            f"{self.source_name} — page {self.page_number} "
            f"→ {self.doc_link}"
        )


@dataclass
class RAGResponse:
    """Full structured response from the RAG chain."""
    query:        str
    answer:       str
    sources:      list[SourceRef] = field(default_factory=list)
    text_context: list[str]       = field(default_factory=list)
    images_used:  list[ImageElement] = field(default_factory=list)

    def format(self) -> str:
        """Pretty-print for terminal output."""
        lines = [
            "",
            "━" * 70,
            f"[bold]Query:[/bold] {self.query}",
            "━" * 70,
            "",
            self.answer,
            "",
        ]
        if self.sources:
            lines.append("─" * 70)
            lines.append("[bold cyan]📚 Sources:[/bold cyan]")
            seen = set()
            for s in self.sources:
                key = (s.source_name, s.page_number, s.element_type)
                if key not in seen:
                    seen.add(key)
                    lines.append(f"  {s}")
        lines.append("━" * 70)
        return "\n".join(lines)


# ── Prompt Builders ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a precise, knowledgeable assistant that answers questions strictly \
based on the provided document context.

Guidelines:
- Answer only from the provided context (text passages, tables, images).
- Cite the source document name and page number when referencing specific information.
- For tables: interpret the data accurately; do not fabricate numbers.
- For images: refer to what you can observe; do not speculate beyond what's visible.
- If the context does not contain enough information, say so clearly.
- Be concise but complete."""


def _build_context_message(
    elements: Sequence[TextElement | TableElement | ImageElement],
    query:    str,
    max_images: int = config.MAX_IMAGES_IN_RESP,
) -> tuple[HumanMessage, list[SourceRef], list[str], list[ImageElement]]:
    """
    Build a HumanMessage with interleaved text and image blocks.
    Returns (message, source_refs, text_passages, images_used).
    """
    content_blocks: list[dict] = []
    source_refs:    list[SourceRef] = []
    text_passages:  list[str]       = []
    images_used:    list[ImageElement] = []
    image_count     = 0

    # ── Preamble ───────────────────────────────────────────────────────────────
    content_blocks.append({
        "type": "text",
        "text": (
            "Below is relevant context extracted from documents. "
            "Use it to answer the question at the end.\n\n"
            "═" * 60 + "\n"
        )
    })

    for i, el in enumerate(elements):
        src = SourceRef(
            element_type = el.element_type,
            source_name  = el.source_name,
            page_number  = el.page_number,
            doc_link     = el.doc_link,
        )
        source_refs.append(src)
        header = (
            f"\n[CONTEXT {i+1} | {el.element_type.upper()} | "
            f"{el.source_name}, page {el.page_number}]\n"
            f"Link: {el.doc_link}\n"
        )

        if isinstance(el, TextElement):
            passage = f"{header}{el.content}"
            content_blocks.append({"type": "text", "text": passage})
            text_passages.append(el.content)

        elif isinstance(el, TableElement):
            table_text = (
                f"{header}"
                f"Table Summary:\n{getattr(el, 'summary', el.content)}\n\n"
                f"Raw Table Data:\n{el.content}\n"
            )
            if el.html_content:
                table_text += f"\nHTML:\n{el.html_content}\n"
            content_blocks.append({"type": "text", "text": table_text})
            text_passages.append(el.content)

        elif isinstance(el, ImageElement) and image_count < max_images:
            if el.image_base64:
                content_blocks.append({"type": "text", "text": header})
                if el.summary:
                    content_blocks.append({
                        "type": "text",
                        "text": f"Image description: {el.summary}\n"
                    })
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{el.mime_type};base64,{el.image_base64}"
                    },
                })
                images_used.append(el)
                image_count += 1

    # ── Question ───────────────────────────────────────────────────────────────
    content_blocks.append({
        "type": "text",
        "text": (
            "\n" + "═" * 60 + "\n"
            f"Question: {query}\n\n"
            "Answer based on the context above. "
            "Always cite document name and page number for each claim."
        )
    })

    return (
        HumanMessage(content=content_blocks),
        source_refs,
        text_passages,
        images_used,
    )


# ── RAG Chain ──────────────────────────────────────────────────────────────────

class MultiModalRAGChain:
    """
    End-to-end multi-modal RAG chain:
    retrieve → build prompt → generate → return RAGResponse.
    """

    def __init__(
        self,
        vector_store: MultiModalVectorStore,
        model_name:   str = config.CHAT_MODEL,
        retriever_k:  int = config.RETRIEVER_K,
    ):
        self.vector_store = vector_store
        self.retriever_k  = retriever_k
        self.llm = ChatGoogleGenerativeAI(
            model       = model_name,
            temperature = 0.1,
            max_tokens  = 4096,
        )

    def invoke(self, query: str) -> RAGResponse:
        """Run the full RAG pipeline for a user query."""

        # Step 1 — Retrieve
        console.print(f"\n[bold cyan]🔍 Retrieving context for:[/bold cyan] {query}")
        elements = self.vector_store.retrieve(query, k=self.retriever_k)

        if not elements:
            return RAGResponse(
                query  = query,
                answer = (
                    "I could not find any relevant information in the indexed documents "
                    "to answer your question. Please ensure documents have been ingested."
                ),
            )

        console.print(
            f"  Retrieved {len(elements)} element(s): "
            + ", ".join(
                f"{sum(1 for e in elements if e.element_type == t)} {t}"
                for t in ("text", "table", "image")
                if any(e.element_type == t for e in elements)
            )
        )

        # Step 2 — Build prompt
        human_msg, source_refs, text_passages, images_used = _build_context_message(
            elements, query
        )

        # Step 3 — Generate
        console.print("[bold cyan]💬 Generating answer…[/bold cyan]")
        try:
            response = self.llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                human_msg,
            ])
            answer = response.content
        except Exception as exc:
            answer = f"⚠ Error generating answer: {exc}"

        return RAGResponse(
            query        = query,
            answer       = answer,
            sources      = source_refs,
            text_context = text_passages,
            images_used  = images_used,
        )

    # ── Streaming variant ──────────────────────────────────────────────────────

    def stream(self, query: str):
        """
        Stream the answer token-by-token.
        Yields str chunks; final item is a RAGResponse (when StopIteration).

        Usage:
            for chunk in chain.stream(query):
                if isinstance(chunk, RAGResponse):
                    response = chunk
                else:
                    print(chunk, end="", flush=True)
        """
        elements = self.vector_store.retrieve(query, k=self.retriever_k)
        if not elements:
            yield RAGResponse(
                query  = query,
                answer = "No relevant documents found.",
            )
            return

        human_msg, source_refs, text_passages, images_used = _build_context_message(
            elements, query
        )

        full_answer = ""
        for chunk in self.llm.stream([SystemMessage(content=SYSTEM_PROMPT), human_msg]):
            token = chunk.content
            full_answer += token
            yield token

        yield RAGResponse(
            query        = query,
            answer       = full_answer,
            sources      = source_refs,
            text_context = text_passages,
            images_used  = images_used,
        )
