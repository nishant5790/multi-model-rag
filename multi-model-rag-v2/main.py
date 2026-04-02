"""
main.py
───────
Interactive CLI for the Multi-Modal RAG system.

Usage:
    # Step 1: ingest your documents first
    python ingest.py --dir ./docs

    # Step 2: query
    python main.py
    python main.py --query "What does the revenue chart show?"
    python main.py --stream           # streaming mode
    python main.py --k 8              # retrieve 8 chunks instead of default
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from document_processor import DocumentProcessor
from image_handler import MultiModalSummariser
from vector_store import MultiModalVectorStore
from rag_chain import MultiModalRAGChain, RAGResponse
import config

console = Console()

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║         🔮  Multi-Modal RAG  ·  Powered by Gemini           ║
║  Embedding : gemini-embedding-2-preview                      ║
║  Chat      : gemini-2.5-flash                                ║
╚══════════════════════════════════════════════════════════════╝
"""


# ── Single-query helper ────────────────────────────────────────────────────────

def run_query(
    chain:  MultiModalRAGChain,
    query:  str,
    stream: bool = False,
) -> RAGResponse:
    if stream:
        console.print("\n[bold cyan]Answer:[/bold cyan] ", end="")
        final: RAGResponse | None = None
        for chunk in chain.stream(query):
            if isinstance(chunk, RAGResponse):
                final = chunk
            else:
                print(chunk, end="", flush=True)
        print()
        if final:
            _print_sources(final)
        return final  # type: ignore
    else:
        response = chain.invoke(query)
        console.print(Panel(
            Markdown(response.answer),
            title="[bold green]Answer[/bold green]",
            border_style="green",
        ))
        _print_sources(response)
        return response


def _print_sources(response: RAGResponse) -> None:
    if not response.sources:
        return
    console.print("\n[bold cyan]📚 Sources:[/bold cyan]")
    seen = set()
    for s in response.sources:
        key = (s.source_name, s.page_number, s.element_type)
        if key not in seen:
            seen.add(key)
            icon = {"text": "📄", "table": "📊", "image": "🖼"}.get(s.element_type, "📎")
            console.print(
                f"  {icon} [bold]{s.source_name}[/bold] — "
                f"page {s.page_number} "
                f"[dim]({s.element_type})[/dim]\n"
                f"     [link={s.doc_link}]{s.doc_link}[/link]"
            )


# ── Full pipeline (ingest + query in one shot) ─────────────────────────────────

def build_pipeline(docs_dir: Path | None = None) -> MultiModalRAGChain:
    """Ingest documents and return a ready-to-use RAG chain."""
    processor  = DocumentProcessor()
    summariser = MultiModalSummariser()
    vs         = MultiModalVectorStore()

    src = docs_dir or config.DOCS_PATH
    console.print(f"[dim]Loading documents from {src}…[/dim]")
    texts, tables, images = processor.process_directory(src)

    if texts or tables or images:
        images, tables = summariser.summarise_all(images, tables)
        vs.index_all(texts, tables, images)
    else:
        console.print("[yellow]⚠ No documents found — using existing index if available.[/yellow]")

    return MultiModalRAGChain(vs)


# ── Interactive REPL ───────────────────────────────────────────────────────────

def interactive_loop(chain: MultiModalRAGChain, stream: bool = False) -> None:
    console.print("[dim]Type your question and press Enter. Type 'exit' to quit.[/dim]\n")

    while True:
        try:
            query = console.input("[bold yellow]❓ Query:[/bold yellow] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            console.print("[dim]Goodbye![/dim]")
            break

        run_query(chain, query, stream=stream)
        console.print()


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-Modal RAG powered by Gemini"
    )
    parser.add_argument(
        "--docs", type=Path, default=None,
        help="Path to documents directory (default: ./docs)",
    )
    parser.add_argument(
        "--query", "-q", type=str, default=None,
        help="Run a single query and exit",
    )
    parser.add_argument(
        "--stream", action="store_true",
        help="Stream the answer token-by-token",
    )
    parser.add_argument(
        "--k", type=int, default=config.RETRIEVER_K,
        help=f"Number of chunks to retrieve (default: {config.RETRIEVER_K})",
    )
    parser.add_argument(
        "--no-ingest", action="store_true",
        help="Skip ingestion, use existing Chroma index",
    )
    args = parser.parse_args()

    console.print(BANNER)

    # ── Build / reuse pipeline ─────────────────────────────────────────────────
    if args.no_ingest:
        vs    = MultiModalVectorStore()
        chain = MultiModalRAGChain(vs, retriever_k=args.k)
        console.print("[dim]Using existing index.[/dim]")
    else:
        # Quick check: if docs dir exists and has files, ingest
        docs_dir = args.docs or config.DOCS_PATH
        if not docs_dir.exists() or not any(docs_dir.iterdir()):
            console.print(
                f"[yellow]⚠ Docs directory '{docs_dir}' is empty or missing.\n"
                "  Add PDFs/docs to that folder, or use --no-ingest to query an "
                "existing index.[/yellow]"
            )
            return
        vs    = MultiModalVectorStore()
        processor  = DocumentProcessor()
        summariser = MultiModalSummariser()
        texts, tables, images = processor.process_directory(docs_dir)
        if texts or tables or images:
            images, tables = summariser.summarise_all(images, tables)
            vs.index_all(texts, tables, images)
        chain = MultiModalRAGChain(vs, retriever_k=args.k)

    # ── Query mode ─────────────────────────────────────────────────────────────
    if args.query:
        run_query(chain, args.query, stream=args.stream)
    else:
        interactive_loop(chain, stream=args.stream)


if __name__ == "__main__":
    main()
