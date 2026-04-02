"""
ingest.py
─────────
One-shot ingestion pipeline. Run this once (or whenever you add new docs).

Usage:
    python ingest.py                        # processes ./docs/
    python ingest.py --dir /path/to/docs
    python ingest.py --file report.pdf

What it does:
  1. Extracts text, tables, images from each document
  2. Generates LLM summaries for images and tables
  3. Embeds summaries with gemini-embedding-2-preview
  4. Stores in Chroma (persist_directory = ./chroma_db)
"""

import argparse
from pathlib import Path
from rich.console import Console

from document_processor import DocumentProcessor
from image_handler import MultiModalSummariser
from vector_store import MultiModalVectorStore
import config

console = Console()


def ingest(source: Path | None = None, file: Path | None = None) -> MultiModalVectorStore:
    """Full ingestion pipeline. Returns the populated vector store."""

    processor  = DocumentProcessor()
    summariser = MultiModalSummariser()
    vs         = MultiModalVectorStore()

    # ── Collect elements ───────────────────────────────────────────────────────
    if file:
        console.print(f"[bold]Ingesting single file:[/bold] {file}")
        texts, tables, images = processor.process(file)
    else:
        src = source or config.DOCS_PATH
        console.print(f"[bold]Ingesting directory:[/bold] {src}")
        texts, tables, images = processor.process_directory(src)

    if not texts and not tables and not images:
        console.print("[red]No content found. Check your docs directory.[/red]")
        return vs

    # ── Generate LLM summaries ─────────────────────────────────────────────────
    images, tables = summariser.summarise_all(images, tables)

    # ── Index into Chroma ──────────────────────────────────────────────────────
    vs.index_all(texts, tables, images)

    console.print("\n[bold green]✅ Ingestion complete![/bold green]")
    console.print(f"   Vector store: [cyan]{config.VECTOR_STORE_PATH}[/cyan]")
    console.print(f"   Images saved: [cyan]{config.IMAGE_STORE_PATH}[/cyan]")

    return vs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Modal RAG Ingestion Pipeline")
    parser.add_argument("--dir",  type=Path, help="Directory of documents to ingest")
    parser.add_argument("--file", type=Path, help="Single file to ingest")
    args = parser.parse_args()

    ingest(source=args.dir, file=args.file)
