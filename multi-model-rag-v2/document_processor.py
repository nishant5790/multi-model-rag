"""
document_processor.py
─────────────────────
Parses PDFs (and other docs) into three typed element lists:
  • TextElement  — plain text / narrative chunks
  • TableElement — structured table content (HTML + text)
  • ImageElement — extracted images (base64 + file path)

Uses `unstructured` for high-fidelity extraction.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List

from rich.console import Console

import config

console = Console()


# ── Data Classes ───────────────────────────────────────────────────────────────

@dataclass
class BaseElement:
    doc_id:       str   # unique ID for this element
    source:       str   # absolute path to source file
    source_name:  str   # display filename
    page_number:  int   # 1-based page number
    doc_link:     str   # human-readable link / reference
    # element_type is set by each subclass via __post_init__
    element_type: str = field(default="base", init=False)

    def to_metadata(self) -> dict:
        return {k: v for k, v in asdict(self).items()
                if k not in ("doc_id",)}


@dataclass
class TextElement(BaseElement):
    content: str = ""

    def __post_init__(self):
        self.element_type = "text"


@dataclass
class TableElement(BaseElement):
    content:      str = ""   # plain-text representation
    html_content: str = ""   # HTML table string
    summary:      str = ""   # filled by LLM summariser

    def __post_init__(self):
        self.element_type = "table"


@dataclass
class ImageElement(BaseElement):
    image_path:   str = ""   # path to saved .jpg/.png
    image_base64: str = ""   # base64-encoded bytes
    mime_type:    str = "image/jpeg"
    summary:      str = ""   # filled by LLM summariser

    def __post_init__(self):
        self.element_type = "image"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_doc_link(source: str, page: int) -> str:
    """Return a file-URI with a page fragment, e.g. file:///path/doc.pdf#page=3"""
    path = Path(source).resolve()
    return f"file://{path}#page={page}"


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type)."""
    suffix = path.suffix.lower()
    mime   = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8"), mime


def _stable_id(source: str, page: int, index: int, kind: str) -> str:
    """Deterministic UUID from document coordinates."""
    seed = f"{source}:{page}:{index}:{kind}"
    return str(uuid.UUID(hashlib.md5(seed.encode()).hexdigest()))


# ── Main Processor ─────────────────────────────────────────────────────────────

class DocumentProcessor:
    """
    Ingests a document file and returns categorised elements.

    Supported formats (via unstructured): PDF, DOCX, PPTX, HTML, MD, TXT.
    For PDFs with images the `hi_res` strategy is recommended.
    """

    def __init__(
        self,
        image_store_path: Path | None = None,
        strategy: str = config.PDF_PARTITION_STRATEGY,
        extract_images: bool = config.EXTRACT_IMAGES_FROM_PDF,
        infer_tables: bool = config.INFER_TABLE_STRUCTURE,
    ):
        self.image_store_path = image_store_path or config.IMAGE_STORE_PATH
        self.strategy         = strategy
        self.extract_images   = extract_images
        self.infer_tables     = infer_tables

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(self, filepath: str | Path) -> tuple[
        list[TextElement], list[TableElement], list[ImageElement]
    ]:
        """
        Parse a document and return (text_elements, table_elements, image_elements).
        """
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"Document not found: {filepath}")

        console.print(f"[bold cyan]📄 Processing:[/bold cyan] {filepath.name}")

        # Each doc gets its own image subdirectory
        img_dir = self.image_store_path / filepath.stem
        img_dir.mkdir(parents=True, exist_ok=True)

        raw_elements = self._partition(filepath, img_dir)
        texts, tables, images = self._categorise(raw_elements, filepath, img_dir)

        console.print(
            f"  [green]✓[/green] {len(texts)} text chunks, "
            f"{len(tables)} tables, {len(images)} images"
        )
        return texts, tables, images

    def process_directory(
        self, directory: str | Path
    ) -> tuple[list[TextElement], list[TableElement], list[ImageElement]]:
        """Process all supported documents in a directory."""
        directory   = Path(directory)
        all_texts:  list[TextElement]  = []
        all_tables: list[TableElement] = []
        all_images: list[ImageElement] = []

        supported = {".pdf", ".docx", ".pptx", ".html", ".htm", ".md", ".txt"}
        docs = [p for p in directory.iterdir() if p.suffix.lower() in supported]

        if not docs:
            console.print("[yellow]⚠ No supported documents found.[/yellow]")
            return all_texts, all_tables, all_images

        for doc in docs:
            t, tb, im = self.process(doc)
            all_texts.extend(t)
            all_tables.extend(tb)
            all_images.extend(im)

        return all_texts, all_tables, all_images

    # ── Private ────────────────────────────────────────────────────────────────

    def _partition(self, filepath: Path, img_dir: Path) -> list:
        """Call unstructured.partition to extract raw elements."""
        try:
            from unstructured.partition.auto import partition

            kwargs: dict = dict(
                filename          = str(filepath),
                strategy          = self.strategy,
                infer_table_structure = self.infer_tables,
                include_metadata  = True,
            )

            if filepath.suffix.lower() == ".pdf":
                kwargs.update(
                    extract_images_in_pdf = self.extract_images,
                    image_output_dir_path = str(img_dir),
                )

            elements = partition(**kwargs)
            return elements

        except ImportError:
            raise ImportError(
                "unstructured is required. Install with:\n"
                "  pip install 'unstructured[all-docs]'"
            )

    def _categorise(
        self,
        raw_elements: list,
        filepath: Path,
        img_dir: Path,
    ) -> tuple[list[TextElement], list[TableElement], list[ImageElement]]:
        """Sort unstructured elements into typed dataclasses."""
        from unstructured.documents.elements import (
            CompositeElement,
            Table,
            Image as UImage,
            NarrativeText,
            Title,
            ListItem,
            Header,
            Footer,
            PageBreak,
        )

        texts:  list[TextElement]  = []
        tables: list[TableElement] = []
        images: list[ImageElement] = []
        source  = str(filepath)
        name    = filepath.name

        for idx, el in enumerate(raw_elements):
            meta  = el.metadata if hasattr(el, "metadata") else {}
            page  = getattr(meta, "page_number", None) or 1
            link  = _make_doc_link(source, page)
            el_id = _stable_id(source, page, idx, type(el).__name__)

            # ── Table ──────────────────────────────────────────────────────────
            if isinstance(el, Table):
                html_text = getattr(meta, "text_as_html", "") or ""
                tables.append(TableElement(
                    doc_id       = el_id,
                    source       = source,
                    source_name  = name,
                    page_number  = page,
                    doc_link     = link,
                    content      = el.text or "",
                    html_content = html_text,
                ))
                continue

            # ── Image ──────────────────────────────────────────────────────────
            if isinstance(el, UImage):
                img_path_str = getattr(meta, "image_path", None) or ""
                img_path = Path(img_path_str) if img_path_str else None

                # Fall back: scan img_dir for the matching image by order
                if not img_path or not img_path.exists():
                    candidates = sorted(img_dir.glob("*.*"))
                    if candidates:
                        img_path = candidates[len(images) % len(candidates)]

                if img_path and img_path.exists():
                    b64, mime = _encode_image(img_path)
                    images.append(ImageElement(
                        doc_id       = el_id,
                        source       = source,
                        source_name  = name,
                        page_number  = page,
                        doc_link     = link,
                        image_path   = str(img_path),
                        image_base64 = b64,
                        mime_type    = mime,
                    ))
                continue

            # ── Skip non-content elements ──────────────────────────────────────
            if isinstance(el, (PageBreak, Header, Footer)):
                continue

            # ── Text ───────────────────────────────────────────────────────────
            text_content = (el.text or "").strip()
            if text_content:
                texts.append(TextElement(
                    doc_id      = el_id,
                    source      = source,
                    source_name = name,
                    page_number = page,
                    doc_link    = link,
                    content     = text_content,
                ))

        # ── Also pick up any loose images saved to img_dir ─────────────────────
        known_paths = {im.image_path for im in images}
        for img_file in sorted(img_dir.glob("*.*")):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".gif", ".webp"}:
                continue
            if str(img_file) in known_paths:
                continue
            b64, mime = _encode_image(img_file)
            idx_extra  = len(images)
            images.append(ImageElement(
                doc_id       = _stable_id(source, 0, idx_extra, "loose_image"),
                source       = source,
                source_name  = name,
                page_number  = 0,
                doc_link     = _make_doc_link(source, 1),
                image_path   = str(img_file),
                image_base64 = b64,
                mime_type    = mime,
            ))

        return texts, tables, images