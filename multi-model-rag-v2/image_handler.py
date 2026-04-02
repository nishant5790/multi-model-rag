"""
image_handler.py
────────────────
Uses Gemini 2.5 Flash to generate rich textual summaries of:
  • Images  → described for embedding + retrieval
  • Tables  → reformatted and summarised

Summaries are what get embedded into the vector store;
the original base64 image is kept in the docstore for final LLM context.
"""

from __future__ import annotations

from typing import Sequence
from tqdm import tqdm
from rich.console import Console

from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from document_processor import ImageElement, TableElement
import config

console = Console()


# ── Prompts ────────────────────────────────────────────────────────────────────

IMAGE_SUMMARY_PROMPT = """\
You are an expert document analyst. Describe the following image in rich detail \
so that a semantic search engine can match it to user queries.

Include:
- What the image shows (charts, diagrams, photos, screenshots, infographics, etc.)
- All visible text, labels, axes, legends, data values
- Colours, layout, spatial relationships
- The apparent purpose / key insight of the image
- Any domain context (medical, financial, engineering, etc.)

Be thorough and precise. Your description will be embedded and searched — \
make every important detail discoverable."""

TABLE_SUMMARY_PROMPT = """\
You are a data analyst. Summarise the following table so it can be searched semantically.

Include:
- What the table represents (title, purpose)
- Column names and what each measures
- Row structure and key groupings
- Notable values, trends, outliers
- Any totals, averages, or summary rows

Then provide a concise 2-3 sentence summary of the table's main insight.

Table content:
{content}

HTML representation:
{html}"""


# ── Summariser ─────────────────────────────────────────────────────────────────

class MultiModalSummariser:
    """
    Generates LLM-powered summaries for images and tables.
    Summaries are stored on the element objects (`.summary` field).
    """

    def __init__(self, model_name: str = config.CHAT_MODEL):
        self.llm = ChatGoogleGenerativeAI(
            model       = model_name,
            temperature = 0.0,
            max_tokens  = 1024,
        )

    # ── Images ─────────────────────────────────────────────────────────────────

    def summarise_image(self, element: ImageElement) -> str:
        """Generate a text description for a single image element."""
        try:
            message = HumanMessage(content=[
                {"type": "text",      "text": IMAGE_SUMMARY_PROMPT},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{element.mime_type};base64,{element.image_base64}"
                    },
                },
            ])
            response = self.llm.invoke([message])
            return response.content.strip()
        except Exception as exc:
            console.print(f"[yellow]⚠ Image summary failed: {exc}[/yellow]")
            return f"Image from {element.source_name} page {element.page_number}"

    def summarise_images(
        self, elements: Sequence[ImageElement], show_progress: bool = True
    ) -> list[ImageElement]:
        """Batch-summarise all images; returns elements with `.summary` filled."""
        if not elements:
            return list(elements)

        console.print(f"[cyan]🖼  Summarising {len(elements)} image(s)…[/cyan]")
        iterable = tqdm(elements, desc="Images", unit="img") if show_progress else elements

        for el in iterable:
            if not el.summary:          # skip if already summarised
                el.summary = self.summarise_image(el)
        return list(elements)

    # ── Tables ─────────────────────────────────────────────────────────────────

    def summarise_table(self, element: TableElement) -> str:
        """Generate a rich text summary for a single table element."""
        try:
            prompt = TABLE_SUMMARY_PROMPT.format(
                content = element.content   or "(no plain-text content)",
                html    = element.html_content or "(no HTML)",
            )
            response = self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as exc:
            console.print(f"[yellow]⚠ Table summary failed: {exc}[/yellow]")
            return element.content or f"Table from {element.source_name} page {element.page_number}"

    def summarise_tables(
        self, elements: Sequence[TableElement], show_progress: bool = True
    ) -> list[TableElement]:
        """Batch-summarise tables; returns elements with `.summary` filled."""
        if not elements:
            return list(elements)

        console.print(f"[cyan]📊 Summarising {len(elements)} table(s)…[/cyan]")
        iterable = tqdm(elements, desc="Tables", unit="tbl") if show_progress else elements

        for el in iterable:
            if not getattr(el, "summary", None):
                el.summary = self.summarise_table(el)   # type: ignore[attr-defined]
        return list(elements)

    # ── Combined ───────────────────────────────────────────────────────────────

    def summarise_all(
        self,
        images: Sequence[ImageElement],
        tables: Sequence[TableElement],
    ) -> tuple[list[ImageElement], list[TableElement]]:
        """Convenience: summarise everything in one call."""
        summarised_images = self.summarise_images(images)
        summarised_tables = self.summarise_tables(tables)
        return summarised_images, summarised_tables
