"""Load PDFs with unstructured (hi-res), produce LangChain documents for text, tables, images."""

from __future__ import annotations

import base64
import mimetypes
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Element, Image, Table
from unstructured.partition.pdf import partition_pdf

from multimodal_rag.links import image_file_link, pdf_doc_link


def _page_num(el: Element) -> int | None:
    n = el.metadata.page_number
    return int(n) if n is not None else None


def _caption_image(image_path: Path, chat: ChatGoogleGenerativeAI) -> str:
    mime, _ = mimetypes.guess_type(str(image_path))
    if not mime:
        mime = "image/png"
    b64 = base64.b64encode(image_path.read_bytes()).decode()
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this figure or image in 2–5 sentences for retrieval. "
                    "Include visible text, chart types, labels, and numbers if any."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ]
    )
    out = chat.invoke([msg])
    text = out.content if isinstance(out.content, str) else str(out.content)
    return text.strip()


def pdf_elements_to_documents(
    pdf_path: Path,
    *,
    image_output_dir: Path,
    chat: ChatGoogleGenerativeAI | None,
    caption_images: bool = True,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Partition a PDF and return LangChain `Document` chunks with link metadata."""
    pdf_path = pdf_path.expanduser().resolve()
    image_output_dir = image_output_dir.expanduser().resolve()
    image_output_dir.mkdir(parents=True, exist_ok=True)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy="hi_res",
        extract_images_in_pdf=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_to_payload=False,
        extract_image_block_output_dir=str(image_output_dir),
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    docs: list[Document] = []

    for el in elements:
        page = _page_num(el)
        doc_link = pdf_doc_link(pdf_path, page)

        if isinstance(el, Table):
            body = (el.metadata.text_as_html or el.text or "").strip()
            if not body:
                continue
            meta = {
                "source": str(pdf_path),
                "page": page or -1,
                "content_type": "table",
                "doc_link": doc_link,
            }
            img_path = el.metadata.image_path
            if img_path:
                meta["image_path"] = str(Path(img_path).resolve())
                il = image_file_link(Path(img_path))
                if il:
                    meta["image_link"] = il
            docs.append(Document(page_content=body, metadata=meta))
            continue

        if isinstance(el, Image):
            raw_path = el.metadata.image_path
            if not raw_path:
                continue
            ip = Path(raw_path).expanduser().resolve()
            if not ip.is_file():
                continue
            if caption_images and chat is not None:
                description = _caption_image(ip, chat)
            else:
                description = (
                    f"Extracted figure (no vision caption). File: {ip.name} "
                    f"page {page or '?'}"
                )
            embed_text = f"[IMAGE] {description}"
            meta = {
                "source": str(pdf_path),
                "page": page or -1,
                "content_type": "image",
                "doc_link": doc_link,
                "image_path": str(ip),
            }
            il = image_file_link(ip)
            if il:
                meta["image_link"] = il
            docs.append(Document(page_content=embed_text, metadata=meta))
            continue

        text = (el.text or "").strip()
        if not text:
            continue

        for chunk in splitter.split_text(text):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": str(pdf_path),
                        "page": page or -1,
                        "content_type": "text",
                        "doc_link": doc_link,
                    },
                )
            )

    return docs
