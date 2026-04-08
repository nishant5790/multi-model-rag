from __future__ import annotations

import tempfile
from pathlib import Path

import httpx
from langchain_core.documents import Document

from agent.config import CHROMA_DIR, RESEARCH_API_URL, RESEARCH_LIMIT
from agent.state import AgentState
from agent.services.report_store import ReportStore
from multimodal_rag import MultiModalRAG


def _download_pdf(url: str, dest_dir: Path) -> Path | None:
    try:
        resp = httpx.get(url, follow_redirects=True, timeout=60)
        resp.raise_for_status()
        filename = url.split("/")[-1]
        if not filename.endswith(".pdf"):
            filename += ".pdf"
        path = dest_dir / filename
        path.write_bytes(resp.content)
        return path
    except Exception:
        return None


def _extract_top_items(raw_results: dict, limit: int) -> dict:
    sources_by_platform: dict[str, list] = {}
    for source_name, source_data in raw_results.items():
        items = source_data.get("items", [])
        sources_by_platform[source_name] = items[:limit]
    return sources_by_platform


def _collect_pdf_urls(sources_by_platform: dict) -> list[str]:
    urls: list[str] = []
    for items in sources_by_platform.values():
        for item in items:
            url = item.get("url", "")
            meta = item.get("metadata", {})
            pdf_url = meta.get("pdf_url", "")
            if pdf_url:
                urls.append(pdf_url)
            elif url.endswith(".pdf"):
                urls.append(url)
    return urls


def _collect_text_content(sources_by_platform: dict) -> list[Document]:
    docs: list[Document] = []
    for source_name, items in sources_by_platform.items():
        for item in items:
            title = item.get("title", "")
            url = item.get("url", "")
            if not title:
                continue
            meta = item.get("metadata", {})
            content = f"# {title}\n\nSource: {source_name}\nURL: {url}\n"
            if meta.get("description"):
                content += f"\n{meta['description']}"
            if meta.get("selftext"):
                content += f"\n{meta['selftext']}"
            if meta.get("summary"):
                content += f"\n{meta['summary']}"
            docs.append(Document(
                page_content=content,
                metadata={"source": url, "content_type": "text", "platform": source_name},
            ))
    return docs


def researcher(state: AgentState, *, report_store: ReportStore) -> dict:
    query = state["query"]
    job_id = state["job_id"]

    try:
        resp = httpx.post(
            f"{RESEARCH_API_URL}/trends/aggregate",
            json={"topic": query, "limit": RESEARCH_LIMIT},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"research_results": None, "status": "error", "error": f"Research API failed: {e}"}

    raw_results = data.get("raw_results", {})
    summary = data.get("summary", {})
    sources_by_platform = _extract_top_items(raw_results, RESEARCH_LIMIT)

    rag = MultiModalRAG(persist_directory=str(CHROMA_DIR))

    pdf_urls = _collect_pdf_urls(sources_by_platform)
    if pdf_urls:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for url in pdf_urls:
                pdf_path = _download_pdf(url, tmp_path)
                if pdf_path:
                    try:
                        rag.ingest_pdf(pdf_path, caption_images=False)
                    except Exception:
                        pass

    text_docs = _collect_text_content(sources_by_platform)
    if text_docs:
        vs = rag._vs
        vs.add_documents(text_docs)

    report = {
        "type": "research_result",
        "summary": summary,
        "sources_by_platform": sources_by_platform,
    }
    report_store.store_query(query, job_id, report)

    return {"research_results": report, "status": "complete"}
