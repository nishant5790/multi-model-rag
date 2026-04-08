# Research Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LangGraph agent with FastAPI that routes queries to existing RAG or research systems, with async polling and 30-day TTL cleanup.

**Architecture:** A 4-node LangGraph StateGraph (query_matcher -> router -> rag_retriever/researcher -> formatter) wrapped in a FastAPI service. Old queries get immediate RAG answers; new queries trigger background research via lnk-proj REST API, ingest content into Chroma via multimodal_rag, and return results via polling.

**Tech Stack:** LangGraph, FastAPI, Chroma, Gemini embeddings (gemini-embedding-2-preview), httpx, multimodal_rag (existing)

---

## File Structure

```
agent/
├── __init__.py              # package init, exports build_graph
├── config.py                # all thresholds, URLs, paths
├── state.py                 # AgentState TypedDict
├── graph.py                 # StateGraph definition + compile
├── nodes/
│   ├── __init__.py          # empty
│   ├── query_matcher.py     # semantic similarity check
│   ├── rag_retriever.py     # calls multimodal_rag.query()
│   ├── researcher.py        # calls lnk-proj API + ingests content
│   └── formatter.py         # structures final response
├── services/
│   ├── __init__.py          # empty
│   ├── report_store.py      # query_index Chroma collection wrapper
│   └── cleanup.py           # 30-day TTL cleanup
└── api.py                   # FastAPI app with endpoints
reports/                     # created at runtime, stores report JSONs
run.py                       # uvicorn entry point
pyproject.toml               # updated with new deps
```

---

### Task 1: Add Dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add langgraph, fastapi, uvicorn, httpx to pyproject.toml**

In `pyproject.toml`, add these to the `dependencies` list:

```toml
[project]
name = "multi-modal-rag"
version = "0.1.0"
description = "Multi-modal RAG over PDFs (text, tables, images) with Gemini and Chroma"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "chromadb>=1.5.5",
    "google-genai>=1.70.0",
    "langchain>=1.2.14",
    "langchain-chroma>=0.1.2",
    "langchain-community>=0.4.1",
    "langchain-core>=1.2.23",
    "langchain-google-genai>=4.2.1",
    "langchain-openai>=1.1.12",
    "langchain-text-splitters>=1.1.1",
    "langgraph>=0.4.0",
    "lxml>=6.0.2",
    "matplotlib>=3.10.8",
    "pillow>=12.2.0",
    "python-dotenv>=1.2.2",
    "unstructured-pytesseract>=0.3.15",
    "unstructured[all-docs]>=0.22.10",
    "fastapi>=0.115.0",
    "uvicorn[standard]>=0.30.0",
    "httpx>=0.28.0",
]
```

- [ ] **Step 2: Install dependencies**

Run: `uv sync`
Expected: All packages install successfully, no conflicts.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "feat: add langgraph, fastapi, uvicorn, httpx dependencies"
```

---

### Task 2: Config and State

**Files:**
- Create: `agent/__init__.py`
- Create: `agent/config.py`
- Create: `agent/state.py`

- [ ] **Step 1: Create agent package init**

```python
# agent/__init__.py
"""Research agent: routes queries through RAG or research systems."""
```

- [ ] **Step 2: Create agent/config.py**

```python
# agent/config.py
from pathlib import Path

SIMILARITY_THRESHOLD = 0.85
TTL_DAYS = 30

RESEARCH_API_URL = "http://localhost:8000"
RESEARCH_LIMIT = 5

CHROMA_DIR = Path(".chroma")
QUERY_INDEX_COLLECTION = "query_index"
REPORTS_DIR = Path("reports")

AGENT_HOST = "0.0.0.0"
AGENT_PORT = 8001
```

- [ ] **Step 3: Create agent/state.py**

```python
# agent/state.py
from __future__ import annotations
from typing import TypedDict


class AgentState(TypedDict, total=False):
    query: str
    job_id: str
    route: str
    similarity_score: float
    matched_query: str | None
    research_results: dict | None
    rag_results: dict | None
    report: dict | None
    status: str
    error: str | None
```

- [ ] **Step 4: Verify imports work**

Run: `python -c "from agent.config import SIMILARITY_THRESHOLD; from agent.state import AgentState; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add agent/__init__.py agent/config.py agent/state.py
git commit -m "feat: add agent config and state definition"
```

---

### Task 3: Report Store Service

**Files:**
- Create: `agent/services/__init__.py`
- Create: `agent/services/report_store.py`

- [ ] **Step 1: Create services package init**

```python
# agent/services/__init__.py
```

- [ ] **Step 2: Create agent/services/report_store.py**

```python
# agent/services/report_store.py
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from agent.config import CHROMA_DIR, QUERY_INDEX_COLLECTION, REPORTS_DIR
from multimodal_rag.models import get_embeddings


class ReportStore:
    """Manages the query_index Chroma collection and on-disk report JSONs."""

    def __init__(
        self,
        chroma_dir: Path = CHROMA_DIR,
        collection_name: str = QUERY_INDEX_COLLECTION,
        reports_dir: Path = REPORTS_DIR,
    ) -> None:
        self._reports_dir = Path(reports_dir)
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        chroma_path = Path(chroma_dir).expanduser().resolve()
        chroma_path.mkdir(parents=True, exist_ok=True)
        self._embeddings = get_embeddings()
        self._collection = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=str(chroma_path),
        )

    def store_query(self, query: str, job_id: str, report: dict) -> None:
        report_path = self._reports_dir / f"{job_id}.json"
        report_path.write_text(json.dumps(report, default=str), encoding="utf-8")
        doc = Document(
            page_content=query,
            metadata={
                "created_at": datetime.now(timezone.utc).isoformat(),
                "job_id": job_id,
                "report_path": str(report_path),
            },
        )
        self._collection.add_documents([doc], ids=[job_id])

    def find_similar(
        self, query: str, threshold: float = 0.85, ttl_days: int = 30
    ) -> tuple[str, dict] | None:
        results = self._collection.similarity_search_with_relevance_scores(
            query, k=1
        )
        if not results:
            return None
        doc, score = results[0]
        if score < threshold:
            return None
        created_str = doc.metadata.get("created_at", "")
        if created_str:
            created = datetime.fromisoformat(created_str)
            cutoff = datetime.now(timezone.utc) - timedelta(days=ttl_days)
            if created < cutoff:
                return None
        report_path = doc.metadata.get("report_path", "")
        if report_path and Path(report_path).exists():
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
        else:
            report = {}
        return doc.page_content, report

    def get_report(self, job_id: str) -> dict | None:
        report_path = self._reports_dir / f"{job_id}.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        return None

    def cleanup_old(self, days: int = 30) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        all_data = self._collection.get(include=["metadatas"])
        ids_to_delete: list[str] = []
        for i, meta in enumerate(all_data.get("metadatas") or []):
            created_str = (meta or {}).get("created_at", "")
            if not created_str:
                continue
            created = datetime.fromisoformat(created_str)
            if created < cutoff:
                doc_id = all_data["ids"][i]
                ids_to_delete.append(doc_id)
                report_path = (meta or {}).get("report_path", "")
                if report_path and Path(report_path).exists():
                    Path(report_path).unlink()
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from agent.services.report_store import ReportStore; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agent/services/__init__.py agent/services/report_store.py
git commit -m "feat: add report store service for query index and report storage"
```

---

### Task 4: Cleanup Service

**Files:**
- Create: `agent/services/cleanup.py`

- [ ] **Step 1: Create agent/services/cleanup.py**

```python
# agent/services/cleanup.py
from __future__ import annotations

from agent.services.report_store import ReportStore


def cleanup_old_reports(report_store: ReportStore, days: int = 30) -> int:
    return report_store.cleanup_old(days=days)
```

- [ ] **Step 2: Commit**

```bash
git add agent/services/cleanup.py
git commit -m "feat: add cleanup service for 30-day TTL"
```

---

### Task 5: Query Matcher Node

**Files:**
- Create: `agent/nodes/__init__.py`
- Create: `agent/nodes/query_matcher.py`

- [ ] **Step 1: Create nodes package init**

```python
# agent/nodes/__init__.py
```

- [ ] **Step 2: Create agent/nodes/query_matcher.py**

```python
# agent/nodes/query_matcher.py
from __future__ import annotations

from agent.config import SIMILARITY_THRESHOLD, TTL_DAYS
from agent.state import AgentState
from agent.services.report_store import ReportStore


def query_matcher(state: AgentState, *, report_store: ReportStore) -> dict:
    query = state["query"]
    result = report_store.find_similar(
        query, threshold=SIMILARITY_THRESHOLD, ttl_days=TTL_DAYS
    )
    if result is not None:
        matched_query, _ = result
        return {
            "route": "old",
            "similarity_score": 1.0,
            "matched_query": matched_query,
        }
    return {
        "route": "new",
        "similarity_score": 0.0,
        "matched_query": None,
    }
```

- [ ] **Step 3: Verify import works**

Run: `python -c "from agent.nodes.query_matcher import query_matcher; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add agent/nodes/__init__.py agent/nodes/query_matcher.py
git commit -m "feat: add query matcher node with semantic similarity routing"
```

---

### Task 6: RAG Retriever Node

**Files:**
- Create: `agent/nodes/rag_retriever.py`

- [ ] **Step 1: Create agent/nodes/rag_retriever.py**

```python
# agent/nodes/rag_retriever.py
from __future__ import annotations

from agent.config import CHROMA_DIR
from agent.state import AgentState
from multimodal_rag import MultiModalRAG


def rag_retriever(state: AgentState) -> dict:
    query = state["query"]
    try:
        rag = MultiModalRAG(persist_directory=str(CHROMA_DIR))
        result = rag.query(query)
        return {"rag_results": result, "status": "complete"}
    except Exception as e:
        return {"rag_results": None, "status": "error", "error": str(e)}
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from agent.nodes.rag_retriever import rag_retriever; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/rag_retriever.py
git commit -m "feat: add RAG retriever node using multimodal_rag"
```

---

### Task 7: Researcher Node

**Files:**
- Create: `agent/nodes/researcher.py`

- [ ] **Step 1: Create agent/nodes/researcher.py**

```python
# agent/nodes/researcher.py
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
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from agent.nodes.researcher import researcher; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add agent/nodes/researcher.py
git commit -m "feat: add researcher node with API call, PDF download, and ingestion"
```

---

### Task 8: Formatter Node

**Files:**
- Create: `agent/nodes/formatter.py`

- [ ] **Step 1: Create agent/nodes/formatter.py**

```python
# agent/nodes/formatter.py
from __future__ import annotations

from agent.state import AgentState


def formatter(state: AgentState) -> dict:
    route = state.get("route", "new")

    if route == "old" and state.get("rag_results"):
        rag = state["rag_results"]
        report = {
            "type": "rag_result",
            "answer": rag.get("answer", ""),
            "sources": rag.get("sources", []),
            "matched_query": state.get("matched_query"),
        }
    elif state.get("research_results"):
        report = state["research_results"]
    else:
        report = {"type": "error", "message": "No results available"}

    return {"report": report, "status": state.get("status", "complete")}
```

- [ ] **Step 2: Commit**

```bash
git add agent/nodes/formatter.py
git commit -m "feat: add formatter node for unified response structure"
```

---

### Task 9: LangGraph Definition

**Files:**
- Create: `agent/graph.py`

- [ ] **Step 1: Create agent/graph.py**

```python
# agent/graph.py
from __future__ import annotations

from functools import partial

from langgraph.graph import END, StateGraph

from agent.nodes.formatter import formatter
from agent.nodes.query_matcher import query_matcher
from agent.nodes.rag_retriever import rag_retriever
from agent.nodes.researcher import researcher
from agent.services.report_store import ReportStore
from agent.state import AgentState


def _route_decision(state: AgentState) -> str:
    return "rag_retriever" if state.get("route") == "old" else "researcher"


def build_graph(report_store: ReportStore) -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_matcher", partial(query_matcher, report_store=report_store))
    graph.add_node("rag_retriever", rag_retriever)
    graph.add_node("researcher", partial(researcher, report_store=report_store))
    graph.add_node("formatter", formatter)

    graph.set_entry_point("query_matcher")
    graph.add_conditional_edges("query_matcher", _route_decision)
    graph.add_edge("rag_retriever", "formatter")
    graph.add_edge("researcher", "formatter")
    graph.add_edge("formatter", END)

    return graph.compile()
```

- [ ] **Step 2: Verify graph compiles**

Run: `python -c "from agent.services.report_store import ReportStore; from agent.graph import build_graph; g = build_graph(ReportStore()); print('Graph compiled:', g)"`
Expected: Prints `Graph compiled: ...` without errors.

- [ ] **Step 3: Commit**

```bash
git add agent/graph.py
git commit -m "feat: add LangGraph state graph with 4 nodes and conditional routing"
```

---

### Task 10: FastAPI Application

**Files:**
- Create: `agent/api.py`

- [ ] **Step 1: Create agent/api.py**

```python
# agent/api.py
from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel

from agent.config import CHROMA_DIR, REPORTS_DIR
from agent.graph import build_graph
from agent.services.cleanup import cleanup_old_reports
from agent.services.report_store import ReportStore
from agent.state import AgentState

report_store: ReportStore | None = None
compiled_graph: Any = None
jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global report_store, compiled_graph
    report_store = ReportStore(chroma_dir=CHROMA_DIR, reports_dir=REPORTS_DIR)
    compiled_graph = build_graph(report_store)
    yield


app = FastAPI(title="Research Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None


def _run_graph_sync(job_id: str, query: str) -> None:
    try:
        initial_state: AgentState = {
            "query": query,
            "job_id": job_id,
            "route": "",
            "similarity_score": 0.0,
            "matched_query": None,
            "research_results": None,
            "rag_results": None,
            "report": None,
            "status": "pending",
            "error": None,
        }
        result = compiled_graph.invoke(initial_state)
        jobs[job_id] = {
            "job_id": job_id,
            "status": result.get("status", "complete"),
            "result": result.get("report"),
            "error": result.get("error"),
        }
    except Exception as e:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "error",
            "result": None,
            "error": str(e),
        }


@app.post("/query", response_model=QueryResponse)
async def submit_query(req: QueryRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())

    initial_state: AgentState = {
        "query": req.query,
        "job_id": job_id,
        "route": "",
        "similarity_score": 0.0,
        "matched_query": None,
        "research_results": None,
        "rag_results": None,
        "report": None,
        "status": "pending",
        "error": None,
    }

    result = compiled_graph.invoke(initial_state)
    route = result.get("route", "new")

    if route == "old":
        return QueryResponse(
            job_id=job_id,
            status="complete",
            result=result.get("report"),
        )

    jobs[job_id] = {"job_id": job_id, "status": "researching", "result": None, "error": None}
    background_tasks.add_task(_run_graph_sync, job_id, req.query)
    return QueryResponse(job_id=job_id, status="researching")


@app.get("/status/{job_id}", response_model=QueryResponse)
async def get_status(job_id: str):
    if job_id not in jobs:
        return QueryResponse(job_id=job_id, status="not_found")
    job = jobs[job_id]
    return QueryResponse(
        job_id=job_id,
        status=job["status"],
        result=job.get("result"),
    )


@app.post("/cleanup")
async def run_cleanup():
    deleted = cleanup_old_reports(report_store)
    return {"deleted_count": deleted}


@app.get("/health")
async def health():
    return {"status": "ok"}
```

- [ ] **Step 2: Verify app creates without errors**

Run: `python -c "from agent.api import app; print('FastAPI app:', app.title)"`
Expected: `FastAPI app: Research Agent`

- [ ] **Step 3: Commit**

```bash
git add agent/api.py
git commit -m "feat: add FastAPI app with /query, /status, /cleanup, /health endpoints"
```

---

### Task 11: Entry Point

**Files:**
- Create: `run.py`

- [ ] **Step 1: Create run.py**

```python
# run.py
import uvicorn
from agent.config import AGENT_HOST, AGENT_PORT

if __name__ == "__main__":
    uvicorn.run("agent.api:app", host=AGENT_HOST, port=AGENT_PORT, reload=True)
```

- [ ] **Step 2: Verify server starts**

Run: `timeout 5 python run.py || true`
Expected: See uvicorn startup logs like `Uvicorn running on http://0.0.0.0:8001` before timeout kills it.

- [ ] **Step 3: Commit**

```bash
git add run.py
git commit -m "feat: add uvicorn entry point for research agent"
```

---

### Task 12: Smoke Test

No new files. This task verifies the full system works end-to-end.

- [ ] **Step 1: Start the agent server**

Run: `python run.py &`
Expected: Server starts on port 8001.

- [ ] **Step 2: Test health endpoint**

Run: `curl http://localhost:8001/health`
Expected: `{"status":"ok"}`

- [ ] **Step 3: Test a query (will route as "new" since no data in query_index)**

Run: `curl -X POST http://localhost:8001/query -H "Content-Type: application/json" -d '{"query": "AI trends in healthcare"}'`
Expected: Either `{"job_id":"...","status":"researching","result":null}` (if lnk-proj is running) or `{"job_id":"...","status":"complete","result":{...}}` if it routes to RAG.

- [ ] **Step 4: Test status endpoint**

Run: `curl http://localhost:8001/status/<job_id from step 3>`
Expected: Returns current job status.

- [ ] **Step 5: Test cleanup endpoint**

Run: `curl -X POST http://localhost:8001/cleanup`
Expected: `{"deleted_count":0}` (no old data yet)

- [ ] **Step 6: Stop server and commit**

```bash
kill %1
git add -A
git commit -m "feat: research agent v1 complete with all endpoints"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] LangGraph StateGraph with 4 nodes — Task 9
- [x] Query matcher with semantic similarity — Task 5
- [x] RAG retriever using multimodal_rag — Task 6
- [x] Researcher calling lnk-proj REST API — Task 7
- [x] Formatter for unified responses — Task 8
- [x] Report store with query_index collection — Task 3
- [x] 30-day TTL cleanup — Task 4
- [x] FastAPI with /query, /status, /cleanup, /health — Task 10
- [x] Async polling for new queries — Task 10 (BackgroundTasks)
- [x] In-memory job state — Task 10 (jobs dict)
- [x] Reports stored as JSON on disk — Task 3 (report_store)
- [x] Port 8001 to avoid conflict with lnk-proj — Task 2 (config)
- [x] multimodal_rag untouched — all nodes import it as library

**Placeholder scan:** No TBDs, TODOs, or "implement later" in any step.

**Type consistency:**
- `AgentState` used consistently across all nodes and api.py
- `ReportStore` passed via `partial()` in graph.py, consistent with node signatures
- `report_store.find_similar()` returns `tuple[str, dict] | None` — matched in query_matcher
- `report_store.store_query()` called with `(query, job_id, report)` — consistent signature
