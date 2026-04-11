from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

from agent.config import CHROMA_DIR
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
    from dotenv import load_dotenv
    load_dotenv()
    report_store = ReportStore(chroma_dir=CHROMA_DIR)
    compiled_graph = build_graph(report_store)
    yield


app = FastAPI(title="Research Agent", lifespan=lifespan)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None
    error: str | None = None


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
    from agent.config import SIMILARITY_THRESHOLD, TTL_DAYS

    job_id = str(uuid.uuid4())

    match = report_store.find_similar(
        req.query, threshold=SIMILARITY_THRESHOLD, ttl_days=TTL_DAYS
    )

    if match is not None:
        initial_state: AgentState = {
            "query": req.query,
            "job_id": job_id,
            "route": "old",
            "similarity_score": 1.0,
            "matched_query": match,
            "research_results": None,
            "rag_results": None,
            "report": None,
            "status": "pending",
            "error": None,
        }
        result = compiled_graph.invoke(initial_state)
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
        error=job.get("error"),
    )


@app.get("/report/{job_id}")
async def get_report(job_id: str):
    """Return the markdown report for a completed job."""
    if job_id not in jobs:
        return {"status": "not_found", "markdown": None}
    job = jobs[job_id]
    if job["status"] != "complete":
        return {"status": job["status"], "markdown": None}
    result = job.get("result") or {}
    markdown = result.get("markdown", "")
    return {
        "status": "complete",
        "markdown": markdown,
        "sources_used": result.get("sources_used", []),
        "generated_at": result.get("generated_at", ""),
    }


@app.get("/report/{job_id}/download")
async def download_report(job_id: str):
    """Download the report as a .md file."""
    if job_id not in jobs:
        return PlainTextResponse("Job not found", status_code=404)
    job = jobs[job_id]
    if job["status"] != "complete":
        return PlainTextResponse(f"Job status: {job['status']}", status_code=202)
    result = job.get("result") or {}
    markdown = result.get("markdown", "No report content")
    return PlainTextResponse(
        content=markdown,
        media_type="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="report-{job_id[:8]}.md"'},
    )


@app.post("/cleanup")
async def run_cleanup():
    deleted = cleanup_old_reports(report_store)
    return {"deleted_count": deleted}


@app.get("/health")
async def health():
    return {"status": "ok"}
