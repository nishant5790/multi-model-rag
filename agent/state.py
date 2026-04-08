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
