from __future__ import annotations

from agent.config import SIMILARITY_THRESHOLD, TTL_DAYS
from agent.state import AgentState
from agent.services.report_store import ReportStore


def query_matcher(state: AgentState, *, report_store: ReportStore) -> dict:
    query = state["query"]
    matched_query = report_store.find_similar(
        query, threshold=SIMILARITY_THRESHOLD, ttl_days=TTL_DAYS
    )
    if matched_query is not None:
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
