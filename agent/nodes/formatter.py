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
