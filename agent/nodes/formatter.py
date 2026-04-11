from __future__ import annotations

from agent.state import AgentState


def formatter(state: AgentState) -> dict:
    route = state.get("route", "new")

    if route == "old" and state.get("rag_results"):
        rag = state["rag_results"]
        docs = rag.get("documents", [])
        report_doc = next(
            (d for d in docs if d.get("metadata", {}).get("content_type") == "research_report"),
            None,
        )
        if report_doc:
            report = {
                "type": "research_report",
                "markdown": report_doc["content"],
                "sources_used": report_doc.get("metadata", {}).get("sources_used", "").split(","),
                "generated_at": report_doc.get("metadata", {}).get("created_at", ""),
            }
        else:
            report = {
                "type": "rag_result",
                "documents": [d["content"] for d in docs],
                "count": len(docs),
                "matched_query": state.get("matched_query"),
            }
    elif state.get("research_results"):
        research = state["research_results"]
        if research.get("type") == "research_report":
            report = {
                "type": "research_report",
                "markdown": research.get("markdown", ""),
                "sources_used": research.get("sources_used", []),
                "generated_at": research.get("generated_at", ""),
            }
        else:
            report = research
    else:
        report = {"type": "error", "message": "No results available"}

    return {"report": report, "status": state.get("status", "complete")}
