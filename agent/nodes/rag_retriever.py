from __future__ import annotations

from agent.state import AgentState
from multimodal_rag import MultiModalRAG


def rag_retriever(state: AgentState) -> dict:
    query = state["query"]
    try:
        rag = MultiModalRAG()
        result = rag.retrieve_docs(query)
        return {"rag_results": result, "status": "complete"}
    except Exception as e:
        return {"rag_results": None, "status": "error", "error": str(e)}
