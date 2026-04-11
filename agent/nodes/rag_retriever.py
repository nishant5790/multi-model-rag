from __future__ import annotations

from agent.config import CHROMA_DIR
from agent.state import AgentState
from multimodal_rag import MultiModalRAG


def rag_retriever(state: AgentState) -> dict:
    query = state["query"]
    try:
        rag = MultiModalRAG(persist_directory=str(CHROMA_DIR))
        result = rag.retrieve_docs(query)
        return {"rag_results": result, "status": "complete"}
    except Exception as e:
        return {"rag_results": None, "status": "error", "error": str(e)}
