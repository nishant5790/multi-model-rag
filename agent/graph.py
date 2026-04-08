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


def build_graph(report_store: ReportStore):
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
