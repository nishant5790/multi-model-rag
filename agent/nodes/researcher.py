from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from langchain_core.documents import Document

from agent.config import (
    CHROMA_DIR,
    DEEP_AGENT_MODEL,
    MCP_SERVER_URL,
    RESEARCH_LIMIT,
)
from agent.state import AgentState
from agent.services.report_store import ReportStore
from multimodal_rag import MultiModalRAG

KNOWN_PLATFORMS = [
    "hacker_news", "youtube", "github", "linkedin",
    "reddit", "rss", "google_news", "podcasts",
]

SYSTEM_PROMPT = f"""\
You are a LinkedIn content strategist and trend scout. Given a topic,
use the available MCP tools to find what's trending RIGHT NOW across
multiple platforms, then produce a research report designed to help
create high-engagement LinkedIn posts in minimal time.

Steps:
1. Search at least 3-5 source tools to find the freshest, most-discussed
   angles on this topic. Pass limit={RESEARCH_LIMIT} to each tool.
2. Focus on: what's getting engagement, what's controversial, what data
   points are surprising, what stories are emerging.
3. Write a structured report in EXACTLY this format:

# Trend Report: {{topic}}

## Top 5 Hooks
Ready-to-use LinkedIn opening lines ranked by engagement potential.
Each hook should stop the scroll -- use numbers, contrarian takes,
or surprising data points.

## Trending Angles
What specific angles are getting the most engagement right now?
For each angle:
- The angle in one sentence
- Why it's trending (data/evidence from sources)
- Engagement signals (upvotes, comments, views, stars)
- Suggested LinkedIn post format (carousel, text, poll, story)

## Key Data Points & Stats
Quotable numbers, metrics, and facts you can drop into posts.
Include the source URL for credibility.

## Platform Pulse
### (platform name)
What each community is saying -- the conversation, sentiment,
and unique angles per platform. Only include platforms with relevant results.

## Raw Source Links
Bullet list of the top URLs with one-line descriptions.

---
Generated: {{date}}
Sources queried: {{list of platforms used}}
"""


def _detect_sources(report_text: str) -> list[str]:
    """Check which known platform names appear in the report."""
    lower = report_text.lower()
    return [p for p in KNOWN_PLATFORMS if p.replace("_", " ") in lower or p in lower]


def _extract_report(result: dict) -> str | None:
    """Walk agent messages in reverse and find the research report."""
    messages = result.get("messages", [])
    best: str | None = None
    best_len = 0

    for msg in reversed(messages):
        content = msg.content if hasattr(msg, "content") else str(msg)
        if not isinstance(content, str):
            continue
        if "# Trend Report:" in content:
            return content
        if len(content) > best_len:
            best = content
            best_len = len(content)

    return best


async def _run_deep_agent(query: str) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from deepagents import create_deep_agent

    async with MultiServerMCPClient(
        {
            "trends": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    ) as mcp_client:
        tools = mcp_client.get_tools()
        agent = create_deep_agent(
            model=DEEP_AGENT_MODEL,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
        )
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": query}]}
        )
    return result


def researcher(state: AgentState, *, report_store: ReportStore) -> dict:
    query = state["query"]
    job_id = state["job_id"]

    try:
        result = asyncio.run(_run_deep_agent(query))
    except Exception as e:
        return {
            "research_results": None,
            "status": "error",
            "error": f"Deep agent failed: {e}",
        }

    markdown = _extract_report(result)
    if not markdown:
        return {
            "research_results": None,
            "status": "error",
            "error": "Deep agent returned no usable report",
        }

    sources_used = _detect_sources(markdown)
    now = datetime.now(timezone.utc).isoformat()

    doc = Document(
        page_content=markdown,
        metadata={
            "content_type": "research_report",
            "query": query,
            "job_id": job_id,
            "sources_used": ",".join(sources_used),
            "created_at": now,
        },
    )
    rag = MultiModalRAG(persist_directory=str(CHROMA_DIR))
    rag._vs.add_documents([doc])

    report_store.store_query(query, job_id)

    return {
        "research_results": {
            "type": "research_report",
            "markdown": markdown,
            "sources_used": sources_used,
            "generated_at": now,
        },
        "status": "complete",
        "error": None,
    }
