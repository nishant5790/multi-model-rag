from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

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
    "hackernews", "youtube", "github", "linkedin",
    "reddit", "rss", "google_news", "podcasts", "arxiv",
]

SYSTEM_PROMPT = f"""\                                                                                                                                                                                                                                 
  You are a LinkedIn content strategist and trend intelligence analyst. \
  Given a topic, use the available MCP tools to find what's trending \
  RIGHT NOW across multiple platforms, then produce a comprehensive \                                                                                                                                                                                   
  research report designed to help create high-engagement LinkedIn \                                                                                                                                                                                    
  posts in minimal time.                                                                                                                                                                                                                                
                                                                                                                                                                                                                                                        
  IMPORTANT RULES:                                                                                                                                                                                                                                      
  - Do NOT use write_todos. Do NOT plan. Just execute immediately.
  - Do NOT use file tools (read_file, write_file, edit_file, ls, glob, grep).                                                                                                                                                                           
  - Do NOT use the execute tool or the task tool.
  - ONLY use the MCP trend-finding tools (find_hackernews_trends, \                                                                                                                                                                                     
  find_youtube_trends, find_github_trends, find_reddit_trends, \                                                                                                                                                                                        
  find_rss_trends, find_google_news_trends, find_podcast_trends, \
  find_arxiv_trends, find_linkedin_trends, etc.).                                                                                                                                                                                                       
  - Call ALL available source tools, then write the full report in your \
  final message.                                                                                                                                                                                                                                        
   
  EXECUTION:                                                                                                                                                                                                                                            
  1. Call ALL available source tools in parallel with topic and \
  limit={RESEARCH_LIMIT}.                                                                                                                                                                                                                               
  2. Once all results are back, analyze and write the complete report \
  in your FINAL message. Do NOT call any more tools after analysis begins.                                                                                                                                                                              
                                                                                                                                                                                                                                                        
  ANALYSIS RULES (apply before writing):
  - DEDUPLICATE: The same story often trends across Reddit, HN, Google \                                                                                                                                                                                
  News simultaneously (e.g., a viral news story). Report it ONCE but \
  cite ALL platforms where it appeared. Cross-platform presence = \                                                                                                                                                                                     
  stronger signal.
  - SIGNAL vs NOISE: Distinguish genuine industry trends from viral \                                                                                                                                                                                   
  outrage, jokes, or clickbait. A practitioner discussion on Reddit \
  with 300 comments is higher signal than a reposted news headline. \                                                                                                                                                                                   
  GitHub repos with high stars/day are stronger than old repos with \                                                                                                                                                                                   
  accumulated stars.                                                                                                                                                                                                                                    
  - REAL NUMBERS ONLY: Only cite metrics that exist in the tool results \                                                                                                                                                                               
  (upvotes, views, stars, stars_per_day, comments, like_ratio). Never \                                                                                                                                                                                 
  fabricate or estimate engagement numbers.                                                                                                                                                                                                             
  - NON-ENGLISH CONTENT: Note its existence in a footnote but do not \                                                                                                                                                                                  
  feature in main analysis unless it signals a geographic trend.                                                                                                                                                                                        
  - PRACTITIONER SIGNAL > MEDIA SIGNAL: GitHub activity + Reddit/HN \                                                                                                                                                                                   
  discussion = builders care about this. News articles alone = media \                                                                                                                                                                                  
  narrative, may not resonate with LinkedIn technical audience.                                                                                                                                                                                         
  - RECENCY: Prioritize items from the last 48-72 hours over older content.
                                                                                                                                                                                                                                                        
  FORMAT YOUR FINAL MESSAGE AS:                                                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
  # Trend Report: {{topic}}                                                                                                                                                                                                                             
  **Date**: {{date}} | **Sources**: {{count}} platforms analyzed
                                                                                                                                                                                                                                                        
  ---
                                                                                                                                                                                                                                                        
  ## Executive Summary                                                                                                                                                                                                                                  
  3-4 sentences: dominant narrative this week, overall sentiment \
  (bullish/cautious/divided), and one surprising finding from the data.                                                                                                                                                                                 
                                                                                                                                                                                                                                                        
  ## Cross-Platform Trending Themes                                                                                                                                                                                                                     
  Rank by signal strength. For EACH theme:                                                                                                                                                                                                              
                                                                                                                                                                                                                                                        
  ### [Theme Name]
  - **Signal**: 🔴 Strong (3+ sources) | 🟡 Emerging (2 sources) | 🟢 Early (1 source)                                                                                                                                                                  
  - **What**: One sentence                                                                                                                                                                                                                              
  - **Why now**: What triggered this in the last 7 days
  - **Evidence**: Specific items with real metrics \                                                                                                                                                                                                    
  (e.g., "Reddit r/cscareerquestions: 1,019 upvotes, 303 comments")                                                                                                                                                                                     
  - **Trajectory**: Rising / Peaking / Fading                                                                                                                                                                                                           
  - **LinkedIn angle**: One-line post idea + suggested format                                                                                                                                                                                           
                                                                                                                                                                                                                                                        
  ## Top 10 LinkedIn Hooks                                                                                                                                                                                                                              
  Ready-to-use scroll-stopping opening lines. For each:                                                                                                                                                                                                 
  1. **Hook**: The opening line                                                                                                                                                                                                                         
     - **Target**: Who engages (engineers / founders / hiring managers / etc.)                                                                                                                                                                          
     - **Format**: Text post / Carousel / Poll / Story                                                                                                                                                                                                  
     - **Backed by**: The data point or source that supports it                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
  ## Engagement Leaderboard                                                                                                                                                                                                                             
  Top 15 items across ALL sources ranked by engagement. Table format:                                                                                                                                                                                   
  | # | Title | Platform | Key Metric | URL |                                                                                                                                                                                                           
                                                                                                                                                                                                                                                        
  ## Platform Pulse                                                                                                                                                                                                                                     
  Only include platforms that returned relevant results. For each:                                                                                                                                                                                      
  ### [Platform Name]                                                                                                                                                                                                                                   
  - **Top signal**: Single most important item + why
  - **Sentiment**: Excited / Skeptical / Debating / Mixed                                                                                                                                                                                               
  - **Unique angle**: Something only visible from this platform                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
  ## Contrarian & Gap Analysis                                                                                                                                                                                                                          
  - **Overhyped**: High volume, low substance — content to AVOID                                                                                                                                                                                        
  - **Underrated**: Low volume, high signal — content OPPORTUNITY                                                                                                                                                                                       
  - **Gap**: What's missing from the conversation that should be there                                                                                                                                                                                  
  - **Emerging**: What could go mainstream in 2-4 weeks                                                                                                                                                                                                 
                                                                                                                                                                                                                                                        
  ## Quotable Stats                                                                                                                                                                                                                                     
  Bullet list of concrete numbers for posts. Each with:                                                                                                                                                                                                 
  - The stat + context (why it matters)                                                                                                                                                                                                                 
  - Source URL for credibility                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                        
  ## This Week's Content Playbook                                                                                                                                                                                                                       
  ### Must-Post (multi-source signal, high conviction)
  1. [Hook] — [Format] — [Why now + data point]                                                                                                                                                                                                         
  2. [Hook] — [Format] — [Why now + data point]                                                                                                                                                                                                         
  3. [Hook] — [Format] — [Why now + data point]                                                                                                                                                                                                         
                                                                                                                                                                                                                                                        
  ### Worth Testing (emerging signal)                                                                                                                                                                                                                   
  1. [Hook] — [Format] — [Why now]                                                                                                                                                                                                                      
  2. [Hook] — [Format] — [Why now]                                                                                                                                                                                                                      
                  
  ### Monitor for Next Week                                                                                                                                                                                                                             
  - [Topic to watch]
  - [Topic to watch]
                                                                                                                                                                                                                                                        
  ## Raw Source Links                                                                                                                                                                                                                                   
  Grouped by platform. Top items only. Format:                                                                                                                                                                                                          
  - [Title](URL) — one-line description (key metric)                                                                                                                                                                                                    
                                                                                                                                                                                                                                                        
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
    skip_keywords = ["write_todos", "Updated todo", "todo list to"]

    def _get_text(msg) -> str | None:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block["text"])
                elif isinstance(block, str):
                    parts.append(block)
            return "\n".join(parts) if parts else None
        return None

    def _is_ai(msg) -> bool:
        return getattr(msg, "type", "") == "ai"

    logger.info("Agent returned %d messages", len(messages))

    for msg in reversed(messages):
        if not _is_ai(msg):
            continue
        text = _get_text(msg)
        if not text or len(text) < 200:
            continue
        if any(kw in text for kw in skip_keywords):
            continue
        return text

    return None


async def _run_deep_agent(query: str) -> dict:
    from langchain_mcp_adapters.client import MultiServerMCPClient
    from deepagents import create_deep_agent

    client = MultiServerMCPClient(
        {
            "trends": {
                "url": MCP_SERVER_URL,
                "transport": "streamable_http",
            }
        }
    )
    try:
        tools = await client.get_tools()
    except Exception:
        logger.warning("streamable_http failed, falling back to sse transport")
        client = MultiServerMCPClient(
            {
                "trends": {
                    "url": MCP_SERVER_URL,
                    "transport": "sse",
                }
            }
        )
        tools = await client.get_tools()

    logger.info("Loaded %d MCP tools", len(tools))
    agent = create_deep_agent(
        model=DEEP_AGENT_MODEL,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": query}]}
    )
    return result


def _run_async(coro):
    """Run an async coroutine from sync code, even inside an existing event loop."""
    def _run():
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run).result()


def researcher(state: AgentState, *, report_store: ReportStore) -> dict:
    query = state["query"]
    job_id = state["job_id"]

    try:
        logger.info("Starting deep agent research for: %s", query)
        result = _run_async(_run_deep_agent(query))
    except Exception as e:
        logger.exception("Deep agent failed for query: %s", query)
        return {
            "research_results": None,
            "status": "error",
            "error": f"Deep agent failed: {e}",
        }

    markdown = _extract_report(result)
    if not markdown:
        logger.error("Deep agent returned no report for: %s", query)
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
    try:
        rag = MultiModalRAG(persist_directory=str(CHROMA_DIR))
        rag._vs.add_documents([doc])
    except Exception:
        pass

    report_store.store_query(query, job_id)

    logger.info(
        "Research complete for '%s': %d chars, sources=%s",
        query, len(markdown), sources_used,
    )

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
