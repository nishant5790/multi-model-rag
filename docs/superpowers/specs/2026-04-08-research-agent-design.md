# Research Agent Design Spec

**Date:** 2026-04-08
**Status:** Draft

## Overview

A LangGraph-based agent that routes user queries through two paths:
- **Old query** (previously researched): RAG retrieval via `multimodal_rag/` with rich results (images, tables, citations)
- **New query**: Research via `lnk-proj` REST API, ingest content into Chroma, return structured report

Single FastAPI service with async polling for new queries.

## Architecture

```
POST /query
    │
    ▼
┌──────────────┐
│ query_matcher │ ── embeds query, checks Chroma "query_index" collection
└──────┬───────┘
       │
   ┌───┴───┐
   │       │
route=old  route=new
   │       │
   ▼       ▼
┌─────────────┐  ┌────────────┐
│rag_retriever│  │ researcher │ ── calls POST /trends/aggregate on lnk-proj
└──────┬──────┘  └─────┬──────┘    downloads PDFs/content, ingests via multimodal_rag
       │               │           stores query+report in query_index
       ▼               ▼
     ┌───────────────────┐
     │    formatter       │ ── structures unified response
     └────────┬──────────┘
              │
              ▼
           response
```

## Systems Used

### multimodal_rag/ (existing, untouched)
- Chroma vector DB with Gemini embeddings (`gemini-embedding-2-preview`, 768 dims)
- PDF ingestion via Unstructured (hi_res strategy)
- Graph-expanded retrieval (BFS over chunk adjacency)
- LLM generation with citations via Gemini 2.5 Flash
- Collection: `"multimodal_rag"`

### lnk-proj REST API (existing, untouched)
- FastAPI on `localhost:8000`
- `POST /trends/aggregate` — hits 9 sources (HN, GitHub, YouTube, arXiv, Reddit, RSS, Google News, LinkedIn, Podcasts)
- Returns: `AggregatedTrends` with `raw_results` (per-source) + `summary` (AI-generated TrendSummary)
- Requires: `GOOGLE_API_KEY`, `YOUTUBE_API_KEY`

## Project Structure

```
multi-model-rag/
├── multimodal_rag/           # existing, untouched
├── agent/
│   ├── __init__.py
│   ├── config.py             # thresholds, URLs, TTL settings
│   ├── state.py              # LangGraph AgentState TypedDict
│   ├── graph.py              # LangGraph StateGraph definition
│   ├── nodes/
│   │   ├── __init__.py
│   │   ├── query_matcher.py
│   │   ├── rag_retriever.py
│   │   ├── researcher.py
│   │   └── formatter.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── report_store.py   # query_index Chroma collection wrapper
│   │   └── cleanup.py        # 30-day TTL cleanup
│   └── api.py                # FastAPI app
├── reports/                  # stored research report JSONs
├── requirements.txt          # updated
└── run.py                    # uvicorn entry point
```

## State Definition

```python
class AgentState(TypedDict):
    query: str                    # user's input query
    job_id: str                   # unique job identifier
    route: str                    # "old" or "new"
    similarity_score: float       # best match score from Chroma
    matched_query: str | None     # the original query it matched against
    research_results: dict | None # raw results from lnk-proj API
    rag_results: dict | None      # results from multimodal_rag retrieval
    report: dict | None           # final structured report
    status: str                   # "pending" | "researching" | "ingesting" | "complete" | "error"
    error: str | None
```

## Node Details

### 1. query_matcher

**Input:** `query`
**Output:** `route`, `similarity_score`, `matched_query`

- Embeds the query using `GoogleGenerativeAIEmbeddings` (model: `gemini-embedding-2-preview`)
- Queries the `"query_index"` Chroma collection via similarity search with score
- If top result similarity >= 0.85 AND `created_at` within 30 days:
  - `route = "old"`, set `matched_query` to the stored query text
- Otherwise:
  - `route = "new"`

**query_index collection schema:**
- Document content: the query text (this gets embedded)
- Metadata: `{created_at: ISO timestamp, job_id: str, report_path: str}` (report JSON stored on disk to avoid Chroma metadata size limits)

### 2. rag_retriever

**Input:** `query`, `matched_query`
**Output:** `rag_results`

- Instantiates `MultiModalRAG(persist_directory=".chroma")`
- Calls `rag.query(query)` using the **current user query** (not the matched query — this gives better retrieval since the user's phrasing may differ)
  1. Vector similarity search (top 6 chunks)
  2. Graph expansion via BFS (1 hop, up to 8 neighbors)
  3. Context formatting with metadata
  4. Gemini 2.5 Flash generation with citations
- Returns: `{"answer": str, "sources": [{"chunk_id", "content_type", "page", "doc_link", "image_path", "image_link", "snippet", "is_expanded"}]}`

### 3. researcher

**Input:** `query`, `job_id`
**Output:** `research_results`, `status`

Steps:
1. Set `status = "researching"`
2. Call `POST http://localhost:8000/trends/aggregate` with `{"topic": query, "limit": 5}`
3. Parse response: extract `TrendSummary` + top 5 items per source
4. Set `status = "ingesting"`
5. Collect downloadable content from results:
   - arXiv: PDF URLs -> download to temp dir
   - Other sources: article URLs, text content
6. Ingest PDFs via `MultiModalRAG.ingest_pdf(pdf_path)` into the RAG Chroma collection
7. For non-PDF text content: create `Document` objects and add directly to Chroma
8. Store query + structured report in `query_index` collection via `report_store.store_query()`
9. Set `status = "complete"`

**Error handling:** If lnk-proj API is unreachable or fails, set `status = "error"` and `error` message. Do not crash.

### 4. formatter

**Input:** `route`, `rag_results` or `research_results`
**Output:** `report`

For **old queries** (route="old"):
```json
{
  "type": "rag_result",
  "answer": "...",
  "sources": [
    {"content_type": "text|table|image", "page": 3, "image_path": "...", "snippet": "..."}
  ],
  "matched_query": "original query text"
}
```

For **new queries** (route="new"):
```json
{
  "type": "research_result",
  "summary": {
    "top_trends": ["..."],
    "content_angles": [{"hook": "...", "angle": "...", "supporting_sources": ["..."]}],
    "analysis": "..."
  },
  "sources_by_platform": {
    "hackernews": [{"title": "...", "url": "...", "metadata": {}}],
    "github": [...],
    "arxiv": [...]
  }
}
```

## Services

### report_store.py

Thin wrapper around a Chroma collection `"query_index"`:

- `store_query(query: str, job_id: str, report: dict)` — embeds query text, stores report as a JSON file on disk at `reports/{job_id}.json`, stores metadata `{created_at, job_id, report_path}` in Chroma (avoids Chroma metadata size limits)
- `find_similar(query: str, threshold: float = 0.85) -> tuple[str, dict] | None` — returns `(matched_query, report_dict)` if above threshold and within 30 days, else None
- `get_report(job_id: str) -> dict | None` — retrieves stored report by filtering on job_id metadata
- `cleanup_old(days: int = 30)` — deletes entries older than `days` from both query_index and associated RAG documents

### cleanup.py

- `cleanup_old_reports(report_store, rag, days=30)` — calls `report_store.cleanup_old(days)` and removes associated ingested documents
- Triggered on every new query (lightweight: just a metadata date comparison)
- Also exposed as `POST /cleanup` for manual trigger

## API Endpoints

| Method | Path | Description | Response |
|--------|------|-------------|----------|
| POST | `/query` | Submit a query | Old: `{job_id, status: "complete", result: {...}}` / New: `{job_id, status: "researching"}` |
| GET | `/status/{job_id}` | Poll for results | `{job_id, status, result (if complete), error (if failed)}` |
| POST | `/cleanup` | Manual 30-day cleanup | `{deleted_count: N}` |
| GET | `/health` | Health check | `{status: "ok"}` |

## Job State Management

- In-memory dict: `jobs: dict[str, AgentState]`
- Background task (FastAPI `BackgroundTasks`) runs the graph for new queries and updates the dict
- No persistence needed — jobs are transient; the durable state is in Chroma

## Configuration (agent/config.py)

```python
SIMILARITY_THRESHOLD = 0.85
TTL_DAYS = 30
RESEARCH_API_URL = "http://localhost:8000"
RESEARCH_LIMIT = 5          # top N items per source
CHROMA_DIR = ".chroma"      # shared with multimodal_rag
QUERY_INDEX_COLLECTION = "query_index"
AGENT_HOST = "0.0.0.0"
AGENT_PORT = 8001           # different from lnk-proj's 8000
```

## Dependencies (new)

- `langgraph` — state graph orchestration
- `fastapi` + `uvicorn` — API server
- `httpx` — async HTTP client for calling lnk-proj API
- All existing `multimodal_rag` dependencies remain

## Key Decisions

1. **Separate Chroma collection for query index** — keeps query matching isolated from document retrieval
2. **Shared Chroma directory** — both query_index and multimodal_rag collections live in the same `.chroma` directory
3. **Agent runs on port 8001** — avoids conflict with lnk-proj on 8000
4. **In-memory job state** — simple, no extra infra; Chroma is the durable store
5. **Cleanup on every new query** — avoids need for cron/scheduler
6. **multimodal_rag untouched** — imported as a library, no modifications
