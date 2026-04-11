import os
from pathlib import Path

SIMILARITY_THRESHOLD = 0.85
TTL_DAYS = 30

RESEARCH_API_URL = os.getenv("RESEARCH_API_URL", "http://localhost:8000")
RESEARCH_LIMIT = 5

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/mcp")
DEEP_AGENT_MODEL = os.getenv("DEEP_AGENT_MODEL", "google_genai:gemini-2.5-flash")

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ".chroma"))
QUERY_INDEX_COLLECTION = "query_index"

AGENT_HOST = "0.0.0.0"
AGENT_PORT = int(os.getenv("PORT", "8001"))
