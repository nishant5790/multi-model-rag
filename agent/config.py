import os
from pathlib import Path

SIMILARITY_THRESHOLD = 0.85
TTL_DAYS = 30

RESEARCH_API_URL = os.getenv("RESEARCH_API_URL", "http://localhost:8000")
RESEARCH_LIMIT = 5

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", ".chroma"))
QUERY_INDEX_COLLECTION = "query_index"
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

AGENT_HOST = "0.0.0.0"
AGENT_PORT = int(os.getenv("PORT", "8001"))
