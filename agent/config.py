import os

SIMILARITY_THRESHOLD = 0.85
TTL_DAYS = 30

RESEARCH_API_URL = os.getenv("RESEARCH_API_URL", "http://localhost:8000")
RESEARCH_LIMIT = 5

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp/mcp")
DEEP_AGENT_MODEL = os.getenv("DEEP_AGENT_MODEL", "google_genai:gemini-2.5-flash")

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION = "multimodal_rag"
QUERY_INDEX_COLLECTION = "query_index"

AGENT_HOST = "0.0.0.0"
AGENT_PORT = int(os.getenv("PORT", "8001"))
