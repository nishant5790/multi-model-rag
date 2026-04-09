"""
config.py — Central configuration for Multi-Modal RAG
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ───────────────────────────────────────────────────────────────────
GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise EnvironmentError(
        "GOOGLE_API_KEY is not set. Add it to your .env file."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ── Models ─────────────────────────────────────────────────────────────────────
EMBEDDING_MODEL: str = os.getenv(
    "EMBEDDING_MODEL",
    "models/gemini-embedding-2-preview",
)
CHAT_MODEL: str = os.getenv("CHAT_MODEL", "gemini-2.5-flash")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR          = Path(__file__).parent
VECTOR_STORE_PATH = Path(os.getenv("VECTOR_STORE_PATH", str(BASE_DIR / "chroma_db")))
DOCSTORE_PATH     = Path(os.getenv("DOCSTORE_PATH",     str(BASE_DIR / "docstore")))
IMAGE_STORE_PATH  = Path(os.getenv("IMAGE_STORE_PATH",  str(BASE_DIR / "extracted_images")))
DOCS_PATH         = Path(os.getenv("DOCS_PATH",         str(BASE_DIR / "docs")))

for _d in [VECTOR_STORE_PATH, DOCSTORE_PATH, IMAGE_STORE_PATH, DOCS_PATH]:
    _d.mkdir(parents=True, exist_ok=True)

# ── Chunking ───────────────────────────────────────────────────────────────────
CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 4000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

# ── Retriever ──────────────────────────────────────────────────────────────────
RETRIEVER_K        = 6   # total docs to retrieve
RETRIEVER_K_TEXT   = 3   # max text chunks
RETRIEVER_K_IMAGE  = 2   # max images
RETRIEVER_K_TABLE  = 2   # max tables
MAX_IMAGES_IN_RESP = int(os.getenv("MAX_IMAGES_IN_RESPONSE", 3))

# ── Unstructured PDF parsing ───────────────────────────────────────────────────
PDF_PARTITION_STRATEGY  = os.getenv("PDF_PARTITION_STRATEGY", "auto")  # "hi_res" | "fast" | "auto"
EXTRACT_IMAGES_FROM_PDF = True
INFER_TABLE_STRUCTURE   = True