# 🔮 Multi-Modal RAG System

A production-grade Retrieval-Augmented Generation pipeline that handles **text, images, and tables** using:

| Component | Technology |
|-----------|------------|
| **Embedding** | `gemini-embedding-2-preview` (Google AI) |
| **Chat/Vision** | `gemini-2.5-flash` |
| **Framework** | LangChain + MultiVectorRetriever |
| **Vector Store** | Chroma (persisted to disk) |
| **Doc Parsing** | `unstructured` (hi-res PDF, DOCX, PPTX) |

---

## Architecture

```
Documents (PDF/DOCX/PPTX)
        │
        ▼
┌──────────────────────┐
│  DocumentProcessor   │  ← unstructured (hi_res)
│  - TextElement       │
│  - TableElement      │
│  - ImageElement      │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  MultiModalSummariser│  ← gemini-2.5-flash (vision)
│  - Image summaries   │
│  - Table summaries   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────────────────┐
│  MultiModalVectorStore                   │
│  ┌────────────────┐  ┌─────────────────┐ │
│  │ Chroma index   │  │ InMemoryByteStore│ │
│  │ (summaries)    │──│ (raw elements)   │ │
│  └────────────────┘  └─────────────────┘ │
└──────────┬───────────────────────────────┘
           │
           ▼ Query
┌──────────────────────┐
│  MultiModalRAGChain  │  ← gemini-2.5-flash
│  - Multi-modal prompt│
│  - Source citations  │
│  - Doc links         │
└──────────────────────┘
```

---

## Setup

### 1. System Dependencies

```bash
# macOS
brew install poppler tesseract

# Ubuntu/Debian
sudo apt-get install -y poppler-utils tesseract-ocr libtesseract-dev
```

### 2. Python Environment

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Environment Variables

```bash
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY
```

Get a free API key at: https://aistudio.google.com/app/apikey

---

## Usage

### Step 1 — Add your documents

```bash
mkdir docs
cp /path/to/your/*.pdf docs/
```

### Step 2 — Ingest (extract + embed)

```bash
# Ingest entire docs/ directory
python ingest.py

# Or ingest a single file
python ingest.py --file report.pdf

# Or specify a custom directory
python ingest.py --dir /path/to/documents
```

### Step 3 — Query

```bash
# Interactive mode (REPL)
python main.py --no-ingest

# Single query
python main.py --no-ingest --query "What does the revenue chart show?"

# Streaming mode
python main.py --no-ingest --stream

# Re-ingest + query
python main.py --docs ./docs --query "Summarise the key findings"

# Retrieve more context (default: 6)
python main.py --no-ingest --k 10 --query "Compare Q3 vs Q4 performance"
```

---

## Programmatic API

```python
from document_processor import DocumentProcessor
from image_handler import MultiModalSummariser
from vector_store import MultiModalVectorStore
from rag_chain import MultiModalRAGChain

# 1. Process documents
processor = DocumentProcessor()
texts, tables, images = processor.process("report.pdf")

# 2. Summarise images and tables
summariser = MultiModalSummariser()
images, tables = summariser.summarise_all(images, tables)

# 3. Build vector store
vs = MultiModalVectorStore()
vs.index_all(texts, tables, images)

# 4. Query
chain = MultiModalRAGChain(vs)
response = chain.invoke("What are the key findings?")

print(response.answer)

# Source citations with doc links
for source in response.sources:
    print(source)
    # → 📄 [TEXT] report.pdf — page 3 → file:///path/to/report.pdf#page=3
    # → 📊 [TABLE] report.pdf — page 7 → file:///path/to/report.pdf#page=7
    # → 🖼 [IMAGE] report.pdf — page 5 → file:///path/to/report.pdf#page=5

# Streaming
for chunk in chain.stream("Explain the diagram on page 4"):
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)
```

---

## Response Structure

```python
@dataclass
class RAGResponse:
    query:        str           # original user query
    answer:       str           # LLM-generated answer
    sources:      list[SourceRef]      # citations (type, name, page, link)
    text_context: list[str]            # text/table passages retrieved
    images_used:  list[ImageElement]   # images passed to LLM
```

Each `SourceRef` contains:
- `element_type` — `"text"` | `"table"` | `"image"`
- `source_name`  — filename (e.g. `"annual_report.pdf"`)
- `page_number`  — page within the document
- `doc_link`     — `file:///absolute/path/doc.pdf#page=N`

---

## Supported Document Formats

| Format | Text | Tables | Images |
|--------|------|--------|--------|
| PDF    | ✅   | ✅     | ✅     |
| DOCX   | ✅   | ✅     | ✅     |
| PPTX   | ✅   | ✅     | ✅     |
| HTML   | ✅   | ✅     | ❌     |
| Markdown | ✅ | ❌     | ❌     |
| TXT    | ✅   | ❌     | ❌     |

---

## Troubleshooting

**`poppler` not found**
```bash
brew install poppler          # macOS
sudo apt install poppler-utils # Linux
```

**`tesseract` not found**
```bash
brew install tesseract        # macOS
sudo apt install tesseract-ocr # Linux
```

**Embedding model name error**
Check the exact model string in Google AI Studio. Common values:
- `models/gemini-embedding-2-preview-05-20`
- `models/gemini-embedding-exp-03-07`
- `models/text-embedding-004` (stable fallback)

Update `EMBEDDING_MODEL` in your `.env` file.

**Rate limits**
Add `time.sleep(1)` between document chunks or use `unstructured`'s chunking parameters to reduce element count.
