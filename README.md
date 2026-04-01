# Multi-modal RAG

Retrieval-augmented generation over **PDFs** that contain **text**, **tables**, and **images**. Documents are chunked and indexed in **Chroma** with **Google Gemini** embeddings (`gemini-embedding-2-preview`) and answers are produced with **Gemini** chat (`gemini-2.5-flash`) via **LangChain**.

**Architecture deep-dive:** see [docs/architecture.md](docs/architecture.md).

---

## Features

- **Hi-res PDF ingestion** with [Unstructured](https://github.com/Unstructured-IO/unstructured) (`partition_pdf`, `strategy="hi_res"`).
- **Text** chunks with recursive character splitting.
- **Tables** as HTML/plain text chunks with page and source links.
- **Figures**: raster crops saved to disk; optional **vision captions** (one Gemini call per image) for better retrieval, or **placeholders** to save quota (`--no-image-captions`).
- **Citations**: answers are prompted to cite `[Source N]`; each source includes `doc_link` (PDF page) and `image_link` / `image_path` when relevant.

---

## Requirements

- **Python** ≥ 3.13  
- **[uv](https://docs.astral.sh/uv/)** for environments and installs  
- **Google AI API key** with access to the configured Gemini models  
- **System dependencies** for Unstructured PDF hi-res processing (e.g. Poppler; see Unstructured docs for your OS)

---

## Setup

```bash
cd multi-modal-RAG
uv sync
```

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_key_here
```

The integration also accepts `GEMINI_API_KEY` (see `langchain-google-genai`).

---

## CLI usage

The package exposes a console script **`multimodal-rag`** and the repo root **`main.py`** delegates to the same CLI.

### Ingest a PDF

```bash
uv run python main.py ingest path/to/document.pdf
# or
uv run multimodal-rag ingest path/to/document.pdf
```

| Option | Description |
|--------|-------------|
| `--chroma-dir` | Where Chroma persists data (default: `.chroma`) |
| `--images-dir` | Directory for extracted images (default: `<pdf_stem>_extracted_images` next to the PDF) |
| `--no-image-captions` | Do not call Gemini vision per image; use placeholder text for image chunks (helps with API quotas) |

### Query the index

```bash
uv run python main.py query "Your question here?"
uv run python main.py query "Your question?" --json
```

| Option | Description |
|--------|-------------|
| `--chroma-dir` | Same store as ingest (default: `.chroma`) |
| `--json` | Print full JSON: `answer` plus structured `sources` |

---

## Python API

```python
from multimodal_rag import MultiModalRAG

rag = MultiModalRAG(persist_directory=".chroma")
rag.ingest_pdf("report.pdf", caption_images=True)  # or False / CLI --no-image-captions
result = rag.query("What are the key risks mentioned?")
print(result["answer"])
print(result["sources"])
```

---

## Project layout

| Path | Role |
|------|------|
| `main.py` | Entry point; calls `multimodal_rag.cli:main` |
| `multimodal_rag/` | Application package |
| `docs/architecture.md` | Diagrams, data flow, metadata schema, design notes |
| `src/test.py` | Ad-hoc Unstructured experiment script (not part of the `multimodal_rag` package) |
| `pyproject.toml` | Dependencies, `multimodal-rag` script, Hatch build |

---

## Package reference: modules, classes, and functions

### `multimodal_rag/__init__.py`

- **`__version__`**: Package version string.  
- **Exports**: `MultiModalRAG` (main API class).

### `multimodal_rag/config.py`

Constants used across the app (no classes):

| Name | Purpose |
|------|---------|
| `EMBEDDING_MODEL` | Default: `gemini-embedding-2-preview` |
| `CHAT_MODEL` | Default: `gemini-2.5-flash` |
| `EMBEDDING_DIMENSIONALITY` | Default: `768` (passed to `GoogleGenerativeAIEmbeddings`) |

### `multimodal_rag/models.py`

Factories for LangChain + Gemini (no classes):

| Function | Returns | Description |
|----------|---------|-------------|
| `get_embeddings()` | `GoogleGenerativeAIEmbeddings` | Embedding model from `config` |
| `get_chat()` | `ChatGoogleGenerativeAI` | Chat model, `temperature=0` |

### `multimodal_rag/links.py`

Pure helpers for stable URIs (no classes):

| Function | Description |
|----------|-------------|
| `pdf_doc_link(source_path, page)` | Builds `file://…/file.pdf#page=N` when `page >= 1` |
| `image_file_link(image_path)` | Returns `file://` URI if the path exists; otherwise a string path |

### `multimodal_rag/store.py`

| Function | Description |
|----------|-------------|
| `get_vectorstore(persist_directory, collection_name=..., embeddings=...)` | Creates or opens a **persistent** `langchain_chroma.Chroma` instance; creates the directory if needed |

### `multimodal_rag/ingestion.py`

Ingestion pipeline from PDF to `langchain_core.documents.Document` list.

| Name | Type | Description |
|------|------|-------------|
| `_page_num` | function | Reads `element.metadata.page_number` |
| `_caption_image` | function | Sends image + prompt to Gemini chat (vision); returns caption string |
| `pdf_elements_to_documents` | function | Runs `partition_pdf`, maps `Table` / `Image` / text elements to documents with metadata; optional captioning |

**`pdf_elements_to_documents` parameters (important):**

- `image_output_dir`: where Unstructured writes extracted image files.  
- `chat`: required if `caption_images=True`; can be `None` if captions are disabled.  
- `caption_images`: if `False`, image chunks use placeholder text instead of vision API.  
- `chunk_size` / `chunk_overlap`: text splitter settings.

### `multimodal_rag/rag.py`

| Name | Type | Description |
|------|------|-------------|
| `_format_context` | function | Formats retrieved `Document`s into a single string with `[Source i]` headers and metadata lines |
| `RAG_PROMPT` | `ChatPromptTemplate` | System + human template: context + question, citation instructions |
| `MultiModalRAG` | class | Orchestrates vector store, retriever, and QA chain |

**`MultiModalRAG`**

| Method | Description |
|--------|-------------|
| `__init__(persist_directory, collection_name="multimodal_rag")` | Builds embeddings, chat, Chroma store, retriever (`k=6`), and `RAG_PROMPT \| chat \| StrOutputParser` |
| `ingest_pdf(pdf_path, image_output_dir=None, caption_images=True)` | Runs `pdf_elements_to_documents`, then `add_documents`; returns chunk count |
| `query(question)` | Retrieves top documents, formats context, runs QA; returns `{"answer": str, "sources": list[dict]}` |

### `multimodal_rag/cli.py`

| Name | Type | Description |
|------|------|-------------|
| `main` | function | `argparse` entry: subcommands `ingest` and `query`; loads `.env` via `load_dotenv()` |

Subcommands:

- **`ingest`**: constructs `MultiModalRAG`, calls `ingest_pdf` with CLI flags.  
- **`query`**: constructs `MultiModalRAG`, calls `query`, prints answer and sources (or JSON).

### `main.py` (repository root)

Thin launcher: `SystemExit(main())` from `multimodal_rag.cli`.

---

## Troubleshooting

- **429 / quota errors on ingest**: Many images each trigger a caption request. Use `--no-image-captions` or `caption_images=False`, or wait for quota reset / upgrade billing.  
- **Empty or poor retrieval**: Increase retriever `k`, improve captions, or add more representative text in image placeholders.  
- **Chroma location**: Delete or change `--chroma-dir` to rebuild the index from scratch.

---

## License / description

Project metadata lives in `pyproject.toml`. Update `description` there when you publish.
