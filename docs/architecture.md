# Architecture

This document describes how the multi-modal RAG pipeline is structured, how data flows, and how components interact.

## High-level diagram

```mermaid
flowchart TB
    subgraph ingest [Ingestion]
        PDF[PDF file]
        UP[partition_pdf hi_res]
        EL[Ordered elements]
        CH[Chunks LangChain Document]
        PDF --> UP --> EL --> CH
    end

    subgraph vision [Optional vision]
        CAP[Gemini caption per Image]
        EL -.-> CAP
        CAP -.-> CH
    end

    subgraph graphBuild [Chunk graph]
        GR[Reading order edges]
        CH --> GR
        GR --> JSON[chunk_graph.json]
    end

    subgraph index [Indexing]
        EMB[Gemini embeddings]
        VS[Chroma persistent store]
        CH --> EMB --> VS
    end

    subgraph query [Query]
        Q[User question]
        RET[Retriever top_k similarity]
        EXP[Graph expand neighbors]
        CTX[Format context with sources]
        GEN[Gemini chat plus RAG prompt]
        A[Answer and source metadata]
        Q --> RET
        VS --> RET
        JSON --> EXP
        RET --> EXP --> CTX --> GEN --> A
    end
```

## Package and module map

```mermaid
flowchart LR
    CLI[multimodal_rag.cli]
    RAG[multimodal_rag.rag]
    ING[multimodal_rag.ingestion]
    ST[multimodal_rag.store]
    MOD[multimodal_rag.models]
    CFG[multimodal_rag.config]
    LNK[multimodal_rag.links]
    GPH[multimodal_rag.graph_store]

    CLI --> RAG
    RAG --> ING
    RAG --> ST
    RAG --> MOD
    RAG --> GPH
    MOD --> CFG
    ING --> LNK
    ING --> UNS[unstructured partition_pdf]
    ST --> CHR[langchain_chroma Chroma]
```

## Ingestion pipeline

```mermaid
flowchart TD
    START[ingest_pdf pdf_path]
    RESOLVE[Resolve pdf_path and image_output_dir]
    BUILD[pdf_elements_to_documents]
    LOC[build_local_graph_from_ordered_docs]
    MERGE[Merge graph into chunk_graph.json]
    ADD[vectorstore.add_documents with ids]
    N[Return chunk count]

    START --> RESOLVE --> BUILD --> LOC --> MERGE --> ADD --> N
```

## Element routing in `pdf_elements_to_documents`

```mermaid
flowchart TD
    LOOP[For each element_index el from partition_pdf]
    PAGE[page_number and doc_link]
    TBL{Table?}
    IMG{Image?}
    TBODY[body from text_as_html or text]
    TEMPTY{body empty?}
    TDOC[One Document table metadata chunk_id]
    IPATH{image_path valid file?}
    CAP{caption_images and chat?}
    VCAP[Gemini vision caption]
    PLH[Placeholder text]
    IDOC[One Document image metadata chunk_id]
    TXT[strip el.text]
    TXEMPTY{text empty?}
    SPLIT[RecursiveCharacterTextSplitter]
    TXDOC[One Document per split chunk_id]

    LOOP --> PAGE --> TBL
    TBL -->|yes| TBODY --> TEMPTY
    TEMPTY -->|yes| LOOP
    TEMPTY -->|no| TDOC --> LOOP
    TBL -->|no| IMG
    IMG -->|yes| IPATH
    IPATH -->|no| LOOP
    IPATH -->|yes| CAP
    CAP -->|yes| VCAP --> IDOC
    CAP -->|no| PLH --> IDOC
    IDOC --> LOOP
    IMG -->|no| TXT --> TXEMPTY
    TXEMPTY -->|yes| LOOP
    TXEMPTY -->|no| SPLIT --> TXDOC --> LOOP
```

## Image caption path

```mermaid
sequenceDiagram
    participant ING as ingestion
    participant FS as local image file
    participant LC as LangChain HumanMessage
    participant GEM as ChatGoogleGenerativeAI

    ING->>FS: read bytes base64
    ING->>LC: text prompt plus image_url data URI
    ING->>GEM: invoke messages
    GEM-->>ING: caption string
    ING->>ING: page_content IMAGE plus caption
```

## Indexing and embedding

```mermaid
flowchart LR
    DOC[Document page_content]
    META[Document metadata]
    EF[GoogleGenerativeAIEmbeddings]
    CHW[Chroma add_documents ids]
    VEC[Vector row keyed by chunk_id]

    DOC --> EF
    META --> CHW
    EF --> CHW
    CHW --> VEC
```

## Chunk graph on disk

- **Path**: `{persist_directory}/chunk_graph.json` (same directory as the Chroma persist folder).
- **Contents**: undirected `neighbors` map (`chunk_id` → list of neighbor ids) and `edge_types` for pairs (`same_element_next` vs `adjacent_element`).
- **Semantics**: consecutive chunks in ingest order are linked. Same `element_index` ⇒ `same_element_next`; different elements ⇒ `adjacent_element` (connects narrative text to adjacent tables or images in reading order).

## Query path: retrieval plus graph expansion

```mermaid
flowchart TD
    Q[query question]
    L[Reload ChunkGraph from disk]
    R[retriever.invoke question]
    S[Collect chunk_id from hits]
    E[expand_chunk_ids BFS max_hops expand_k]
    G[Chroma get by expanded ids]
    M[Concat retrieved then expanded]
    F[_format_context]
    P[RAG prompt pipe chat parser]
    SRC[Sources list graph_expanded flag]
    OUT[Return answer and sources]

    Q --> L --> R --> S --> E --> G --> M --> F --> P --> SRC --> OUT
```

## Sequence: query path

```mermaid
sequenceDiagram
    participant User
    participant R as MultiModalRAG
    participant Ret as Chroma retriever
    participant G as chunk_graph.json
    participant CH as Chroma get
    participant F as _format_context
    participant QA as RAG prompt Chat Parser

    User->>R: query(question)
    R->>G: load graph
    R->>Ret: invoke(question)
    Ret-->>R: top_k Documents
    R->>R: expand neighbor chunk ids
    R->>CH: get(ids expanded)
    CH-->>R: extra Documents
    R->>F: format all docs
    F-->>R: context string
    R->>QA: invoke context question
    QA-->>R: answer
    R-->>User: answer plus sources
```

## Context assembly

```mermaid
flowchart LR
    D[Each Document]
    H[Header Source N type page doc_link chunk_id image fields]
    B[Body source path plus page_content]
    J[join with separator]

    D --> H --> B --> J
```

## Design choices

### Single vector collection

All chunk types (text, table, image-derived text) live in **one Chroma collection**. Initial hits come from **embedding similarity**; **graph expansion** then pulls in adjacent chunks (same element splits and neighbors in document order) so figures and nearby narrative tend to appear together in context.

### How images participate in search

Image pixels are **not** embedded directly. Each extracted image is either:

1. **Captioned** with Gemini 2.5 Flash (vision): the caption text is embedded and stored, or  
2. **Placeholder text** only (`caption_images=False` / CLI `--no-image-captions`): a short deterministic description is embedded instead.

Option 2 avoids one `generate_content` call per image, which matters on **strict API quotas** (for example, free-tier daily limits on chat requests).

### Document “links”

The system does not host documents over HTTP. “Links” are **local file URIs** so users can open the PDF at a page or open an extracted image in a viewer:

| Metadata field | Meaning |
|----------------|---------|
| `doc_link` | `file:///…/document.pdf#page=N` |
| `image_link` | `file:///…/extracted.png` (when the file exists) |
| `source` | Absolute path to the source PDF |

### Table handling

`Table` elements from Unstructured prefer `metadata.text_as_html`, falling back to plain `text`. Tables are stored as **one chunk per table element** (no recursive splitting in the current implementation).

### Text handling

Non-table, non-image elements with text are split with `RecursiveCharacterTextSplitter` (default chunk size 1200, overlap 200).

## Metadata schema (per `Document`)

Values are chosen to be **Chroma-friendly** (strings, ints; `-1` used when page is unknown).

| Key | Typical values |
|-----|----------------|
| `chunk_id` | UUID string; Chroma row id and graph node id |
| `element_index` | Index of the source element in `partition_pdf` order |
| `split_index` | 0-based index within splits of that element (text); 0 for single-chunk table/image |
| `split_count` | Number of chunks emitted for that element |
| `source` | Absolute path to PDF |
| `page` | 1-based page number, or `-1` |
| `content_type` | `"text"` \| `"table"` \| `"image"` |
| `doc_link` | PDF file URI with `#page=` fragment |
| `image_path` | Filesystem path to crop (images and sometimes tables) |
| `image_link` | File URI to crop, when file exists |

## Extension points

- **Models**: change names or dimensions in `multimodal_rag/config.py`; wiring in `models.py`.
- **Chunking**: `pdf_elements_to_documents` parameters `chunk_size` / `chunk_overlap`.
- **Retrieval**: `MultiModalRAG(..., retrieve_k=..., graph_expand_k=..., max_graph_hops=..., use_graph_expand=...)` in `rag.py`.
- **Graph store**: `multimodal_rag/graph_store.py` (`ChunkGraph`, edge type constants).
- **Prompting**: `RAG_PROMPT` in `rag.py`.

## Dependencies (conceptual)

| Layer | Technology |
|-------|------------|
| PDF parsing | `unstructured` (hi-res strategy; layout + optional table/image extraction) |
| Embeddings | `langchain-google-genai` → Gemini Embedding API |
| Vector store | `langchain-chroma` → ChromaDB on disk |
| Chat | `langchain-google-genai` → Gemini generateContent |
| CLI / env | `python-dotenv`, `argparse` |

## Limitations

- **Quotas**: Gemini API limits apply separately to embedding calls, chat completion, and (if enabled) per-image captioning.
- **PDFs only** in the ingestion path as implemented; other formats would need new partitioners.
- **No deduplication** across repeated ingest runs: ingesting the same PDF twice adds duplicate vectors and duplicate graph edges unless the collection and `chunk_graph.json` are cleared or you use a fresh persist directory.
- **Re-ingest**: Prefer clearing the Chroma directory and removing `chunk_graph.json` before re-indexing the same corpus to avoid stale edges pointing at removed ids.
