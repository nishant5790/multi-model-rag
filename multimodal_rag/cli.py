"""CLI: ingest PDFs and ask questions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from multimodal_rag import MultiModalRAG


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    p = argparse.ArgumentParser(description="Multi-modal RAG (Gemini + Qdrant + unstructured PDFs)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Index a PDF into the vector store")
    ing.add_argument("pdf", type=Path, help="Path to PDF file")
    ing.add_argument(
        "--images-dir",
        type=Path,
        default=None,
        help="Directory for extracted images (default: <pdf_stem>_extracted_images next to PDF)",
    )
    ing.add_argument(
        "--no-image-captions",
        action="store_true",
        help="Skip Gemini vision calls per image (avoids quota limits; uses placeholder text for image chunks)",
    )

    ask = sub.add_parser("query", help="Ask a question against the indexed store")
    ask.add_argument("question", help="Natural language question")
    ask.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON (answer + sources with links)",
    )

    args = p.parse_args(argv)

    if args.cmd == "ingest":
        rag = MultiModalRAG()
        n = rag.ingest_pdf(
            args.pdf,
            image_output_dir=args.images_dir,
            caption_images=not args.no_image_captions,
        )
        print(f"Indexed {n} chunks from {args.pdf}")
        return 0

    if args.cmd == "query":
        rag = MultiModalRAG()
        out = rag.query(args.question)
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(out["answer"])
            print("\n--- Sources ---")
            for s in out["sources"]:
                print(
                    f"[{s['index']}] {s.get('content_type')} page={s.get('page')} "
                    f"link={s.get('doc_link')}"
                )
                if s.get("image_link"):
                    print(f"    image: {s['image_link']}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
