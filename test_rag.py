"""Test script for multi-modal RAG — runs diverse queries and saves results."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from multimodal_rag import MultiModalRAG

load_dotenv()

QUERIES = [
    # --- Annual Report queries ---
    {
        "id": 1,
        "query": "What are the key financial highlights from the annual report?",
        "category": "text-retrieval",
        "target_pdf": "Annual-Report-Analysis.pdf",
    },
    {
        "id": 2,
        "query": "What are the main risks and challenges mentioned in the report?",
        "category": "text-retrieval",
        "target_pdf": "Annual-Report-Analysis.pdf",
    },
    {
        "id": 3,
        "query": "Are there any tables showing revenue or financial data? What do they contain?",
        "category": "table-retrieval",
        "target_pdf": "Annual-Report-Analysis.pdf",
    },
    {
        "id": 4,
        "query": "What charts or figures are included in the annual report and what do they show?",
        "category": "image-retrieval",
        "target_pdf": "Annual-Report-Analysis.pdf",
    },
    {
        "id": 5,
        "query": "What is the company's strategy for future growth?",
        "category": "text-retrieval",
        "target_pdf": "Annual-Report-Analysis.pdf",
    },
    # --- Eval Agents queries ---
    {
        "id": 6,
        "query": "What is the main topic of the eval agents document?",
        "category": "text-retrieval",
        "target_pdf": "Eval-agents.md.pdf",
    },
    {
        "id": 7,
        "query": "How are agents evaluated in the document? What metrics or benchmarks are used?",
        "category": "text-retrieval",
        "target_pdf": "Eval-agents.md.pdf",
    },
    {
        "id": 8,
        "query": "What are the different types of agents discussed?",
        "category": "text-retrieval",
        "target_pdf": "Eval-agents.md.pdf",
    },
    # --- Cross-document / edge case queries ---
    {
        "id": 9,
        "query": "Compare the topics covered in both documents.",
        "category": "cross-document",
        "target_pdf": "both",
    },
    {
        "id": 10,
        "query": "What is quantum computing?",
        "category": "out-of-scope",
        "target_pdf": "none",
    },
]


def run_tests():
    rag = MultiModalRAG(persist_directory=".chroma")
    results = []
    print(f"Running {len(QUERIES)} test queries...\n")

    for q in QUERIES:
        print(f"[Query {q['id']}] {q['query']}")
        start = time.time()
        try:
            result = rag.query(q["query"])
            elapsed = round(time.time() - start, 2)
            source_types = [s["content_type"] for s in result["sources"]]
            source_pages = [s["page"] for s in result["sources"]]
            has_images = any(s.get("image_link") or s.get("image_path") for s in result["sources"])

            entry = {
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "target_pdf": q["target_pdf"],
                "answer": result["answer"],
                "sources": result["sources"],
                "source_types_retrieved": source_types,
                "source_pages": source_pages,
                "has_image_sources": has_images,
                "num_sources": len(result["sources"]),
                "latency_seconds": elapsed,
                "status": "success",
                "error": None,
            }
            print(f"  -> {elapsed}s | {len(result['sources'])} sources | types: {set(source_types)}")
            print(f"  -> Answer preview: {result['answer'][:150]}...\n")
        except Exception as e:
            elapsed = round(time.time() - start, 2)
            entry = {
                "id": q["id"],
                "query": q["query"],
                "category": q["category"],
                "target_pdf": q["target_pdf"],
                "answer": None,
                "sources": [],
                "source_types_retrieved": [],
                "source_pages": [],
                "has_image_sources": False,
                "num_sources": 0,
                "latency_seconds": elapsed,
                "status": "error",
                "error": str(e),
            }
            print(f"  -> ERROR ({elapsed}s): {e}\n")

        results.append(entry)

    return results


def build_report(results):
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful) if successful else 0
    all_source_types = set()
    for r in successful:
        all_source_types.update(r["source_types_retrieved"])

    report = {
        "test_run": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_queries": len(results),
            "successful": len(successful),
            "failed": len(failed),
            "avg_latency_seconds": round(avg_latency, 2),
            "source_types_seen": sorted(all_source_types),
            "image_sources_found": sum(1 for r in successful if r["has_image_sources"]),
        },
        "results": results,
    }
    return report


if __name__ == "__main__":
    results = run_tests()
    report = build_report(results)

    out_path = Path("test_results.json")
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n{'='*60}")
    print(f"Test Report Summary")
    print(f"{'='*60}")
    print(f"Total queries:        {report['test_run']['total_queries']}")
    print(f"Successful:           {report['test_run']['successful']}")
    print(f"Failed:               {report['test_run']['failed']}")
    print(f"Avg latency:          {report['test_run']['avg_latency_seconds']}s")
    print(f"Source types seen:    {report['test_run']['source_types_seen']}")
    print(f"Queries w/ images:    {report['test_run']['image_sources_found']}")
    print(f"\nFull results saved to: {out_path.resolve()}")
