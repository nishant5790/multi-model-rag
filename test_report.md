# Multi-Modal RAG Test Report

**Date:** 2026-04-02  
**PDFs Ingested:** Annual-Report-Analysis.pdf (278 chunks), Eval-agents.md.pdf (381 chunks)  
**Total Chunks:** 659  
**Image Captioning:** Disabled (--no-image-captions)  
**Model:** gemini-2.5-flash | **Embeddings:** gemini-embedding-2-preview (768-dim) | **Retriever k:** 6

---

## Summary

| Metric | Value |
|---|---|
| Total Queries | 10 |
| Successful | 10 |
| Failed | 0 |
| Avg Latency | 4.95s |
| Source Types Retrieved | text, table, image |
| Queries with Image Sources | 3/10 |

---

## Detailed Results

### Query 1 — Text Retrieval (Annual Report)
**Q:** What are the key financial highlights from the annual report?  
**Latency:** 5.36s | **Sources:** 6 (5 text, 1 table) | **Has images:** Yes  
**Answer:** AUM grew 18% (3,817 -> 4,505), Total Revenue up 65% (486 -> 800), PAT up 148% (56 -> 139). Data correctly pulled from table on page 9.  
**Assessment:** GOOD — Retrieved financial table and cited specific numbers accurately.

---

### Query 2 — Text Retrieval (Annual Report)
**Q:** What are the main risks and challenges mentioned in the report?  
**Latency:** 4.88s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Identified risks including asset quality challenges in rural B2C segment, winding down of 2W/3W financing, higher delinquencies, and regulatory change impacts.  
**Assessment:** GOOD — Relevant sources from page 13 (Risks & Mitigation section). Minor issue: 1 source from Eval-agents doc (cross-contamination).

---

### Query 3 — Table Retrieval (Annual Report)
**Q:** Are there any tables showing revenue or financial data? What do they contain?  
**Latency:** 4.80s | **Sources:** 6 (3 text, 1 table, 1 image) | **Has images:** Yes  
**Answer:** Correctly identified table on page 9 with AUM, Revenue, and PAT data for FY24/FY25. Referenced image path for the table.  
**Assessment:** EXCELLENT — Multi-modal retrieval working well. Retrieved table content + image reference + supporting text.

---

### Query 4 — Image Retrieval (Annual Report)
**Q:** What charts or figures are included in the annual report and what do they show?  
**Latency:** 4.43s | **Sources:** 6 (2 text, 4 image) | **Has images:** Yes  
**Answer:** Found figures on pages 1 and 9. Acknowledged that no vision captions were available to describe them.  
**Assessment:** PARTIAL — Correctly retrieved image sources but couldn't describe content (expected, since captioning was disabled). Also pulled 2 images from the wrong PDF (Eval-agents).

---

### Query 5 — Text Retrieval (Annual Report)
**Q:** What is the company's strategy for future growth?  
**Latency:** 5.11s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Comprehensive answer covering LRS FY25-29 implementation, new product lines (leasing, green financing), customer share growth, and senior management changes.  
**Assessment:** EXCELLENT — All 6 sources from the correct PDF (pages 8-11). Rich, well-cited answer.

---

### Query 6 — Text Retrieval (Eval Agents)
**Q:** What is the main topic of the eval agents document?  
**Latency:** 6.38s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Evaluation of coding agents, focusing on impact of actively used context on real-world software engineering tasks.  
**Assessment:** GOOD — Correctly identified the document's focus on coding agent evaluation with AGENTBENCH.

---

### Query 7 — Text Retrieval (Eval Agents)
**Q:** How are agents evaluated in the document? What metrics or benchmarks are used?  
**Latency:** 6.22s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Described AGENTBENCH benchmark for real-world software engineering tasks, focusing on actively used context impact.  
**Assessment:** GOOD — Relevant retrieval from the eval agents document.

---

### Query 8 — Text Retrieval (Eval Agents)
**Q:** What are the different types of agents discussed?  
**Latency:** 4.10s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Identified coding agents: Claude Code, Codex, and Qwen Code.  
**Assessment:** GOOD — Correctly extracted agent names from the document.

---

### Query 9 — Cross-Document
**Q:** Compare the topics covered in both documents.  
**Latency:** 5.70s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** Distinguished Annual Report (financial analysis) from Eval Agents (coding agent evaluation). Brief comparison.  
**Assessment:** PARTIAL — Retrieved sources from both docs but comparison was shallow. Only 2 sources from each doc would be ideal; got a mixed bag.

---

### Query 10 — Out-of-Scope
**Q:** What is quantum computing?  
**Latency:** 2.51s | **Sources:** 6 (6 text) | **Has images:** No  
**Answer:** "The provided context does not contain information about what quantum computing is."  
**Assessment:** EXCELLENT — Correctly refused to answer with hallucinated content. Fastest response (2.51s).

---

## Observations & Analysis

### What's Working Well
1. **Text retrieval** is strong — relevant chunks are retrieved with proper page citations
2. **Table retrieval** works — HTML table content is embedded and retrievable with accurate data extraction
3. **Out-of-scope handling** is excellent — model correctly declines when context is insufficient
4. **Source citations** are consistent and properly formatted with [Source N] references
5. **Latency** is reasonable at ~5s average per query

### Issues Found
1. **Cross-document contamination** — Query 2 (about annual report risks) pulled 1 source from Eval-agents.md.pdf. Query 4 (about annual report charts) pulled 2 images from Eval-agents.
2. **Image content unknown** — With `--no-image-captions`, image chunks are just placeholders. The model can point to image files but can't describe what they contain.
3. **Cross-document comparison is weak** — Query 9 produced a shallow comparison because retrieval doesn't optimize for balanced multi-doc coverage.
4. **Some snippets are very short** — Several retrieved chunks are just headers ("Key Operating Highlights", "Strategy") with minimal content, suggesting chunking could be improved.

### Recommendations
1. **Enable image captioning** (`caption_images=True`) for richer image understanding
2. **Consider per-document filtering** — Allow queries to target specific PDFs to avoid cross-contamination
3. **Improve chunking** — Short header-only chunks dilute retrieval quality; consider merging small chunks or filtering by minimum content length
4. **Increase k for cross-doc queries** — Cross-document comparison would benefit from higher k or multi-query retrieval

---

## Raw Data
Full structured results with all source metadata saved to: `test_results.json`
