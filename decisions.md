# DECISIONS.md

This document records the key technical and architectural decisions made during the development of the SEC Filings Grounded Knowledge Search system. It also captures the reasoning, trade-offs, and changes made based on experimental results.

## Decision 1: Use PyMuPDF4LLM for PDF Extraction

**Decision:** Use PyMuPDF4LLM to extract content from SEC filing PDFs.

**Reason:** The dataset contains lengthy financial documents with complex layouts, headings, and tables. PyMuPDF4LLM provides Markdown-oriented extraction, helping preserve document structure and tabular content for downstream processing.

**Trade-off:** Complex tables, multi-column layouts, and scanned pages may still require extraction validation or additional processing.

**Implementation:** Extracted content is organized by PDF page so that page-level metadata can be preserved during enrichment, chunking, retrieval, and citation.

## Decision 2: Preserve Original PDF Page Numbers

**Decision:** Store the physical PDF page number with every extracted page and chunk.

**Reason:** Every generated answer must cite the exact source document and PDF page. Page-level metadata makes evidence traceable to the original filing.

**Trade-off:** PDF page numbers may differ from the printed page numbers inside a filing. The system consistently uses physical PDF page numbers for citations.

## Decision 3: Use the Manifest as the Source of Document Metadata

**Decision:** Use `manifest.csv` to enrich extracted documents with metadata.

**Reason:** The manifest contains consistent information about each filing, including company, sector, desk, document type, and reporting period. This avoids manually assigning metadata to individual documents.

**Trade-off:** Each extracted JSON filename must match a valid `doc_id` in the manifest.

## Decision 4: Use Semantic Chunking

**Decision:** Use LangChain's `SemanticChunker` with the `BAAI/bge-small-en-v1.5` embedding model.

**Reason:** Fixed-size chunking can split related financial statements or legal disclosures at arbitrary positions. Semantic chunking attempts to preserve semantically related text together.

**Trade-off:** Semantic chunking requires embedding calculations during ingestion and produces chunks of varying sizes. Financial tables also require special attention to ensure their structure is preserved.

## Decision 5: Add Contextual Metadata to Chunks

**Decision:** Add company, filing type, reporting period, document ID, and PDF page information to each chunk before embedding.

**Reason:** Contextual information helps the embedding model understand the source and subject of each passage, particularly when chunks contain references without repeating the company or filing name.

**Trade-off:** Contextual headers increase the amount of text embedded and stored.

## Decision 6: Use Qdrant Cloud for Vector Storage

**Decision:** Use Qdrant Cloud instead of maintaining a local vector database.

**Reason:** Qdrant provides vector similarity search, metadata filtering, and persistent remote storage.

**Trade-off:** The implementation depends on network connectivity and Qdrant Cloud credentials. Payload indexes are required for the metadata fields used in filtering.

## Decision 7: Use Deterministic IDs for Ingestion

**Decision:** Generate deterministic UUIDs using the document ID, page number, and content hash. Delete existing points for a document before re-ingesting it.

**Reason:** Re-ingesting an updated filing should not leave duplicate or outdated chunks in the collection.

**Trade-off:** Re-ingestion replaces all chunks belonging to that document rather than updating only the changed chunks.

## Decision 8: Enforce Role-Based Access During Retrieval

**Decision:** Apply company-level access filters directly in Qdrant retrieval.

**Reason:** Different user roles should only retrieve filings belonging to their authorized companies. Enforcing access at retrieval prevents unauthorized evidence from reaching the answer-generation model.

**Implementation:** Four roles are supported: `consumer`, `health_industrials`, `tech_telecom`, and `lead`.

**Trade-off:** Role-to-company mappings must be maintained correctly as the corpus changes.

## Decision 9: Use Dense Vector Retrieval as the Baseline

**Decision:** Start with dense vector similarity search using BGE embeddings.

**Reason:** Dense retrieval provides semantic matching between natural-language questions and filing passages without requiring exact keyword overlap.

**Trade-off:** It may miss relevant passages when the question uses terminology different from the source document.

**Experiment:** Experiment 1 established the baseline using the original PepsiCo legal-proceedings question. The two target evidence chunks were not retrieved within the Top-10 results. Page 25 appeared only when retrieval depth increased to Top-20, while page 26 remained missing even at Top-30.

## Decision 10: Introduce SEC-Specific Query Expansion

**Decision:** Expand user questions with terminology commonly used in SEC filings.

**Reason:** The baseline query used the phrase "materially important ongoing legal battles," while the filing used terms such as litigation, claims, legal proceedings, inquiries, and investigations.

**Experiment:** Experiment 2 added SEC-specific terminology to the original question. The relevant page 25 and page 26 chunks appeared at ranks 1 and 4, respectively.

**Trade-off:** Expanded queries may introduce irrelevant terms and retrieve additional noise. Further evaluation across the development questions is required.

## Decision 11: Use Groq for Answer Generation

**Decision:** Use Groq-hosted LLM inference for generating answers from retrieved evidence.

**Reason:** Groq provides access to capable language models without requiring a locally hosted large language model.

**Trade-off:** The system depends on an external inference API and its availability, rate limits, and model behavior.

## Decision 12: Require Grounded Answers and Explicit Citations

**Decision:** Restrict answer generation to retrieved filing evidence and require document ID and PDF page citations for factual claims.

**Reason:** Financial and legal answers must be traceable to their original SEC disclosures. The model must not invent facts or rely on outside knowledge.

**Experiment:** Experiment 3 exposed an interpretation error in which the model treated management's statement about no material adverse financial effect as proof that no individual legal proceeding was material.

**Decision:** Strengthen the prompt to distinguish:
- The existence of legal proceedings.
- The materiality of individual proceedings.
- The expected aggregate financial impact.

The revised prompt instructs the model not to infer individual case materiality from a statement about aggregate financial impact.

## Decision 13: Support Partial Answers and Refusals

**Decision:** Support separate answer statuses for fully answered, partially answered, and unsupported questions.

**Reason:** A question may request information for multiple reporting periods, while retrieved evidence may cover only some of them. Missing evidence for one period must not be treated as evidence that no relevant event occurred.

**Implementation Goal:**
- `answered`: Sufficient evidence supports all requested parts.
- `partially_answered`: Evidence supports some requested parts, but others remain unverified.
- `refused`: Available evidence is insufficient to answer the question.

**Trade-off:** Status determination requires evidence coverage logic rather than simply checking whether retrieval returned any chunks.

## Decision 14: Strict CLI Interface Contract & Clean JSON Output

**Decision:** Implement the exact command-line interface contract (`ingest <corpus_dir>`, `search --role <role> --k <n> "<question>"`, and `ask --role <role> "<question>"`) with strict, unpolluted JSON stdout outputs and cross-platform executable wrapper scripts.

**Reason:** Automated grading scripts and evaluation harnesses parse stdout as pure JSON (`json.loads(stdout)`). Any diagnostic logs or library warnings printed to stdout break automated parsers.

**Implementation:**
- `search` outputs a JSON array: `[{"doc_id": "...", "page": 60, "chunk_id": "...", "score": 0.0, "text": "..."}]` where page is a 1-indexed physical PDF page number.
- `ask` outputs a single JSON object: `{"status": "answered" | "refused", "answer": "...", "citations": [{"doc_id": "...", "page": 60, "chunk_id": "..."}]}`. When refused, citations is an empty list (`[]`).
- Diagnostic outputs (filters, rewritten queries, progress) are redirected strictly to `stderr`.
- Executable wrappers (`search`, `ask`, `ingest` for POSIX bash, `.bat` for Windows CMD, and `.ps1` for PowerShell) ensure compatibility regardless of whether the grading harness executes commands directly or via `python main.py`.

---

## Decision Log Summary

The system follows a document-processing and retrieval pipeline:

1. Extract PDF content using PyMuPDF4LLM.
2. Preserve physical PDF page numbers.
3. Enrich extracted documents using the manifest.
4. Split documents using semantic chunking.
5. Add contextual metadata to chunks.
6. Generate embeddings using BGE.
7. Store vectors and metadata in Qdrant Cloud.
8. Enforce role-based access during retrieval.
9. Retrieve relevant evidence using dense search and query expansion.
10. Rerank candidate evidence using Cohere Rerank.
11. Generate grounded answers using Cohere Command models.
12. Return document and page citations with clean JSON output conforming to the evaluation interface contract.

**Documentation note:** These decisions describe the current implementation and the experiments performed so far. Decisions and implementation goals should be updated as the system evolves.