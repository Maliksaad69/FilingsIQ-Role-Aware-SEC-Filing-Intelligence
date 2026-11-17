# FilingsIQ — Role-Aware SEC Filing Intelligence

**An evidence-grounded Retrieval-Augmented Generation (RAG) system for searching and analyzing SEC filings with role-based access control.**

FilingsIQ is a financial document intelligence system built for equity research analysts. It enables analysts to query across 119 SEC filings (10-K, 10-Q, 8-K, and earnings releases) covering 12 public US companies from 2015 to 2024. The system combines local dense semantic embeddings, Qdrant Cloud vector search, multi-query expansion, statement-level deterministic pinning, Cohere neural reranking, and Cohere Command generation to deliver traceable, audited answers grounded strictly in primary filing disclosures.

---

## Architecture

```text
                           SEC Filing PDFs
                                  │
                                  ▼
                        PyMuPDF4LLM Extraction
                     (1-indexed physical PDF pages)
                                  │
                                  ▼
                         Metadata Enrichment
                     (manifest.csv desk & sector)
                                  │
                                  ▼
                       Chunking & Context Headers
                   (Recursive splitter + Doc/Page meta)
                                  │
                                  ▼
                     BAAI/bge-small-en-v1.5 (Local)
                                  │
                                  ▼
                         Qdrant Cloud Vectors
                       (Payload index on company)
                                  │
  User Query ───────► Query Rewriter (Cohere Chat)
                          │   (Multi-query expansion)
                          ▼
                 Role-Based Filter (RBAC)
            (Strict desk whitelist applied at Qdrant)
                          │
                          ▼
             Dense Retrieval + Statement Pinning
             (Top 30 candidates + financial tables)
                          │
                          ▼
               Cohere Rerank (rerank-v5.0-pro)
             (Top 8 high-signal evidence blocks)
                          │
                          ▼
              Cohere Generation (command-a-03-2025)
          (Temperature 0, evidence-only, refusal protocol)
                          │
                          ▼
            JSON Response: Status, Answer & Page Citations
```

### Retrieval & Generation Highlights
1. **Contextual Embedding Headers:** Each chunk is prefixed with document ID, company name, filing type, reporting period, and physical PDF page before embedding. This prevents semantic drift when financial tables lack explicit inline headings.
2. **Pre-Retrieval RBAC Filtering:** Access control is enforced directly inside Qdrant search filters, preventing unauthorized desk data from entering context.
3. **Multi-Query Expansion:** Decomposes natural queries into formal SEC terminology (e.g., translating "legal battles" into "Item 3. Legal Proceedings", "litigation", "commitments and contingencies").
4. **Financial Statement Pinning:** A regex and full-text index detects queries requiring balance sheets, income statements, or cash flows, directly pinning candidate financial statement pages so tabular data is never lost to prose in reranking.
5. **Strict Grounding & Refusal Protocols:** Constrained prompting forces the model to refuse (`STATUS: REFUSED`) whenever required financial periods or ratio inputs are absent, eliminating hallucinations.

---

## Role-Based Access Control (RBAC)

Filings are partitioned into four operational desks. Access restrictions are enforced at vector retrieval:

| Role / Desk | Authorized Companies | Scope |
|---|---|---|
| `consumer` | PepsiCo, Best Buy, MGM Resorts, Ulta Beauty | Consumer products & retail |
| `health_industrials` | Johnson & Johnson, Pfizer, 3M, Boeing | Healthcare, defense & industrials |
| `tech_telecom` | AMD, Adobe, Corning, Verizon | Semiconductors, software & telecom |
| `lead` | All 12 public companies | Full corpus visibility |

Enforcing RBAC in Qdrant payload filters ensures unauthorized evidence is never retrieved, guaranteeing zero information leakage at zero token overhead.

---

## Evaluation Results

The system was evaluated against 27 diverse questions sourced from the FinanceBench benchmark and custom financial analysis queries across all 12 companies and desks. Retrieval recall was evaluated deterministically against ground-truth document IDs and page numbers, while answer quality was graded using a Ragas LLM-as-a-judge framework (`openai/gpt-oss-20b` via Groq) with discrete pass/fail criteria.

### Benchmark Scorecard (27 Questions)

| Evaluation Metric | Score (%) | Passed / Total | Description |
|---|---:|:---:|---|
| **Gold Document Retrieved** | **96.3%** | **26 / 27** | Top candidate evidence contains the ground-truth SEC filing |
| **Answer Relevancy** | **92.6%** | **25 / 27** | Answer directly addresses the prompt without extraneous noise |
| **Gold Document Cited** | **85.2%** | **23 / 27** | Answer citations reference the correct primary SEC filing |
| **Answer Correctness** | **77.8%** | **21 / 27** | Factual alignment and numerical accuracy against gold answer |
| **Answer Completeness** | **77.8%** | **21 / 27** | Fully covers all requested periods, line items, and qualitative conditions |
| **Citation Correctness** | **48.1%** | **13 / 27** | Strict alignment with gold document and exact physical page references |

### Performance by Desk

| Role / Desk | Samples | Gold Doc Retr. | Gold Doc Cited | Correctness | Relevancy | Completeness | Citation Corr. |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `consumer` | 7 | 100.0% (7/7) | 85.7% (6/7) | 85.7% (6/7) | 100.0% (7/7) | 100.0% (7/7) | 57.1% (4/7) |
| `health_industrials` | 12 | 91.7% (11/12) | 83.3% (10/12) | 83.3% (10/12) | 91.7% (11/12) | 83.3% (10/12) | 41.7% (5/12) |
| `tech_telecom` | 8 | 100.0% (8/8) | 87.5% (7/8) | 62.5% (5/8) | 87.5% (7/8) | 50.0% (4/8) | 50.0% (4/8) |
| **Total Corpus** | **27** | **96.3% (26/27)** | **85.2% (23/27)** | **77.8% (21/27)** | **92.6% (25/27)** | **77.8% (21/27)** | **48.1% (13/27)** |

### Evaluation Analysis
- **High Retrieval Recall (96.3%):** Query rewriting combined with contextual metadata headers achieved 96.3% gold document retrieval across all 119 filings.
- **High Relevancy (92.6%):** Cross-encoder neural reranking (`rerank-v5.0-pro`) reliably surfaces high-signal chunks to the top 8 positions.
- **Citation Correctness Nuance (48.1%):**
  1. *Physical PDF vs Printed Page Offset:* FinanceBench gold answers frequently cite printed report page numbers, whereas FilingsIQ consistently cites physical 1-indexed PDF pages (typically an offset of 1–3 pages due to covers and tables of contents).
  2. *Multi-Year Recurring Statements:* For broad questions lacking a specific fiscal year (e.g. Boeing government services market share), the system cited all valid disclosures spanning 2017–2022 filings, which the single-document gold benchmark penalized.
  3. *Appropriate Refusals:* When complex multi-statement metrics were missing required inputs, the system returned `STATUS: REFUSED` rather than guessing, which appropriately withheld citations.

---

## How to Run

### 1. Environment Setup

Clone repository and install dependencies:

```bash
git clone https://github.com/Maliksaad69/FilingsIQ-Role-Aware-SEC-Filing-Intelligence.git
cd filings-iq
pip install -r requirements.txt
```

Create `.env` using [.env.example](file:///D:/AI%20Engineer%20Assignment/Assignment/.env.example):

```env
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key
COHERE_API_KEY=your_cohere_api_key
COHERE_RERANK_MODEL=rerank-v5.0-pro
COHERE_GENERATION_MODEL=command-a-03-2025
GROQ_API_KEY=your_groq_api_key
```

### 2. Interface Contract (CLI)

FilingsIQ strictly implements the command-line interface contract specified for evaluation. Commands can be invoked directly as standalone executables (`search`, `ask`, `ingest`) or via `python main.py <command>` across Windows, Linux, and macOS.

#### Supported Roles
* `consumer`: PepsiCo, Best Buy, MGM Resorts, Ulta Beauty
* `health_industrials`: Johnson & Johnson, Pfizer, 3M, Boeing
* `tech_telecom`: AMD, Adobe, Corning, Verizon
* `lead`: All 12 public companies (full corpus access)

---

#### `search --role <role> --k <n> "<question>"`
Retrieves the top `<n>` candidate evidence chunks ordered by relevance score.

```bash
# Invocation (either form supported)
search --role consumer --k 10 "What legal proceedings has PepsiCo disclosed?"
# OR
python main.py search --role consumer --k 10 "What legal proceedings has PepsiCo disclosed?"
```

**Output:** Prints a JSON list, best match first. Physical PDF `page` is counted from 1:

```json
[
  {
    "doc_id": "3M_2018_10K",
    "page": 60,
    "chunk_id": "3M_2018_10K_p60_c0268",
    "score": 0.8421,
    "text": "Item 3. Legal Proceedings..."
  }
]
```

---

#### `ask --role <role> "<question>"`
Generates an evidence-grounded answer with document and page citations, or returns `status: "refused"` when filings do not contain sufficient evidence.

```bash
# Invocation (either form supported)
ask --role consumer "What legal proceedings has PepsiCo disclosed?"
# OR
python main.py ask --role consumer "What legal proceedings has PepsiCo disclosed?"
```

**Output:** Prints one JSON object:

*When Answered:*
```json
{
  "status": "answered",
  "answer": "Pepsico has not reported any individual legal battles as materially important for FY2022 or FY2021. However, the company acknowledges being party to various litigation, claims, and regulatory proceedings related to advertising, product labels, intellectual property, employment, and other matters [PEPSICO_2021_10K, PDF page 25; PEPSICO_2022_10K, PDF page 25]. Management believes the final outcome will not have a material adverse effect on financial condition [PEPSICO_2021_10K, PDF page 26; PEPSICO_2022_10K, PDF page 25].",
  "citations": [
    {
      "doc_id": "PEPSICO_2022_10K",
      "page": 25,
      "chunk_id": "PEPSICO_2022_10K_p25_c0069"
    },
    {
      "doc_id": "PEPSICO_2022_10K",
      "page": 26,
      "chunk_id": "PEPSICO_2022_10K_p26_c0073"
    }
  ]
}
```

*When Refused (Insufficient Evidence):*
```json
{
  "status": "refused",
  "answer": "The available filing evidence is insufficient to answer this question. REASON: The evidence does not provide the accounts payable and inventory figures for FY2019 and FY2020 necessary to calculate Days Payable Outstanding.",
  "citations": []
}
```

---

#### `ingest <corpus_dir>`
Parses every PDF in `<corpus_dir>` with PyMuPDF4LLM, preserves physical page numbers, enriches chunks with metadata from `manifest.csv`, chunks text with contextual headers, and embeds into Qdrant collection `sec_filings`. Re-ingesting a changed filing replaces existing points by document ID without creating duplicates.

```bash
# Invocation (either form supported)
ingest ./data/pdfs
# OR
python main.py ingest ./data/pdfs
```

### 3. Running Batch Evaluations
To run batch inference and evaluation scoring:
```bash
# Generate batch RAG responses over test questions
python eval/run_batch.py

# Run Ragas LLM-as-a-judge evaluation suite
python eval/rag_eval/evals.py
```

---

## Known Limitations

1. **Multi-Statement Accounting Formulas:** Financial metrics requiring simultaneous line items across multiple statements and comparative fiscal years (e.g., Days Payable Outstanding requiring balance sheet Accounts Payable and Inventory alongside income statement COGS across two years) can fail if all comparative statement pages are not co-retrieved.
2. **Fiscal Year Calendar Mismatches:** Retailers with non-calendar fiscal years (e.g., Ulta Beauty's fiscal year ending in January 2023 labeled as fiscal 2022) require explicit period disambiguation.
3. **Filing Cover Page Disclosures:** Exchange-listed debt security classes and ticker tables located on page 1 of Form 10-K/10-Q are occasionally deprioritized by semantic search against narrative debt footnote queries.

---

## Project Structure

```text
├── main.py                     # Primary CLI interface (search, ask, ingest)
├── manifest.csv                # Corpus metadata (company, sector, desk, filing type)
├── DECISIONS.md                # Technical & architectural decision log
├── EXPERIMENTS.md              # Retrieval experiments & 3 worst failures analysis
├── AI_NOTES.md                 # AI usage reflections and mitigation log
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment configuration
├── search, ask, ingest         # Executable wrapper scripts (POSIX bash & Windows cmd/ps1)
├── Enrichment/
│   └── join.py                 # Document metadata enrichment pipeline
├── Notebooks/
│   ├── Extraction.ipynb        # PyMuPDF4LLM extraction notebook
│   ├── Injestion and testing.ipynb # Chunking & Qdrant vector indexing
│   └── Testing_with_Single_Document_organized.ipynb # Prototype validation notebook
├── data/
│   ├── chunks/                 # 119 Processed chunks with contextual headers
│   └── enriched/               # 119 Metadata-enriched document JSONs
└── eval/
    ├── cohere_sec_rag.py       # Core RAG retrieval and generation engine
    ├── run_batch.py            # Batch question execution script
    ├── questions.jsonl         # Benchmark evaluation dataset (30 questions)
    ├── results.csv             # Batch generation outputs with page citations
    └── rag_eval/
        ├── evals.py            # Ragas LLM-as-a-judge evaluation script
        ├── results.jsonl       # Ragas evaluation inputs
        └── evals/experiments/  # Experiment results (cranky_pearl.csv - 27 evaluated)
```
