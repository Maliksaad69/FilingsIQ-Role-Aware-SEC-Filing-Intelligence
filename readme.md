# FilingsIQ — Role-Aware SEC Filing Intelligence

**An evidence-grounded Retrieval-Augmented Generation (RAG) system for searching and analyzing SEC filings with role-based access control.**

FilingsIQ is an AI-powered financial document intelligence system that enables users to retrieve relevant information and generate evidence-backed answers from SEC filings. It combines local semantic embeddings, vector search, neural reranking, and LLM-based answer generation to provide accurate, traceable responses grounded in original filing documents.

The system is designed to handle long financial documents, preserve page-level references, enforce role-based document access, and reduce unsupported AI-generated answers.

## Key Features

- **SEC Document Processing:** Extracts text and tables from PDF filings using PyMuPDF4LLM.
- **Semantic Chunking:** Splits extracted content into meaningful chunks while preserving document metadata and PDF page numbers.
- **Local Embeddings:** Uses `BAAI/bge-small-en-v1.5` for semantic vector embeddings.
- **Vector Search:** Stores and retrieves document chunks using Qdrant Cloud.
- **Query Expansion:** Expands financial and legal questions with relevant SEC terminology to improve retrieval.
- **Neural Reranking:** Uses Cohere Rerank to reorder retrieved evidence based on query relevance.
- **Grounded Answer Generation:** Uses Cohere Command models to generate answers from retrieved filing evidence.
- **Role-Based Access Control:** Restricts document retrieval according to user roles.
- **Evidence Citations:** Provides document IDs, PDF page numbers, and chunk references for supporting evidence.
- **Hallucination Control:** Supports insufficient-evidence refusals and distinguishes disclosed facts from management assessments.

## Dataset

The project uses a corpus of SEC filings from 12 publicly listed US companies.

| Dataset attribute | Description                        |
| ----------------- | ---------------------------------- |
| Companies         | 12 US-listed companies             |
| Documents         | 119 SEC filings                    |
| Reporting period  | 2015–2024                          |
| Document formats  | PDF                                |
| Document types    | 10-K, 10-Q, 8-K, earnings releases |

### Role-Based Document Access

Documents are grouped into four access roles:

| Role                 | Authorized companies                        |
| -------------------- | ------------------------------------------- |
| `consumer`           | PepsiCo, Best Buy, MGM Resorts, Ulta Beauty |
| `health_industrials` | Johnson & Johnson, Pfizer, 3M, Boeing       |
| `tech_telecom`       | AMD, Adobe, Corning, Verizon                |
| `lead`               | All 12 companies                            |

Access restrictions are applied during vector retrieval rather than relying solely on the language model's instructions.

## Architecture

```text
                 SEC Filing PDFs
                        |
                        v
                PDF Text Extraction
                   PyMuPDF4LLM
                        |
                        v
                 Metadata Enrichment
                        |
                        v
                  Semantic Chunking
                        |
                        v
              Local BGE Embeddings
                        |
                        v
                  Qdrant Cloud
                        |
                 User Question
                        |
                        v
                 Query Expansion
                        |
                        v
             Role-Filtered Retrieval
                        |
                        v
               Dense Search (Top 30)
                        |
                        v
              Cohere Reranking
                        |
                        v
                Relevant Evidence
                        |
                        v
              Cohere Answer Generation
                        |
                        v
             Grounded Answer + Citations
```

## Technology Stack

| Component            | Technology                |
| -------------------- | ------------------------- |
| Programming language | Python                    |
| PDF extraction       | PyMuPDF4LLM               |
| Document chunking    | LangChain SemanticChunker |
| Embeddings           | BAAI/bge-small-en-v1.5    |
| Vector database      | Qdrant Cloud              |
| Reranking            | Cohere Rerank             |
| Answer generation    | Cohere Command            |
| Configuration        | Python dotenv             |

## Project Structure

```text
filings-iq/
│
├── pdfs/                   # Source SEC filing PDFs
├── extracted/              # Extracted document content
├── data/
│   ├── enriched/            # Documents with metadata
│   └── chunks/              # Processed document chunks
├── manifest.csv             # Document metadata
├── dev_questions.jsonl      # Development evaluation questions
├── DECISIONS.md             # Architecture and design decisions
├── EXPERIMENTS.md           # Retrieval and generation experiments
├── AI_NOTES.md              # AI usage and lessons learned
├── requirements.txt
├── .env.example
└── README.md
```

## Configuration

Create a `.env` file in the project root:

```env
# Qdrant Cloud
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_api_key

# Cohere
COHERE_API_KEY=your_cohere_api_key
COHERE_RERANK_MODEL=rerank-v5.0-pro
COHERE_GENERATION_MODEL=command-a-03-2025
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Keep API keys in environment variables. Never commit `.env` or private API credentials.

## Usage

The intended command-line interface supports two operations:

**Search for relevant filing evidence:**

```bash
python main.py search --role consumer --k 10 "What legal proceedings has PepsiCo disclosed?"
```

**Generate an evidence-grounded answer:**

```bash
python main.py ask --role consumer "What legal proceedings has PepsiCo disclosed?"
```

The search operation returns document IDs, page numbers, chunk IDs, relevance scores, and evidence text.

The ask operation returns an answer with supporting citations or an insufficient-evidence response when the retrieved documents do not support an answer.

## Evaluation and Experiments

The system is evaluated through controlled experiments that compare retrieval and answer-generation approaches.

Experiments include:

- Baseline dense retrieval.
- SEC-specific query expansion.
- Grounded answer generation.
- Cohere neural reranking.
- Retrieval and answer-quality comparisons.

Evaluation focuses on retrieval relevance, evidence coverage, citation accuracy, answer correctness, and unsupported claims.

Experiment results and architectural decisions are documented in:

- `EXPERIMENTS.md`
- `DECISIONS.md`
- `AI_NOTES.md`

## Design Principles

1. Answers must be grounded in retrieved SEC filing evidence.
2. Every factual claim should have a traceable document and page citation.
3. Access control must be enforced before evidence reaches the language model.
4. Financial disclosures must be interpreted carefully without overstating their implications.
5. Missing evidence must result in an explicit limitation or refusal rather than an invented answer.
6. Re-ingesting an updated filing should not create duplicate vector records.

## Important Note

The source SEC filings are not included in this repository. Users must obtain the documents separately and configure their own Qdrant and Cohere credentials.

---

**FilingsIQ** — Turning SEC filings into searchable, traceable financial intelligence.
