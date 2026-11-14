### Experiment 1: Baseline Dense Retrieval

**Hypothesis:**

The BGE embedding model combined with Qdrant dense vector similarity search can retrieve relevant SEC filing chunks using the original natural-language question without query expansion.

**Configuration:**

- Embedding model: `BAAI/bge-small-en-v1.5`
- Vector database: Qdrant Cloud
- Retrieval method: Dense vector similarity search
- Query: Original user question (no query expansion)
- Role filter: `consumer`
- Top-K values: 5, 10, 20, and 30

**Test Question:**

"Has PepsiCo reported any materially important ongoing legal battles from FY2022 and FY2021?"

**Expected Evidence:**

- `PEPSICO_2022_10K_p25_c0069` — PDF page 25
- `PEPSICO_2022_10K_p26_c0073` — PDF page 26

**Results:**

| Top-K | Page 25 Retrieved | Page 26 Retrieved |
|---|---|---|
| 5 | No | No |
| 10 | No | No |
| 20 | Yes | No |
| 30 | Yes | No |

Page 25 was retrieved at a rank between 11 and 20, while page 26 was not retrieved even within the top 30 results.

**Analysis:**

The baseline dense retrieval struggled to identify the relevant legal proceedings disclosures using the original question. The wording "materially important ongoing legal battles" differed from the terminology used in the SEC filing, such as "litigation," "claims," "legal proceedings," and "investigations."

Increasing the retrieval depth improved recall slightly but did not retrieve both relevant evidence chunks within the top 10.

**Decision:**

The baseline was insufficient for reliable evidence retrieval. Query expansion using SEC-specific terminology was selected as the next experiment to improve retrieval recall.

### Experiment 2: SEC Query Expansion

**Hypothesis:**
Expanding user queries with terminology commonly found in
SEC filings will improve retrieval of relevant evidence.

**Baseline:**
Original natural-language query with dense retrieval, Top-10.

**Change:**
Added SEC-related terms:
- Litigation
- Legal proceedings
- Legal claims
- Regulatory proceedings
- Lawsuits
- Investigations
- Inquiries
- Potential liabilities
- Material adverse effect

**Results:**
Original query:
- Page 25: Not retrieved in Top-10.
- Page 26: Not retrieved in Top-10.

Expanded query:
- Page 25: Rank 1.
- Page 26: Rank 4.

**Decision:**
Query expansion improved retrieval for this test case.
Retain it for further evaluation across the question set.


### Experiment 3: Grounded Answer Generation — Final

**Hypothesis:** Groq can interpret SEC filing evidence without making unsupported conclusions.

**Evidence:** PEPSICO_2022_10K, PDF pages 25–26.

**Initial failure:**
The model incorrectly inferred that no individual legal proceedings were material from management's statement about aggregate financial impact.

**Modification:**
Strengthened the prompt to distinguish:
- Existence of legal proceedings.
- Materiality of individual proceedings.
- Expected aggregate financial impact.

**Final results:**
- Correctly identified disclosed litigation.
- Preserved management's actual material adverse effect statement.
- Did not infer individual case materiality.
- Correctly identified missing FY2021 evidence.
- Included document and PDF page citations.

**Decision:** Accept the prompt improvement and retain this question as a regression test.

## Experiment 4: Cohere Reranking

**Objective:** Test whether Cohere reranking improves the relevance ordering of candidates retrieved through BGE dense retrieval.

**Hypothesis:** Cohere reranking will place the most relevant SEC filing evidence higher in the final results.

**Configuration:**
- Embedding model: `BAAI/bge-small-en-v1.5` (local)
- Vector database: Qdrant Cloud
- Initial retrieval: BGE dense retrieval
- Initial candidate count: 30
- Query: SEC-specific expanded question
- Reranker: Cohere `rerank-v5.0-pro`
- Final results: Top 10

**Results:**

| Evidence | Dense rank | Reranked position |
|---|---:|---:|
| `PEPSICO_2022_10K_p25_c0069` | 1 | 1 |
| `PEPSICO_2022_10K_p26_c0073` | 4 | 2 |

**Observation:** Both relevant evidence chunks remained in the final top 10. The page 26 chunk improved from rank 4 to rank 2, while the page 25 chunk remained at rank 1.

**Decision:** Retain Cohere reranking in the retrieval pipeline. It improves the ordering of relevant evidence without requiring changes to the existing embeddings or Qdrant collection.

**Limitation:** This is one question-level experiment. More questions across different companies, document types, and roles are required before concluding that reranking consistently improves retrieval performance.

