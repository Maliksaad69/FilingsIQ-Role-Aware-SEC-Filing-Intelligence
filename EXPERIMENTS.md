# EXPERIMENTS.md

## Experiment Log Summary

| Experiment | Change Tested | Hypothesis | Metrics & Outcome | Decision |
|---|---|---|---|---|
| **1. Baseline Dense Retrieval** | Raw natural-language question with `bge-small-en-v1.5` dense vector search in Qdrant | Dense semantic similarity alone will retrieve relevant legal disclosures at top-10 | Target evidence missed at Top-10 (Page 25 retrieved only at Top-20; Page 26 missed at Top-30) | **Rejected** as standalone retrieval; recall too low for domain terminology |
| **2. SEC Query Expansion** | Multi-query expansion injecting formal SEC terminology (litigation, proceedings, legal claims) | Domain-specific term expansion bridges vocabulary gap between questions and filings | Target chunk Page 25 reached **Rank 1**; Page 26 reached **Rank 4** in Top-10 | **Kept** in retrieval pipeline |
| **3. Grounded Answer Generation** | Strict interpretation prompt separating case existence, individual materiality, and aggregate impact | Constrained prompt prevents hallucinating that absence of aggregate impact implies no material cases | Eliminated false negative claims; correctly preserved management disclosures and noted missing FY21 evidence | **Kept** in generation prompt |
| **4. Neural Reranking** | Cohere Rerank (`rerank-v5.0-pro`) applied to top-30 retrieved candidates against original question | Neural cross-encoder reranking places the highest-signal evidence chunks at top positions | Page 25 held Rank 1; Page 26 improved from Rank 4 to **Rank 2** | **Kept** in retrieval pipeline |

---

## Detailed Experiment Logs

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
The baseline dense retrieval struggled to identify the relevant legal proceedings disclosures using the original question. The wording "materially important ongoing legal battles" differed from the terminology used in the SEC filing, such as "litigation," "claims," "legal proceedings," and "investigations." Increasing retrieval depth improved recall slightly but did not retrieve both relevant evidence chunks within the top 10.

**Decision:**
The baseline was insufficient for reliable evidence retrieval. Query expansion using SEC-specific terminology was selected as the next experiment to improve retrieval recall.

---

### Experiment 2: SEC Query Expansion

**Hypothesis:**
Expanding user queries with terminology commonly found in SEC filings will improve retrieval of relevant evidence.

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
Query expansion improved retrieval for this test case. Retain it for further evaluation across the question set.

---

### Experiment 3: Grounded Answer Generation

**Hypothesis:**
Constrained prompting allows the language model to interpret SEC filing disclosures without making unsupported conclusions or false inferences.

**Evidence:**
`PEPSICO_2022_10K`, PDF pages 25–26.

**Initial failure:**
The model incorrectly inferred that no individual legal proceedings were material from management's statement about aggregate financial impact ("management believes the final outcome will not have a material adverse effect on the company's financial condition").

**Modification:**
Strengthened the prompt to distinguish:
- Existence of legal proceedings.
- Materiality of individual proceedings.
- Expected aggregate financial impact.

**Final results:**
- Correctly identified disclosed litigation.
- Preserved management's actual material adverse effect statement.
- Did not infer individual case materiality; explicitly stated individual materiality was not established.
- Correctly identified missing FY2021 evidence.
- Included document and PDF page citations.

**Decision:**
Accept the prompt improvement and retain this question as a regression test.

---

### Experiment 4: Cohere Reranking

**Objective:**
Test whether Cohere reranking improves the relevance ordering of candidates retrieved through BGE dense retrieval.

**Hypothesis:**
Cohere reranking will place the most relevant SEC filing evidence higher in the final results.

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

**Observation:**
Both relevant evidence chunks remained in the final top 10. The page 26 chunk improved from rank 4 to rank 2, while the page 25 chunk remained at rank 1.

**Decision:**
Retain Cohere reranking in the retrieval pipeline. It improves the ordering of relevant evidence without requiring changes to the existing embeddings or Qdrant collection.

---

## 3 Worst Failures & Root Cause Analysis

From the 27-question evaluation suite (`cranky_pearl.csv`), the following three cases represent the system's most significant failure modes:

### Failure 1: Corning Working Capital Calculation (`fb_00005`)
- **Question:** *"Does Corning have positive working capital based on FY2022 data? If working capital is not a useful or relevant metric for this company, then please state that and explain why."*
- **What Was Retrieved:** MD&A narrative chunks (`CORNING_2022_10K_p34_c0154`) ranked at #1 stating working capital was $2,278 million; the Consolidated Balance Sheet (`CORNING_2022_10K_p60`) was retrieved at rank #2.
- **Model Output:** Model answered that Corning had positive working capital of $2,278 million citing page 34, and stated the filing did not mention if working capital was useful.
- **Gold Answer:** Yes. Positive operating working capital of $831 million computed strictly from operating current assets and liabilities on the balance sheet (PDF page 60).
- **Root Cause:** The reranker and generation model prioritized the MD&A's stated total working capital figure over calculating operating working capital directly from the balance sheet table. Additionally, the LLM failed to answer the second qualitative sub-question regarding why working capital may or may not be useful for a manufacturing company.
- **What to Try Next:** (1) Implement statement-table precedence prompting: when balance sheet metrics are queried, balance sheet tables must supersede MD&A narrative numbers. (2) Decompose compound multi-part questions into discrete sub-questions so both quantitative and qualitative prompts are guaranteed to be addressed.

### Failure 2: Corning Days Payable Outstanding (`fb_10130`)
- **Question:** *"Based on the information provided primarily in the balance sheet and the statement of income, what is FY2020 days payable outstanding (DPO) for Corning? DPO is defined as: 365 * (average accounts payable between FY2019 and FY2020) / (FY2020 COGS + change in inventory between FY2019 and FY2020). Round your answer to two decimal places."*
- **What Was Retrieved:** Income Statement (`CORNING_2020_10K_p70`) and MD&A chunks (`p43`, `p126`). The balance sheet pages containing accounts payable and inventory across both FY2019 and FY2020 (`p70`, `p72`) were not in the top reranked chunks.
- **Model Output:** `STATUS: REFUSED`. Model correctly recognized it lacked accounts payable figures for FY2019/2020 and refused rather than hallucinating random numbers.
- **Gold Answer:** 63.86
- **Root Cause:** DPO requires four separate accounting line items across two consecutive fiscal years spanning two distinct financial statements (Balance Sheet and Income Statement). Dense semantic search prioritized the income statement and general cost discussions, missing the comparative balance sheet table chunks needed for average accounts payable.
- **What to Try Next:** Multi-statement formula co-retrieval. When query rewriting detects a working capital or turnover ratio requiring multiple statements (e.g., DPO, DSO, Cash Conversion Cycle), programmatically inject both the Consolidated Balance Sheets and Statements of Operations for the target fiscal years.

### Failure 3: 3M Capital-Intensive Ratio Assessment (`fb_00499`)
- **Question:** *"Is 3M a capital-intensive business based on FY2022 data?"*
- **What Was Retrieved:** General narrative and business segment chunks from `3M_2022_10K` (`p34`, `p37`, `p39`, `p54`) and `3M_2021_10K`. The primary financial statements (Balance Sheet p50, Cash Flow p52) were ranked outside the top 8.
- **Model Output:** `STATUS: REFUSED`. Model refused because it lacked total assets, revenue, and detailed capital expenditure figures required to assess capital intensity.
- **Gold Answer:** "No, the company is managing its CAPEX and Fixed Assets pretty efficiently, which is evident from below key metrics: CAPEX/Revenue Ratio: 5.1%, Fixed assets/Total Assets: 20%, Return on Assets= 12.4%."
- **Root Cause:** The query did not contain the explicit words "balance sheet" or "cash flows", so rule-based statement lookup did not trigger. Dense semantic similarity struggled to link the abstract conceptual phrase "capital-intensive" to the specific financial statement tables (Income Statement, Balance Sheet, Cash Flows) needed to compute capex/revenue and asset turnover ratios.
- **What to Try Next:** Expand financial ratio rule mappings in `extract_filters()` so terms like "capital-intensive", "capital intensity", "operating leverage", and "asset turnover" automatically trigger multi-statement pinning for the Consolidated Statement of Income, Balance Sheet, and Statement of Cash Flows.
