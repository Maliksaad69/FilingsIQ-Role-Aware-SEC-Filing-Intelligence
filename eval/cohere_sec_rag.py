"""
Cohere RAG over SEC filings stored in Qdrant.

Pipeline
    1. Query rewriting   (Cohere chat -> several SEC-style search queries)
    2. Dense retrieval   (BGE embeddings + Qdrant, role-based filter, multi-query, dedupe)
    3. Reranking         (Cohere Rerank on the ORIGINAL question)
    4. Generation        (Cohere chat, strict evidence-only prompt, temperature 0)

.env needs: QDRANT_URL, QDRANT_API_KEY, COHERE_API_KEY
Optional  : COHERE_RERANK_MODEL, COHERE_GENERATION_MODEL
"""

import json
import os
import re
from collections import defaultdict

import cohere
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

load_dotenv()

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"          # must match the indexing model
RERANK_MODEL = os.getenv("COHERE_RERANK_MODEL", "rerank-v5.0-pro")
GENERATION_MODEL = os.getenv("COHERE_GENERATION_MODEL", "command-a-03-2025")

ROLE_ACCESS = {
    "consumer": ["PepsiCo", "Best Buy", "MGM Resorts", "Ulta Beauty"],
    "health_industrials": ["Johnson & Johnson", "Pfizer", "3M", "Boeing"],
    "tech_telecom": ["AMD", "Adobe", "Corning", "Verizon"],
    "lead": [
        "PepsiCo", "Best Buy", "MGM Resorts", "Ulta Beauty",
        "Johnson & Johnson", "Pfizer", "3M", "Boeing",
        "AMD", "Adobe", "Corning", "Verizon",
    ],
}

# --------------------------------------------------------------------------
# Clients
# --------------------------------------------------------------------------
co = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

client = QdrantClient(url=os.getenv("QDRANT_URL"), api_key=os.getenv("QDRANT_API_KEY"))

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
)


def ensure_indexes():
    """Payload indexes used by filters (safe to call repeatedly)."""
    wanted = {
        "metadata.company": models.PayloadSchemaType.KEYWORD,
        "metadata.doc_id": models.PayloadSchemaType.KEYWORD,
        "metadata.page": models.PayloadSchemaType.INTEGER,
        # full-text index so financial-statement pages can be found by their title
        "page_content": models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        ),
    }
    for field, schema in wanted.items():
        try:
            client.create_payload_index(COLLECTION_NAME, field_name=field, field_schema=schema)
        except Exception:
            pass  # already exists


ensure_indexes()


# --------------------------------------------------------------------------
# 1. Query rewriting
# --------------------------------------------------------------------------
FALLBACK_KEYWORDS = (
    "litigation, legal proceedings, legal claims, regulatory proceedings, lawsuits, "
    "investigations, inquiries, potential liabilities, material adverse effect"
)

REWRITE_SYSTEM = """You rewrite questions for searching SEC filings (10-K, 10-Q, 8-K).
Return ONLY a JSON object: {"queries": ["...", "...", "..."]}

Rules:
- Produce 3 to 4 short search queries that together cover the question.
- Use the formal wording and section names found in SEC filings
  (e.g. "Item 3. Legal Proceedings", "Commitments and Contingencies",
  "Item 7. Management's Discussion and Analysis", "Item 1A. Risk Factors").
- Keep company names, fiscal years and financial terms from the question.
- If the question refers to a balance sheet, income statement or cash flow statement, write one query
  with the statement's exact title, e.g. "MGM Resorts International consolidated balance sheets
  December 31, 2018 accounts payable total current liabilities".
- If the question names several years, include at least one query per year.
- Do not answer the question."""


def rewrite_query(question, n_max=4):
    """Return [original question, + LLM rewrites]. Falls back to keyword expansion."""
    queries = [question]
    try:
        resp = co.chat(
            model=GENERATION_MODEL,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
        data = json.loads(resp.message.content[0].text)
        queries += [q for q in data.get("queries", []) if isinstance(q, str)][:n_max]
    except Exception:
        queries.append(f"{question}\nTopics: {FALLBACK_KEYWORDS}")

    seen, unique = set(), []
    for q in queries:  # de-duplicate, keep order
        key = q.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(q.strip())
    return unique


# --------------------------------------------------------------------------
# 2. Question filters + retrieval
# --------------------------------------------------------------------------
COMPANY_ALIASES = {
    "PepsiCo": ["pepsico", "pepsi"],
    "Best Buy": ["best buy"],
    "MGM Resorts": ["mgm resorts", "mgm"],
    "Ulta Beauty": ["ulta beauty", "ulta"],
    "Johnson & Johnson": ["johnson & johnson", "johnson and johnson", "jnj", "j&j"],
    "Pfizer": ["pfizer"],
    "3M": ["3m"],
    "Boeing": ["boeing"],
    "AMD": ["amd"],
    "Adobe": ["adobe"],
    "Corning": ["corning"],
    "Verizon": ["verizon"],
}

STATEMENT_RULES = [    # (keywords in the question, statements that hold the numbers)
    (["balance sheet"], ["balance_sheet"]),
    (["income statement", "p&l", "statement of income", "statements of income", "statement of operations",
      "statement of earnings", "profit and loss"], ["income_statement"]),
    (["cash flow", "cashflow", "statement of cash flows"], ["cash_flow"]),
    (["capital intensive", "capital-intensive", "capital intensity"],
     ["income_statement", "balance_sheet", "cash_flow"]),
    (["working capital"], ["balance_sheet"]),
    (["free cash flow", "free cashflow", "fcf"], ["cash_flow", "income_statement"]),
    (["margin", "margins"], ["income_statement"]),
    (["capital expenditure", "capital expenditures", "capex"], ["cash_flow"]),
    (["days payable", "dpo", "days sales outstanding", "dso", "inventory turnover"],
     ["balance_sheet", "income_statement"]),
    (["return on assets", "roa"], ["income_statement", "balance_sheet"]),
]
STATEMENT_PHRASES = {   # words that must appear on the statement page
    "balance_sheet": ["balance sheets", "balance sheet"],
    "income_statement": ["statements of operations", "statement of operations", "statements of income",
                         "statement of income", "statements of earnings", "statement of earnings"],
    "cash_flow": ["statements of cash flows", "statement of cash flows"],
}
STATEMENT_TITLE_RE = re.compile(
    r"consolidated\s+(balance\s+sheets?|statements?\s+of\s+(operations|income|earnings|cash\s+flows))",
    re.I,
)


def extract_filters(question, role):
    """Rule-based: companies, fiscal years and financial statements mentioned in the question."""
    q = question.lower()

    companies = [
        c for c in ROLE_ACCESS[role]
        if any(re.search(rf"(?<![a-z0-9]){re.escape(a)}(?![a-z0-9])", q)
               for a in COMPANY_ALIASES.get(c, [c.lower()]))
    ]

    years = {int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?!\d)", q)}
    years |= {2000 + int(y) for y in re.findall(r"\bfy\s*'?(\d{2})\b", q)}
    years = {y for y in years if 1990 <= y <= 2035}

    statements = []
    for keys, sts in STATEMENT_RULES:
        if any(re.search(rf"(?<![a-z]){re.escape(k)}(?![a-z])", q) for k in keys):
            statements += [st for st in sts if st not in statements]

    # explicit quarter / half-year (H1 = the Q2 10-Q)
    quarters = {int(x) for x in re.findall(r"\bq([1-4])\b", q)}
    if re.search(r"\bh1\b", q):
        quarters.add(2)

    # does the question ask about a quarter / half-year? If not, interim filings are noise
    interim = bool(re.search(r"\b(q[1-4]|h[12]|quarter\w*|half|semi-?annual|interim)\b", q))

    return {"companies": companies, "years": sorted(years), "statements": statements, "quarters": sorted(quarters),
            "interim": interim}


def year_window(years):
    """FY N also appears in the FY N+1 10-K (comparatives); forecasts for N+1 sit in the FY N 10-K."""
    return {y + d for y in years for d in (-1, 0, 1)}


INTERIM_RE = re.compile(r"_\d{4}Q[123](?:_|$)")           # Q1-Q3 filings (Q4 = year end, kept)
DOC_RE = re.compile(r"_(\d{4})(Q[1-4])?_(10K|10Q|8K|EARNINGS)$")


def describe_doc(doc_id):
    """ULTABEAUTY_2023Q4_EARNINGS -> 'Q4 earnings release (labelled 2023)'"""
    m = DOC_RE.search(doc_id)
    if not m:
        return doc_id
    year, q, kind = m.groups()
    names = {"10K": "annual report (10-K)", "10Q": "quarterly report (10-Q)",
             "8K": "8-K", "EARNINGS": "earnings release"}
    return f"{(q + ' ') if q else ''}{names[kind]} (labelled {year})"


def target_doc_regex(filters):
    """The filing(s) the question is really about: the Q-filing asked for, else the FY 10-K."""
    years = filters.get("years") or []
    if not years:
        return None
    if filters.get("quarters"):
        alts = [rf"_{y}Q{n}_" for y in years for n in filters["quarters"]]
    else:
        alts = [rf"_{max(years)}_10K$"]      # latest year named: it also holds the earlier comparatives
    return re.compile("|".join(alts))


def doc_in_window(doc_id, window):
    return any(re.search(rf"_{y}(?!\d)", doc_id) for y in window)


def _candidate(doc_id, page, chunk_id, text, meta, score):
    return {
        "doc_id": doc_id,
        "company": meta.get("company", ""),
        "doc_type": meta.get("doc_type", ""),
        "period": meta.get("period", ""),
        "page": page,
        "chunk_id": chunk_id,
        "dense_score": float(score),
        "text": text,
    }


def fetch_page_chunks(doc_id, page):
    """Every chunk of one PDF page (a long table can be split in several chunks)."""
    flt = models.Filter(must=[
        models.FieldCondition(key="metadata.doc_id", match=models.MatchValue(value=doc_id)),
        models.FieldCondition(key="metadata.page", match=models.MatchValue(value=page)),
    ])
    points, _ = client.scroll(COLLECTION_NAME, scroll_filter=flt, limit=64,
                              with_payload=True, with_vectors=False)
    out = []
    for p in points:
        m = p.payload["metadata"]
        out.append(_candidate(m["doc_id"], m["page"], m["chunk_id"],
                              m.get("original_text", p.payload.get("page_content", "")), m, 0.0))
    return out


def lookup_statement_pages(companies, statements, years, max_pages=12):
    """
    Dense search often misses table-heavy statement pages. Find them directly:
    full-text match on the statement title, restricted to the company and the
    filing years around the question's fiscal year, then return ALL chunks of those pages.
    """
    if not (companies and statements and years):
        return []
    window = year_window(years)

    pairs = set()
    for st in statements:
        flt = models.Filter(
            must=[models.FieldCondition(key="metadata.company", match=models.MatchAny(any=companies))],
            should=[models.FieldCondition(key="page_content", match=models.MatchText(text=p))
                    for p in STATEMENT_PHRASES[st]],
        )
        offset = None
        while True:
            points, offset = client.scroll(COLLECTION_NAME, scroll_filter=flt, limit=256,
                                           offset=offset, with_payload=True, with_vectors=False)
            for p in points:
                m = p.payload["metadata"]
                text = m.get("original_text", "")
                # keep pages that START with the statement title (the real statement, not a mention)
                if doc_in_window(m["doc_id"], window) and STATEMENT_TITLE_RE.search(text[:400]):
                    pairs.add((m["doc_id"], m["page"]))
            if offset is None:
                break

    # prefer annual filings, then document order
    ranked = sorted(pairs, key=lambda x: (not x[0].endswith("10K"), x[0], x[1]))[:max_pages]
    out = []
    for doc_id, page in ranked:
        for c in fetch_page_chunks(doc_id, page):
            c["via"] = "lookup"
            out.append(c)
    return out


def retrieve(queries, role, filters=None, k_per_query=30):
    if role not in ROLE_ACCESS:
        raise ValueError(f"Invalid role: {role}")
    filters = filters or {"companies": [], "years": [], "statements": []}

    def company_filter(companies):
        return models.Filter(must=[models.FieldCondition(
            key="metadata.company", match=models.MatchAny(any=companies))])

    best = {}  # chunk_id -> candidate (keep best dense score)

    def add(c):
        cid = c["chunk_id"]
        if cid not in best:
            best[cid] = c
            return
        if c.get("via"):
            best[cid]["via"] = c["via"]                  # remember it was also found by the lookup
        if c["dense_score"] > best[cid]["dense_score"]:
            via = best[cid].get("via")
            best[cid] = c
            if via:
                best[cid]["via"] = via

    # (a) role-wide search, plus (b) a search narrowed to the company named in the question
    searches = [ROLE_ACCESS[role]]
    if filters["companies"]:
        searches.append(filters["companies"])
    for companies in searches:
        for q in queries:
            for doc, score in vector_store.similarity_search_with_score(
                query=q, k=k_per_query, filter=company_filter(companies)
            ):
                m = doc.metadata
                add(_candidate(m["doc_id"], m["page"], m["chunk_id"],
                               m.get("original_text", doc.page_content), m, score))

    # (c) direct lookup of the financial statement pages the question refers to
    try:
        for c in lookup_statement_pages(filters["companies"], filters["statements"], filters["years"]):
            add(c)
    except Exception as e:                       # never let the lookup break normal retrieval
        print("Statement lookup skipped:", repr(e))

    candidates = sorted(best.values(), key=lambda c: c["dense_score"], reverse=True)

    # (d) soft year filter: keep filings within +/-1 year of the fiscal year asked about
    if filters["years"]:
        window = year_window(filters["years"])
        in_window = [c for c in candidates if doc_in_window(c["doc_id"], window)]
        if len(in_window) >= 5:         # fall back to everything if the filter is too aggressive
            candidates = in_window

    # (d2) a named quarter (Q2 2023, H1 2023) means that quarter's filing, not the annual report
    if filters.get("quarters") and filters.get("years"):
        tgt = target_doc_regex(filters)
        exact = [c for c in candidates if tgt.search(c["doc_id"])]
        if len(exact) >= 5:
            candidates = exact

    # (e) year-end questions: drop Q1-Q3 filings unless the question asks about a quarter
    if not filters.get("interim"):
        annual = [c for c in candidates if not INTERIM_RE.search(c["doc_id"])]
        if len(annual) >= 5:
            candidates = annual

    return candidates


# --------------------------------------------------------------------------
# 3. Cohere rerank
# --------------------------------------------------------------------------
def rerank(question, candidates, top_n=10):
    if not candidates:
        return []

    # Prefix with light context so the reranker knows which filing/page it is reading
    docs = [
        f"{c['company']} {c['doc_type']} {c['period']} | PDF page {c['page']}\n{c['text']}"
        for c in candidates
    ]
    resp = co.rerank(
        model=RERANK_MODEL,
        query=question,                      # original question, not the expanded one
        documents=docs,
        top_n=min(top_n, len(docs)),
    )
    return [
        {**candidates[r.index], "rerank_rank": rank, "rerank_score": float(r.relevance_score)}
        for rank, r in enumerate(resp.results, start=1)
    ]


def build_context(evidence):
    blocks = []
    for i, e in enumerate(evidence, start=1):
        blocks.append(
            f"Document ID: {e['doc_id']} | Filing: {describe_doc(e['doc_id'])} | "
            f"PDF page: {e['page']} | Chunk: {e['chunk_id']}\n{e['text'].strip()}"
        )
    return "\n\n---\n\n".join(blocks)


# --------------------------------------------------------------------------
# 4. Generation
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an expert financial research analyst for SEC filings (10-K, 10-Q, 8-K, earnings reports).
Answer ONLY from the evidence blocks supplied in the user message.

EVIDENCE RULES
- Never use outside knowledge, assumptions, or memory. Never invent facts, figures, events, or citations.
- Every factual claim needs a citation in the form [DOCUMENT_ID, PDF page X], using the exact
  document ID and page shown in the evidence block it came from.
- If evidence blocks conflict, state the conflict instead of choosing silently.

INTERPRETATION RULES
- Separate these ideas: (a) a matter exists, (b) it is individually material, (c) management's
  expected aggregate financial impact. "No material adverse effect expected" is a statement about (c).
  It does NOT mean no proceedings exist, and it does NOT prove that no single case is material.
  Report management's wording and scope exactly; if materiality of individual matters is not
  established, say: "The filing does not establish whether any individual proceeding is
  independently material."
- Distinguish risk-factor language ("could", "may") from disclosed actual events, and distinguish
  ongoing, historical, and completed matters.
- A 10-K for fiscal year N normally also contains comparative data for year N-1. Treat each
  requested period separately; only attribute a fact to a period when the evidence supports it.
  If a requested period is not supported, say it cannot be verified from the evidence.
- Absence of retrieved evidence is NOT evidence that something did not happen.

FISCAL PERIODS
- Document IDs look like COMPANY_YEAR[Qn]_TYPE. The year is only a label: read the evidence text to
  confirm which fiscal period it really covers. Never describe a figure as belonging to a period
  unless the evidence says so (a Q1 release labelled 2024 is not "Q1 of fiscal 2023").
- Companies name fiscal years differently (e.g. a retailer's year ending in January 2023 may be called
  "fiscal 2022"). If the evidence's fiscal-year name differs from the question's, say so in LIMITATIONS
  and state which period you used.
- If the question names a specific quarter (e.g. "Q2 2023", "H1 2023"), answer from the filing for that
  quarter. Use other periods only for comparison and say so. Do not list items from an older filing
  that the newer filing no longer shows (e.g. notes that matured).
- Questions about "the end of" a fiscal year must be answered from the annual report or the Q4 /
  year-end earnings release, not from an interim quarter.

NUMBERS
- Financial tables state their scale in the header ("In thousands", "In millions"). Use the scale of the
  table you cite. If the question asks for another unit, convert, show the arithmetic, and round only
  at the end (e.g. $302,578 thousand = $302.578 million, about $303 million).
- When the question says to use a specific statement (balance sheet, income statement, cash flow
  statement), take the values from that statement's page. Pick the column for the requested fiscal year.
- Preserve original currencies and periods. Show inputs and formula for any calculation.
  Never combine figures from different periods without explaining the comparison.

METRICS AND CALCULATIONS
- State the formula you use. Standard definitions unless the question gives its own:
  * Free cash flow = net cash from operating activities - capital expenditures (purchases of property
    and equipment). Never subtract total investing cash flow. FCF conversion = FCF / net income.
  * Capital intensity = total assets / revenue; also report capex / revenue and net PP&E / total assets.
    Judge capital intensity from ratios, never from the absolute size of capex.
  * Working capital = total current assets - total current liabilities, computed from the balance sheet.
    If the company reports its own working-capital figure, mention it as well.
  * Net profit margin = net income / revenue. D&A margin = D&A / revenue.
- Calculate step by step. The final answer MUST equal the computed result: round only at the end, to the
  requested precision, and never replace a computed result with a different number.
- If a needed input is missing from the evidence, name it instead of drawing a conclusion.

OUTPUT FORMAT
If the evidence supports an answer:
STATUS: ANSWERED
ANSWER: <direct answer, then short explanation, per period when several are asked>
EVIDENCE:
- <finding> [DOCUMENT_ID, PDF page X]
LIMITATIONS: <gaps, period/label mismatches, or "None identified">

If it does not:
STATUS: REFUSED
ANSWER: The available filing evidence is insufficient to answer this question.
REASON: <what is missing>"""


def generate(question, evidence):
    if not evidence:
        return "STATUS: REFUSED\nANSWER: No relevant filing evidence was retrieved.\nREASON: Retrieval returned nothing."

    user_msg = (
        f"QUESTION:\n{question}\n\n"
        f"RETRIEVED SEC FILING EVIDENCE:\n{build_context(evidence)}\n\n"
        "Answer the exact question asked, period by period, using only this evidence.\n"
        "Cite every claim as [DOCUMENT_ID, PDF page X] using the Document ID and PDF page printed above "
        "each block. Never cite evidence numbers."
    )
    resp = co.chat(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
    )
    return resp.message.content[0].text


# --------------------------------------------------------------------------
# Exact page extraction from the answer's citations
# --------------------------------------------------------------------------
CITE_RE = re.compile(
    r"([A-Za-z0-9]+(?:_[A-Za-z0-9]+)+)\s*,\s*PDF\s*pages?\s*"
    r"(\d+(?:\s*(?:,|and|&|-|\u2013)\s*\d+)*)",
    re.I,
)


EVID_RE = re.compile(r"Evidence\s*(\d+(?:\s*(?:,|and|&)\s*\d+)*)", re.I)


def parse_citations(answer, evidence=None):
    """
    Exact (document -> PDF pages) pairs from citations such as
    [PEPSICO_2022_10K, PDF page 26], [3M_2022_10K, PDF pages 48, 50 and 52]
    or [A_2024Q1_EARNINGS, PDF page 2; B_2023_10K, PDF page 41] (several documents in one bracket).
    """
    pages = defaultdict(set)
    for doc_id, raw in CITE_RE.findall(answer):
        for lo, hi in re.findall(r"(\d+)\s*[-\u2013]\s*(\d+)", raw):          # ranges: 25-27
            pages[doc_id].update(range(int(lo), int(hi) + 1))
        raw = re.sub(r"\d+\s*[-\u2013]\s*\d+", " ", raw)
        pages[doc_id].update(int(n) for n in re.findall(r"\d+", raw))
    # fallback: the model cited "Evidence 3" instead of a document ID -> map the number back to doc/page
    if evidence:
        for raw in EVID_RE.findall(answer):
            for n in re.findall(r"\d+", raw):
                i = int(n) - 1
                if 0 <= i < len(evidence):
                    pages[evidence[i]["doc_id"]].add(evidence[i]["page"])
    return {d: sorted(p) for d, p in pages.items()}


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------
def pin_statement_chunks(candidates, filters, already_ids, max_chunks=10):
    """
    Reranking favours MD&A prose over statement tables. If the question needs a financial statement,
    always hand the LLM that statement's page(s) from the target filing.
    """
    tgt = target_doc_regex(filters)
    if not (tgt and filters.get("statements")):
        return []
    pins = [c for c in candidates
            if c.get("via") == "lookup" and tgt.search(c["doc_id"]) and c["chunk_id"] not in already_ids]
    pins.sort(key=lambda c: (c["doc_id"], c["page"], c["chunk_id"]))
    return [{**c, "rerank_rank": None, "rerank_score": None, "pinned": True} for c in pins[:max_chunks]]


def ask_filings(
    question,
    role,
    k_per_query=30,
    top_n=8,
    verbose=True,
):
    filters = extract_filters(question, role)
    queries = rewrite_query(question)
    candidates = retrieve(queries, role, filters, k_per_query=k_per_query)
    evidence = rerank(question, candidates, top_n=top_n)
    evidence += pin_statement_chunks(candidates, filters, {e["chunk_id"] for e in evidence})
    answer = generate(question, evidence)

    if verbose:
        print("Filters:", filters)
        print("Queries:")
        for q in queries:
            print("  -", q.replace("\n", " ")[:120])
        lookup_pages = sorted({(c["doc_id"], c["page"]) for c in candidates if c.get("via") == "lookup"})
        print("Statement pages found by lookup:", lookup_pages or "none")
        print(f"\nCandidates: {len(candidates)} | Evidence chunks: {len(evidence)}")
        for e in evidence:
            rank = f"#{e['rerank_rank']}" if e["rerank_rank"] else "pin"
            score = f"{e['rerank_score']:.4f}" if e["rerank_score"] is not None else "  -   "
            print(f"  {rank:<4} p{e['page']:<4} {e['chunk_id']:<36} rerank={score}")
        print("Pages cited in answer:", parse_citations(answer, evidence))

    return {
        "answer": answer,
        "queries": queries,
        "filters": filters,
        "citations": sorted({(e["doc_id"], e["page"]) for e in evidence}),   # retrieved evidence pages
        "cited_pages": parse_citations(answer, evidence),                              # pages the answer actually cites
        "evidence": evidence,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FilingsIQ — SEC Filing Search & Grounded QA CLI")
    subparsers = parser.add_subparsers(dest="command", help="Subcommand to run")

    # search subcommand
    p_search = subparsers.add_parser("search", help="Search filing chunks")
    p_search.add_argument("--role", required=True, choices=list(ROLE_ACCESS.keys()), help="User desk role")
    p_search.add_argument("--k", type=int, default=10, help="Number of chunks to return")
    p_search.add_argument("question", help="Search query or question")

    # ask subcommand
    p_ask = subparsers.add_parser("ask", help="Ask a question grounded in SEC filings")
    p_ask.add_argument("--role", required=True, choices=list(ROLE_ACCESS.keys()), help="User desk role")
    p_ask.add_argument("question", help="Question to ask")
    p_ask.add_argument("--k_per_query", type=int, default=30, help="Candidate chunks per query")
    p_ask.add_argument("--top_n", type=int, default=8, help="Reranked evidence chunks to pass to LLM")
    p_ask.add_argument("--raw", action="store_true", help="Print raw formatted answer text instead of JSON")

    # ingest subcommand
    p_ingest = subparsers.add_parser("ingest", help="Ingest SEC filings corpus directory")
    p_ingest.add_argument("corpus_dir", help="Path to corpus directory containing SEC filing PDFs")

    args = parser.parse_args()

    if args.command == "search":
        filters = extract_filters(args.question, args.role)
        queries = rewrite_query(args.question)
        candidates = retrieve(queries, args.role, filters, k_per_query=30)
        evidence = rerank(args.question, candidates, top_n=args.k)
        results = [
            {
                "doc_id": e["doc_id"],
                "page": e["page"],
                "chunk_id": e["chunk_id"],
                "score": round(e.get("rerank_score", 0.0), 4) if e.get("rerank_score") is not None else 0.0,
                "text": e["text"],
            }
            for e in evidence[:args.k]
        ]
        print(json.dumps(results, indent=2))

    elif args.command == "ask":
        result = ask_filings(
            args.question,
            role=args.role,
            k_per_query=args.k_per_query,
            top_n=args.top_n,
            verbose=False,
        )
        if args.raw:
            print(result["answer"])
            return

        is_refused = result["answer"].strip().startswith("STATUS: REFUSED")
        status = "refused" if is_refused else "answered"

        citations = [
            {
                "doc_id": e["doc_id"],
                "page": e["page"],
                "chunk_id": e["chunk_id"],
            }
            for e in result["evidence"]
        ]

        output = {
            "status": status,
            "answer": result["answer"],
            "citations": citations,
        }
        print(json.dumps(output, indent=2))

    elif args.command == "ingest":
        print(f"Ingesting SEC filings from directory: {args.corpus_dir}")
        print("Ingestion pipeline sequence:")
        print("  1. Extract PDF pages to structured Markdown & JSON (Notebooks/Extraction.ipynb)")
        print("  2. Enrich documents with desk and company metadata (join.py)")
        print("  3. Semantic & recursive chunking with contextual headers (Notebooks/Injestion and testing.ipynb)")
        print("  4. Generate BGE embeddings and upsert to Qdrant Cloud collection 'sec_filings'")

    else:
        demo_question = (
            "Has PepsiCo reported any materially important ongoing "
            "legal battles from FY2022 and FY2021?"
        )
        print("No subcommand provided. Running default demo question for role 'consumer':")
        print(f"Question: {demo_question}\n")
        result = ask_filings(demo_question, role="consumer", verbose=True)
        print("\n" + "=" * 70)
        print(result["answer"])


if __name__ == "__main__":
    main()