import os
import sys
import json
import asyncio
import time
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv  

from ragas import Dataset, experiment
from ragas.llms import llm_factory
from ragas.metrics import DiscreteMetric


# ============================================================
# Configuration
# ============================================================
load_dotenv()
BASE_DIR = Path(__file__).parent
JSONL_FILE =  BASE_DIR / "results.jsonl"

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError(
        "MISTRAL_API_KEY environment variable is not set."
    )


# ============================================================
# Mistral client
# ============================================================

# Mistral provides an OpenAI-compatible API.
groq_client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
)

# LLM judge
JUDGE_MODEL = "openai/gpt-oss-20b"

# Mistral model used as the evaluation judge
judge_llm = llm_factory(
    JUDGE_MODEL,
    client=groq_client,
)


# ============================================================
# Load JSONL
# ============================================================

def load_jsonl(path: Path):

    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}"
        )

    dataset = Dataset(
        name="sec_rag_evaluation",
        backend="local/csv",
        root_dir="evals",
    )

    count = 0

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Invalid JSON on line {line_number}: {e}"
                )
                continue

            # Required fields
            required = [
                "question",
                "gold_answer",
                "model_answer",
            ]

            missing = [
                field
                for field in required
                if field not in row
            ]

            if missing:
                print(
                    f"WARNING: Line {line_number} is missing: {missing}"
                )
                continue

            dataset.append({
                "id": str(row.get("id", "")),
                "role": str(row.get("role", "")),

                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "model_answer": row["model_answer"],

                # Retrieval ground truth
                "gold_doc_id": str(
                    row.get("gold_doc_id", "")
                ),

                "gold_pages": json.dumps(
                    row.get("gold_pages", [])
                ),

                # Citation information
                "cited_pages": json.dumps(
                    row.get("cited_pages", {})
                ),

                "evidence_pages": json.dumps(
                    row.get("evidence_pages", {})
                ),

                "top_evidence": json.dumps(
                    row.get("top_evidence", [])
                ),

                # Existing deterministic metrics
                "gold_doc_retrieved": str(
                    row.get("gold_doc_retrieved", "")
                ),

                "gold_doc_cited": str(
                    row.get("gold_doc_cited", "")
                ),

                "recall_retrieved": str(
                    row.get("recall_retrieved", "")
                ),

                "recall_cited": str(
                    row.get("recall_cited", "")
                ),

                "error": str(
                    row.get("error", "")
                ),
            })

            count += 1

    dataset.save()

    print(f"Loaded {count} evaluation samples.")

    return dataset


# ============================================================
# Metric 1 — Answer Correctness
# ============================================================

correctness_metric = DiscreteMetric(
    name="correctness",

    prompt="""
You are an expert evaluator for a financial SEC document
question-answering system.

Determine whether the MODEL ANSWER correctly answers the QUESTION
when compared against the GOLD ANSWER.

Important:

- Do not require identical wording.
- Paraphrases are acceptable.
- Additional correct information is acceptable.
- Do not penalize the answer merely because it contains more detail.
- Penalize factual contradictions.
- Penalize claims that materially change the meaning of the gold answer.
- Focus on the actual question.

Return exactly:

pass

or

fail

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{response}
""",

    allowed_values=[
        "pass",
        "fail",
    ],
)


# ============================================================
# Metric 2 — Answer Relevancy
# ============================================================

relevancy_metric = DiscreteMetric(
    name="relevancy",

    prompt="""
Evaluate whether the MODEL ANSWER directly and appropriately
answers the QUESTION.

Return "pass" when:

- The answer addresses the actual question.
- The response is focused on the requested information.
- The answer does not spend most of its content discussing
  unrelated information.

Return "fail" when:

- The answer is irrelevant.
- The answer avoids the question.
- The response mainly discusses unrelated information.

Do not judge whether the answer is factually correct here.
Judge only relevance.

Return exactly:

pass

or

fail

QUESTION:
{question}

MODEL ANSWER:
{response}
""",

    allowed_values=[
        "pass",
        "fail",
    ],
)


# ============================================================
# Metric 3 — Completeness
# ============================================================

completeness_metric = DiscreteMetric(
    name="completeness",

    prompt="""
Evaluate whether the MODEL ANSWER contains the important
information necessary to answer the QUESTION.

Compare it against the GOLD ANSWER.

Return "pass" when the model answer contains the essential
information required by the gold answer.

Return "fail" when important information from the gold answer
is missing, causing the answer to be incomplete or misleading.

Do not require identical wording.

Return exactly:

pass

or

fail

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{response}
""",

    allowed_values=[
        "pass",
        "fail",
    ],
)


# ============================================================
# Metric 4 — Citation Correctness
# ============================================================

citation_metric = DiscreteMetric(
    name="citation_correctness",

    prompt="""
You are evaluating citations produced by an SEC RAG system.

Determine whether the citations in the MODEL ANSWER are
consistent with the GOLD DOCUMENT and GOLD PAGES.

The model answer may cite multiple documents and pages.

The important question is whether the citations supporting
the answer point to the expected source material.

Return "pass" when:

- The answer cites the gold document and relevant gold page(s),
  OR
- The cited evidence is clearly appropriate for the answer even
  if the exact gold page representation differs.

Return "fail" when:

- The answer relies on unrelated documents/pages.
- The cited pages contradict the expected evidence.
- The answer claims evidence from pages that do not support
  the stated conclusion.

Return exactly:

pass

or

fail

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

GOLD DOCUMENT:
{gold_doc_id}

GOLD PAGES:
{gold_pages}

MODEL ANSWER:
{response}

ACTUAL CITED PAGES:
{cited_pages}
""",

    allowed_values=[
        "pass",
        "fail",
    ],
)


# ============================================================
# Run experiment
# ============================================================

@experiment()
async def run_experiment(row):

    question = row["question"]
    gold_answer = row["gold_answer"]
    model_answer = row["model_answer"]

    print(
        f"\nEvaluating: {row['id']}"
    )

    # --------------------------------------------------------
    # LLM-as-judge metrics
    # --------------------------------------------------------

    correctness = correctness_metric.score(
        llm=judge_llm,
        question=question,
        gold_answer=gold_answer,
        response=model_answer,
    )
    time.sleep(10)
    relevancy = relevancy_metric.score(
        llm=judge_llm,
        question=question,
        response=model_answer,
    )

    time.sleep(10)

    completeness = completeness_metric.score(
        llm=judge_llm,
        question=question,
        gold_answer=gold_answer,
        response=model_answer,
    )
    time.sleep(10)


    citation_correctness = citation_metric.score(
        llm=judge_llm,
        question=question,
        gold_answer=gold_answer,
        gold_doc_id=row["gold_doc_id"],
        gold_pages=row["gold_pages"],
        response=model_answer,
        cited_pages=row["cited_pages"],
    )
    time.sleep(10)


    # --------------------------------------------------------
    # Return complete evaluation result
    # --------------------------------------------------------

    return {
        # Identity
        "id": row["id"],
        "role": row["role"],

        # Question / answers
        "question": question,
        "gold_answer": gold_answer,
        "model_answer": model_answer,

        # LLM judge scores
        "correctness": correctness.value,
        "relevancy": relevancy.value,
        "completeness": completeness.value,
        "citation_correctness":
            citation_correctness.value,

        # Existing retrieval metrics
        "gold_doc_id": row["gold_doc_id"],
        "gold_pages": row["gold_pages"],
        "cited_pages": row["cited_pages"],
        "evidence_pages": row["evidence_pages"],
        "top_evidence": row["top_evidence"],

        "gold_doc_retrieved":
            row["gold_doc_retrieved"],

        "gold_doc_cited":
            row["gold_doc_cited"],

        "recall_retrieved":
            row["recall_retrieved"],

        "recall_cited":
            row["recall_cited"],

        "error": row["error"],
    }


# ============================================================
# Main
# ============================================================

async def main():

    print("=" * 70)
    print("SEC RAG Evaluation")
    print("=" * 70)

    print(
        f"\nInput JSONL: {JSONL_FILE}"
    )

    print(
        "\nJudge model: mistral-small-latest"
    )

    dataset = load_jsonl(JSONL_FILE)

    print(
        "\nRunning experiment..."
    )

    results = await run_experiment.arun(dataset)

    print(
        "\n" + "=" * 70
    )
    print("Experiment completed successfully!")
    print("=" * 70)

    print(
        f"\nResults:\n{results}"
    )

    # Save RAGAS experiment results
    results.save()

    csv_path = (
        Path(".")
        / "experiments"
        / f"{results.name}.csv"
    )

    print(
        f"\nResults saved to:"
        f"\n{csv_path.resolve()}"
    )


if __name__ == "__main__":
    asyncio.run(main())