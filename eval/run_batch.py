"""
Send every question in questions.jsonl through the Cohere RAG pipeline and
report the EXACT pages used for each answer.

Usage
    python run_batch.py                              # all questions
    python run_batch.py --limit 5                    # first 5 only
    python run_batch.py --top_n 10 --sleep 3         # more evidence, slower pace (rate limits)
    python run_batch.py --fresh                      # ignore previous results and start over

Output
    results.jsonl   one JSON record per question (resumable: finished ids are skipped)
    results.csv     flat table for Excel / pandas
"""

import argparse
import csv
import json
import time
from pathlib import Path

from cohere_sec_rag import ask_filings


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall(gold, found):
    gold = set(gold)
    return len(gold & set(found)) / len(gold) if gold else 0.0


def process(row, top_n):
    res = ask_filings(row["question"], role=row["role"], top_n=top_n, verbose=False)

    gold_doc, gold_pages = row["doc_id"], row["pages"]

    # exact pages shown to the LLM (retrieved evidence), per document
    evidence_pages = {}
    for e in res["evidence"]:
        evidence_pages.setdefault(e["doc_id"], set()).add(e["page"])
    evidence_pages = {d: sorted(p) for d, p in evidence_pages.items()}

    cited = res["cited_pages"]                       # pages the answer itself cites

    return {
        "id": row["id"],
        "role": row["role"],
        "question": row["question"],
        "gold_answer": row["answer"],
        "gold_doc_id": gold_doc,
        "gold_pages": gold_pages,
        "model_answer": res["answer"],
        "cited_pages": cited,                         # {doc_id: [pages]}  <- exact pages in the answer
        "evidence_pages": evidence_pages,             # {doc_id: [pages]}  <- exact pages retrieved
        "top_evidence": [
            {"rank": e.get("rerank_rank"), "doc_id": e["doc_id"], "page": e["page"],
             "chunk_id": e["chunk_id"],
             "rerank_score": round(e["rerank_score"], 4) if e.get("rerank_score") is not None else None,
             "pinned": bool(e.get("pinned"))}
            for e in res["evidence"]
        ],
        "queries": res["queries"],
        "gold_doc_retrieved": gold_doc in evidence_pages,
        "gold_doc_cited": gold_doc in cited,
        "recall_retrieved": recall(gold_pages, evidence_pages.get(gold_doc, [])),
        "recall_cited": recall(gold_pages, cited.get(gold_doc, [])),
        "error": None,
    }


def fmt(pages_by_doc):
    if not pages_by_doc:
        return "-"
    return "; ".join(f"{d} p{','.join(map(str, p))}" for d, p in pages_by_doc.items())


def summarize(records):
    ok = [r for r in records if not r["error"]]
    if not ok:
        print("No successful records.")
        return
    n = len(ok)
    print("\n" + "=" * 78)
    print(f"SUMMARY ({n} answered, {len(records) - n} errors)")
    print(f"  Gold document retrieved : {sum(r['gold_doc_retrieved'] for r in ok)}/{n}")
    print(f"  Gold document cited     : {sum(r['gold_doc_cited'] for r in ok)}/{n}")
    print(f"  Avg gold-page recall    : retrieved {sum(r['recall_retrieved'] for r in ok) / n:.2f} | "
          f"cited {sum(r['recall_cited'] for r in ok) / n:.2f}")
    print(f"  Any gold page retrieved : {sum(r['recall_retrieved'] > 0 for r in ok)}/{n}")
    print(f"  All gold pages retrieved: {sum(r['recall_retrieved'] == 1 for r in ok)}/{n}")
    for role in sorted({r["role"] for r in ok}):
        sub = [r for r in ok if r["role"] == role]
        print(f"    {role:<20} any-page hit {sum(r['recall_retrieved'] > 0 for r in sub)}/{len(sub)}")


def write_csv(records, path):
    cols = ["id", "role", "question", "gold_answer", "model_answer", "gold_doc_id", "gold_pages",
            "cited_pages", "evidence_pages", "gold_doc_retrieved", "gold_doc_cited",
            "recall_retrieved", "recall_cited", "error"]
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in records:
            w.writerow([
                r.get(c) if c not in ("gold_pages", "cited_pages", "evidence_pages")
                else (",".join(map(str, r[c])) if c == "gold_pages" else fmt(r[c]))
                for c in cols
            ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="questions.jsonl")
    ap.add_argument("--output", default="results.jsonl")
    ap.add_argument("--top_n", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between questions")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fresh", action="store_true", help="discard existing results")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    if args.limit:
        rows = rows[: args.limit]

    out_path = Path(args.output)
    if args.fresh and out_path.exists():
        out_path.unlink()
    records = load_jsonl(out_path) if out_path.exists() else []
    done = {r["id"] for r in records if not r.get("error")}
    records = [r for r in records if not r.get("error")]       # retry failed ones

    for i, row in enumerate(rows, start=1):
        if row["id"] in done:
            continue

        print(f"\n[{i}/{len(rows)}] {row['id']} ({row['role']})")
        print("Q:", row["question"][:160].replace("\n", " "))

        try:
            rec = process(row, args.top_n)
        except Exception as e:                                  # keep going on API errors
            rec = {"id": row["id"], "role": row["role"], "question": row["question"],
                   "gold_doc_id": row["doc_id"], "gold_pages": row["pages"], "error": repr(e)}
            print("ERROR:", rec["error"])
        else:
            hit = "HIT " if rec["recall_retrieved"] > 0 else "MISS"
            print(f"Gold      : {row['doc_id']} p{','.join(map(str, row['pages']))}")
            print(f"Cited     : {fmt(rec['cited_pages'])}")
            print(f"Retrieved : {fmt(rec['evidence_pages'])}")
            print(f"Page match: {hit} (retrieved recall {rec['recall_retrieved']:.2f}, "
                  f"cited recall {rec['recall_cited']:.2f})")
            print("A:", rec["model_answer"].strip().replace("\n", " ")[:300])

        records.append(rec)
        with open(out_path, "w", encoding="utf-8") as f:           # save after every question
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        time.sleep(args.sleep)

    write_csv(records, out_path.with_suffix(".csv"))
    summarize(records)
    print(f"\nSaved {out_path} and {out_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()