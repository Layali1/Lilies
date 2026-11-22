"""Shows what retrieval returns for one evaluation question (ranked, with distances).

  python -m eval.peek 7 --k 15

Tells you whether the expected page is retrieved at all, and at which rank.
(Smaller distance = closer. This is the FIRST retrieval, before any query rewrite or reranking.)
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from core.ingest import get_store

load_dotenv(override=True)
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int, help="question number, as in the results table (Q7 -> 7)")
ap.add_argument("--k", type=int, default=15)
args = ap.parse_args()

c = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))[args.n - 1]
print(f"Q{args.n}: {c['question']}\nfile: {c['source']} | expected page: {c.get('page', '-')} | evidence: {c.get('evidence', '-')}\n")
store = get_store("chroma_db")
res = store.similarity_search_with_score(c["question"], k=args.k, filter={"source": c["source"]})
found = None
for rank, (d, dist) in enumerate(res, 1):
    hit = d.metadata["page"] == c.get("page")
    found = found or (rank if hit else None)
    print(f"{rank:>2}. dist={dist:.3f}  p.{d.metadata['page']:<4} {d.page_content[:90]!r}{'   <== expected page' if hit else ''}")
print(f"\nexpected page first appears at rank: {found if found else f'not in top {args.k}'}")