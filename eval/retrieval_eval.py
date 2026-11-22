"""Retrieval-only evaluation: no LLM calls, so no API limits and no answer-generation noise.

  python -m eval.retrieval_eval --k 10 --name vector
  python -m eval.retrieval_eval --k 10 --rerank 30 --name rerank30
  python -m eval.retrieval_eval --k 10 --hybrid 1 --name hybrid

For every answerable question: at which rank is the FIRST retrieved chunk that contains the 'evidence' text?
Hit@k = evidence found in the top k chunks.  MRR = average of 1/rank (0 when not found).
"""
import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--rerank", type=int, default=0, help="candidates to fetch and rerank (0 = no reranking)")
ap.add_argument("--quant", type=int, default=None)
ap.add_argument("--hybrid", type=int, default=0)
ap.add_argument("--name", default="latest")
args = ap.parse_args()
if args.quant is not None:
    os.environ["LILIES_RERANK_QUANT"] = str(args.quant)

from core.ingest import get_store  # noqa: E402
from core.retrieval import retrieve_docs  # noqa: E402
from core.utils import clean  # noqa: E402

AR_DIGITS = {0x0660 + i: str(i) for i in range(10)} | {0x06F0 + i: str(i) for i in range(10)}
squash = lambda t: re.sub(r"\s+", "", re.sub(r"[\u2010-\u2015\u2212]", "-", clean(t))).translate(AR_DIGITS).lower()

store = get_store("chroma_db")
cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
rows, ranks, start = [], [], time.time()
for n, c in enumerate(cases, 1):
    if not c.get("answerable", True):
        continue
    q = clean(c["question"])
    docs = retrieve_docs(store, q, q, args.k, source=c["source"], n_cand=args.rerank,
                         use_hybrid=bool(args.hybrid), mode="always")
    rank = next((i for i, d in enumerate(docs, 1)
                 if all(squash(e) in squash(d.page_content) for e in c["evidence"])), None)
    ranks.append(rank)
    rows.append(f"| Q{n} | {c['question'][:45].replace('|', '/')} | {rank or '-'} |")
    print(rows[-1])

hit = lambda k: sum(1 for r in ranks if r and r <= k)
total = len(ranks)
mrr = sum(1 / r for r in ranks if r) / total
seconds = time.time() - start
lines = [f"- Config: k={args.k}, rerank={args.rerank}, quant={os.getenv('LILIES_RERANK_QUANT', '0')}, hybrid={args.hybrid}",
         f"- Hit@1: **{hit(1)}/{total}**, Hit@3: **{hit(3)}/{total}**, Hit@5: **{hit(5)}/{total}**"
         + (f", Hit@{args.k}: **{hit(args.k)}/{total}**" if args.k not in (1, 3, 5) else ""),
         f"- MRR: **{mrr:.3f}**",
         f"- Time: {seconds:.0f} s for {total} questions (no LLM calls)"]
report = (f"# Retrieval evaluation: {args.name}\n\n" + "\n".join(lines)
          + "\n\n| # | Question | Rank of the first chunk containing the evidence |\n|---|---|---|\n" + "\n".join(rows) + "\n")
Path(f"eval/retrieval_{args.name}.md").write_text(report, encoding="utf-8")
print("\n" + "\n".join(lines))
print(f"saved: eval/retrieval_{args.name}.md")