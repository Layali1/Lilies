"""Evaluation: python -m eval.evaluate "AI Engineering.pdf" "AINewsletter27Aug.pdf" --name baseline

Each question searches ONLY its own file (the 'source' field), like the chat will.
  --fresh          use a temporary index instead of chroma_db (needed for chunk-size experiments)
  --k N            how many chunks to retrieve (default 5)
  --rerank N       fetch N candidates and rerank them (0 = off; default: LILIES_RERANK from .env)
  --rerank-mode M  "always" = rerank every retrieval, "retry" = only when the grader said no (faster)
  --quant 0/1      int8-quantized reranker (faster on CPU; check that accuracy holds)
  --hybrid 0/1     add BM25 keyword search to the vector search (default: LILIES_HYBRID from .env)
  --pause S        seconds between questions, to respect API rate limits (default 2)
  --name X         results are written to eval/results_X.md
  --only 6,8       run only these question numbers (1-based) to investigate failures quickly
"""
import argparse
import json
import os
import re
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)
ap = argparse.ArgumentParser()
ap.add_argument("pdfs", nargs="+")
ap.add_argument("--fresh", action="store_true")
ap.add_argument("--k", type=int, default=5)
ap.add_argument("--pause", type=float, default=2.0)
ap.add_argument("--name", default="latest")
ap.add_argument("--only", default="")
ap.add_argument("--rerank", type=int, default=None)
ap.add_argument("--rerank-mode", default=None, choices=["always", "retry"])
ap.add_argument("--quant", type=int, default=None)
ap.add_argument("--hybrid", type=int, default=None)
args = ap.parse_args()
if args.quant is not None:  # must be set before the reranker is loaded
    os.environ["LILIES_RERANK_QUANT"] = str(args.quant)

from core.graph import build_graph  # noqa: E402
from core.ingest import CHUNK_OVERLAP, CHUNK_SIZE, add_pdf, get_store  # noqa: E402
from core.timing import summary_rows  # noqa: E402
from core.utils import NOT_FOUND, clean  # noqa: E402

store = get_store(tempfile.mkdtemp() if args.fresh else "chroma_db")
for pdf in map(Path, args.pdfs):
    if not store.get(where={"source": pdf.name}, limit=1)["ids"]:
        print(f"indexing {pdf.name} ...")
        add_pdf(store, str(pdf), pdf.name, progress=lambda p: print(f"  {int(p * 100)}%", end="\r"))
        print()

graphs = {}


def graph_for(source):
    if source not in graphs:
        graphs[source] = build_graph(store, k=args.k, source=source, rerank_candidates=args.rerank,
                                     hybrid=args.hybrid, rerank_mode=args.rerank_mode)
    return graphs[source]


# compare answers fairly: plain hyphens, and Arabic-Indic digits (٥٠) turned into 50
AR_DIGITS = {0x0660 + i: str(i) for i in range(10)} | {0x06F0 + i: str(i) for i in range(10)}
norm = lambda t: re.sub(r"[\u2010-\u2015\u2212]", "-", clean(t)).translate(AR_DIGITS).lower()
mark = lambda ok: "✅" if ok else "❌"
pct = lambda a, b: f"{a / b:.0%} ({a}/{b})" if b else "n/a"

cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
only = {int(x) for x in args.only.split(",") if x.strip()}
selected = [(n, c) for n, c in enumerate(cases, 1) if not only or n in only]
n_ans = sum(c.get("answerable", True) for _, c in selected)
n_ref = len(selected) - n_ans
rows, fails, ans_ok, page_ok, refuse_ok = [], [], 0, 0, 0
start = time.time()

for n, c in selected:
    out = graph_for(c["source"]).invoke({"question": c["question"]})
    pages = [d.metadata["page"] for d in out["docs"]]
    q = c["question"][:40].replace("|", "/")
    if c.get("answerable", True):
        a = bool(out["relevant"]) and any(norm(k) in norm(out["answer"]) for k in c["keywords"])
        p = c["page"] in pages
        ans_ok += a
        page_ok += p
        ok = a and p
        rows.append(f"| Q{n} | {q} | answerable | {mark(a)} | {mark(p)} |")
    else:
        ok = out["answer"] in NOT_FOUND.values()
        refuse_ok += ok
        rows.append(f"| Q{n} | {q} | out-of-file | {mark(ok)} | - |")
    print(rows[-1])
    if not ok:
        note = (f"Q{n}: relevant={out['relevant']} | tries={out.get('tries')} | query={out.get('query', '')[:80]!r} | "
                f"expected p.{c.get('page', '-')} | retrieved pages={pages} | "
                f"answer={out['answer'][:220].replace(chr(10), ' ')!r}")
        fails.append(note)
        print("     " + note)
    time.sleep(args.pause)

minutes = (time.time() - start) / 60
busy = minutes * 60 - args.pause * len(selected)  # time without the deliberate pauses
per_q = busy / len(selected)
stages = "\n".join(f"| {name} | {calls} | {total:.0f} s | {avg:.2f} s |" for name, calls, total, avg in summary_rows())
report = (f"# Evaluation: {args.name}\n\n"
          f"- Config: chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}, k={args.k}, "
          f"model={os.getenv('LILIES_MODEL', 'default')}, "
          f"rerank={args.rerank if args.rerank is not None else os.getenv('LILIES_RERANK', '0')}, "
          f"rerank_mode={args.rerank_mode or os.getenv('LILIES_RERANK_MODE', 'always')}, "
          f"quant={os.getenv('LILIES_RERANK_QUANT', '0')}, "
          f"hybrid={args.hybrid if args.hybrid is not None else os.getenv('LILIES_HYBRID', '0')}, "
          f"index={'temporary' if args.fresh else 'chroma_db'}"
          f"{', only=' + args.only if only else ''}\n"
          f"- Answer accuracy (keyword match): **{pct(ans_ok, n_ans)}**\n"
          f"- Source accuracy (right page retrieved): **{pct(page_ok, n_ans)}**\n"
          f"- Refusal accuracy (out-of-file questions): **{pct(refuse_ok, n_ref)}**\n"
          f"- Time: **{per_q:.1f} s per question** (run {minutes:.1f} min, including the {args.pause:g}s pause between questions)\n\n"
          "| # | Question | Type | Answer / Refusal | Source |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n"
          "\n## Time per stage\n\n(retrieve includes rerank_model; a question that fails the first check runs some stages twice)\n\n"
          "| Stage | Calls | Total | Avg per call |\n|---|---|---|---|\n" + stages + "\n"
          + ("\n## Failures\n\n" + "\n".join(f"- {f}" for f in fails) + "\n" if fails else ""))
Path(f"eval/results_{args.name}.md").write_text(report, encoding="utf-8")
print("\n" + report.split("\n\n")[1])
print("\nTime per stage:\n| Stage | Calls | Total | Avg per call |\n" + stages)
print(f"\nsaved: eval/results_{args.name}.md")