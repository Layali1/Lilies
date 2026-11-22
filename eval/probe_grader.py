"""Shows exactly what the relevance grader says for one evaluation question.

  python -m eval.probe_grader 9                              (same settings as the graph)
  python -m eval.probe_grader 9 --chars 800                  (show the grader whole chunks)
  python -m eval.probe_grader 9 --max-tokens 1500            (give the model room to think)
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from core.ingest import get_store

load_dotenv(override=True)
ap = argparse.ArgumentParser()
ap.add_argument("n", type=int, help="question number, as in the results table (Q9 -> 9)")
ap.add_argument("--chars", type=int, default=500, help="characters of each chunk shown to the grader")
ap.add_argument("--max-tokens", type=int, default=300)
args = ap.parse_args()

c = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))[args.n - 1]
store = get_store("chroma_db")
docs = store.similarity_search(c["question"], k=5, filter={"source": c["source"]})
ctx = "\n---\n".join(d.page_content[:args.chars] for d in docs)
prompt = (f"Question: {c['question']}\n\nPassages:\n{ctx}\n\n"
          "Do the passages contain information that answers the question? Reply yes or no only.")
grader = ChatGroq(model=os.getenv("LILIES_GRADER_MODEL", "openai/gpt-oss-20b"),
                  temperature=0, max_tokens=args.max_tokens)
r = grader.invoke(prompt)

evidence = [e for e in c.get("evidence", [])]
shown = " ".join(d.page_content[:args.chars] for d in docs).replace("\n", " ")
print(f"Q{args.n}: {c['question']}")
print(f"retrieved pages: {[d.metadata['page'] for d in docs]} (expected {c.get('page', '-')})")
print(f"evidence {evidence} visible to the grader: {[e.lower() in shown.lower() for e in evidence]}")
print(f"settings: chars={args.chars}, max_tokens={args.max_tokens}")
print(f"\ngrader said: {r.content!r}")
print(f"finish_reason: {r.response_metadata.get('finish_reason')} | tokens: {r.response_metadata.get('token_usage')}")