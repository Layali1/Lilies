"""Verifies the ground truth in eval/questions.json against your real PDFs.

  python -m eval.check_questions "AI Engineering.pdf" "AINewsletter27Aug.pdf"

For every answerable question it checks that the 'evidence' text really appears on its 'page'
(page = position in the PDF file, starting at 1). Run it before trusting any evaluation numbers.
"""
import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

from core.utils import clean

squash = lambda t: re.sub(r"\s+", "", clean(t)).lower()   # ignore spaces/line breaks/case

readers = {Path(p).name: PdfReader(p) for p in sys.argv[1:]}
cache = {}


def page_text(name, page):  # page is 1-based
    key = (name, page)
    if key not in cache:
        cache[key] = squash(readers[name].pages[page - 1].extract_text() or "")
    return cache[key]


cases = json.loads(Path("eval/questions.json").read_text(encoding="utf-8"))
bad = 0
for n, c in enumerate(cases, 1):
    if not c.get("answerable", True):
        continue
    name, page = c["source"], c["page"]
    if name not in readers:
        print(f"[SKIP ] Q{n}: {name} was not given on the command line")
        continue
    if page > len(readers[name].pages):
        bad += 1
        print(f"[CHECK] Q{n}: {name} has only {len(readers[name].pages)} pages, question says {page}")
        continue
    if all(squash(e) in page_text(name, page) for e in c["evidence"]):
        label = readers[name].page_labels[page - 1]
        print(f"[ OK  ] Q{n}: {name} p.{page} (printed {label})")
    else:
        bad += 1
        near = [p for p in range(max(1, page - 3), min(len(readers[name].pages), page + 3) + 1)
                if all(squash(e) in page_text(name, p) for e in c["evidence"])]
        print(f"[CHECK] Q{n}: evidence {c['evidence']} not on {name} p.{page}"
              + (f" -> found on page(s) {near}" if near else " -> not found within +-3 pages"))
print(f"\n{bad} question(s) need a look" if bad else "\nall ground-truth pages verified")
sys.exit(1 if bad else 0)