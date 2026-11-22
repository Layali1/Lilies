"""Sanity checks for Lilies: the basics that must always hold.

  python sanity_check.py --offline     fast, no heavy packages, no internet
  python sanity_check.py               + embeddings and Chroma (packages installed)
  python sanity_check.py --full        + LLM end-to-end through Groq (needs GROQ_API_KEY)
  python sanity_check.py --pdf f.pdf   + checks your own PDF
  python sanity_check.py --rerank      + reranker check (downloads the reranker model)

Think of it like asserting that softmax weights sum to 1: cheap checks that catch silent bugs.
"""
import argparse
import math
import os
import sys
import tempfile

CHECKS, _cache = [], {}
PDF_PATH = None


class Skip(Exception):
    pass


def check(group, name):
    def deco(fn):
        CHECKS.append((group, name, fn))
        return fn
    return deco


def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(v): return math.sqrt(sum(x * x for x in v))
def close(a, b, tol=1e-3): return all(abs(x - y) <= tol for x, y in zip(a, b))


# ---------------------------------------------------------------- offline
@check("offline", "clean: Arabic presentation forms -> normal letters")
def _():
    from core.utils import clean
    out = clean("\ufedf\ufee0\ufee3\ufea9\ufee3\ufe8e\ufe97")
    assert all(not (0xFB50 <= ord(c) <= 0xFEFF) for c in out), f"forms left: {out!r}"
    assert "\u0644" in out, f"expected letter lam in {out!r}"


@check("offline", "clean: removes invisible marks and extra whitespace")
def _():
    from core.utils import clean
    out = clean("a\u200f\u202b  b\n\n\n\nc")
    assert out == "a b\n\nc", repr(out)


@check("offline", "detect_language: clear Arabic / English")
def _():
    from core.utils import detect_language
    assert detect_language("كم مدة الإجازة السنوية؟") == "ar"
    assert detect_language("How many leave days do I get?") == "en"
    assert detect_language("ايش معنى RAG؟") == "ar"


@check("offline", "question pasted from a PDF (presentation forms) is detected as Arabic after clean")
def _():
    from core.utils import clean, detect_language
    q = " ".join(["\ufedf\ufee0\ufee3\ufea9\ufee3\ufe8e\ufe97"] * 3)
    assert detect_language(clean(q)) == "ar"


@check("offline", "route: relevant -> answer, weak -> rewrite once, then not_found")
def _():
    from core.utils import route_after_grade as r
    assert r({"relevant": True, "tries": 0}) == "answer"
    assert r({"relevant": True, "tries": 1}) == "answer"
    assert r({"relevant": False, "tries": 0}) == "rewrite"
    assert r({"relevant": False, "tries": 1}) == "not_found"


@check("offline", "graph loop always terminates (grade never relevant)")
def _():
    from core.utils import route_after_grade as r, MAX_REWRITES
    state, steps = {"relevant": False, "tries": 0}, 0
    while True:
        steps += 1
        assert steps <= 10, "infinite loop"
        nxt = r(state)
        if nxt == "rewrite":
            state["tries"] += 1
        else:
            break
    assert nxt == "not_found" and state["tries"] == MAX_REWRITES, (nxt, state)


@check("offline", "cite: plain page, or page + printed label when they differ")
def _():
    from core.utils import cite
    assert cite({"page": 3}) == "p.3"
    assert cite({"page": 3, "label": "3"}) == "p.3"
    assert cite({"page": 341, "label": "317"}) == "p.341 (printed 317)"


@check("offline", "not-found messages exist in both languages")
def _():
    from core.utils import NOT_FOUND
    assert set(NOT_FOUND) == {"ar", "en"} and all(NOT_FOUND.values())
    assert any("\u0600" <= c <= "\u06FF" for c in NOT_FOUND["ar"])
    assert not any("\u0600" <= c <= "\u06FF" for c in NOT_FOUND["en"])


# ---------------------------------------------------------------- libs
def emb():
    if "emb" not in _cache:
        from core.ingest import make_embeddings
        _cache["emb"] = make_embeddings()
    return _cache["emb"]


DOCS = [("تستحق الموظفة إجازة سنوية مدتها 30 يوماً مدفوعة الأجر.", "hr.pdf", 1),
        ("Employees must submit expense reports within 14 days of travel.", "finance.pdf", 2),
        ("يبدأ الدوام الرسمي من الساعة الثامنة صباحاً وينتهي الرابعة عصراً.", "hr.pdf", 3)]


def tiny_store():
    if "store" not in _cache:
        from core.ingest import get_store
        from langchain_core.documents import Document
        s = get_store(tempfile.mkdtemp(), emb())
        s.add_documents([Document(page_content=t, metadata={"source": f, "page": p}) for t, f, p in DOCS],
                        ids=[f"d{i}" for i in range(len(DOCS))])
        _cache["store"] = s
    return _cache["store"]


@check("libs", "chunking: size limit, metadata kept, no text lost")
def _():
    from core.ingest import split_pages, CHUNK_SIZE
    from langchain_core.documents import Document
    n = 120
    text = " ".join(f"جملة رقم {i}." for i in range(n))
    chunks = split_pages([Document(page_content=text, metadata={"source": "x.pdf", "page": 7})])
    assert len(chunks) > 1, "text was not split"
    assert all(0 < len(c.page_content) <= CHUNK_SIZE for c in chunks), "chunk empty or too long"
    assert all(c.metadata == {"source": "x.pdf", "page": 7} for c in chunks), "metadata lost"
    joined = " ".join(c.page_content for c in chunks)
    missing = [i for i in range(n) if f"جملة رقم {i}." not in joined]
    assert not missing, f"lost sentences: {missing[:5]}"


@check("libs", "e5 prefixes: query uses 'query:' only, documents use 'passage:'")
def _():
    from langchain_huggingface import HuggingFaceEmbeddings as Base
    e, t = emb(), "كم مدة الإجازة؟"
    q = e.embed_query(t)
    assert close(q, Base.embed_documents(e, ["query: " + t])[0]), "query prefix wrong (double prefix?)"
    assert not close(q, Base.embed_documents(e, [t])[0]), "query prefix missing"
    assert close(e.embed_documents([t])[0], Base.embed_documents(e, ["passage: " + t])[0]), "passage prefix wrong"


@check("libs", "embeddings: dimension 384 and unit length (norm = 1)")
def _():
    from core.ingest import EMBED_DIM
    vecs = emb().embed_documents(["إجازة سنوية", "annual leave", "x"]) + [emb().embed_query("سؤال")]
    assert all(len(v) == EMBED_DIM for v in vecs), [len(v) for v in vecs]
    assert all(abs(norm(v) - 1) < 1e-3 for v in vecs), [round(norm(v), 4) for v in vecs]


@check("libs", "embeddings: deterministic (same text -> same vector)")
def _():
    a, b = emb().embed_query("اختبار"), emb().embed_query("اختبار")
    assert close(a, b, 1e-6)


@check("libs", "embeddings: Arabic meaning ~ English meaning > unrelated")
def _():
    ar, en, other = emb().embed_documents(
        ["الموظف يحصل على إجازة سنوية", "The employee gets annual leave", "The recipe needs two cups of rice"])
    s_same, s_other = dot(ar, en), dot(ar, other)
    assert s_same > s_other + 0.03, f"same-meaning {s_same:.3f} vs unrelated {s_other:.3f}"
    return f"same={s_same:.3f} unrelated={s_other:.3f}"


@check("libs", "retrieval: top result is the right file/page (AR, EN, cross-language)")
def _():
    cases = [("كم مدة الإجازة السنوية؟", "hr.pdf", 1),
             ("How many days to submit expense reports?", "finance.pdf", 2),
             ("How long is the annual leave?", "hr.pdf", 1),
             ("متى يبدأ الدوام؟", "hr.pdf", 3)]
    bad = []
    for q, f, p in cases:
        m = tiny_store().similarity_search(q, k=1)[0].metadata
        if (m["source"], m["page"]) != (f, p):
            bad.append(f"{q!r} -> {m}")
    assert not bad, "; ".join(bad)


@check("libs", "retrieval: returns k results with content + source + page")
def _():
    res = tiny_store().similarity_search("إجازة", k=3)
    assert len(res) == 3
    for d in res:
        assert d.page_content.strip() and d.metadata.get("source")
        assert isinstance(d.metadata.get("page"), int) and d.metadata["page"] >= 1


@check("libs", "source filter: only the chosen file's chunks come back")
def _():
    res = tiny_store().similarity_search("إجازة", k=3, filter={"source": "finance.pdf"})
    assert res and all(d.metadata["source"] == "finance.pdf" for d in res), [d.metadata for d in res]


@check("libs", "re-adding the same chunks does not duplicate them")
def _():
    from core.ingest import get_store, add_chunks
    from langchain_core.documents import Document
    store = get_store(tempfile.mkdtemp(), emb())
    chunks = [Document(page_content=t, metadata={"source": f, "page": p}) for t, f, p in DOCS]
    add_chunks(store, chunks)
    first = len(store.get()["ids"])
    add_chunks(store, chunks)
    second = len(store.get()["ids"])
    assert first == second == len(DOCS), f"before={first} after re-add={second}"


# ---------------------------------------------------------------- full (LLM)
def graph():
    if "graph" not in _cache:
        if not os.getenv("GROQ_API_KEY"):
            raise Skip("GROQ_API_KEY is not set")
        from core.graph import build_graph
        _cache["graph"] = build_graph(tiny_store())
    return _cache["graph"]


@check("full", "graph: Arabic in-document question -> relevant, grounded answer")
def _():
    out = graph().invoke({"question": "كم مدة الإجازة السنوية؟"})
    assert out["relevant"] is True and out["answer"].strip()
    assert any(d.metadata["source"] == "hr.pdf" for d in out["docs"]), "wrong sources"
    assert "30" in out["answer"], f"expected 30 in answer: {out['answer'][:120]!r}"


@check("full", "graph: English in-document question -> relevant, grounded answer")
def _():
    out = graph().invoke({"question": "Within how many days must employees submit expense reports?"})
    assert out["relevant"] is True, "grader said NOT relevant for an English in-document question"
    assert "14" in out["answer"], out["answer"][:120]


@check("full", "graph: out-of-document question -> refuses (AR and EN)")
def _():
    from core.utils import NOT_FOUND
    for q, lang in [("ما هي طريقة عمل الكبسة؟", "ar"), ("What is the capital of Japan?", "en")]:
        out = graph().invoke({"question": q})
        assert out["relevant"] is False and out["tries"] == 1, (q, out.get("relevant"), out.get("tries"))
        assert out["answer"] == NOT_FOUND[lang], (q, out["answer"][:80])


# ---------------------------------------------------------------- rerank (downloads ~2 GB the first time)
@check("rerank", "reranker puts the right passage first (Arabic and English)")
def _():
    from core.rerank import rerank
    from langchain_core.documents import Document
    docs = [Document(page_content=t, metadata={"source": f, "page": p}) for t, f, p in DOCS]
    for q, want in [("كم مدة الإجازة السنوية؟", ("hr.pdf", 1)),
                    ("Within how many days must employees submit expense reports?", ("finance.pdf", 2))]:
        top = rerank(q, docs, 2)
        got = (top[0].metadata["source"], top[0].metadata["page"])
        assert len(top) == 2 and got == want, (q, got)


# ---------------------------------------------------------------- your PDF
@check("pdf", "your PDF: has text, pages recorded, no leftover presentation forms")
def _():
    from core.ingest import get_store, add_pdf
    store = get_store(tempfile.mkdtemp(), emb())
    n = add_pdf(store, PDF_PATH, os.path.basename(PDF_PATH))
    assert n > 0, "no text extracted (scanned PDF? it needs OCR)"
    data = store.get()
    assert all(isinstance(m["page"], int) and m["page"] >= 1 for m in data["metadatas"])
    left = sum(1 for t in data["documents"] for c in t if 0xFB50 <= ord(c) <= 0xFEFF)
    assert left == 0, f"{left} presentation-form characters left"
    return f"{n} chunks from {len({m['page'] for m in data['metadatas']})} pages"


def main():
    global PDF_PATH
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--pdf")
    ap.add_argument("--rerank", action="store_true")
    args = ap.parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
    except ImportError:
        pass
    PDF_PATH = args.pdf
    groups = {"offline"} if args.offline else {"offline", "libs"}
    if args.full and not args.offline:
        groups.add("full")
    if args.pdf:
        groups.add("pdf")
    if args.rerank:
        groups.add("rerank")
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}
    for group, name, fn in CHECKS:
        if group not in groups:
            continue
        try:
            note = fn()
            status, msg = "PASS", note or ""
        except Skip as e:
            status, msg = "SKIP", str(e)
        except ImportError as e:
            status, msg = "SKIP", f"missing package: {e.name}"
        except AssertionError as e:
            status, msg = "FAIL", str(e)
        except Exception as e:  # unexpected error is also a failure
            status, msg = "FAIL", f"{type(e).__name__}: {e}"
        counts[status] += 1
        print(f"[{status}] {name}" + (f"  ->  {msg}" if msg else ""))
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()