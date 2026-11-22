
"""LangGraph pipeline: detect language -> retrieve -> grade -> (rewrite -> retrieve) -> answer."""

import os
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

from core.retrieval import retrieve_docs
from core.timing import stage
from core.utils import NOT_FOUND, cite, clean, detect_language, route_after_grade


class State(TypedDict, total=False):
    question: str
    query: str
    lang: str
    docs: list
    relevant: bool
    tries: int
    answer: str


def build_graph(store, k: int = 5, source=None, rerank_candidates=None, hybrid=None, rerank_mode=None):

    # how many chunks to fetch before reranking (0 = no reranking)
    n_cand = int(os.getenv("LILIES_RERANK", "0")) if rerank_candidates is None else rerank_candidates

    # "always" = rerank every retrieval, "retry" = only after the grader said no
    mode = os.getenv("LILIES_RERANK_MODE", "always") if rerank_mode is None else rerank_mode

    # hybrid search = vector search + BM25 keyword search
    use_hybrid = (os.getenv("LILIES_HYBRID", "0") == "1") if hybrid is None else bool(hybrid)

    llm = ChatGroq(
        model=os.getenv("LILIES_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        max_tokens=2000
    )

    grader = ChatGroq(
        model=os.getenv("LILIES_GRADER_MODEL", "openai/gpt-oss-20b"),
        temperature=0,
        max_tokens=1500
    )

    def detect_lang(s: State):
        q = clean(s["question"])
        return {
            "question": q,
            "lang": detect_language(q),
            "query": q,
            "tries": 0
        }

    def retrieve(s: State):
        with stage("retrieve"):
            docs = retrieve_docs(
                store,
                s["query"],
                s["question"],
                k,
                source=source,
                n_cand=n_cand,
                use_hybrid=use_hybrid,
                mode=mode,
                tries=s["tries"]
            )
        return {"docs": docs}

    def grade(s: State):
        ctx = "\n---\n".join(d.page_content for d in s["docs"])

        with stage("grade"):
            r = grader.invoke(
                f"Question: {s['question']}\n\n"
                f"Passages:\n{ctx}\n\n"
                "Do the passages contain information that answers the question? "
                "Reply yes or no only."
            )

        return {"relevant": r.content.strip().lower().startswith("yes")}

    def rewrite(s: State):
        with stage("rewrite"):
            r = llm.invoke(
                "Rewrite this question as a short search query with key terms, "
                "same language, no explanation:\n"
                f"{s['question']}"
            )

        return {
            "query": r.content.strip(),
            "tries": s["tries"] + 1
        }

    def answer(s: State):
        ctx = "\n\n".join(
            f"[{d.metadata['source']} {cite(d.metadata)}]\n{d.page_content}"
            for d in s["docs"]
        )

        with stage("answer"):
            r = llm.invoke(
                "Answer ONLY from the context below, in the same language as the question. "
                "Cite sources like [file p.N]. If the context is insufficient, say so.\n\n"
                f"Context:\n{ctx}\n\n"
                f"Question: {s['question']}"
            )

        return {"answer": r.content}

    def not_found(s: State):
        return {"answer": NOT_FOUND[s["lang"]]}

    g = StateGraph(State)

    for name, fn in [
        ("detect_lang", detect_lang),
        ("retrieve", retrieve),
        ("grade", grade),
        ("rewrite", rewrite),
        ("answer", answer),
        ("not_found", not_found)
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "detect_lang")
    g.add_edge("detect_lang", "retrieve")
    g.add_edge("retrieve", "grade")

    g.add_conditional_edges(
        "grade",
        route_after_grade,
        ["answer", "rewrite", "not_found"]
    )

    g.add_edge("rewrite", "retrieve")
    g.add_edge("answer", END)
    g.add_edge("not_found", END)

    return g.compile()