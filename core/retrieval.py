"""Decides how chunks are fetched: vector or hybrid search, with optional reranking.

mode="always": rerank every retrieval (most accurate, slowest).
mode="retry" : normal fast retrieval first; rerank only on the retry that happens when the grader said no.
               That retry searches with the ORIGINAL question, because the rewritten query can drift away
               from the right chunk (rerank can only reorder candidates, it cannot recover missing ones).
"""
from core.rerank import rerank


def retrieve_docs(store, query, question, k, source=None, n_cand=0, use_hybrid=False, mode="always", tries=0):
    rerank_now = n_cand > k and (mode == "always" or tries >= 1)
    fetch = n_cand if rerank_now else k  # fetch more candidates only when we will rerank them
    search_query = question if (rerank_now and mode == "retry") else query
    if use_hybrid:
        from core.hybrid import hybrid_search  # only needed when hybrid search is switched on
        docs = hybrid_search(store, search_query, fetch, source=source, pool=max(50, fetch))
    else:
        extra = {"filter": {"source": source}} if source else {}
        docs = store.similarity_search(search_query, k=fetch, **extra)
    return rerank(question, docs, k) if rerank_now else docs