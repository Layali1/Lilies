"""Hybrid retrieval: BM25 (exact words) + vector search (meaning), merged with Reciprocal Rank Fusion.

BM25 is written here directly (no extra package). It only matches words that literally appear in the
chunk, so it helps with names, numbers and rare terms, and it cannot match across languages.
"""
import math
import re
from collections import Counter, defaultdict

from core.utils import clean

AR_DIGITS = {0x0660 + i: str(i) for i in range(10)} | {0x06F0 + i: str(i) for i in range(10)}
_MARKS = re.compile(r"[\u064B-\u065F\u0670\u0640]")  # Arabic diacritics and tatweel


def _norm_word(w: str) -> str:
    w = _MARKS.sub("", w.translate(AR_DIGITS).lower())
    return re.sub("[أإآ]", "ا", w).replace("ى", "ي").replace("ة", "ه")


_STOP_RAW = ("a an the of in on to and or is are was were what which who how does do did for with by at from that "
             "this it as be many much من في على إلى عن ما ماذا هل كم هو هي التي الذي أن إن حسب").split()
STOP = {_norm_word(w) for w in _STOP_RAW}


def tokenize(text: str) -> list:
    out = []
    text = _MARKS.sub("", clean(text))  # strip diacritics first: they would otherwise split words apart
    for w in re.findall(r"\w+", text):
        w = _norm_word(w)
        if not w or w in STOP:
            continue
        if w.startswith("ال") and len(w) > 4:  # light stemming: drop the Arabic definite article
            w = w[2:]
        out.append(w)
    return out


class BM25:
    def __init__(self, docs: list, k1: float = 1.5, b: float = 0.75):
        self.docs, self.k1, self.b = docs, k1, b
        self.lengths, self.index = [], defaultdict(list)  # index: term -> [(doc number, term count)]
        for i, d in enumerate(docs):
            tokens = tokenize(d.page_content)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.index[term].append((i, tf))
        self.avg = sum(self.lengths) / len(self.lengths) if self.lengths else 0

    def search(self, query: str, k: int) -> list:
        n, scores = len(self.docs), defaultdict(float)
        for term in set(tokenize(query)):
            postings = self.index.get(term)
            if not postings:
                continue
            idf = math.log((n - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for i, tf in postings:
                norm = 1 - self.b + self.b * self.lengths[i] / self.avg
                scores[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return [self.docs[i] for i in sorted(scores, key=scores.get, reverse=True)[:k]]


_cache = {}


def _bm25_for(store, source):
    where = {"source": source} if source else None
    n_ids = len(store.get(where=where, include=["metadatas"])["ids"])  # changes when files are added
    key = (id(store), source, n_ids)
    if key not in _cache:
        from langchain_core.documents import Document
        got = store.get(where=where, include=["documents", "metadatas"])
        docs = [Document(page_content=t, metadata=m) for t, m in zip(got["documents"], got["metadatas"])]
        _cache.clear()  # keep one index in memory
        _cache[key] = BM25(docs)
    return _cache[key]


def hybrid_search(store, query: str, k: int, source=None, pool: int = 30, rrf_k: int = 60) -> list:
    """Top `pool` from vector search and from BM25, merged by Reciprocal Rank Fusion, best `k` returned."""
    extra = {"filter": {"source": source}} if source else {}
    rankings = [store.similarity_search(query, k=pool, **extra), _bm25_for(store, source).search(query, pool)]
    scores, docs = defaultdict(float), {}
    for ranking in rankings:
        for rank, d in enumerate(ranking, 1):
            key = (d.metadata.get("page"), d.page_content)
            docs[key] = d
            scores[key] += 1 / (rrf_k + rank)
    return [docs[key] for key in sorted(scores, key=scores.get, reverse=True)[:k]]