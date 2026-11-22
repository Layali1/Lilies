"""Optional reranking: retrieve many candidates, let a cross-encoder re-order them, keep the best few."""
import os

from core.timing import stage

RERANK_MODEL = os.getenv("LILIES_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # multilingual (incl. Arabic)
_model = None


def _load():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL, max_length=512)


def get_reranker():
    global _model
    if _model is None:  # loaded once, only when reranking is actually used
        model = _load()
        if os.getenv("LILIES_RERANK_QUANT", "0") == "1":  # int8 weights: usually faster on CPU, slightly less exact
            try:
                import torch
                from torch.ao.quantization import quantize_dynamic
                # in place on the whole model, so its structure (and how sentence-transformers calls it) stays the same
                quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)
                model.predict([("test", "test")])  # smoke test now, not in the middle of a question
                print("[rerank] int8 quantization enabled")
            except Exception as e:  # never break the app because of an optimization
                print(f"[rerank] quantization failed ({type(e).__name__}), using the normal model")
                model = _load()
        _model = model
    return _model


def rerank(query: str, docs: list, top_n: int) -> list:
    if len(docs) <= 1:
        return docs
    model = get_reranker()
    with stage("rerank_model"):
        scores = model.predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(scores, docs), key=lambda pair: pair[0], reverse=True)
    return [d for _, d in ranked[:top_n]]