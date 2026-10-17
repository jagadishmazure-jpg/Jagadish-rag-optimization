"""Two-stage retrieval: cheap recall, then precise rerank.

Stage 1 retrieves a wide candidate set (n=20) with a fast retriever.
Stage 2 scores each candidate with a richer (simulated) cross-encoder
and keeps the top_k. The cross-encoder here is deterministic —
term overlap plus position features — and is documented as simulated;
in production, swap for a real cross-encoder model.
"""

from .embeddings import tokenize
from .retrieve import ScoredChunk


def simulated_cross_encoder_score(query: str, chunk_text: str) -> float:
    """Deterministic stand-in for a cross-encoder relevance score.

    Features: query-term coverage, early-term position bonus, and a
    length penalty for very long chunks. Returns a 0..1-ish score.
    """
    qterms = set(tokenize(query))
    if not qterms:
        return 0.0
    ctoks = tokenize(chunk_text)
    cset = set(ctoks)
    coverage = len(qterms & cset) / len(qterms)
    # Position bonus: query terms appearing early in the chunk score higher.
    first_hits = []
    for t in qterms & cset:
        try:
            first_hits.append(ctoks.index(t))
        except ValueError:
            pass
    pos_bonus = 0.0
    if first_hits:
        avg_pos = sum(first_hits) / len(first_hits)
        pos_bonus = 1.0 / (1.0 + avg_pos / 20.0)
    # Mild penalty for very long chunks (diluted relevance).
    length_penalty = 1.0 if len(ctoks) <= 120 else 120.0 / len(ctoks)
    return 0.6 * coverage + 0.3 * pos_bonus + 0.1 * length_penalty


class Reranker:
    """Wraps a base retriever with a rerank stage."""

    def __init__(self, base_retriever, top_n: int = 20, top_k: int = 5):
        self.base = base_retriever
        self.top_n = top_n
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int = None) -> list:
        k = top_k or self.top_k
        candidates = self.base.retrieve(query, top_k=self.top_n)
        rescored = [
            ScoredChunk(s.chunk_id, s.doc_id,
                        simulated_cross_encoder_score(query, s.text), s.text)
            for s in candidates
        ]
        rescored.sort(key=lambda s: s.score, reverse=True)
        return rescored[:k]
