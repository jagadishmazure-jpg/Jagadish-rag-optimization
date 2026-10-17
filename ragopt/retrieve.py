"""Retrievers: dense, BM25, and hybrid (reciprocal rank fusion).

All scoring is implemented by hand (stdlib only) so the lab has no
heavy dependencies. BM25 follows the standard Robertson/Spärck-Jones
formulation.
"""

import math
from dataclasses import dataclass

from .embeddings import tokenize


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    score: float
    text: str


def cosine(a: list, b: list) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class DenseRetriever:
    """Cosine similarity over precomputed chunk embeddings."""

    name = "dense"

    def __init__(self, chunks: list, embeddings):
        self.chunks = chunks
        self.embeddings = embeddings
        self._vecs = {c.chunk_id: embeddings.embed(c.text) for c in chunks}

    def retrieve(self, query: str, top_k: int = 10) -> list:
        qv = self.embeddings.embed(query)
        scored = [
            ScoredChunk(c.chunk_id, c.doc_id, cosine(qv, self._vecs[c.chunk_id]), c.text)
            for c in self.chunks
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class BM25Retriever:
    """Real BM25 over chunk term frequencies.

    score(D, Q) = sum over query terms of
        IDF(t) * f(t, D) * (k1 + 1) / (f(t, D) + k1 * (1 - b + b * |D| / avgdl))
    with IDF(t) = ln(1 + (N - n(t) + 0.5) / (n(t) + 0.5)).
    """

    name = "bm25"

    def __init__(self, chunks: list, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._tf = []   # per-chunk term frequency dicts
        self._dl = []   # per-chunk doc lengths
        df = {}
        for c in chunks:
            toks = tokenize(c.text)
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self._tf.append(tf)
            self._dl.append(len(toks))
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self._df = df
        self._N = len(chunks)
        self._avgdl = sum(self._dl) / len(self._dl) if self._dl else 0.0

    def _idf(self, term: str) -> float:
        n = self._df.get(term, 0)
        return math.log(1 + (self._N - n + 0.5) / (n + 0.5))

    def score_chunk(self, query_terms: list, idx: int) -> float:
        score = 0.0
        tf = self._tf[idx]
        dl = self._dl[idx]
        for t in query_terms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            idf = self._idf(t)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
            score += idf * f * (self.k1 + 1) / denom
        return score

    def retrieve(self, query: str, top_k: int = 10) -> list:
        qterms = tokenize(query)
        scored = [
            ScoredChunk(c.chunk_id, c.doc_id,
                        self.score_chunk(qterms, i), c.text)
            for i, c in enumerate(self.chunks)
        ]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


class HybridRetriever:
    """Reciprocal Rank Fusion of dense + BM25 rankings.

    RRF(d) = sum over rankings of 1 / (k + rank(d)), with k=60.
    Fusion needs no score normalization and is robust when the two
    retrievers disagree on scale.
    """

    name = "hybrid"

    def __init__(self, dense: DenseRetriever, bm25: BM25Retriever, k: int = 60):
        self.dense = dense
        self.bm25 = bm25
        self.k = k
        # chunk_id -> ScoredChunk (for text/doc lookup)
        self._by_id = {c.chunk_id: c for c in dense.chunks}

    def retrieve(self, query: str, top_k: int = 10) -> list:
        pool = max(top_k * 3, 50)
        r_dense = self.dense.retrieve(query, top_k=pool)
        r_bm25 = self.bm25.retrieve(query, top_k=pool)
        fused = {}
        for rank, s in enumerate(r_dense, start=1):
            fused[s.chunk_id] = fused.get(s.chunk_id, 0.0) + 1.0 / (self.k + rank)
        for rank, s in enumerate(r_bm25, start=1):
            fused[s.chunk_id] = fused.get(s.chunk_id, 0.0) + 1.0 / (self.k + rank)
        ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        out = []
        for chunk_id, score in ranked[:top_k]:
            c = self._by_id[chunk_id]
            out.append(ScoredChunk(chunk_id, c.doc_id, score, c.text))
        return out
