"""Per-technique ablation: change ONE thing from the baseline and measure it.

The A/B harness compares two whole pipelines; when the optimised pipeline
stacks several techniques you cannot tell which one earned the gain. This
module adds one technique at a time to the same baseline (fixed 200/50
chunks + dense retrieval, top 5) and reports the change, with a paired
bootstrap 95% confidence interval on the recall@5 delta.

Sample-size warning: the gold set has 15 questions. Intervals are wide, and a
delta whose interval includes 0 is not evidence of a real gain.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from .chunking import FixedChunker, ParagraphChunker, SlidingWindowChunker
from .compress import ContextCompressor
from .corpus import build_corpus, build_gold_questions
from .embeddings import CachedEmbeddings, FakeEmbeddings
from .eval import keyword_hit_rate, recall_at_k
from .pipeline import RAGPipeline
from .rerank import Reranker
from .retrieve import BM25Retriever, DenseRetriever, HybridRetriever, ScoredChunk, cosine
from .rewrite import HyDE, QueryRewriter

K = 5


class RewritingRetriever:
    """Expands the query with synonyms, then calls the wrapped retriever."""

    def __init__(self, base, rewriter: QueryRewriter):
        self.base, self.rewriter = base, rewriter

    def retrieve(self, query: str, top_k: int = 10) -> list:
        return self.base.retrieve(self.rewriter.rewrite(query), top_k=top_k)


class HyDERetriever:
    """Embeds a hypothetical answer instead of the query (dense only)."""

    def __init__(self, dense: DenseRetriever, hyde: HyDE):
        self.dense, self.hyde = dense, hyde

    def retrieve(self, query: str, top_k: int = 10) -> list:
        qv = self.hyde.embed(query)
        scored = [ScoredChunk(c.chunk_id, c.doc_id, cosine(qv, self.dense._vecs[c.chunk_id]), c.text)
                  for c in self.dense.chunks]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:top_k]


def _chunks(chunker, docs):
    out = []
    for d in docs:
        out += chunker.chunk(d.doc_id, d.text)
    return out


def build_variants(docs) -> list[RAGPipeline]:
    emb = CachedEmbeddings(FakeEmbeddings())
    fixed = FixedChunker(200, 50)
    base_chunks = _chunks(fixed, docs)
    dense = DenseRetriever(base_chunks, emb)
    bm25 = BM25Retriever(base_chunks)
    hybrid = HybridRetriever(dense, bm25)
    para, slide = ParagraphChunker(400), SlidingWindowChunker(60, 15)
    return [
        RAGPipeline("baseline (fixed 200/50 + dense)", fixed, dense),
        RAGPipeline("chunking: paragraph", para, DenseRetriever(_chunks(para, docs), emb)),
        RAGPipeline("chunking: sliding 60/15", slide, DenseRetriever(_chunks(slide, docs), emb)),
        RAGPipeline("retrieval: BM25 only", fixed, bm25),
        RAGPipeline("retrieval: hybrid (RRF)", fixed, hybrid),
        RAGPipeline("rerank on dense", fixed, dense, reranker=Reranker(dense, top_n=20, top_k=K)),
        RAGPipeline("query rewrite (synonyms)", fixed, RewritingRetriever(dense, QueryRewriter())),
        RAGPipeline("HyDE", fixed, HyDERetriever(dense, HyDE(emb))),
        RAGPipeline("compression (600-token budget)", fixed, dense,
                    compressor=ContextCompressor(token_budget=600)),
        RAGPipeline("all: hybrid + rerank + compression", fixed, hybrid,
                    reranker=Reranker(hybrid, top_n=20, top_k=K),
                    compressor=ContextCompressor(token_budget=600)),
    ]


def per_question(pipe: RAGPipeline, gold) -> dict:
    rec, hit, tok = [], [], []
    for g in gold:
        ids = [s.doc_id for s in pipe.ranked(g.question, top_k=K)]
        rec.append(recall_at_k(set(g.relevant_doc_ids), ids, K))
        ans = pipe.run(g.question)
        hit.append(keyword_hit_rate(ans.text, g.expected_keywords))
        tok.append(ans.tokens_used)
    return {"recall": rec, "hit": hit, "tokens": tok}


def bootstrap_ci(deltas: list[float], n_boot: int = 2000, seed: int = 7) -> tuple[float, float]:
    """Paired bootstrap 95% CI of the mean delta (deterministic seed)."""
    rng = random.Random(seed)
    n = len(deltas)
    means = sorted(sum(rng.choice(deltas) for _ in range(n)) / n for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


@dataclass
class AblationRow:
    name: str
    recall: float
    delta: float
    ci: tuple[float, float]
    keyword_hit: float
    avg_tokens: float


def run_ablation(docs=None, gold=None) -> list[AblationRow]:
    docs = docs or build_corpus()
    gold = gold or build_gold_questions()
    variants = build_variants(docs)
    base = per_question(variants[0], gold)
    rows = []
    n = len(gold)
    for pipe in variants:
        q = base if pipe is variants[0] else per_question(pipe, gold)
        deltas = [b - a for a, b in zip(base["recall"], q["recall"])]
        rows.append(AblationRow(pipe.name, sum(q["recall"]) / n, sum(deltas) / n,
                                bootstrap_ci(deltas), sum(q["hit"]) / n, sum(q["tokens"]) / n))
    return rows


def markdown(rows: list[AblationRow], n_questions: int) -> str:
    lines = [f"| Variant (one change from baseline) | recall@{K} | delta | 95% CI of delta | "
             "keyword hit | avg tokens |", "|---|---:|---:|---|---:|---:|"]
    for r in rows:
        sig = "" if r.ci[0] > 0 or r.ci[1] < 0 or r.delta == 0 else " (includes 0)"
        lines.append(f"| {r.name} | {r.recall:.3f} | {r.delta:+.3f} | "
                     f"[{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]{sig} | {r.keyword_hit:.3f} | {r.avg_tokens:.1f} |")
    lines.append(f"\n{n_questions} gold questions; intervals are paired bootstrap (2,000 resamples, seed 7).")
    return "\n".join(lines)


if __name__ == "__main__":
    gold = build_gold_questions()
    print(markdown(run_ablation(gold=gold), len(gold)))
