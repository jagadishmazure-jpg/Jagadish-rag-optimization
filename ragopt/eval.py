"""Evaluation: retrieval metrics and A/B testing over the gold set.

Every optimization must earn its keep: ABTest runs two pipelines over all
gold questions and reports per-metric deltas plus a winner with margin.
"""

from dataclasses import dataclass


def recall_at_k(relevant: set, ranked_ids: list, k: int) -> float:
    """Fraction of relevant docs present in the top-k."""
    if not relevant:
        return 0.0
    top = set(ranked_ids[:k])
    return len(relevant & top) / len(relevant)


def precision_at_k(relevant: set, ranked_ids: list, k: int) -> float:
    """Fraction of the top-k that is relevant."""
    if k == 0:
        return 0.0
    top = ranked_ids[:k]
    return sum(1 for i in top if i in relevant) / k


def keyword_hit_rate(answer_text: str, expected_keywords: list) -> float:
    """Answer-quality proxy: fraction of expected keywords in the answer."""
    if not expected_keywords:
        return 0.0
    low = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in low)
    return hits / len(expected_keywords)


@dataclass
class ABResult:
    pipeline_a: str
    pipeline_b: str
    metrics: dict   # metric -> {"a": x, "b": y, "delta": y - x}
    winner: str     # pipeline name or "tie"
    margin: float   # winning delta on the primary metric

    def summary(self) -> str:
        lines = [f"A/B: {self.pipeline_a} vs {self.pipeline_b}"]
        for m, v in self.metrics.items():
            lines.append(
                f"  {m}: A={v['a']:.3f} B={v['b']:.3f} delta={v['delta']:+.3f}")
        lines.append(f"  winner: {self.winner} (margin {self.margin:+.3f})")
        return "\n".join(lines)


class ABTest:
    """Runs two RAGPipeline configs over the gold questions."""

    def __init__(self, gold_questions: list, k: int = 5):
        self.gold = gold_questions
        self.k = k

    def _score(self, pipeline) -> dict:
        recalls, precisions, hits, tokens = [], [], [], []
        for g in self.gold:
            # Score the pipeline's final ranking (after rerank), not just the
            # first-stage retriever; otherwise a reranker is never measured.
            ranked = pipeline.ranked(g.question, top_k=self.k)
            ids = [s.doc_id for s in ranked]
            relevant = set(g.relevant_doc_ids)
            recalls.append(recall_at_k(relevant, ids, self.k))
            precisions.append(precision_at_k(relevant, ids, self.k))
            ans = pipeline.run(g.question)
            hits.append(keyword_hit_rate(ans.text, g.expected_keywords))
            tokens.append(ans.tokens_used)
        n = len(self.gold)
        return {
            f"recall@{self.k}": sum(recalls) / n,
            f"precision@{self.k}": sum(precisions) / n,
            "keyword_hit_rate": sum(hits) / n,
            "avg_tokens": sum(tokens) / n,
        }

    def run(self, pipe_a, pipe_b, primary_metric: str = None) -> ABResult:
        primary = primary_metric or f"recall@{self.k}"
        sa = self._score(pipe_a)
        sb = self._score(pipe_b)
        metrics = {
            m: {"a": sa[m], "b": sb[m], "delta": sb[m] - sa[m]}
            for m in sa
        }
        delta = metrics[primary]["delta"]
        if abs(delta) < 1e-9:
            winner, margin = "tie", 0.0
        elif delta > 0:
            winner, margin = pipe_b.name, delta
        else:
            winner, margin = pipe_a.name, -delta
        return ABResult(pipe_a.name, pipe_b.name, metrics, winner, margin)
