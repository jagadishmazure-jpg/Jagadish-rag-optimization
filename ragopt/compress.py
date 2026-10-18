"""Context compression: fit retrieved chunks into a token budget.

Every token sent to the model costs money and latency. Compression keeps
the highest-scoring chunks that fit the budget and prunes marginal ones,
reporting exactly how many tokens were saved.
"""



def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return max(1, len(text) // 4)


class ContextCompressor:
    """Budget-aware chunk selection and marginal-chunk pruning."""

    def __init__(self, token_budget: int = 800):
        self.token_budget = token_budget

    def compress(self, scored_chunks: list) -> dict:
        """Keep top chunks (already ranked) that fit the token budget.

        Returns dict with kept chunks, tokens_used, tokens_saved
        (vs. sending all input chunks), and count dropped.
        """
        total_all = sum(estimate_tokens(s.text) for s in scored_chunks)
        kept = []
        used = 0
        for s in scored_chunks:
            t = estimate_tokens(s.text)
            if used + t <= self.token_budget:
                kept.append(s)
                used += t
        return {
            "chunks": kept,
            "tokens_used": used,
            "tokens_saved": total_all - used,
            "dropped": len(scored_chunks) - len(kept),
        }

    def prune_marginal(self, scored_chunks: list, score_threshold: float) -> list:
        """Drop chunks scoring below the threshold (likely irrelevant)."""
        return [s for s in scored_chunks if s.score >= score_threshold]
