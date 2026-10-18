# Optimization Guide

A catalog of every RAG optimization in this lab: what it does, when it
helps, when it hurts, and what it costs. The A/B harness in `eval.py`
exists so none of this is taken on faith.

## 1. Chunking

**What:** how documents are split before embedding.
**Helps when:** retrieval misses answers that span chunk boundaries
(raise overlap), or when chunks mix topics (use paragraph boundaries).
**Hurts when:** overlap is cranked up blindly — more chunks mean more
embeddings, more storage, and a noisier candidate pool.
**Measured here:** `FixedChunker` vs `ParagraphChunker` vs
`SlidingWindowChunker` — swap the chunker in `RAGPipeline` and re-run
the A/B test.

## 2. Hybrid retrieval (dense + BM25, RRF)

**What:** combine semantic (dense) and lexical (BM25) rankings with
reciprocal rank fusion — no score normalization needed.
**Helps when:** queries contain rare exact terms (model numbers, error
codes, names) that dense embeddings smear, or synonyms the keyword
index misses. Each covers the other's blind spot.
**Hurts when:** the corpus is tiny or jargon-free — then it's pure
latency overhead for no gain.
**Measured here:** hybrid (RRF) lifts recall@5 from 0.633 to 0.867 over
the dense baseline, but BM25 alone reaches 1.000 on this corpus, because the
offline "dense" embedding is a hashed bag of words and the gold questions
share exact terms with their documents. With real embeddings the ranking of
these variants may change. Full per-technique table, with paired bootstrap
intervals: `python -m ragopt.ablation` (also in the README).

## 3. Two-stage reranking

**What:** retrieve a wide candidate set cheaply (top-20), then score
precisely and keep top-5.
**Helps when:** the base retriever has decent recall but poor ordering —
reranking fixes ranking without re-indexing anything.
**Hurts when:** stage-1 recall is bad (reranking can't find what was
never retrieved), or when the reranker is so slow it dominates latency.
**Measured here:** `Reranker` wraps any retriever; the simulated
cross-encoder uses term coverage + position features.

## 4. Query rewriting & HyDE

**What:** synonym expansion attacks vocabulary mismatch ("cost" vs
"price"); HyDE embeds a hypothetical answer instead of the raw query,
pulling retrieval toward answer-shaped text.
**Helps when:** users phrase queries differently from the documents.
**Hurts when:** expansion adds noisy terms that drag in irrelevant
chunks — always A/B it, never assume.
**Measured here:** `QueryRewriter` (built-in synonym map) and `HyDE`
(deterministic template; swap for LLM generation in production).

## 5. Context compression

**What:** pack only the top chunks that fit a token budget; prune
marginal chunks below a score threshold.
**Helps when:** you're paying per token or hitting context limits —
this is the cheapest win in the whole catalog.
**Hurts when:** the budget is set below what the question needs, or the
pruning threshold is tuned on too few examples.
**Measured here:** `ContextCompressor.compress` reports `tokens_saved`
per query; the pipeline test asserts `tokens_used <= budget`.

## 6. Embedding caching

**What:** cache query/document embeddings; repeated questions skip
recomputation.
**Helps when:** query traffic has repeats (it always does in
production) or documents are re-indexed often.
**Hurts when:** the cache grows unbounded — pair with eviction in
production.
**Measured here:** `CachedEmbeddings` tracks hits/misses/hit-rate.

## The rule

No optimization ships on theory. Run `ABTest`, read the delta on
`recall@k` **and** the delta on `avg_tokens`, and keep only what earns
its keep on both.
