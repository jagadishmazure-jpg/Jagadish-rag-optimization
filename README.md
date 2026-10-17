# Jagadish-rag-optimization

A measurement lab for RAG optimizations. Every technique — chunking, hybrid
retrieval, reranking, query rewriting, HyDE, context compression, caching —
is A/B tested against a golden eval set, so each optimization must **earn its
keep** with measured recall/precision gains, not claims.

All data is synthetic (fictional Acme Robotics corpus). Everything is
deterministic and offline: no API keys, no network, no GPUs.

## The problem

RAG quality and RAG cost are set long before the LLM is called. Chunk
boundaries decide what can be retrieved; the retriever decides what is found;
context packing decides what the model sees and what you pay for. Each stage
has a catalog of optimizations — and each one trades something. This repo
measures the trades.

## Quickstart

```bash
python -m pytest tests/ -q
python - <<'EOF'
from ragopt import *

docs = build_corpus()
gold = build_gold_questions()

chunks = []
for d in docs:
    chunks += FixedChunker(200, 50).chunk(d.doc_id, d.text)

emb = CachedEmbeddings(FakeEmbeddings())
dense = DenseRetriever(chunks, emb)
hybrid = HybridRetriever(dense, BM25Retriever(chunks))

base = RAGPipeline("baseline", FixedChunker(200, 50), dense)
opt  = RAGPipeline("hybrid+rerank", FixedChunker(200, 50), hybrid,
                   reranker=Reranker(hybrid, top_n=20, top_k=5),
                   compressor=ContextCompressor(token_budget=600))

print(ABTest(gold, k=5).run(base, opt).summary())
print("embedding cache:", emb.stats())
EOF
```

On the bundled corpus the optimized pipeline beats the dense baseline by
**+0.300 recall@5** (0.567 → 0.867) while the compressor keeps every answer
inside its token budget.

## What's inside

| Module | Optimization |
|---|---|
| `chunking.py` | Fixed / paragraph / sliding-window chunkers |
| `embeddings.py` | Deterministic fake embeddings + cache with hit/miss stats |
| `retrieve.py` | Dense (cosine), real BM25, hybrid via reciprocal rank fusion |
| `rerank.py` | Two-stage retrieve-then-rerank (simulated cross-encoder) |
| `rewrite.py` | Synonym query expansion, HyDE |
| `compress.py` | Token-budget packing, marginal-chunk pruning |
| `pipeline.py` | Composable RAG pipeline with citations |
| `eval.py` | recall@k, precision@k, keyword hit rate, A/B harness |
| `corpus.py` | 30 synthetic docs, 15 gold questions |

See [`docs/optimization-guide.md`](docs/optimization-guide.md) for the
technique catalog: when each optimization helps, when it hurts, and what it
costs.

## How to read an A/B report

```
A/B: baseline vs hybrid+rerank
  recall@5: A=0.567 B=0.867 delta=+0.300
  precision@5: A=0.160 B=0.253 delta=+0.093
  keyword_hit_rate: A=0.233 B=0.833 delta=+0.600
  avg_tokens: A=189.067 B=211.133 delta=+22.067
  winner: hybrid+rerank (margin +0.300)
```

`delta` is B − A on each metric. The winner is decided on the primary metric
(default `recall@5`); `avg_tokens` is reported alongside so you can see what
the quality gain costs. A technique that wins recall but doubles tokens may
still lose the argument — that's the point of measuring both.

## Production path

- `FakeEmbeddings` → a real embedding model (e.g. sentence-transformers) or a
  managed embedding API; keep the `CachedEmbeddings` wrapper.
- In-memory chunk lists → a vector database (Qdrant, pgvector, Pinecone).
- `simulated_cross_encoder_score` → a real cross-encoder reranker.
- `HyDE.hypothetical_doc` → LLM-generated hypothetical answers.
- Extractive answers → the compressed context fed to an LLM with citations.

## Notes

- Synthetic/fictional data only; no real products or customers.
- BM25 and reciprocal rank fusion are implemented by hand (stdlib only) —
  read `retrieve.py` as the reference.
