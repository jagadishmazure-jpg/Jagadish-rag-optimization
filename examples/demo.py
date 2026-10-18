"""A/B test, then a one-change-at-a-time ablation with bootstrap intervals.

Synthetic corpus (fictional company), fake hash embeddings and a simulated
cross-encoder: results show what the harness measures, not what a real
embedding model would score. 15 gold questions only.
"""
from ragopt import (
    ABTest,
    BM25Retriever,
    CachedEmbeddings,
    ContextCompressor,
    DenseRetriever,
    FakeEmbeddings,
    FixedChunker,
    HybridRetriever,
    RAGPipeline,
    Reranker,
    build_corpus,
    build_gold_questions,
)
from ragopt.ablation import markdown, run_ablation

docs, gold = build_corpus(), build_gold_questions()
print(f"{len(docs)} synthetic docs, {len(gold)} gold questions\n")
chunks = []
for d in docs:
    chunks += FixedChunker(200, 50).chunk(d.doc_id, d.text)
emb = CachedEmbeddings(FakeEmbeddings())
dense = DenseRetriever(chunks, emb)
hybrid = HybridRetriever(dense, BM25Retriever(chunks))
base = RAGPipeline("baseline", FixedChunker(200, 50), dense)
opt = RAGPipeline("hybrid+rerank", FixedChunker(200, 50), hybrid,
                  reranker=Reranker(hybrid, top_n=20, top_k=5),
                  compressor=ContextCompressor(token_budget=600))
print(ABTest(gold, k=5).run(base, opt).summary())
print("\nAblation (one change from the baseline):")
print(markdown(run_ablation(docs, gold), len(gold)))
