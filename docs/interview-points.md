# Interview points

Short, honest talking points for Jagadish-rag-optimization. Every number quoted is from the
real run shown in the README.

- My harness had a bug: it measured retrieval before rerank, so the reranker always looked useless.
- BM25 alone won here because the embeddings are lexical; I say that instead of claiming hybrid always wins.
- With 15 questions, a +0.10 change can be noise; the bootstrap interval shows it.
- Report tokens with quality: paragraph chunking added about 75% more tokens per answer for an unproven gain.

## Likely follow-up questions

- How many gold questions do you need? (Enough that the interval on the change you care about excludes zero.)
- Why paired bootstrap? (Same questions in both arms; pairing removes question difficulty from the noise.)
- When does reranking hurt? (When first-stage recall is poor; it can only reorder what it gets.)
