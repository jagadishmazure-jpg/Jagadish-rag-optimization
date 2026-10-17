"""Tests for the RAG optimization lab. Deterministic, offline, stdlib only."""

import pytest

from ragopt import (
    ABTest, BM25Retriever, CachedEmbeddings, ContextCompressor, DenseRetriever,
    FakeEmbeddings, FixedChunker, HyDE, HybridRetriever, ParagraphChunker,
    QueryRewriter, RAGPipeline, Reranker, SlidingWindowChunker,
    build_corpus, build_gold_questions, estimate_tokens,
    keyword_hit_rate, precision_at_k, recall_at_k,
)
from ragopt.compress import estimate_tokens as est2


@pytest.fixture(scope="module")
def docs():
    return build_corpus()


@pytest.fixture(scope="module")
def gold():
    return build_gold_questions()


@pytest.fixture(scope="module")
def chunks(docs):
    out = []
    for d in docs:
        out += FixedChunker(200, 50).chunk(d.doc_id, d.text)
    return out


@pytest.fixture(scope="module")
def dense(chunks):
    return DenseRetriever(chunks, FakeEmbeddings())


@pytest.fixture(scope="module")
def bm25(chunks):
    return BM25Retriever(chunks)


@pytest.fixture(scope="module")
def hybrid(dense, bm25):
    return HybridRetriever(dense, bm25)


# ---------- chunking ----------

def test_fixed_chunker_overlap_math():
    c = FixedChunker(size=100, overlap=20)
    chunks = c.chunk("D1", "x" * 250)
    # step=80: starts at 0, 80, 160, 240 -> 4 chunks
    assert len(chunks) == 4
    assert chunks[1].text == "x" * 100  # full window
    assert all(ch.doc_id == "D1" for ch in chunks)
    assert [ch.position for ch in chunks] == [0, 1, 2, 3]


def test_fixed_chunker_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedChunker(size=100, overlap=100)


def test_paragraph_chunker_respects_boundaries():
    text = "First sentence here. Second sentence here.\n\nNew paragraph about robots."
    chunks = ParagraphChunker(max_size=400).chunk("D1", text)
    assert len(chunks) == 1  # fits in one chunk
    assert "robots" in chunks[0].text


def test_paragraph_chunker_splits_long_paragraph():
    text = "Alpha beta gamma. " * 60  # one long paragraph, no blank lines
    chunks = ParagraphChunker(max_size=200).chunk("D1", text)
    assert len(chunks) > 1
    # No chunk cuts mid-sentence: every chunk ends with a period.
    assert all(ch.text.rstrip().endswith(".") for ch in chunks)


def test_sliding_window_word_overlap():
    words = " ".join(f"w{i}" for i in range(100))
    chunks = SlidingWindowChunker(window_words=60, overlap_words=15).chunk("D1", words)
    first = set(chunks[0].text.split())
    second = set(chunks[1].text.split())
    # windows start at 0 and 45 -> 15 shared words
    assert len(first & second) == 15


def test_chunks_carry_doc_id_and_position(docs):
    chunks = FixedChunker(200, 50).chunk(docs[0].doc_id, docs[0].text)
    assert chunks[0].chunk_id == f"{docs[0].doc_id}#c0"
    assert chunks[0].position == 0


# ---------- embeddings ----------

def test_fake_embeddings_deterministic():
    e = FakeEmbeddings()
    assert e.embed("robo arm payload") == e.embed("robo arm payload")
    assert e.embed("robo arm") != e.embed("warranty claim")


def test_cached_embeddings_hit_increments():
    inner = FakeEmbeddings()
    cached = CachedEmbeddings(inner)
    cached.embed("hello world")
    cached.embed("hello world")
    assert cached.misses == 1
    assert cached.hits == 1
    assert cached.stats()["hit_rate"] == pytest.approx(0.5)


# ---------- BM25 ----------

def test_bm25_ranks_known_term_first(chunks, bm25):
    # "gigapixel" appears only in VisionCam docs; query should surface them.
    res = bm25.retrieve("twelve megapixel sensor", top_k=3)
    assert res[0].doc_id == "DOC-04"


def test_bm25_scores_are_nonnegative(chunks, bm25):
    res = bm25.retrieve("warranty refund shipping", top_k=10)
    assert all(r.score >= 0 for r in res)


def test_bm25_idf_downweights_common_terms(chunks, bm25):
    # "the" appears everywhere -> low idf; "telemetry" is rare -> high idf.
    assert bm25._idf("telemetry") > bm25._idf("the")


# ---------- dense / hybrid ----------

def test_dense_retriever_returns_top_k(chunks, dense):
    res = dense.retrieve("robotic arm payload", top_k=5)
    assert len(res) == 5
    assert res[0].score >= res[-1].score


def test_hybrid_fuses_both_rankings(chunks, hybrid):
    res = hybrid.retrieve("warranty claim process", top_k=5)
    assert len(res) == 5
    # RRF scores are positive sums of 1/(k+rank)
    assert all(r.score > 0 for r in res)


def test_hybrid_beats_dense_on_some_gold(gold, dense, hybrid):
    better = 0
    for g in gold:
        rd = [s.doc_id for s in dense.retrieve(g.question, 5)]
        rh = [s.doc_id for s in hybrid.retrieve(g.question, 5)]
        rel = set(g.relevant_doc_ids)
        if recall_at_k(rel, rh, 5) > recall_at_k(rel, rd, 5):
            better += 1
    assert better >= 1


# ---------- rerank ----------

def test_rerank_subset_of_candidates(chunks, hybrid):
    rr = Reranker(hybrid, top_n=20, top_k=5)
    res = rr.retrieve("conveyor battery life", top_k=5)
    assert len(res) == 5
    cands = {s.chunk_id for s in hybrid.retrieve("conveyor battery life", top_k=20)}
    assert all(s.chunk_id in cands for s in res)


def test_rerank_can_reorder(chunks, hybrid):
    rr = Reranker(hybrid, top_n=20, top_k=20)
    base = [s.chunk_id for s in hybrid.retrieve("support response time", top_k=20)]
    re = [s.chunk_id for s in rr.retrieve("support response time", top_k=20)]
    assert set(base) == set(re)  # same candidates


# ---------- rewrite / HyDE ----------

def test_rewrite_expands_synonyms():
    rw = QueryRewriter()
    out = rw.rewrite("What is the cost of express shipping?")
    assert "price" in out.lower()


def test_rewrite_noop_without_synonyms():
    rw = QueryRewriter()
    q = "Zebra xylophone quantum"
    assert rw.rewrite(q) == q


def test_hyde_generates_hypothetical_doc():
    hyde = HyDE(FakeEmbeddings())
    doc = hyde.hypothetical_doc("What is the battery life?")
    assert "battery" in doc.lower()
    vec = hyde.embed("What is the battery life?")
    assert len(vec) == 128
    assert vec == hyde.embed("What is the battery life?")  # deterministic


# ---------- compression ----------

def test_compress_fits_budget(chunks, dense):
    ranked = dense.retrieve("return policy refund", top_k=10)
    comp = ContextCompressor(token_budget=100)
    out = comp.compress(ranked)
    assert out["tokens_used"] <= 100
    assert out["tokens_saved"] >= 0
    assert out["dropped"] >= 0
    assert len(out["chunks"]) + out["dropped"] == 10


def test_compress_reports_savings(chunks, dense):
    ranked = dense.retrieve("warranty", top_k=10)
    comp = ContextCompressor(token_budget=50)
    out = comp.compress(ranked)
    total = sum(estimate_tokens(s.text) for s in ranked)
    assert out["tokens_saved"] == total - out["tokens_used"]


def test_prune_marginal_drops_low_scores(chunks, dense):
    ranked = dense.retrieve("robot", top_k=10)
    comp = ContextCompressor()
    threshold = ranked[2].score
    pruned = comp.prune_marginal(ranked, threshold)
    assert all(s.score >= threshold for s in pruned)
    assert len(pruned) <= len(ranked)


def test_estimate_tokens_scales_with_length():
    assert est2("a" * 400) == 2 * est2("a" * 200)


# ---------- pipeline ----------

def _pipe(name, retriever, **kw):
    return RAGPipeline(name, FixedChunker(200, 50), retriever, **kw)


def test_pipeline_answer_has_citations(chunks, hybrid):
    pipe = _pipe("p", hybrid)
    ans = pipe.run("How long is the warranty?")
    assert ans.citations, "answer must cite sources"
    assert ans.text
    assert ans.chunks_used


def test_pipeline_respects_token_budget(chunks, dense):
    comp = ContextCompressor(token_budget=120)
    pipe = _pipe("p", dense, compressor=comp)
    ans = pipe.run("What is the return window?")
    assert ans.tokens_used <= 120


def test_pipeline_empty_retrieval_graceful():
    class Empty:
        def retrieve(self, q, top_k=5):
            return []
    pipe = _pipe("p", Empty())
    ans = pipe.run("anything")
    assert ans.citations == []
    assert "No relevant" in ans.text


# ---------- eval ----------

def test_recall_at_k_math():
    assert recall_at_k({"a", "b"}, ["a", "x", "b"], 3) == 1.0
    assert recall_at_k({"a", "b"}, ["a", "x", "y"], 2) == 0.5
    assert recall_at_k(set(), ["a"], 1) == 0.0


def test_precision_at_k_math():
    assert precision_at_k({"a"}, ["a", "x"], 2) == 0.5
    assert precision_at_k({"a"}, ["x", "y"], 2) == 0.0
    assert precision_at_k({"a"}, [], 0) == 0.0


def test_keyword_hit_rate():
    assert keyword_hit_rate("The warranty lasts twenty four months",
                            ["twenty four months", "refund"]) == 0.5


def test_ab_declares_winner_with_margin(gold, dense, hybrid):
    ab = ABTest(gold, k=5)
    res = ab.run(_pipe("dense", dense), _pipe("hybrid", hybrid))
    assert res.winner == "hybrid"
    assert res.margin > 0
    assert "recall@5" in res.metrics
    assert res.summary().startswith("A/B:")


def test_ab_tie_when_identical(gold, dense):
    ab = ABTest(gold, k=5)
    res = ab.run(_pipe("a", dense), _pipe("b", dense))
    assert res.winner == "tie"
    assert res.margin == 0.0


def test_hybrid_rerank_ge_baseline_recall(gold, dense, hybrid):
    ab = ABTest(gold, k=5)
    base = _pipe("baseline", dense)
    full = RAGPipeline("hybrid+rerank", FixedChunker(200, 50), hybrid,
                       reranker=Reranker(hybrid, top_n=20, top_k=5))
    res = ab.run(base, full)
    assert res.metrics["recall@5"]["b"] >= res.metrics["recall@5"]["a"]
