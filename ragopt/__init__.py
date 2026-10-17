"""RAG optimization lab: measure every technique against golden evals."""

from .chunking import Chunker, FixedChunker, ParagraphChunker, SlidingWindowChunker
from .compress import ContextCompressor, estimate_tokens
from .corpus import Document, GoldQuestion, build_corpus, build_gold_questions
from .embeddings import CachedEmbeddings, FakeEmbeddings, tokenize
from .eval import ABResult, ABTest, keyword_hit_rate, precision_at_k, recall_at_k
from .pipeline import Answer, RAGPipeline
from .rerank import Reranker, simulated_cross_encoder_score
from .retrieve import BM25Retriever, DenseRetriever, HybridRetriever, ScoredChunk, cosine
from .rewrite import HyDE, QueryRewriter

__all__ = [
    "Chunker", "FixedChunker", "ParagraphChunker", "SlidingWindowChunker",
    "ContextCompressor", "estimate_tokens",
    "Document", "GoldQuestion", "build_corpus", "build_gold_questions",
    "CachedEmbeddings", "FakeEmbeddings", "tokenize",
    "ABResult", "ABTest", "keyword_hit_rate", "precision_at_k", "recall_at_k",
    "Answer", "RAGPipeline",
    "Reranker", "simulated_cross_encoder_score",
    "BM25Retriever", "DenseRetriever", "HybridRetriever", "ScoredChunk", "cosine",
    "HyDE", "QueryRewriter",
]

__version__ = "0.1.0"
