"""Composable RAG pipeline: chunker -> retriever -> [rerank] -> [compress] -> answer."""

from dataclasses import dataclass, field

from .compress import ContextCompressor, estimate_tokens


@dataclass
class Answer:
    text: str
    citations: list          # doc_ids cited
    chunks_used: list        # chunk_ids that formed the answer
    tokens_used: int
    pipeline_name: str = "rag"


class RAGPipeline:
    """Assembles chunking, retrieval, optional rerank/compress into answers.

    The "answer" is extractive (top chunk text + citations) — deterministic
    and offline. In production, the compressed context feeds an LLM here.
    """

    def __init__(self, name: str, chunker, retriever,
                 reranker=None, compressor: ContextCompressor = None,
                 top_k: int = 5):
        self.name = name
        self.chunker = chunker
        self.retriever = retriever
        self.reranker = reranker
        self.compressor = compressor or ContextCompressor(token_budget=10 ** 9)
        self.top_k = top_k

    def run(self, question: str) -> Answer:
        ranked = self.retriever.retrieve(question, top_k=self.top_k)
        if self.reranker is not None:
            ranked = self.reranker.retrieve(question, top_k=self.top_k)
        packed = self.compressor.compress(ranked)
        kept = packed["chunks"]
        if kept:
            text = kept[0].text
        else:
            text = "No relevant context found."
        citations = sorted({s.doc_id for s in kept})
        return Answer(
            text=text,
            citations=citations,
            chunks_used=[s.chunk_id for s in kept],
            tokens_used=packed["tokens_used"],
            pipeline_name=self.name,
        )
