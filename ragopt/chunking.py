"""Chunking strategies: how documents are split before embedding.

Chunking is the first optimization lever in RAG: chunk size and boundaries
directly control retrieval precision vs. context cost.
"""

from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    position: int  # 0-based index within the document
    text: str


class Chunker:
    """Interface: split a document into chunks."""

    name = "base"

    def chunk(self, doc_id: str, text: str) -> list:
        raise NotImplementedError


class FixedChunker(Chunker):
    """Fixed-size character windows with configurable overlap.

    Overlap preserves cross-boundary context at the cost of more chunks
    (more embeddings, more storage, larger candidate pool).
    """

    name = "fixed"

    def __init__(self, size: int = 200, overlap: int = 50):
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def chunk(self, doc_id: str, text: str) -> list:
        chunks = []
        step = self.size - self.overlap
        pos = 0
        start = 0
        while start < len(text):
            piece = text[start:start + self.size]
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#c{pos}", doc_id=doc_id,
                position=pos, text=piece))
            pos += 1
            start += step
        return chunks


class ParagraphChunker(Chunker):
    """Split on paragraph boundaries (double newlines), falling back to
    sentence splits for oversized paragraphs.

    Respects semantic boundaries: a chunk never cuts a sentence in half,
    which improves embedding quality at the cost of variable chunk sizes.
    """

    name = "paragraph"

    def __init__(self, max_size: int = 400):
        self.max_size = max_size

    def chunk(self, doc_id: str, text: str) -> list:
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            # Fall back to sentence splitting on single-paragraph text.
            paragraphs = [s.strip() for s in text.split(". ") if s.strip()]
        chunks = []
        pos = 0
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= self.max_size:
                current = (current + "\n\n" + para).strip() if current else para
            else:
                if current:
                    chunks.append(Chunk(
                        chunk_id=f"{doc_id}#c{pos}", doc_id=doc_id,
                        position=pos, text=current))
                    pos += 1
                # Split an oversized paragraph on sentence boundaries.
                if len(para) > self.max_size:
                    sentences = [s.strip() for s in para.split(". ") if s.strip()]
                    buf = ""
                    for s in sentences:
                        s = s if s.endswith(".") else s + "."
                        if len(buf) + len(s) + 1 <= self.max_size:
                            buf = (buf + " " + s).strip()
                        else:
                            chunks.append(Chunk(
                                chunk_id=f"{doc_id}#c{pos}", doc_id=doc_id,
                                position=pos, text=buf))
                            pos += 1
                            buf = s
                    current = buf
                else:
                    current = para
        if current:
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#c{pos}", doc_id=doc_id,
                position=pos, text=current))
        return chunks


class SlidingWindowChunker(Chunker):
    """Word-level sliding window: fixed word count, fixed word overlap.

    Finer control than character windows; overlap is measured in whole words
    so chunks never split mid-word.
    """

    name = "sliding"

    def __init__(self, window_words: int = 60, overlap_words: int = 15):
        if overlap_words >= window_words:
            raise ValueError("overlap_words must be smaller than window_words")
        self.window_words = window_words
        self.overlap_words = overlap_words

    def chunk(self, doc_id: str, text: str) -> list:
        words = text.split()
        chunks = []
        step = self.window_words - self.overlap_words
        pos = 0
        start = 0
        while start < len(words):
            piece = " ".join(words[start:start + self.window_words])
            chunks.append(Chunk(
                chunk_id=f"{doc_id}#c{pos}", doc_id=doc_id,
                position=pos, text=piece))
            pos += 1
            start += step
        return chunks
