"""Deterministic fake embeddings with a caching wrapper.

FakeEmbeddings produces stable hash-based vectors so the whole lab is
deterministic and offline. In production, swap for a real embedding model
(e.g. sentence-transformers) behind the same interface.
"""

import hashlib
import math
import re


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeEmbeddings:
    """Hash-based bag-of-words vectors, L2-normalized.

    Each token hashes into one of `dim` buckets; the vector is the
    L2-normalized bucket histogram. Deterministic across runs and processes.
    """

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dim

    def embed(self, text: str) -> list:
        vec = [0.0] * self.dim
        for tok in tokenize(text):
            vec[self._bucket(tok)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class CachedEmbeddings:
    """Wraps an embedding model with an in-memory cache.

    Tracks hits/misses so the A/B harness can report the latency/cost
    savings from caching repeated queries.
    """

    def __init__(self, inner):
        self.inner = inner
        self.hits = 0
        self.misses = 0
        self._cache = {}

    def embed(self, text: str) -> list:
        if text in self._cache:
            self.hits += 1
            return self._cache[text]
        self.misses += 1
        vec = self.inner.embed(text)
        self._cache[text] = vec
        return vec

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": (self.hits / total) if total else 0.0,
        }
