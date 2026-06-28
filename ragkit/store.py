"""In-memory vector store with cosine similarity search.

Deliberately dependency-light: we keep embeddings in a NumPy matrix instead of
pulling in a heavyweight vector database. For a portfolio / single-machine RAG
this is fast, transparent and trivially serialisable.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class Document:
    """A source document before chunking."""

    doc_id: str
    text: str
    source: str  # filename or URL


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    chunk_id: str
    doc_id: str
    source: str
    text: str
    # populated once embedded
    embedding: list[float] | None = field(default=None, repr=False)


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks on sentence-ish boundaries.

    We greedily pack sentences up to ``chunk_size`` characters, then step back
    by ``overlap`` characters so adjacent chunks share context.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence terminators but keep them.
    sentences = re.split(r"(?<=[.!?。！？])\s+", text)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 <= chunk_size:
            current = f"{current} {sent}".strip()
        else:
            if current:
                chunks.append(current)
            # Start the next chunk with an overlapping tail of the previous one.
            tail = current[-overlap:] if overlap and current else ""
            current = f"{tail} {sent}".strip()
    if current:
        chunks.append(current)
    return chunks


class VectorStore:
    """Holds chunks + their embedding matrix and does top-k cosine search."""

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None  # shape (n_chunks, dim), L2-normalised

    def add(self, chunks: Iterable[Chunk]) -> None:
        new = [c for c in chunks if c.embedding is not None]
        if not new:
            return
        self.chunks.extend(new)
        mat = np.array([c.embedding for c in new], dtype=np.float32)
        mat = _l2_normalize(mat)
        self._matrix = mat if self._matrix is None else np.vstack([self._matrix, mat])

    def search(self, query_embedding: list[float], top_k: int) -> list[tuple[Chunk, float]]:
        if self._matrix is None or not self.chunks:
            return []
        q = _l2_normalize(np.array([query_embedding], dtype=np.float32))[0]
        scores = self._matrix @ q  # cosine similarity (both normalised)
        top_idx = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx]

    def __len__(self) -> int:
        return len(self.chunks)

    # --- persistence -----------------------------------------------------
    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = {"chunks": [asdict(c) for c in self.chunks]}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        store = cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store.add(Chunk(**c) for c in data["chunks"])
        return store


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms
