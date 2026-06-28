"""RagEngine — the orchestrator that ties retrieval and generation together.

Pipeline (the "agentic" part):

    question
       │
       ├─►  (1) query rewrite: an LLM turns a possibly-vague, conversational
       │        question into a standalone search query, optionally expanding
       │        it with synonyms. This is what lifts retrieval quality above a
       │        naive "embed the raw question" baseline.
       │
       ├─►  (2) retrieve: top-k chunks by cosine similarity.
       │
       └─►  (3) answer: the LLM answers ONLY from the retrieved context and is
                instructed to cite sources as [1], [2] ... mapping back to the
                chunks it was given, so every claim is traceable.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import embeddings, llm
from .config import settings
from .store import Chunk, Document, VectorStore, split_text


@dataclass
class Citation:
    marker: int
    source: str
    snippet: str
    score: float


@dataclass
class Answer:
    question: str
    rewritten_query: str
    text: str
    citations: list[Citation]


_REWRITE_SYSTEM = (
    "You rewrite a user's question into a single, self-contained search query "
    "for a document retrieval system. Resolve pronouns using the chat history, "
    "drop conversational filler, and keep key entities. Reply with ONLY the "
    "rewritten query, nothing else."
)

_ANSWER_SYSTEM = (
    "You are a precise assistant that answers strictly from the provided context "
    "passages. Each passage is numbered. Cite the passages you use with bracketed "
    "markers like [1] or [2]. If the answer is not contained in the context, say "
    "you don't have enough information — do not invent facts."
)


class RagEngine:
    def __init__(self, store: VectorStore | None = None) -> None:
        self.store = store or VectorStore()

    # --- ingestion -------------------------------------------------------
    def add_document(self, text: str, source: str) -> int:
        """Chunk, embed and index a document. Returns number of chunks added."""
        doc_id = hashlib.sha1(source.encode("utf-8")).hexdigest()[:12]
        Document(doc_id=doc_id, text=text, source=source)  # validation/clarity
        pieces = split_text(text, settings.chunk_size, settings.chunk_overlap)
        if not pieces:
            return 0
        vectors = embeddings.embed_texts(pieces)
        chunks = [
            Chunk(
                chunk_id=f"{doc_id}-{i}",
                doc_id=doc_id,
                source=source,
                text=piece,
                embedding=vec,
            )
            for i, (piece, vec) in enumerate(zip(pieces, vectors))
        ]
        self.store.add(chunks)
        return len(chunks)

    # --- query -----------------------------------------------------------
    def rewrite_query(self, question: str, history: list[dict] | None = None) -> str:
        messages = [{"role": "system", "content": _REWRITE_SYSTEM}]
        if history:
            messages.extend(history[-4:])  # a little context for pronoun resolution
        messages.append({"role": "user", "content": question})
        try:
            rewritten = llm.chat(
                messages, model=settings.rewrite_model, temperature=0.0, max_tokens=80
            ).strip()
            return rewritten or question
        except llm.GroqError:
            # Degrade gracefully — a failed rewrite shouldn't break the answer.
            return question

    def retrieve(self, query: str) -> list[tuple[Chunk, float]]:
        q_vec = embeddings.embed_query(query)
        return self.store.search(q_vec, settings.top_k)

    def _build_context(self, hits: list[tuple[Chunk, float]]) -> tuple[str, list[Citation]]:
        blocks, citations = [], []
        for i, (chunk, score) in enumerate(hits, start=1):
            blocks.append(f"[{i}] (source: {chunk.source})\n{chunk.text}")
            citations.append(
                Citation(
                    marker=i,
                    source=chunk.source,
                    snippet=chunk.text[:160] + ("…" if len(chunk.text) > 160 else ""),
                    score=round(score, 4),
                )
            )
        return "\n\n".join(blocks), citations

    def ask(self, question: str, history: list[dict] | None = None) -> Answer:
        """Full agentic RAG: rewrite → retrieve → answer with citations."""
        rewritten = self.rewrite_query(question, history)
        hits = self.retrieve(rewritten)
        context, citations = self._build_context(hits)

        if not hits:
            return Answer(
                question=question,
                rewritten_query=rewritten,
                text="I don't have any indexed documents to answer from yet.",
                citations=[],
            )

        user_msg = (
            f"Context passages:\n\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the passages above and cite them."
        )
        text = llm.chat(
            [
                {"role": "system", "content": _ANSWER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.2,
        )
        return Answer(
            question=question, rewritten_query=rewritten, text=text, citations=citations
        )
