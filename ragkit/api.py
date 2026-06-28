"""FastAPI server exposing the RAG engine over HTTP.

Run with:
    uvicorn ragkit.api:app --reload

Endpoints
---------
    POST /documents   — index a document          {"text": "...", "source": "..."}
    POST /ask         — ask a question            {"question": "...", "history": [...]}
    GET  /healthz     — liveness probe
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__
from .engine import RagEngine

app = FastAPI(
    title="ragkit — Agentic RAG API",
    version=__version__,
    description="Upload documents and ask grounded, cited questions, powered by Groq.",
)

# A single in-process engine. For multi-tenant use you'd key these by user/session.
engine = RagEngine()


class DocumentIn(BaseModel):
    text: str = Field(..., min_length=1, description="Raw document text")
    source: str = Field(..., min_length=1, description="A label/filename for citations")


class AskIn(BaseModel):
    question: str = Field(..., min_length=1)
    history: list[dict] | None = Field(default=None, description="Prior chat turns")


class CitationOut(BaseModel):
    marker: int
    source: str
    snippet: str
    score: float


class AnswerOut(BaseModel):
    question: str
    rewritten_query: str
    text: str
    citations: list[CitationOut]


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "indexed_chunks": len(engine.store)}


@app.post("/documents")
def add_document(doc: DocumentIn) -> dict:
    n = engine.add_document(doc.text, doc.source)
    if n == 0:
        raise HTTPException(status_code=400, detail="Document produced no chunks (empty?)")
    return {"indexed_chunks": n, "total_chunks": len(engine.store)}


@app.post("/ask", response_model=AnswerOut)
def ask(body: AskIn) -> AnswerOut:
    if len(engine.store) == 0:
        raise HTTPException(status_code=409, detail="No documents indexed yet. POST /documents first.")
    ans = engine.ask(body.question, history=body.history)
    return AnswerOut(
        question=ans.question,
        rewritten_query=ans.rewritten_query,
        text=ans.text,
        citations=[CitationOut(**c.__dict__) for c in ans.citations],
    )
