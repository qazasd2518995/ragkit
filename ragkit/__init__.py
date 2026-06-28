"""ragkit — a lightweight, dependency-light Agentic RAG toolkit powered by Groq.

Public API:
    from ragkit import RagEngine, Document
"""
from .engine import RagEngine
from .store import Document, Chunk

__all__ = ["RagEngine", "Document", "Chunk"]
__version__ = "0.1.0"
