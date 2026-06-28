"""Central configuration, read from environment with sane defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass

try:  # Load a local .env if python-dotenv is installed; harmless otherwise.
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass


@dataclass(frozen=True)
class Settings:
    """Runtime settings for the RAG engine.

    All values can be overridden via environment variables so the same code
    runs unchanged in the CLI, the API and the test-suite.
    """

    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_base_url: str = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

    # Generation model used to answer questions.
    chat_model: str = os.getenv("RAG_CHAT_MODEL", "llama-3.3-70b-versatile")
    # Smaller/faster model used for the agentic query-rewriting step.
    rewrite_model: str = os.getenv("RAG_REWRITE_MODEL", "llama-3.1-8b-instant")
    # Local CPU embedding model (no API needed, runs via onnxruntime).
    embed_model: str = os.getenv("RAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    # Chunking.
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "120"))

    # Retrieval.
    top_k: int = int(os.getenv("RAG_TOP_K", "4"))

    def require_key(self) -> str:
        if not self.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key, "
                "or export GROQ_API_KEY in your shell."
            )
        return self.groq_api_key


settings = Settings()
