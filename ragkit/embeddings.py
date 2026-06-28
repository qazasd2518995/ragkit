"""Local, CPU-only embeddings via fastembed (ONNX runtime).

No API key, no GPU and no network access required after the model is cached.
The model is loaded lazily so that importing ragkit stays fast and the test
suite can monk-patch the embedder without downloading anything.
"""
from __future__ import annotations

from functools import lru_cache

from .config import settings


@lru_cache(maxsize=1)
def _model(name: str):
    # Imported lazily — fastembed pulls in onnxruntime which is heavy to import.
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=name)


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    """Embed a batch of texts into dense vectors."""
    if not texts:
        return []
    model = _model(model_name or settings.embed_model)
    return [vec.tolist() for vec in model.embed(texts)]


def embed_query(text: str, model_name: str | None = None) -> list[float]:
    """Embed a single query string."""
    return embed_texts([text], model_name=model_name)[0]
