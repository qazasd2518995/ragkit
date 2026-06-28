"""Unit tests for the vector store search + persistence (no network needed)."""
import math

from ragkit.store import VectorStore, Chunk


def _chunk(cid: str, vec: list[float]) -> Chunk:
    return Chunk(chunk_id=cid, doc_id="d", source="s", text=f"text-{cid}", embedding=vec)


def test_search_returns_nearest_first():
    store = VectorStore()
    store.add([
        _chunk("a", [1.0, 0.0, 0.0]),
        _chunk("b", [0.0, 1.0, 0.0]),
        _chunk("c", [0.9, 0.1, 0.0]),
    ])
    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert [c.chunk_id for c, _ in results] == ["a", "c"]
    # cosine of identical vectors is 1.0
    assert math.isclose(results[0][1], 1.0, rel_tol=1e-5)


def test_search_empty_store():
    assert VectorStore().search([1.0, 0.0], top_k=3) == []


def test_save_and_load_roundtrip(tmp_path):
    store = VectorStore()
    store.add([_chunk("a", [1.0, 0.0]), _chunk("b", [0.0, 1.0])])
    path = tmp_path / "store.json"
    store.save(path)

    loaded = VectorStore.load(path)
    assert len(loaded) == 2
    res = loaded.search([1.0, 0.0], top_k=1)
    assert res[0][0].chunk_id == "a"
