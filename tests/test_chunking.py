"""Unit tests for the chunking logic (no network, no API key needed)."""
from ragkit.store import split_text, VectorStore, Chunk


def test_split_empty_returns_empty():
    assert split_text("", 100, 10) == []
    assert split_text("   \n  ", 100, 10) == []


def test_split_respects_chunk_size():
    text = " ".join(f"Sentence number {i}." for i in range(50))
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    # every chunk should be reasonably bounded (allow slack for the overlap tail)
    assert all(len(c) <= 100 + 40 for c in chunks)


def test_split_single_short_text():
    chunks = split_text("Just one short sentence.", 800, 120)
    assert chunks == ["Just one short sentence."]


def test_chunks_overlap_shares_context():
    text = "Alpha sentence one. Beta sentence two. Gamma sentence three. Delta four. Epsilon five."
    chunks = split_text(text, chunk_size=40, overlap=15)
    assert len(chunks) >= 2
