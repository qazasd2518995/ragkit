"""Test the RAG pipeline orchestration with the LLM + embeddings mocked out.

This verifies the *agentic* wiring (rewrite → retrieve → cited answer) without
making a single network call, so it runs in CI with no API key.
"""
import ragkit.engine as engine_mod
from ragkit.engine import RagEngine


def _fake_embed_texts(texts, model_name=None):
    # Deterministic toy embedding: vector of [len, vowel_count].
    out = []
    for t in texts:
        vowels = sum(t.lower().count(v) for v in "aeiou")
        out.append([float(len(t)), float(vowels), 1.0])
    return out


def _fake_embed_query(text, model_name=None):
    return _fake_embed_texts([text])[0]


def test_full_pipeline_is_cited(monkeypatch):
    # Stub embeddings (used by both ingestion and retrieval).
    monkeypatch.setattr(engine_mod.embeddings, "embed_texts", _fake_embed_texts)
    monkeypatch.setattr(engine_mod.embeddings, "embed_query", _fake_embed_query)

    # Stub the LLM: rewrite echoes the question; answer cites [1].
    calls = {"rewrite": 0, "answer": 0}

    def fake_chat(messages, **kwargs):
        system = messages[0]["content"]
        if "rewrite" in system.lower():
            calls["rewrite"] += 1
            return "rewritten standalone query"
        calls["answer"] += 1
        return "The policy allows refunds within 30 days [1]."

    monkeypatch.setattr(engine_mod.llm, "chat", fake_chat)

    eng = RagEngine()
    added = eng.add_document(
        "Our refund policy. Customers may request a refund within 30 days of purchase.",
        source="policy.txt",
    )
    assert added >= 1

    ans = eng.ask("can I get my money back?")
    assert calls["rewrite"] == 1
    assert calls["answer"] == 1
    assert ans.rewritten_query == "rewritten standalone query"
    assert "[1]" in ans.text
    assert ans.citations and ans.citations[0].source == "policy.txt"


def test_ask_with_no_documents_is_graceful(monkeypatch):
    monkeypatch.setattr(engine_mod.embeddings, "embed_query", _fake_embed_query)
    monkeypatch.setattr(engine_mod.llm, "chat", lambda *a, **k: "should not be called")
    eng = RagEngine()
    ans = eng.ask("anything?")
    assert ans.citations == []
    assert "don't have" in ans.text.lower()
