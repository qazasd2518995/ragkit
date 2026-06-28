<div align="center">

# 🧠 ragkit — Agentic RAG, powered by Groq

**Ask grounded, *cited* questions over your own documents — with sub-second LLM inference on Groq and zero-cost local embeddings.**

[![CI](https://github.com/qazasd2518995/groq-agentic-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/qazasd2518995/groq-agentic-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Groq](https://img.shields.io/badge/LLM-Groq-orange)

</div>

---

`ragkit` is a small, readable, **dependency-light** Retrieval-Augmented Generation toolkit. It indexes your PDFs / Markdown / text files, retrieves the most relevant passages with **local CPU embeddings** (no embedding API, no GPU), and answers your questions with a Groq-hosted LLM — **citing every source** so nothing is hallucinated without a trace.

It's "**agentic**" because each question first goes through an LLM **query-rewriting** step that turns vague, conversational questions into sharp standalone search queries — measurably better retrieval than embedding the raw question.

## ✨ Highlights

- 🔎 **Agentic retrieval** — LLM rewrites/expands your query before searching.
- 📚 **Cited answers** — every claim maps back to a `[1]`, `[2]` source passage.
- ⚡ **Groq inference** — `llama-3.3-70b` for answers, `llama-3.1-8b-instant` for the cheap rewrite step.
- 🧮 **Local embeddings** — `fastembed` (ONNX, CPU-only). No embedding API, runs offline.
- 🪶 **Tiny footprint** — NumPy vector store, no heavyweight vector DB to operate.
- 🧰 **Three interfaces** — Python library, CLI, and a FastAPI REST server.
- ✅ **Tested & CI'd** — pipeline logic is unit-tested with the network mocked (runs without an API key).

## 🏗️ How it works

```
                ┌──────────────────────────────────────────────┐
   question ───►│  1. query rewrite   (llama-3.1-8b-instant)   │
                │     "can I get money back?"                   │
                │        → "refund policy time limit"          │
                └───────────────────┬──────────────────────────┘
                                    ▼
                ┌──────────────────────────────────────────────┐
                │  2. retrieve top-k   (fastembed + cosine)     │
                │     local CPU embeddings, NumPy similarity     │
                └───────────────────┬──────────────────────────┘
                                    ▼
                ┌──────────────────────────────────────────────┐
                │  3. grounded answer  (llama-3.3-70b)          │
                │     answers ONLY from retrieved passages,      │
                │     cites them as [1] [2] …                    │
                └──────────────────────────────────────────────┘
```

## 🚀 Quickstart

```bash
git clone https://github.com/qazasd2518995/groq-agentic-rag.git
cd groq-agentic-rag

python -m venv venv && source venv/bin/activate
pip install -e .

cp .env.example .env          # then paste your Groq key (https://console.groq.com/keys)
```

### CLI

```bash
# One-off question over the bundled sample docs
ragkit ask "How many vacation days do employees get?" --docs ./samples

# Interactive chat (keeps conversation history for pronoun resolution)
ragkit chat --docs ./samples
```

Example output:

```
✓ Indexed 2 chunks from 1 document(s)

According to the context, full-time employees accrue 20 days of paid vacation
per year [1]. Unused vacation of up to 5 days may be carried over into the next
calendar year; anything beyond that is forfeited on December 31st [1].

Sources:
  [1] samples/company_handbook.md  (score 0.7149)
      # Acme Corp Employee Handbook (excerpt) ## Remote Work Policy …
```

### REST API

```bash
uvicorn ragkit.api:app --reload
```

```bash
# index a document
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"text":"Our refund window is 30 days.","source":"policy"}'

# ask
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"how long do I have to ask for a refund?"}'
```

Interactive docs are auto-generated at **http://localhost:8000/docs**.

### Python library

```python
from ragkit import RagEngine

engine = RagEngine()
engine.add_document(open("report.md").read(), source="report.md")

answer = engine.ask("What were the Q3 revenue drivers?")
print(answer.text)
for c in answer.citations:
    print(c.marker, c.source, c.score)
```

## ⚙️ Configuration

Everything is environment-driven (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | **required** |
| `RAG_CHAT_MODEL` | `llama-3.3-70b-versatile` | answer model |
| `RAG_REWRITE_MODEL` | `llama-3.1-8b-instant` | query-rewrite model |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | local embedding model |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | `800` / `120` | chunking |
| `RAG_TOP_K` | `4` | passages retrieved per query |

## 🧪 Tests

```bash
pip install -e . pytest
pytest -q          # 9 tests, no API key needed (LLM + embeddings are mocked)
```

## 🗂️ Project layout

```
ragkit/
  config.py      env-driven settings
  store.py       chunking + NumPy cosine vector store (+ save/load)
  embeddings.py  fastembed CPU embeddings (lazy-loaded)
  llm.py         Groq chat client (sync + streaming, retries)
  engine.py      the agentic pipeline: rewrite → retrieve → cite
  loaders.py     txt / md / pdf loaders
  cli.py         `ragkit ask` / `ragkit chat`
  api.py         FastAPI server
tests/           unit tests (network mocked)
```

## 📋 Roadmap

- [ ] Streaming answers in the API (`text/event-stream`)
- [ ] Re-ranking pass with a cross-encoder
- [ ] Persisted on-disk index across restarts
- [ ] Multi-tenant sessions

## 📄 License

MIT © Justin
