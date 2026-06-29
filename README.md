# ragkit

[![CI](https://github.com/qazasd2518995/ragkit/actions/workflows/ci.yml/badge.svg)](https://github.com/qazasd2518995/ragkit/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A lightweight, dependency-light agentic Retrieval-Augmented Generation toolkit. It pairs local CPU embeddings (`fastembed`) with a Groq-hosted LLM, adds an agentic query-rewrite step before retrieval, and returns answers with citations back to the source passages. It can be used as a Python library, a command-line tool, or a FastAPI server.

The query-rewrite step is what makes the pipeline "agentic": each question is first passed through a small LLM that turns vague, conversational input into a standalone search query, which generally improves retrieval over embedding the raw question directly. Answers are produced strictly from the retrieved passages and cite them as `[1]`, `[2]`, so every claim is traceable to a source.

## Features

- Agentic retrieval: an LLM rewrites and expands the query before searching.
- Cited answers: each claim maps back to a numbered source passage.
- Groq inference: `llama-3.3-70b-versatile` for answers, `llama-3.1-8b-instant` for the rewrite step.
- Local embeddings: `fastembed` (ONNX, CPU-only). No embedding API, runs offline.
- Small footprint: a NumPy vector store, with no separate vector database to operate.
- Three interfaces: Python library, CLI, and a FastAPI REST server.
- Tested in CI: the pipeline logic is unit-tested with the network mocked, so the suite runs without an API key.

## How it works

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

Groq is the inference backend for both the rewrite and answer steps. Embeddings are computed locally and never leave the machine.

## Installation

```bash
git clone https://github.com/qazasd2518995/ragkit.git
cd ragkit

python -m venv venv && source venv/bin/activate
pip install -e .

cp .env.example .env          # then add your Groq API key (https://console.groq.com/keys)
```

A Groq API key is required for the rewrite and answer steps. Embeddings run locally and need no key.

## Usage

### Library

```python
from ragkit import RagEngine

engine = RagEngine()
engine.add_document(open("report.md").read(), source="report.md")

answer = engine.ask("What were the Q3 revenue drivers?")
print(answer.text)
for c in answer.citations:
    print(c.marker, c.source, c.score)
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
Indexed 2 chunks from 1 document(s)

According to the context, full-time employees accrue 20 days of paid vacation
per year [1]. Unused vacation of up to 5 days may be carried over into the next
calendar year; anything beyond that is forfeited on December 31st [1].

Sources:
  [1] samples/company_handbook.md  (score 0.7149)
      # Acme Corp Employee Handbook (excerpt) ## Remote Work Policy …
```

### API

```bash
uvicorn ragkit.api:app --reload
```

```bash
# index a document
curl -X POST localhost:8000/documents \
  -H 'content-type: application/json' \
  -d '{"text":"Our refund window is 30 days.","source":"policy"}'

# ask a question
curl -X POST localhost:8000/ask \
  -H 'content-type: application/json' \
  -d '{"question":"how long do I have to ask for a refund?"}'
```

Interactive API documentation is auto-generated at http://localhost:8000/docs.

## Configuration

Configuration is environment-driven (see `.env.example`):

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | Groq API key (required) |
| `GROQ_BASE_URL` | `https://api.groq.com/openai/v1` | Groq API base URL |
| `RAG_CHAT_MODEL` | `llama-3.3-70b-versatile` | Answer model |
| `RAG_REWRITE_MODEL` | `llama-3.1-8b-instant` | Query-rewrite model |
| `RAG_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Local embedding model |
| `RAG_CHUNK_SIZE` | `800` | Chunk size in characters |
| `RAG_CHUNK_OVERLAP` | `120` | Chunk overlap in characters |
| `RAG_TOP_K` | `4` | Passages retrieved per query |

## Testing

```bash
pip install -e . pytest
pytest -q          # 9 tests, no API key needed (the LLM and embeddings are mocked)
```

## Project layout

```
ragkit/
  config.py      env-driven settings
  store.py       chunking + NumPy cosine vector store (with save/load)
  embeddings.py  fastembed CPU embeddings (lazy-loaded)
  llm.py         Groq chat client (sync and streaming, with retries)
  engine.py      the agentic pipeline: rewrite, retrieve, cite
  loaders.py     txt / md / pdf loaders
  cli.py         ragkit ask / ragkit chat
  api.py         FastAPI server
tests/           unit tests (network mocked)
```

## License

MIT
