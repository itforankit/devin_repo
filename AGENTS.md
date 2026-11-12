# AGENTS.md

Guidance for humans and AI agents working in this repository.

## What this is

An Agentic RAG (Retrieval-Augmented Generation) system: documents are ingested into a
local Chroma vector store, and an OpenAI function-calling agent decides when to retrieve,
(re)formulates queries, and answers with inline citations. A Streamlit UI and an eval
harness sit on top.

## Project structure

```
app.py                     # Streamlit chat UI (upload docs, chat, citations, latency)
src/
  ingestion.py             # loaders (PDF/DOCX/TXT/MD), chunking, embedding, Chroma persistence
  retrieval.py             # similarity-search wrapper returning chunks + metadata
  agent.py                 # agentic loop (OpenAI function calling, retrieve tool, citations)
tests/                     # pytest suite; OpenAI is always mocked
eval/
  evaluate.py              # agentic vs. naive baseline, LLM-as-judge + hit-rate
  questions.json           # fixed Q&A set
  data/                    # sample corpus for eval
  results.md               # latest eval output
.agents/skills/test-app/   # how to install, run and test the app (follow before app PRs)
chroma_db/                 # local vector store (git-ignored)
```

## Conventions

- **Framework choice:** OpenAI Python SDK for embeddings + chat/function calling;
  LangChain is used **only** for text splitting (`langchain_text_splitters`). Do not mix in
  LlamaIndex.
- **Models:** embeddings `text-embedding-3-small`; chat model configurable via
  `OPENAI_CHAT_MODEL` (default `gpt-4o-mini`).
- **Secrets:** never hardcode keys. Read `OPENAI_API_KEY` from the environment (optionally
  loaded from a local, git-ignored `.env` via `python-dotenv`; see `.env.example`).
- **Tests are required for core logic** (`src/`). Use `pytest`. Tests must not make network
  calls — mock the OpenAI client and use a temporary Chroma directory.
- **Lint/format:** run `ruff check .` and `ruff format --check .` before every PR; both must
  be clean. Config lives in `pyproject.toml`.
- **Small commits, small PRs:** one logical slice per PR; each PR must leave `pytest` green.
- **App changes:** any PR touching `app.py` (or behaviour visible in the UI) must follow
  `.agents/skills/test-app/SKILL.md`, including the manual smoke test.

## Environment variables

| Variable            | Required | Purpose                                   |
|---------------------|----------|-------------------------------------------|
| `OPENAI_API_KEY`    | yes      | OpenAI API access (embeddings + chat)     |
| `OPENAI_CHAT_MODEL` | no       | Chat model override (default `gpt-4o-mini`) |
| `CHROMA_DIR`        | no       | Vector store path (default `./chroma_db`) |

## Pre-PR checklist

```bash
pip install -r requirements.txt
ruff check . && ruff format --check .
pytest -q
```
