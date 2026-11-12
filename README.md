# Agentic RAG

An Agentic Retrieval-Augmented Generation system:

- **Ingestion** of PDF / DOCX / TXT / MD into a local Chroma vector store
  (OpenAI `text-embedding-3-small`).
- **Agent** built on OpenAI function calling that decides whether retrieval is needed,
  reformulates queries, and answers with inline citations.
- **Eval harness** comparing the agent with a naive single-shot RAG baseline.
- **Streamlit UI** with upload, streamed answers, citations, and latency breakdown.

## Requirements

- Python 3.10+
- `OPENAI_API_KEY` environment variable (required). Never commit it — copy `.env.example`
  to `.env` for local use if you prefer.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export OPENAI_API_KEY=...        # or put it in .env
pytest -q                        # tests (OpenAI mocked)
ruff check .                     # lint
streamlit run app.py             # UI (available once Phase 5 lands)
```

See [AGENTS.md](AGENTS.md) for structure and conventions, and
[.agents/skills/test-app/SKILL.md](.agents/skills/test-app/SKILL.md) for the full test
procedure.
