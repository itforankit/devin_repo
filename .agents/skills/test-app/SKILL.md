---
name: test-app
description: How to install, run, and test the Agentic RAG app (pytest + manual Streamlit smoke test). Follow before every PR that touches the app.
---

# Test the Agentic RAG app

Follow every step below before opening any PR that touches `app.py` or app-visible
behaviour (`src/`). Record the manual smoke test when the PR changes the UI.

## 1. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

`OPENAI_API_KEY` must be set in the environment (or in a git-ignored `.env`). Never paste
the key into code, commits, or PR text.

```bash
test -n "$OPENAI_API_KEY" && echo "key present"
```

## 2. Automated checks

```bash
ruff check . && ruff format --check .
pytest -q
```

Both must pass. Tests mock OpenAI, so they run without network access.

## 3. Run the app

```bash
streamlit run app.py --server.port 8501 --server.headless true
```

Open http://localhost:8501.

## 4. Manual UI smoke test

1. **Upload a document:** use the sidebar file uploader to upload a PDF, DOCX, or TXT
   (e.g. `eval/data/` sample doc). Confirm a success message showing the number of chunks
   ingested.
2. **Ask a question** answerable from the document via the chat input.
3. **Verify the answer:**
   - The answer streams in and is grounded in the document.
   - Inline citations like `[1]`, `[2]` appear in the text.
   - A sources/citations list renders below the answer with filename + chunk index for each
     cited number.
4. **Verify latency display:** under the answer, the per-message latency shows
   retrieval ms, LLM ms, and total ms (non-zero, total >= retrieval + LLM is not required but
   total must be >= each component).
5. **No-retrieval path:** send a greeting such as "hi". Expect a short reply with no
   citations and retrieval ms of 0.
6. Check the terminal running Streamlit for tracebacks — there should be none.

## 5. Clean up

Stop Streamlit (Ctrl+C). Remove `./chroma_db` if you want a clean store for the next run.
