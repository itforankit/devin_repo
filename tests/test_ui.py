from __future__ import annotations

from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import agent, ingestion
from src.agent import AgentResult, Citation
from src.ui import format_latency, format_sources, to_message

APP_PATH = Path(__file__).resolve().parent.parent / "app.py"

RESULT = AgentResult(
    answer="The K7 flies for 110 minutes [1].",
    sources=[Citation(1, "manual.pdf", 2, "typical flight endurance is 110 minutes")],
    queries=["K7 flight endurance"],
    retrieval_ms=120.4,
    llm_ms=850.6,
    total_ms=990.2,
)


def test_format_latency():
    assert format_latency(120.4, 850.6, 990.2) == ("⏱ retrieval 120 ms · LLM 851 ms · total 990 ms")


def test_format_sources_lists_filename_and_chunk():
    cites = [Citation(1, "a.pdf", 0, "x"), Citation(2, "b.docx", 7, "y")]
    assert format_sources(cites) == "- **[1]** `a.pdf` — chunk 0\n- **[2]** `b.docx` — chunk 7"


def test_to_message():
    msg = to_message(RESULT)
    assert msg["role"] == "assistant"
    assert msg["content"] == RESULT.answer
    assert msg["sources"] == RESULT.sources
    assert msg["latency"] == {"retrieval_ms": 120.4, "llm_ms": 850.6, "total_ms": 990.2}


def test_list_sources_and_clear(collection, fake_openai):
    ingestion.ingest_text("x" * 2500, "big.txt", collection=collection, client=fake_openai)
    ingestion.ingest_text("small", "a.md", collection=collection, client=fake_openai)
    assert ingestion.list_sources(collection) == {"a.md": 1, "big.txt": 3}
    ingestion.clear_collection(collection)
    assert ingestion.list_sources(collection) == {}
    ingestion.clear_collection(collection)


@pytest.fixture
def app(monkeypatch, tmp_path):
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path / "chroma"))
    st.cache_resource.clear()
    calls = []

    def fake_run_agent(question, history=None, on_token=None, **_):
        calls.append({"question": question, "history": history})
        for token in ("The K7 ", "flies for ", "110 minutes [1]."):
            on_token(token)
        return RESULT

    monkeypatch.setattr(agent, "run_agent", fake_run_agent)
    at = AppTest.from_file(str(APP_PATH), default_timeout=30)
    at.calls = calls
    yield at
    st.cache_resource.clear()


def test_app_renders_empty_state(app):
    app.run()
    assert not app.exception
    assert app.title[0].value == "📚 Agentic RAG"
    assert any("No documents yet" in c.value for c in app.sidebar.caption)


def test_app_chat_shows_answer_citations_and_latency(app):
    app.run()
    app.chat_input[0].set_value("How long can the K7 fly?").run()

    assert not app.exception
    markdown = [m.value for m in app.markdown]
    assert "How long can the K7 fly?" in markdown
    assert RESULT.answer in markdown
    assert any("**[1]** `manual.pdf` — chunk 2" in m for m in markdown)
    captions = [c.value for c in app.caption]
    assert "⏱ retrieval 120 ms · LLM 851 ms · total 990 ms" in captions
    assert any("K7 flight endurance" in c for c in captions)

    app.chat_input[0].set_value("And with the thermal camera?").run()
    assert app.calls[1]["history"] == [
        {"role": "user", "content": "How long can the K7 fly?"},
        {"role": "assistant", "content": RESULT.answer},
    ]


def test_app_shows_agent_errors(app, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("OPENAI_API_KEY is not set.")

    monkeypatch.setattr(agent, "run_agent", boom)
    app.run()
    app.chat_input[0].set_value("hi").run()
    assert "Agent error: OPENAI_API_KEY is not set." in app.error[0].value
