"""Streamlit chat UI for the Agentic RAG system.

Run with: streamlit run app.py   (requires OPENAI_API_KEY in the environment)
"""

from __future__ import annotations

import hashlib

import streamlit as st

from src import agent, ingestion
from src.ui import format_latency, format_sources, to_message

st.set_page_config(page_title="Agentic RAG", page_icon="📚", layout="wide")


@st.cache_resource
def get_collection():
    return ingestion.get_collection()


def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("ingested", set())


def render_assistant_extras(msg: dict) -> None:
    if msg.get("sources"):
        st.markdown("**Sources**\n\n" + format_sources(msg["sources"]))
        with st.expander("Show cited passages"):
            for c in msg["sources"]:
                st.markdown(f"**[{c.number}] {c.label}**")
                st.code(c.text, language=None, wrap_lines=True)
    if msg.get("queries"):
        st.caption("Search queries: " + " · ".join(f"“{q}”" for q in msg["queries"]))
    if msg.get("latency"):
        st.caption(format_latency(**msg["latency"]))


def sidebar(collection) -> None:
    with st.sidebar:
        st.header("📄 Documents")
        uploads = st.file_uploader(
            "Upload PDF, DOCX, TXT or MD",
            type=[ext.lstrip(".") for ext in ingestion.SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
        )
        for upload in uploads or []:
            data = upload.getvalue()
            key = (upload.name, hashlib.sha256(data).hexdigest())
            if key in st.session_state.ingested:
                continue
            try:
                with st.spinner(f"Ingesting {upload.name}…"):
                    result = ingestion.ingest_bytes(upload.name, data, collection=collection)
            except Exception as exc:  # surface any loader/API failure in the UI
                st.error(f"Failed to ingest {upload.name}: {exc}")
                continue
            st.session_state.ingested.add(key)
            st.success(f"Ingested **{result.source}**: {result.num_chunks} chunks")

        sources = ingestion.list_sources(collection)
        st.subheader("Knowledge base")
        if sources:
            st.markdown("\n".join(f"- `{name}` — {n} chunks" for name, n in sources.items()))
        else:
            st.caption("No documents yet. Upload one to get started.")
        col1, col2 = st.columns(2)
        if col1.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if col2.button("Clear documents", use_container_width=True, disabled=not sources):
            ingestion.clear_collection(collection)
            st.session_state.ingested = set()
            st.rerun()


def main() -> None:
    init_state()
    collection = get_collection()
    sidebar(collection)

    st.title("📚 Agentic RAG")
    st.caption(
        "Ask about your uploaded documents. The agent decides when to search and cites "
        "its sources as [1], [2], …"
    )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_assistant_extras(msg)

    prompt = st.chat_input("Ask a question about your documents")
    if not prompt:
        return

    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        streamed: list[str] = []

        def on_token(token: str) -> None:
            streamed.append(token)
            placeholder.markdown("".join(streamed) + "▌")

        try:
            with st.spinner("Thinking…"):
                result = agent.run_agent(prompt, history=history, on_token=on_token)
        except Exception as exc:  # e.g. missing OPENAI_API_KEY, quota or network errors
            placeholder.empty()
            st.error(f"Agent error: {exc}")
            st.session_state.messages.pop()
            return

        placeholder.markdown(result.answer)
        msg = to_message(result)
        render_assistant_extras(msg)
    st.session_state.messages.append(msg)


main()
