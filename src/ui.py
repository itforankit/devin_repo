"""Pure formatting helpers for the Streamlit UI (kept separate so they are unit-testable)."""

from __future__ import annotations

from collections.abc import Sequence

from src.agent import AgentResult, Citation

PREVIEW_CHARS = 300


def format_latency(retrieval_ms: float, llm_ms: float, total_ms: float) -> str:
    return f"⏱ retrieval {retrieval_ms:.0f} ms · LLM {llm_ms:.0f} ms · total {total_ms:.0f} ms"


def format_sources(sources: Sequence[Citation]) -> str:
    """Markdown list mapping each citation number to filename + chunk index."""
    return "\n".join(f"- **[{c.number}]** `{c.source}` — chunk {c.chunk_index}" for c in sources)


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def to_message(result: AgentResult) -> dict:
    """Serialisable chat-history entry for an assistant turn."""
    return {
        "role": "assistant",
        "content": result.answer,
        "sources": list(result.sources),
        "queries": list(result.queries),
        "latency": {
            "retrieval_ms": result.retrieval_ms,
            "llm_ms": result.llm_ms,
            "total_ms": result.total_ms,
        },
    }
