"""Agentic RAG loop built on OpenAI function calling.

The model decides whether to call the ``retrieve`` tool (e.g. greetings need no
retrieval), writes/rewrites the search query, may search again with a reformulated
query, and answers with inline ``[n]`` citations that map to retrieved chunks.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src import config
from src.retrieval import DEFAULT_TOP_K, RetrievedChunk, retrieve

Retriever = Callable[[str], list[RetrievedChunk]]
TokenCallback = Callable[[str], None]

MAX_STEPS = 4

SYSTEM_PROMPT = """You are a helpful assistant answering questions about the user's \
documents, which live in a searchable knowledge base.

Decide first whether the knowledge base is needed:
- Greetings, small talk, thanks, or questions about yourself: answer directly and briefly \
WITHOUT calling any tool and WITHOUT citations.
- Anything that could be answered from the documents: call `retrieve`.

When retrieving:
- Write a concise, standalone search query (resolve pronouns and references using the \
conversation; use key terms likely to appear in the documents).
- If the results do not contain the answer, you may call `retrieve` again with a \
reformulated query (different wording, synonyms, or a narrower/broader focus).
- Answer ONLY from retrieved sources. Cite every factual claim inline with the source \
number in square brackets, e.g. "The limit is 5 GB [2]." Use separate brackets for \
multiple sources: [1][3]. Never invent source numbers.
- If the sources do not contain the answer, say you could not find it in the documents."""

RETRIEVE_TOOL = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the user's document knowledge base and return the most relevant "
            "chunks, each labelled with a citation number like [1]."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Standalone search query optimised for semantic search.",
                }
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    number: int
    source: str
    chunk_index: int
    text: str

    @property
    def label(self) -> str:
        return f"{self.source} (chunk {self.chunk_index})"


@dataclass
class AgentResult:
    answer: str
    sources: list[Citation] = field(default_factory=list)
    queries: list[str] = field(default_factory=list)
    retrieval_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0

    @property
    def used_retrieval(self) -> bool:
        return bool(self.queries)


class SourceRegistry:
    """Assigns stable per-turn citation numbers to retrieved chunks."""

    def __init__(self) -> None:
        self._by_id: dict[str, int] = {}
        self.chunks: dict[int, RetrievedChunk] = {}

    def add(self, chunk: RetrievedChunk) -> int:
        if chunk.id not in self._by_id:
            number = len(self._by_id) + 1
            self._by_id[chunk.id] = number
            self.chunks[number] = chunk
        return self._by_id[chunk.id]

    def format_results(self, chunks: Sequence[RetrievedChunk]) -> str:
        if not chunks:
            return "No matching documents found."
        blocks = []
        for chunk in chunks:
            n = self.add(chunk)
            blocks.append(
                f"[{n}] (source: {chunk.source}, chunk {chunk.chunk_index})\n{chunk.text}"
            )
        return "\n\n".join(blocks)


def finalize_citations(
    answer: str, chunks: dict[int, RetrievedChunk]
) -> tuple[str, list[Citation]]:
    """Renumber cited sources 1..n by first appearance and drop unknown numbers.

    ``[1, 3]`` style groups are normalised to ``[1][3]``.
    """
    mapping: dict[int, int] = {}
    citations: list[Citation] = []

    def replace(match: re.Match) -> str:
        out = []
        for raw in match.group(1).split(","):
            old = int(raw)
            if old not in chunks:
                continue
            if old not in mapping:
                mapping[old] = len(mapping) + 1
                c = chunks[old]
                citations.append(Citation(mapping[old], c.source, c.chunk_index, c.text))
            out.append(f"[{mapping[old]}]")
        return "".join(out)

    text = CITATION_RE.sub(replace, answer)
    text = re.sub(r"[ \t]+([.,;:!?])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text, citations


def _assistant_message(message) -> dict:
    out: dict = {"role": "assistant", "content": message.content or ""}
    if getattr(message, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.function.name, "arguments": tc.function.arguments},
            }
            for tc in message.tool_calls
        ]
    return out


class _Message:
    def __init__(self, content: str | None, tool_calls: list | None):
        self.content = content
        self.tool_calls = tool_calls or None


class _ToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id = id
        self.type = "function"
        self.function = _Function(name, arguments)


class _Function:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


def _collect_stream(stream, on_token: TokenCallback) -> _Message:
    content: list[str] = []
    calls: dict[int, dict] = {}
    for event in stream:
        if not event.choices:
            continue
        delta = event.choices[0].delta
        if getattr(delta, "content", None):
            content.append(delta.content)
            on_token(delta.content)
        for tc in getattr(delta, "tool_calls", None) or []:
            slot = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
            if tc.id:
                slot["id"] = tc.id
            if tc.function is not None:
                slot["name"] += tc.function.name or ""
                slot["arguments"] += tc.function.arguments or ""
    tool_calls = [_ToolCall(c["id"], c["name"], c["arguments"]) for _, c in sorted(calls.items())]
    return _Message("".join(content) or None, tool_calls)


def run_agent(
    question: str,
    history: Sequence[dict] | None = None,
    *,
    client=None,
    retriever: Retriever | None = None,
    model: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    max_steps: int = MAX_STEPS,
    on_token: TokenCallback | None = None,
) -> AgentResult:
    """Answer ``question`` with the agentic loop.

    ``history`` is prior turns as ``{"role": "user"|"assistant", "content": str}``.
    ``on_token`` (optional) receives streamed answer text as it is generated.
    """
    total_start = time.perf_counter()
    client = client or config.get_openai_client()
    model = model or config.chat_model()
    if retriever is None:

        def retriever(q: str) -> list[RetrievedChunk]:
            return retrieve(q, k=top_k, client=client)

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend({"role": m["role"], "content": m["content"]} for m in history or [])
    messages.append({"role": "user", "content": question})

    registry = SourceRegistry()
    result = AgentResult(answer="")

    for step in range(max_steps):
        force_answer = step == max_steps - 1
        llm_start = time.perf_counter()
        kwargs = {
            "model": model,
            "messages": messages,
            "tools": [RETRIEVE_TOOL],
            "tool_choice": "none" if force_answer else "auto",
            "temperature": 0,
        }
        if on_token is not None:
            message = _collect_stream(
                client.chat.completions.create(**kwargs, stream=True), on_token
            )
        else:
            message = client.chat.completions.create(**kwargs).choices[0].message
        result.llm_ms += (time.perf_counter() - llm_start) * 1000

        if not message.tool_calls:
            result.answer, result.sources = finalize_citations(
                message.content or "", registry.chunks
            )
            break

        messages.append(_assistant_message(message))
        for call in message.tool_calls:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": _run_tool(call, retriever, registry, result),
                }
            )

    result.total_ms = (time.perf_counter() - total_start) * 1000
    return result


def _run_tool(call, retriever: Retriever, registry: SourceRegistry, result: AgentResult) -> str:
    if call.function.name != "retrieve":
        return f"Error: unknown tool '{call.function.name}'."
    try:
        query = str(json.loads(call.function.arguments or "{}").get("query", "")).strip()
    except json.JSONDecodeError:
        return "Error: arguments must be JSON with a 'query' string."
    if not query:
        return "Error: 'query' must be a non-empty string."
    result.queries.append(query)
    start = time.perf_counter()
    chunks = retriever(query)
    result.retrieval_ms += (time.perf_counter() - start) * 1000
    return registry.format_results(chunks)
