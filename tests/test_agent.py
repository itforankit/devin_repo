from __future__ import annotations

import pytest

from src.agent import RETRIEVE_TOOL, SourceRegistry, finalize_citations, run_agent
from src.retrieval import RetrievedChunk
from tests.fakes import FakeChatClient, message, tool_call


def chunk(source: str, idx: int, text: str = "") -> RetrievedChunk:
    return RetrievedChunk(
        id=f"{source}::chunk-{idx}",
        text=text or f"{source} {idx}",
        source=source,
        chunk_index=idx,
        score=0.9,
    )


class FakeRetriever:
    def __init__(self, results: dict[str, list[RetrievedChunk]] | None = None, default=None):
        self.results = results or {}
        self.default = default or []
        self.queries: list[str] = []

    def __call__(self, query: str) -> list[RetrievedChunk]:
        self.queries.append(query)
        return self.results.get(query, self.default)


class TestNoRetrievalPath:
    def test_greeting_answers_directly_without_tool_call(self):
        client = FakeChatClient([message("Hello! How can I help with your documents?")])
        retriever = FakeRetriever()

        result = run_agent("hi", client=client, retriever=retriever, model="m")

        assert result.answer == "Hello! How can I help with your documents?"
        assert result.sources == []
        assert result.queries == []
        assert not result.used_retrieval
        assert result.retrieval_ms == 0
        assert retriever.queries == []
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["tools"] == [RETRIEVE_TOOL]
        assert call["tool_choice"] == "auto"
        assert call["messages"][-1] == {"role": "user", "content": "hi"}

    def test_stray_citations_without_retrieval_are_removed(self):
        client = FakeChatClient([message("Hi there [1].")])
        result = run_agent("hello", client=client, retriever=FakeRetriever(), model="m")
        assert result.answer == "Hi there."
        assert result.sources == []


class TestRetrievalPath:
    def test_retrieves_then_answers_with_citations(self):
        retriever = FakeRetriever(
            {"refund policy duration": [chunk("policy.pdf", 3, "30 days"), chunk("faq.md", 0)]}
        )
        client = FakeChatClient(
            [
                message(tool_calls=[tool_call("refund policy duration")]),
                message("Refunds are accepted within 30 days [1]. See the FAQ [2]."),
            ]
        )

        result = run_agent("How long do refunds take?", client=client, retriever=retriever)

        assert result.used_retrieval
        assert result.queries == ["refund policy duration"]
        assert result.answer == "Refunds are accepted within 30 days [1]. See the FAQ [2]."
        assert [(c.number, c.source, c.chunk_index) for c in result.sources] == [
            (1, "policy.pdf", 3),
            (2, "faq.md", 0),
        ]
        assert result.sources[0].text == "30 days"
        assert result.sources[0].label == "policy.pdf (chunk 3)"

        second_messages = client.calls[1]["messages"]
        assistant, tool_msg = second_messages[-2], second_messages[-1]
        assert assistant["tool_calls"][0]["function"]["name"] == "retrieve"
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_1"
        assert tool_msg["content"].startswith('<source id="1" file="policy.pdf">\n30 days')
        assert '<source id="2" file="faq.md">' in tool_msg["content"]

    def test_reformulation_reuses_numbers_for_duplicate_chunks(self):
        a, b = chunk("a.txt", 0), chunk("b.txt", 1)
        retriever = FakeRetriever({"first try": [a], "reformulated": [a, b]})
        client = FakeChatClient(
            [
                message(tool_calls=[tool_call("first try", "c1")]),
                message(tool_calls=[tool_call("reformulated", "c2")]),
                message("Answer from b [2] and a [1]."),
            ]
        )

        result = run_agent("q", client=client, retriever=retriever)

        assert result.queries == ["first try", "reformulated"]
        second_tool_output = client.calls[2]["messages"][-1]["content"]
        assert second_tool_output.startswith('<source id="1" file="a.txt">')
        assert '<source id="2" file="b.txt">' in second_tool_output
        assert result.answer == "Answer from b [1] and a [2]."
        assert [c.source for c in result.sources] == ["b.txt", "a.txt"]

    def test_history_is_passed_to_model(self):
        client = FakeChatClient([message("Sure.")])
        history = [
            {"role": "user", "content": "What is X?"},
            {"role": "assistant", "content": "X is Y [1]."},
        ]
        run_agent("and Z?", history=history, client=client, retriever=FakeRetriever())
        msgs = client.calls[0]["messages"]
        assert msgs[0]["role"] == "system"
        assert msgs[1:] == [*history, {"role": "user", "content": "and Z?"}]

    def test_empty_results_tell_model_nothing_found(self):
        client = FakeChatClient(
            [message(tool_calls=[tool_call("unknown")]), message("I couldn't find that.")]
        )
        result = run_agent("q", client=client, retriever=FakeRetriever())
        assert client.calls[1]["messages"][-1]["content"] == "No matching documents found."
        assert result.sources == []

    def test_forces_final_answer_after_max_steps(self):
        client = FakeChatClient(
            [
                message(tool_calls=[tool_call("q1", "c1")]),
                message(tool_calls=[tool_call("q2", "c2")]),
                message("Best effort [1]."),
            ]
        )
        retriever = FakeRetriever(default=[chunk("a.txt", 0)])
        result = run_agent("q", client=client, retriever=retriever, max_steps=3)
        assert [c["tool_choice"] for c in client.calls] == ["auto", "auto", "none"]
        assert result.answer == "Best effort [1]."

    @pytest.mark.parametrize(
        ("name", "raw", "expected"),
        [
            ("search_web", None, "unknown tool"),
            ("retrieve", "not json", "must be JSON"),
            ("retrieve", '{"query": "  "}', "non-empty"),
        ],
    )
    def test_bad_tool_calls_return_errors_to_model(self, name, raw, expected):
        client = FakeChatClient(
            [message(tool_calls=[tool_call("x", name=name, raw=raw)]), message("ok")]
        )
        retriever = FakeRetriever()
        run_agent("q", client=client, retriever=retriever)
        assert expected in client.calls[1]["messages"][-1]["content"]
        assert retriever.queries == []

    def test_streaming_emits_tokens_and_handles_tool_calls(self):
        retriever = FakeRetriever(default=[chunk("doc.txt", 2)])
        client = FakeChatClient(
            [
                message(tool_calls=[tool_call("streamed query")]),
                message("Streaming answer with a citation [1]."),
            ]
        )
        tokens: list[str] = []

        result = run_agent("q", client=client, retriever=retriever, on_token=tokens.append)

        assert all(call["stream"] is True for call in client.calls)
        assert retriever.queries == ["streamed query"]
        assert "".join(tokens) == "Streaming answer with a citation [1]."
        assert result.answer == "Streaming answer with a citation [1]."
        assert result.sources[0].source == "doc.txt"

    def test_latency_breakdown_is_recorded(self):
        client = FakeChatClient([message(tool_calls=[tool_call("q")]), message("A [1].")])
        result = run_agent("q", client=client, retriever=FakeRetriever(default=[chunk("a", 0)]))
        assert result.llm_ms > 0
        assert result.retrieval_ms > 0
        assert result.total_ms >= result.llm_ms
        assert result.total_ms >= result.retrieval_ms


class TestCitationFormatting:
    def chunks(self, n: int) -> dict[int, RetrievedChunk]:
        return {i: chunk(f"f{i}.txt", i) for i in range(1, n + 1)}

    def test_renumbers_by_first_appearance(self):
        text, cites = finalize_citations("C [3]. A [1]. C again [3].", self.chunks(3))
        assert text == "C [1]. A [2]. C again [1]."
        assert [(c.number, c.source) for c in cites] == [(1, "f3.txt"), (2, "f1.txt")]

    def test_grouped_citations_are_split(self):
        text, cites = finalize_citations("Both [1, 2].", self.chunks(2))
        assert text == "Both [1][2]."
        assert len(cites) == 2

    def test_unknown_numbers_are_dropped(self):
        text, cites = finalize_citations("Real [1], fake [9], mixed [2, 7].", self.chunks(2))
        assert text == "Real [1], fake, mixed [2]."
        assert [c.number for c in cites] == [1, 2]

    def test_non_citation_brackets_untouched(self):
        text, cites = finalize_citations("Use list[str] or [a] here.", self.chunks(1))
        assert text == "Use list[str] or [a] here."
        assert cites == []

    def test_registry_formats_tagged_blocks(self):
        reg = SourceRegistry()
        out = reg.format_results([chunk("a.txt", 0, "alpha"), chunk("b.txt", 4, "beta")])
        assert out.startswith(
            '<source id="1" file="a.txt">\nalpha\n</source>\n\n'
            '<source id="2" file="b.txt">\nbeta\n</source>'
        )
        assert reg.add(chunk("a.txt", 0)) == 1
