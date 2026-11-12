from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval import evaluate as ev
from src.ingestion import ingest_text
from src.retrieval import RetrievedChunk
from tests.conftest import FakeOpenAI
from tests.fakes import FakeChatClient, message, tool_call

ITEM = ev.EvalItem(id="q1", question="How long?", answer="30 days", evidence="within  30\ndays")


def chunk(text: str, source: str = "a.txt", idx: int = 0) -> RetrievedChunk:
    return RetrievedChunk(f"{source}::chunk-{idx}", text, source, idx, 0.9)


class Combined:
    """Fake client exposing both embeddings and a scripted chat API."""

    def __init__(self, script):
        self.embeddings = FakeOpenAI().embeddings
        self.chat = FakeChatClient(script).chat

    @property
    def calls(self):
        return self.chat.completions.calls


def judge_msg(c: int, a: int, why: str = "ok"):
    return message(json.dumps({"correctness": c, "citation_accuracy": a, "rationale": why}))


def test_questions_file_is_valid_and_evidence_is_in_corpus():
    items = ev.load_items(ev.EVAL_DIR / "questions.json")
    assert 8 <= len(items) <= 10
    assert len({i.id for i in items}) == len(items)
    corpus = " ".join(p.read_text() for p in (ev.EVAL_DIR / "data").glob("*"))
    for item in items:
        assert ev.evidence_hit(item.evidence, [chunk(corpus)]), item.id


def test_evidence_hit_normalizes_whitespace_and_case():
    assert ev.evidence_hit(ITEM.evidence, [chunk("x"), chunk("Refunds WITHIN 30 days only")])
    assert not ev.evidence_hit(ITEM.evidence, [chunk("within 60 days")])
    assert not ev.evidence_hit(ITEM.evidence, [])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"correctness": 2, "citation_accuracy": 1, "rationale": "r"}', (1.0, 0.5, "r")),
        ('{"correctness": 7, "citation_accuracy": -1}', (1.0, 0.0, "")),
        ('{"correctness": "bad"}', (0.0, 0.0, "")),
    ],
)
def test_parse_judgement(raw, expected):
    j = ev.parse_judgement(raw)
    assert (j.correctness, j.citation_accuracy, j.rationale) == expected


def test_parse_judgement_invalid_json():
    j = ev.parse_judgement("not json")
    assert (j.correctness, j.citation_accuracy) == (0.0, 0.0)
    assert "unparseable" in j.rationale


def test_baseline_is_single_retrieval_and_single_llm_call():
    retrieved = [chunk("Refunds within 30 days.")]
    calls = []

    def retriever(q):
        calls.append(q)
        return retrieved

    retriever.retrieved = retrieved
    client = FakeChatClient([message("Within 30 days [1].")])
    out = ev.run_baseline(ITEM, retriever, client, "m")

    assert calls == [ITEM.question]
    assert len(client.calls) == 1
    assert "tools" not in client.calls[0]
    user = client.calls[0]["messages"][-1]["content"]
    assert '<source id="1" file="a.txt">\nRefunds within 30 days.\n</source>' in user
    assert out.answer == "Within 30 days [1]."
    assert [c.source for c in out.sources] == ["a.txt"]


def test_judge_receives_cited_source_text():
    client = FakeChatClient([judge_msg(2, 2)])
    out = ev.SystemOutput("A [1].", [ev.Citation(1, "a.txt", 0, "chunk body")], [], 1.0)
    j = ev.judge(ITEM, out, client, "judge")
    assert (j.correctness, j.citation_accuracy) == (1.0, 1.0)
    call = client.calls[0]
    assert call["model"] == "judge"
    assert call["response_format"] == {"type": "json_object"}
    assert "[1] (a.txt (chunk 0))\nchunk body" in call["messages"][-1]["content"]
    assert "Reference answer: 30 days" in call["messages"][-1]["content"]


def test_evaluate_end_to_end_with_fakes(collection, tmp_path):
    seed = FakeOpenAI()
    ingest_text("Refunds are accepted within 30 days.", "policy.txt", collection, seed)
    client = Combined(
        [
            message(tool_calls=[tool_call("refund window")]),  # agentic: tool call
            message("Within 30 days [1]."),  # agentic: answer
            judge_msg(2, 2),  # judge agentic
            message("30 days [1]."),  # baseline: single call
            judge_msg(1, 0, "partial"),  # judge baseline
        ]
    )

    rows = ev.evaluate([ITEM], collection, client, model="m", judge_model="j", k=2)

    assert [r.system for r in rows] == ["agentic", "baseline"]
    assert all(r.hit for r in rows)
    assert rows[0].output.sources[0].source == "policy.txt"
    summaries = {s.system: s for s in ev.summarize(rows)}
    assert summaries["agentic"].correctness == 1.0
    assert summaries["baseline"].correctness == 0.5
    assert summaries["baseline"].citation_accuracy == 0.0
    assert summaries["agentic"].hit_rate == 1.0

    md = ev.render_markdown(rows, "m", "j", 2)
    assert "| agentic | 1 | 1.00 | 1.00 | 1.00 |" in md
    assert "| baseline | 1 | 0.50 | 0.00 | 1.00 |" in md
    assert "| q1 | baseline | 0.5 | 0.0 | yes |" in md
    assert "partial" in md


def test_main_writes_results(monkeypatch, tmp_path, capsys):
    data = tmp_path / "data"
    data.mkdir()
    (data / "doc.txt").write_text("Refunds are accepted within 30 days.")
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([ITEM.__dict__]))
    client = Combined(
        [message("Within 30 days."), judge_msg(2, 0), message("30 days [1]."), judge_msg(2, 2)]
    )
    monkeypatch.setattr(ev.config, "get_openai_client", lambda: client)
    out = tmp_path / "results.md"

    ev.main(
        [
            "--questions",
            str(questions),
            "--data",
            str(data),
            "--out",
            str(out),
            "--chroma-dir",
            str(tmp_path / "db"),
            "--model",
            "m",
            "--judge-model",
            "j",
        ]
    )

    printed = capsys.readouterr().out
    assert "ingested doc.txt: 1 chunks" in printed
    assert "| System |" in printed
    text = Path(out).read_text()
    assert text.startswith("# Eval results")
    assert "| agentic | 1 | 1.00 | 0.00 | 0.00 |" in text
    assert "| baseline | 1 | 1.00 | 1.00 | 1.00 |" in text
