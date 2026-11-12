"""Compare the agentic RAG system with a naive single-shot RAG baseline.

Usage:
    python -m eval.evaluate [--questions eval/questions.json] [--data eval/data]
                            [--out eval/results.md] [--k 4] [--judge-model gpt-4o-mini]

Metrics per system:
- correctness        LLM-as-judge vs. the reference answer (0 / 0.5 / 1)
- citation accuracy  LLM-as-judge: are claims cited and do cited chunks support them (0 / 0.5 / 1)
- retrieval hit-rate fraction of questions where a retrieved chunk contains the evidence span
- latency            mean wall-clock ms per question (excluding judging)
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src import config
from src.agent import Citation, SourceRegistry, finalize_citations, run_agent
from src.ingestion import get_collection, ingest_directory
from src.retrieval import DEFAULT_TOP_K, RetrievedChunk, retrieve

EVAL_DIR = Path(__file__).parent
EVAL_COLLECTION = "eval_corpus"
SYSTEMS = ("agentic", "baseline")

BASELINE_PROMPT = """Answer the question using ONLY the numbered sources below. Cite every \
factual claim inline with its source number in square brackets, e.g. [1]. If the sources do \
not contain the answer, say so."""

JUDGE_PROMPT = """You are a strict grader for a retrieval-augmented QA system.
Given a question, a reference answer, the system's answer, and the text of each source the \
answer cites, grade:

correctness (compare against the reference answer):
  2 = fully correct and complete; 1 = partially correct or missing key details;
  0 = wrong, hallucinated, or says it cannot answer.
citation_accuracy:
  2 = every factual claim has an inline citation [n] AND each cited source supports its claim;
  1 = some claims uncited, or some citations do not support their claim;
  0 = no citations, or citations are mostly wrong/unsupported.

Respond with JSON only:
{"correctness": 0|1|2, "citation_accuracy": 0|1|2, "rationale": "<one sentence>"}"""


@dataclass
class EvalItem:
    id: str
    question: str
    answer: str
    evidence: str


@dataclass
class SystemOutput:
    answer: str
    sources: list[Citation]
    retrieved: list[RetrievedChunk]
    latency_ms: float


@dataclass
class Judgement:
    correctness: float
    citation_accuracy: float
    rationale: str = ""


@dataclass
class Row:
    item: EvalItem
    system: str
    output: SystemOutput
    judgement: Judgement
    hit: bool


class RecordingRetriever:
    """Retriever bound to a collection that records every chunk it returns."""

    def __init__(self, collection, client, k: int = DEFAULT_TOP_K):
        self.collection, self.client, self.k = collection, client, k
        self.retrieved: list[RetrievedChunk] = []

    def __call__(self, query: str) -> list[RetrievedChunk]:
        chunks = retrieve(query, k=self.k, collection=self.collection, client=self.client)
        self.retrieved.extend(chunks)
        return chunks


def load_items(path: Path) -> list[EvalItem]:
    return [EvalItem(**raw) for raw in json.loads(Path(path).read_text())]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def evidence_hit(evidence: str, chunks: Sequence[RetrievedChunk]) -> bool:
    needle = _normalize(evidence)
    return any(needle in _normalize(c.text) for c in chunks)


def run_agentic(item: EvalItem, retriever, client, model: str) -> SystemOutput:
    start = time.perf_counter()
    result = run_agent(item.question, client=client, retriever=retriever, model=model)
    return SystemOutput(result.answer, result.sources, list(retriever.retrieved), _ms_since(start))


def run_baseline(item: EvalItem, retriever, client, model: str) -> SystemOutput:
    """Naive RAG: top-k retrieval on the raw question, then exactly one LLM call."""
    start = time.perf_counter()
    chunks = retriever(item.question)
    registry = SourceRegistry()
    context = registry.format_results(chunks)
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": BASELINE_PROMPT},
            {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {item.question}"},
        ],
    )
    answer, sources = finalize_citations(response.choices[0].message.content or "", registry.chunks)
    return SystemOutput(answer, sources, list(retriever.retrieved), _ms_since(start))


def judge(item: EvalItem, output: SystemOutput, client, model: str) -> Judgement:
    cited = "\n\n".join(f"[{c.number}] ({c.label})\n{c.text}" for c in output.sources) or "(none)"
    user = (
        f"Question: {item.question}\n\nReference answer: {item.answer}\n\n"
        f"System answer: {output.answer}\n\nCited sources:\n{cited}"
    )
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": JUDGE_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    return parse_judgement(response.choices[0].message.content or "")


def parse_judgement(raw: str) -> Judgement:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return Judgement(0.0, 0.0, f"unparseable judge output: {raw[:80]!r}")

    def score(key: str) -> float:
        try:
            return min(max(float(data.get(key, 0)), 0.0), 2.0) / 2.0
        except (TypeError, ValueError):
            return 0.0

    return Judgement(
        score("correctness"), score("citation_accuracy"), str(data.get("rationale", ""))
    )


def evaluate(
    items: Sequence[EvalItem],
    collection,
    client,
    model: str,
    judge_model: str,
    k: int = DEFAULT_TOP_K,
) -> list[Row]:
    runners = {"agentic": run_agentic, "baseline": run_baseline}
    rows: list[Row] = []
    for item in items:
        for system in SYSTEMS:
            retriever = RecordingRetriever(collection, client, k=k)
            output = runners[system](item, retriever, client, model)
            rows.append(
                Row(
                    item=item,
                    system=system,
                    output=output,
                    judgement=judge(item, output, client, judge_model),
                    hit=evidence_hit(item.evidence, output.retrieved),
                )
            )
    return rows


@dataclass
class Summary:
    system: str
    n: int
    correctness: float
    citation_accuracy: float
    hit_rate: float
    avg_latency_ms: float
    extra: dict = field(default_factory=dict)


def summarize(rows: Sequence[Row]) -> list[Summary]:
    out = []
    for system in SYSTEMS:
        rs = [r for r in rows if r.system == system]
        if not rs:
            continue
        n = len(rs)
        out.append(
            Summary(
                system=system,
                n=n,
                correctness=sum(r.judgement.correctness for r in rs) / n,
                citation_accuracy=sum(r.judgement.citation_accuracy for r in rs) / n,
                hit_rate=sum(r.hit for r in rs) / n,
                avg_latency_ms=sum(r.output.latency_ms for r in rs) / n,
            )
        )
    return out


def summary_table(summaries: Sequence[Summary]) -> str:
    lines = [
        "| System | N | Correctness | Citation accuracy | Retrieval hit-rate | Avg latency (ms) |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {s.system} | {s.n} | {s.correctness:.2f} | {s.citation_accuracy:.2f} | "
        f"{s.hit_rate:.2f} | {s.avg_latency_ms:.0f} |"
        for s in summaries
    ]
    return "\n".join(lines)


def per_question_table(rows: Sequence[Row]) -> str:
    lines = [
        "| ID | System | Correct | Citations | Hit | Latency (ms) | Judge rationale |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        rationale = r.judgement.rationale.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| {r.item.id} | {r.system} | {r.judgement.correctness:.1f} | "
            f"{r.judgement.citation_accuracy:.1f} | {'yes' if r.hit else 'no'} | "
            f"{r.output.latency_ms:.0f} | {rationale} |"
        )
    return "\n".join(lines)


def render_markdown(rows: Sequence[Row], model: str, judge_model: str, k: int) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "# Eval results: agentic RAG vs. naive baseline\n\n"
        f"- Generated: {stamp}\n- Answer model: `{model}`; judge model: `{judge_model}`; "
        f"top-k: {k}\n- Questions: {len({r.item.id for r in rows})} "
        "(`eval/questions.json`) over `eval/data/`\n\n"
        "Scores are means in [0, 1]. Correctness and citation accuracy are LLM-as-judge "
        "(0 / 0.5 / 1); hit-rate = share of questions where any retrieved chunk contains the "
        "reference evidence span.\n\n"
        "## Summary\n\n" + summary_table(summarize(rows)) + "\n\n"
        "## Per question\n\n" + per_question_table(rows) + "\n"
    )


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--data", type=Path, default=EVAL_DIR / "data")
    parser.add_argument("--out", type=Path, default=EVAL_DIR / "results.md")
    parser.add_argument("--chroma-dir", type=Path, default=EVAL_DIR / "chroma_db")
    parser.add_argument("--k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--model", default=config.chat_model())
    parser.add_argument("--judge-model", default=config.chat_model())
    args = parser.parse_args(argv)

    client = config.get_openai_client()
    collection = get_collection(args.chroma_dir, name=EVAL_COLLECTION)
    for r in ingest_directory(args.data, collection=collection, client=client):
        print(f"ingested {r.source}: {r.num_chunks} chunks")

    rows = evaluate(
        load_items(args.questions), collection, client, args.model, args.judge_model, args.k
    )
    print()
    print(summary_table(summarize(rows)))
    args.out.write_text(render_markdown(rows, args.model, args.judge_model, args.k))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
