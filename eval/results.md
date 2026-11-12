# Eval results: agentic RAG vs. naive baseline

- Generated: 2026-11-12 20:12 UTC
- Answer model: `gpt-4o-mini`; judge model: `gpt-4o-mini`; top-k: 4
- Questions: 10 (`eval/questions.json`) over `eval/data/`

Scores are means in [0, 1]. Correctness and citation accuracy are LLM-as-judge (0 / 0.5 / 1); hit-rate = share of questions where any retrieved chunk contains the reference evidence span.

## Summary

| System | N | Correctness | Citation accuracy | Retrieval hit-rate | Avg latency (ms) |
|---|---|---|---|---|---|
| agentic | 10 | 1.00 | 1.00 | 1.00 | 3218 |
| baseline | 10 | 1.00 | 1.00 | 1.00 | 1434 |

## Per question

| ID | System | Correct | Citations | Hit | Latency (ms) | Judge rationale |
|---|---|---|---|---|---|---|
| q1 | agentic | 1.0 | 1.0 | yes | 2932 | The system answer is fully correct and complete, and the cited source directly supports the claim made. |
| q1 | baseline | 1.0 | 1.0 | yes | 1174 | The system answer is fully correct and complete, and the cited source directly supports the claim made. |
| q2 | agentic | 1.0 | 1.0 | yes | 3988 | The system answer accurately reflects the reference answer regarding the wait time and default altitude, and the cited source supports all claims made. |
| q2 | baseline | 1.0 | 1.0 | yes | 1585 | The system answer is fully correct and complete, and the cited source supports all claims made. |
| q3 | agentic | 1.0 | 1.0 | yes | 2916 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q3 | baseline | 1.0 | 1.0 | yes | 1247 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q4 | agentic | 1.0 | 1.0 | yes | 2520 | The system answer is fully correct and complete, and the cited source directly supports the claims made. |
| q4 | baseline | 1.0 | 1.0 | yes | 1126 | The system answer is fully correct and complete, and the cited source directly supports the claims made. |
| q5 | agentic | 1.0 | 1.0 | yes | 2052 | The system answer is fully correct and complete, and the cited source directly supports the claim about the battery warranty. |
| q5 | baseline | 1.0 | 1.0 | yes | 1123 | The system answer is fully correct and complete, and the cited source directly supports the claim about the battery warranty. |
| q6 | agentic | 1.0 | 1.0 | yes | 3203 | The system answer is fully correct and complete, accurately reflecting the reference answer and all claims are well-supported by the cited source. |
| q6 | baseline | 1.0 | 1.0 | yes | 1980 | The system answer is fully correct and complete, and all claims are accurately supported by the cited source. |
| q7 | agentic | 1.0 | 1.0 | yes | 3598 | The system answer is fully correct and complete, accurately reflecting the reference answer and is well-supported by the cited source. |
| q7 | baseline | 1.0 | 1.0 | yes | 1492 | The system answer is fully correct and complete, accurately reflecting the reference answer, and the cited source supports all claims made. |
| q8 | agentic | 1.0 | 1.0 | yes | 2952 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q8 | baseline | 1.0 | 1.0 | yes | 1066 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q9 | agentic | 1.0 | 1.0 | yes | 4951 | The system answer is fully correct and complete, and all claims are accurately supported by the cited source. |
| q9 | baseline | 1.0 | 1.0 | yes | 2418 | The system answer is fully correct and complete, and it accurately cites a source that supports all claims made. |
| q10 | agentic | 1.0 | 1.0 | yes | 3065 | The system answer accurately reflects the reference answer and is fully supported by the cited source. |
| q10 | baseline | 1.0 | 1.0 | yes | 1129 | The system answer accurately reflects the reference answer and is fully supported by the cited source. |
