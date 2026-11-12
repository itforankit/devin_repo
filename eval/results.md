# Eval results: agentic RAG vs. naive baseline

- Generated: 2026-11-12 20:05 UTC
- Answer model: `gpt-4o-mini`; judge model: `gpt-4o-mini`; top-k: 4
- Questions: 10 (`eval/questions.json`) over `eval/data/`

Scores are means in [0, 1]. Correctness and citation accuracy are LLM-as-judge (0 / 0.5 / 1); hit-rate = share of questions where any retrieved chunk contains the reference evidence span.

## Summary

| System | N | Correctness | Citation accuracy | Retrieval hit-rate | Avg latency (ms) |
|---|---|---|---|---|---|
| agentic | 10 | 1.00 | 1.00 | 1.00 | 3614 |
| baseline | 10 | 1.00 | 1.00 | 1.00 | 1528 |

## Per question

| ID | System | Correct | Citations | Hit | Latency (ms) | Judge rationale |
|---|---|---|---|---|---|---|
| q1 | agentic | 1.0 | 1.0 | yes | 5036 | The system answer is fully correct and complete, accurately reflecting the reference answer and providing additional relevant details, with all claims supported by the cited source. |
| q1 | baseline | 1.0 | 1.0 | yes | 3193 | The system answer is fully correct and complete, and the cited source directly supports the claim made. |
| q2 | agentic | 1.0 | 1.0 | yes | 4192 | The system answer accurately reflects the reference answer and is fully supported by the cited source. |
| q2 | baseline | 1.0 | 1.0 | yes | 1507 | The system answer accurately reflects the reference answer and is fully supported by the cited source. |
| q3 | agentic | 1.0 | 1.0 | yes | 3155 | The system answer is fully correct and complete, and the cited source directly supports the claim made. |
| q3 | baseline | 1.0 | 1.0 | yes | 2353 | The system answer is fully correct and complete, and the cited source directly supports the claim made. |
| q4 | agentic | 1.0 | 1.0 | yes | 3031 | The system's answer is fully correct and complete, and the cited source directly supports the claims made. |
| q4 | baseline | 1.0 | 1.0 | yes | 987 | The system answer is fully correct and complete, and the cited source directly supports the claims made. |
| q5 | agentic | 1.0 | 1.0 | yes | 2088 | The system answer is fully correct and complete, and the cited source directly supports the claim about the battery warranty. |
| q5 | baseline | 1.0 | 1.0 | yes | 809 | The system answer is fully correct and complete, and the cited source directly supports the claim about the battery warranty. |
| q6 | agentic | 1.0 | 1.0 | yes | 3710 | The system answer is fully correct and complete, accurately reflecting the details of KestrelCare Plus and correctly cites a source that supports all claims. |
| q6 | baseline | 1.0 | 1.0 | yes | 1261 | The system answer is fully correct and complete, and all claims are accurately supported by the cited source. |
| q7 | agentic | 1.0 | 1.0 | yes | 3311 | The system answer is fully correct and complete, accurately reflecting the reference answer and supported by the cited source. |
| q7 | baseline | 1.0 | 1.0 | yes | 1268 | The system answer is fully correct and complete, accurately reflecting the reference answer, and the cited source supports all claims made. |
| q8 | agentic | 1.0 | 1.0 | yes | 2568 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q8 | baseline | 1.0 | 1.0 | yes | 1229 | The system answer is fully correct and complete, and the cited source directly supports the claim. |
| q9 | agentic | 1.0 | 1.0 | yes | 4166 | The system answer is fully correct and complete, accurately reflecting the reference answer and properly citing a source that supports all claims. |
| q9 | baseline | 1.0 | 1.0 | yes | 1462 | The system answer is fully correct and complete, and it accurately cites a source that supports all claims made. |
| q10 | agentic | 1.0 | 1.0 | yes | 4878 | The system answer accurately reflects the ranges of both the live video link and the command-and-control link, and the cited source supports these claims. |
| q10 | baseline | 1.0 | 1.0 | yes | 1214 | The system answer accurately reflects the reference answer and is fully supported by the cited source. |
