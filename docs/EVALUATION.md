# Evaluation Strategy

Evaluate the system at multiple layers rather than relying on answer similarity alone.

| Layer | Suggested metrics |
|---|---|
| Retrieval | Recall@K, Precision@K, MRR, nDCG |
| Context | relevance, redundancy, authorization correctness |
| Generation | groundedness/faithfulness, answer relevance |
| Citations | coverage, correctness, source/chunk match |
| Agent | route accuracy, rewrite quality, loop success, tool errors |
| Safety | injection resistance, data leakage, policy adherence |
| Operations | p50/p95 latency, retries, cache hit rate, tokens/cost |

Maintain a versioned golden dataset containing query, expected relevant document/chunk IDs, expected route, safety label and reference assertions. Run deterministic retrieval metrics in CI and model-judge evaluations in a controlled evaluation environment.
