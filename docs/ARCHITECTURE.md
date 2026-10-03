# Architecture

## Request path

1. FastAPI validates the request.
2. A request-scoped `RuntimeContext` captures retrieval/model controls.
3. Guardrail checks the query before retrieval.
4. Retriever applies `top_k`, search mode and category filters.
5. Evidence is graded/adapted; insufficient evidence triggers bounded rewrite/retry.
6. Source lineage is constructed directly from structured retrieval metadata.
7. Retrieved text is wrapped as untrusted evidence before generation.
8. Response returns answer, sources, reasoning metadata, attempts and trace ID.

## Component boundaries

- `app/main.py`: transport/API.
- `app/models.py`: external contracts.
- `app/agent/`: state, runtime context, security and lineage.
- `app/retrieval/`: interchangeable retrieval adapters.
- `app/services/`: orchestration.
- `app/evaluation/`: deterministic quality metrics.

## Enterprise extension points

Use dependency injection to add an LLM adapter, embedding provider, OpenSearch hybrid search pipeline, Redis cache, Langfuse tracer and identity-aware authorization filter. Keep tenant/project/ACL filters in retrieval, not only in the UI or prompt.

## Trust boundaries

User input and retrieved documents are both untrusted. Secrets never belong in prompts or repository files. Production OpenSearch/Redis must be private and authenticated. Retrieval authorization must be enforced before evidence reaches the model.
