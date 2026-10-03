# Validation Record

This file records the repository validation gate.

Automated CI must run:
- Ruff lint
- mypy type checking
- pytest with coverage

Validated design contracts include compiled LangGraph orchestration, bounded retrieval loops, structured source lineage, request-scoped retrieval controls, OpenSearch BM25/hybrid query construction, category filtering, Ollama provider abstraction, prompt-injection boundary, non-root API container and Docker service DNS.
