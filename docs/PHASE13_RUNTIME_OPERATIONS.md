# Phase 13 Runtime Operations

Phase 13 adds dependency diagnostics for the production RAG runtime.

The liveness endpoint checks the API process. The readiness endpoint checks PostgreSQL, Redis, OpenSearch, and Ollama. Ollama readiness also verifies that the configured generation and embedding models are installed.

Operators can run the runtime check script in normal or JSON output mode. It exits successfully only when every required dependency is ready.

CI validates diagnostics logic plus the existing PostgreSQL, Redis, and real OpenSearch integration path. Large Ollama models are not downloaded in CI, so deployment environments must provision the configured models separately.
