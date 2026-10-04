# Phase 14 Live RAG Smoke

Phase 14 adds an opt-in validation path for the complete AI runtime.

The live smoke verifies the configured Ollama embedding model, OpenSearch vector ingestion, tenant-scoped hybrid retrieval, and Ollama answer generation against known evidence. The temporary smoke index is deleted after the run.

The normal pull-request CI remains lightweight and deterministic. The Live RAG Smoke workflow is manual and targets a self-hosted runner where Ollama and the configured models are already provisioned. PostgreSQL, Redis, and OpenSearch are supplied by the workflow.

Before running the workflow, install the configured generation and embedding models on the Ollama host. The runtime readiness check runs before the smoke and fails when either model is missing.

This separation prevents ordinary code validation from downloading large model artifacts while still providing an executable production-runtime gate when a model-capable runner is available.
