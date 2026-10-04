from __future__ import annotations

import argparse
import asyncio
import json
from uuid import uuid4

from opensearchpy import OpenSearch

from app.config import get_settings
from app.embeddings.ollama import OllamaEmbeddingProvider
from app.ingestion.models import IngestDocument
from app.ingestion.opensearch_index import OpenSearchChunkIndex
from app.ingestion.service import IngestionService
from app.llm.ollama import OllamaProvider
from app.models import AskRequest
from app.retrieval.opensearch import OpenSearchRetriever
from app.services.agentic_rag import AgenticRAGService


async def run_smoke() -> dict[str, object]:
    settings = get_settings()
    client = OpenSearch(hosts=[settings.opensearch_url])
    index_name = f"rag-live-smoke-{uuid4().hex}"
    embeddings = OllamaEmbeddingProvider(
        settings.ollama_url, settings.embedding_model, settings.embedding_dimensions
    )
    index = OpenSearchChunkIndex(
        client, index_name, settings.opensearch_vector_field, embeddings.dimensions
    )
    ingestion = IngestionService(index, embeddings, chunk_size=400, overlap=40)
    retriever = OpenSearchRetriever(
        client, embeddings, index=index_name, vector_field=settings.opensearch_vector_field
    )
    service = AgenticRAGService(
        retriever,
        OllamaProvider(settings.ollama_url),
        max_attempts=1,
        guardrail_threshold=settings.guardrail_threshold,
    )
    tenant = "phase14-smoke"
    project = uuid4().hex
    try:
        indexed = await ingestion.ingest([
            IngestDocument(
                document_id="smoke-doc",
                title="Phase 14 Smoke Evidence",
                text=(
                    "The production RAG smoke verification code is ORBIT-731. "
                    "This value exists only to validate grounded retrieval and generation."
                ),
                tenant_id=tenant,
                project_id=project,
            )
        ])
        response = await service.ask(
            AskRequest(
                query="What is the production RAG smoke verification code?",
                top_k=3,
                use_hybrid=True,
                model=settings.default_model,
                tenant_id=tenant,
                project_id=project,
            )
        )
        passed = (
            indexed.chunks_indexed > 0
            and bool(response.sources)
            and "ORBIT-731" in response.answer.upper()
        )
        return {
            "status": "passed" if passed else "failed",
            "index": index_name,
            "chunks_indexed": indexed.chunks_indexed,
            "sources": len(response.sources),
            "answer_contains_verification_code": "ORBIT-731" in response.answer.upper(),
            "retrieval_attempts": response.retrieval_attempts,
        }
    finally:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)


async def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live Ollama + OpenSearch RAG smoke test")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = await run_smoke()
    print(json.dumps(result, sort_keys=True) if args.json else result)
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
