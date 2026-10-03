import os
from uuid import uuid4

import pytest
from opensearchpy import OpenSearch

from app.agent.context import RuntimeContext
from app.ingestion.models import IngestDocument
from app.ingestion.opensearch_index import OpenSearchChunkIndex
from app.ingestion.service import IngestionService
from app.retrieval.opensearch import OpenSearchRetriever


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_OPENSEARCH_INTEGRATION") != "1",
    reason="OpenSearch integration service is not enabled",
)


class DeterministicEmbeddings:
    dimensions = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [
            [1.0, 0.0, 0.0] if "alpha" in text.lower() else [0.0, 1.0, 0.0]
            for text in texts
        ]


@pytest.mark.asyncio
async def test_ingest_then_hybrid_retrieve_with_tenant_isolation() -> None:
    client = OpenSearch(hosts=[os.getenv("OPENSEARCH_URL", "http://localhost:9200")])
    index_name = f"rag-ci-{uuid4().hex}"
    index = OpenSearchChunkIndex(client, index_name, "embedding", 3)
    embeddings = DeterministicEmbeddings()
    ingestion = IngestionService(index, embeddings, chunk_size=200, overlap=0)
    try:
        await ingestion.ingest([
            IngestDocument(
                document_id="tenant-a-doc", title="Alpha A",
                text="alpha release requirements",
                tenant_id="tenant-a", project_id="project-a",
            ),
            IngestDocument(
                document_id="tenant-b-doc", title="Alpha B",
                text="alpha private requirements",
                tenant_id="tenant-b", project_id="project-b",
            ),
        ])
        retriever = OpenSearchRetriever(client, embeddings, index=index_name)
        context = RuntimeContext(
            top_k=5, use_hybrid=True, model="test", categories=(),
            max_retrieval_attempts=2, guardrail_threshold=70,
            tenant_id="tenant-a", project_id="project-a",
        )
        results = await retriever.search("alpha requirements", context)
        assert results
        assert {item["document_id"] for item in results} == {"tenant-a-doc"}
        assert all(item["tenant_id"] == "tenant-a" for item in results)
    finally:
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
