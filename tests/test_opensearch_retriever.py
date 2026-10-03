import pytest

from app.agent.context import RuntimeContext
from app.retrieval.opensearch import OpenSearchRetriever


class FakeEmbeddings:
    dimensions = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.5] for _ in texts]


class FakeClient:
    def __init__(self) -> None:
        self.bodies: list[dict[str, object]] = []

    def search(self, *, index: str, body: dict[str, object], params=None):
        self.bodies.append(body)
        if "knn" in body["query"]:
            return {"hits": {"hits": [
                {"_id": "b", "_score": 0.9, "_source": {"chunk_id": "b", "document_id": "2", "text": "vector"}},
                {"_id": "a", "_score": 0.8, "_source": {"chunk_id": "a", "document_id": "1", "text": "shared"}},
            ]}}
        return {"hits": {"hits": [
            {"_id": "a", "_score": 10.0, "_source": {"chunk_id": "a", "document_id": "1", "text": "shared"}},
            {"_id": "c", "_score": 9.0, "_source": {"chunk_id": "c", "document_id": "3", "text": "lexical"}},
        ]}}


def context(hybrid: bool = True) -> RuntimeContext:
    return RuntimeContext(
        top_k=2, use_hybrid=hybrid, model="test", categories=("requirements",),
        max_retrieval_attempts=2, guardrail_threshold=70,
        tenant_id="tenant-a", project_id="project-a",
    )


@pytest.mark.asyncio
async def test_hybrid_uses_query_embedding_knn_and_rrf() -> None:
    client = FakeClient()
    retriever = OpenSearchRetriever(client, FakeEmbeddings(), rrf_k=60, candidate_multiplier=2)
    docs = await retriever.search("grounded rag", context())
    assert [item["chunk_id"] for item in docs] == ["a", "b"]
    assert all(item["search_mode"] == "hybrid_rrf" for item in docs)
    knn = client.bodies[1]["query"]["knn"]["embedding"]
    assert knn["vector"] == [1.0, 0.0, 0.5]
    filters = knn["filter"]["bool"]["filter"]
    assert {"term": {"tenant_id": "tenant-a"}} in filters
    assert {"term": {"project_id": "project-a"}} in filters


@pytest.mark.asyncio
async def test_bm25_does_not_embed_query() -> None:
    client = FakeClient()
    retriever = OpenSearchRetriever(client, None)
    docs = await retriever.search("rag", context(False))
    assert len(client.bodies) == 1
    assert docs[0]["search_mode"] == "bm25"


class WrongDimensions(FakeEmbeddings):
    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 2.0]]


@pytest.mark.asyncio
async def test_hybrid_rejects_wrong_query_vector_dimensions() -> None:
    with pytest.raises(RuntimeError, match="dimension"):
        await OpenSearchRetriever(FakeClient(), WrongDimensions()).search("rag", context())
