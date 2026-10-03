import pytest

from app.agent.context import RuntimeContext
from app.retrieval.opensearch import OpenSearchRetriever


class FakeEmbeddings:
    dimensions = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeOpenSearch:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    def search(self, *, index: str, body: dict, params=None) -> dict:
        self.bodies.append(body)
        return {"hits": {"hits": []}}


def context(hybrid: bool) -> RuntimeContext:
    return RuntimeContext(
        top_k=5,
        use_hybrid=hybrid,
        model="test",
        categories=("cs.AI",),
        max_retrieval_attempts=2,
        guardrail_threshold=70,
    )


@pytest.mark.asyncio
async def test_hybrid_query_contains_lexical_and_knn_clauses() -> None:
    client = FakeOpenSearch()
    retriever = OpenSearchRetriever(client, FakeEmbeddings())
    await retriever.search("agentic rag", context(True))
    assert "bool" in client.bodies[0]["query"]
    assert "knn" in client.bodies[1]["query"]
    assert client.bodies[1]["query"]["knn"]["embedding"]["vector"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_hybrid_requires_embedding_provider() -> None:
    retriever = OpenSearchRetriever(FakeOpenSearch())
    with pytest.raises(RuntimeError, match="embedding provider"):
        await retriever.search("agentic rag", context(True))


@pytest.mark.asyncio
async def test_bm25_does_not_require_embedding_provider() -> None:
    client = FakeOpenSearch()
    retriever = OpenSearchRetriever(client)
    await retriever.search("agentic rag", context(False))
    assert "bool" in client.bodies[0]["query"]
