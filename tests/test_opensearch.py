import pytest

from app.agent.context import RuntimeContext
from app.retrieval.opensearch import OpenSearchRetriever


class FakeOpenSearch:
    def __init__(self) -> None:
        self.body = None
        self.params = None

    def search(self, *, index: str, body: dict, params=None) -> dict:
        self.body = body
        self.params = params
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
async def test_hybrid_query_contains_lexical_and_neural_clauses() -> None:
    client = FakeOpenSearch()
    retriever = OpenSearchRetriever(
        client, neural_model_id="model-1", search_pipeline="hybrid-pipeline"
    )
    await retriever.search("agentic rag", context(True))
    hybrid = client.body["query"]["hybrid"]
    assert len(hybrid["queries"]) == 2
    assert "match" in hybrid["queries"][0]["bool"]["must"][0]
    assert "neural" in hybrid["queries"][1]
    assert client.params == {"search_pipeline": "hybrid-pipeline"}


@pytest.mark.asyncio
async def test_hybrid_requires_neural_model() -> None:
    retriever = OpenSearchRetriever(FakeOpenSearch())
    with pytest.raises(RuntimeError, match="OPENSEARCH_NEURAL_MODEL_ID"):
        await retriever.search("agentic rag", context(True))


@pytest.mark.asyncio
async def test_bm25_does_not_require_neural_model() -> None:
    client = FakeOpenSearch()
    retriever = OpenSearchRetriever(client)
    await retriever.search("agentic rag", context(False))
    assert "bool" in client.body["query"]
