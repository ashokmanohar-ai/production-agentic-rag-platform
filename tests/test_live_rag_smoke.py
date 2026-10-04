import pytest

from app import models
from scripts import live_rag_smoke


class FakeIndices:
    def exists(self, index: str) -> bool:
        return True

    def delete(self, index: str) -> None:
        return None


class FakeOpenSearch:
    def __init__(self, hosts) -> None:
        self.indices = FakeIndices()


class FakeEmbeddingProvider:
    def __init__(self, *args) -> None:
        self.dimensions = 3


class FakeIndex:
    def __init__(self, *args) -> None:
        self.index = args[1]
        self.vector_field = args[2]


class FakeIndexed:
    chunks_indexed = 1


class FakeIngestion:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ingest(self, documents):
        assert documents[0].tenant_id == "phase14-smoke"
        return FakeIndexed()


class FakeService:
    def __init__(self, *args, **kwargs) -> None:
        pass

    async def ask(self, request):
        return models.AskResponse(
            query=request.query,
            answer="The verification code is ORBIT-731 [1].",
            sources=[
                models.SourceItem(
                    document_id="smoke-doc",
                    chunk_id="chunk-1",
                    title="Phase 14 Smoke Evidence",
                )
            ],
            retrieval_attempts=1,
            search_mode="hybrid",
            trace_id="trace",
        )


@pytest.mark.asyncio
async def test_live_smoke_contract(monkeypatch) -> None:
    monkeypatch.setattr(live_rag_smoke, "OpenSearch", FakeOpenSearch)
    monkeypatch.setattr(live_rag_smoke, "OllamaEmbeddingProvider", FakeEmbeddingProvider)
    monkeypatch.setattr(live_rag_smoke, "OpenSearchChunkIndex", FakeIndex)
    monkeypatch.setattr(live_rag_smoke, "IngestionService", FakeIngestion)
    monkeypatch.setattr(live_rag_smoke, "OpenSearchRetriever", lambda *args, **kwargs: object())
    monkeypatch.setattr(live_rag_smoke, "OllamaProvider", lambda *args: object())
    monkeypatch.setattr(live_rag_smoke, "AgenticRAGService", FakeService)

    result = await live_rag_smoke.run_smoke()
    assert result["status"] == "passed"
    assert result["answer_contains_verification_code"] is True
    assert result["sources"] == 1
