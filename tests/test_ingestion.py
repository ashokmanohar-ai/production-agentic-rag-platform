import pytest

from app.ingestion.models import IngestDocument
from app.ingestion.service import IngestionService


class FakeEmbeddings:
    dimensions = 3

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 0.0, 1.0] for text in texts]


class FakeIndex:
    index = "test-index"
    vector_field = "embedding"

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []
        self.ensured = False

    async def ensure_index(self) -> None:
        self.ensured = True

    async def index_chunks(self, chunks: list[dict[str, object]]) -> int:
        self.records.extend(chunks)
        return len(chunks)


@pytest.mark.asyncio
async def test_ingestion_preserves_lineage_and_vectors() -> None:
    index = FakeIndex()
    service = IngestionService(index, FakeEmbeddings(), chunk_size=20, overlap=5)
    response = await service.ingest(
        [
            IngestDocument(
                document_id="req-101",
                title="RAG Requirements",
                text="Agentic RAG needs grounded answers with citations and evaluation.",
                category="requirements",
            )
        ]
    )
    assert index.ensured
    assert response.documents_indexed == 1
    assert response.chunks_indexed == len(index.records)
    assert index.records[0]["document_id"] == "req-101"
    assert index.records[0]["embedding"]
