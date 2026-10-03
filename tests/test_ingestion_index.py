import pytest

from app.ingestion.opensearch_index import OpenSearchChunkIndex


class FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None

    def exists(self, *, index: str) -> bool:
        return False

    def create(self, *, index: str, body: dict[str, object]) -> dict[str, object]:
        self.created = body
        return {"acknowledged": True}


class FakeClient:
    def __init__(self) -> None:
        self.indices = FakeIndices()
        self.bulk_body: list[dict[str, object]] = []

    def bulk(self, *, body: list[dict[str, object]], refresh: bool) -> dict[str, object]:
        self.bulk_body = body
        return {"errors": False}


@pytest.mark.asyncio
async def test_index_bootstrap_has_knn_vector_mapping() -> None:
    client = FakeClient()
    index = OpenSearchChunkIndex(client, "chunks", "embedding", 3)
    await index.ensure_index()
    assert client.indices.created is not None
    mappings = client.indices.created["mappings"]
    assert mappings["properties"]["embedding"]["dimension"] == 3


@pytest.mark.asyncio
async def test_bulk_index_uses_stable_chunk_id_as_document_id() -> None:
    client = FakeClient()
    index = OpenSearchChunkIndex(client, "chunks", "embedding", 3)
    count = await index.index_chunks(
        [{"chunk_id": "chunk-123", "document_id": "doc-1", "text": "hello"}]
    )
    assert count == 1
    assert client.bulk_body[0]["index"]["_id"] == "chunk-123"
