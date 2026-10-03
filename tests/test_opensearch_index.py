import pytest

from app.ingestion.opensearch_index import OpenSearchChunkIndex


class Indices:
    def __init__(self, exists: bool, dimension: int = 3) -> None:
        self._exists = exists
        self.dimension = dimension
        self.created = None

    def exists(self, *, index: str) -> bool:
        return self._exists

    def get_mapping(self, *, index: str):
        return {index: {"mappings": {"properties": {"embedding": {"type": "knn_vector", "dimension": self.dimension}}}}}

    def create(self, *, index: str, body):
        self.created = body


class Client:
    def __init__(self, exists: bool, dimension: int = 3) -> None:
        self.indices = Indices(exists, dimension)


@pytest.mark.asyncio
async def test_existing_index_dimension_is_validated() -> None:
    await OpenSearchChunkIndex(Client(True, 3), "idx", "embedding", 3).ensure_index()
    with pytest.raises(RuntimeError, match="dimension"):
        await OpenSearchChunkIndex(Client(True, 2), "idx", "embedding", 3).ensure_index()


@pytest.mark.asyncio
async def test_new_index_uses_configured_dimension() -> None:
    client = Client(False)
    await OpenSearchChunkIndex(client, "idx", "embedding", 768).ensure_index()
    vector = client.indices.created["mappings"]["properties"]["embedding"]
    assert vector["dimension"] == 768
