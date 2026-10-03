import asyncio
from typing import Any, Protocol


class IndexClient(Protocol):
    class Indices(Protocol):
        def exists(self, *, index: str) -> bool:
            ...

        def create(self, *, index: str, body: dict[str, object]) -> dict[str, Any]:
            ...

    indices: Indices

    def bulk(self, *, body: list[dict[str, object]], refresh: bool) -> dict[str, Any]:
        ...


class OpenSearchChunkIndex:
    def __init__(
        self,
        client: IndexClient,
        index: str,
        vector_field: str,
        dimensions: int,
    ) -> None:
        self.client = client
        self.index = index
        self.vector_field = vector_field
        self.dimensions = dimensions

    async def ensure_index(self) -> None:
        exists = await asyncio.to_thread(self.client.indices.exists, index=self.index)
        if exists:
            return
        body: dict[str, object] = {
            "settings": {"index": {"knn": True}},
            "mappings": {
                "properties": {
                    "document_id": {"type": "keyword"},
                    "chunk_id": {"type": "keyword"},
                    "title": {"type": "text"},
                    "text": {"type": "text"},
                    "category": {"type": "keyword"},
                    "url": {"type": "keyword", "index": False},
                    "authors": {"type": "keyword"},
                    "chunk_index": {"type": "integer"},
                    self.vector_field: {
                        "type": "knn_vector",
                        "dimension": self.dimensions,
                    },
                }
            },
        }
        await asyncio.to_thread(self.client.indices.create, index=self.index, body=body)

    async def index_chunks(self, chunks: list[dict[str, object]]) -> int:
        if not chunks:
            return 0
        body: list[dict[str, object]] = []
        for chunk in chunks:
            chunk_id = str(chunk["chunk_id"])
            body.append({"index": {"_index": self.index, "_id": chunk_id}})
            body.append(chunk)
        result = await asyncio.to_thread(self.client.bulk, body=body, refresh=True)
        if result.get("errors"):
            raise RuntimeError("OpenSearch bulk indexing reported one or more failures")
        return len(chunks)
