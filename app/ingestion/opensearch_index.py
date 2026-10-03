import asyncio
from typing import Any


class OpenSearchChunkIndex:
    def __init__(
        self,
        client: Any,
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
            mapping = await asyncio.to_thread(self.client.indices.get_mapping, index=self.index)
            properties = mapping[self.index]["mappings"].get("properties", {})
            vector = properties.get(self.vector_field, {})
            for field in ("tenant_id", "project_id"):
                if properties.get(field, {}).get("type") != "keyword":
                    raise RuntimeError(f"OpenSearch field {field} must be mapped as keyword")
            actual = vector.get("dimension")
            if actual != self.dimensions:
                raise RuntimeError(
                    f"OpenSearch vector dimension {actual} does not match configured dimension "
                    f"{self.dimensions}"
                )
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
                    "tenant_id": {"type": "keyword"},
                    "project_id": {"type": "keyword"},
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

    async def delete_document(self, document_id: str) -> None:
        exists = await asyncio.to_thread(self.client.indices.exists, index=self.index)
        if not exists:
            return
        await asyncio.to_thread(
            self.client.delete_by_query,
            index=self.index,
            body={"query": {"term": {"document_id": document_id}}},
            refresh=True,
            conflicts="proceed",
        )

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
