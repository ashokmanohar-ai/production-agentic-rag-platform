from app.embeddings.base import EmbeddingProvider
from app.ingestion.chunking import chunk_text
from app.ingestion.models import IngestDocument, IngestResponse
from app.ingestion.opensearch_index import OpenSearchChunkIndex


class IngestionService:
    def __init__(
        self,
        index: OpenSearchChunkIndex,
        embeddings: EmbeddingProvider,
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> None:
        self.index = index
        self.embeddings = embeddings
        self.chunk_size = chunk_size
        self.overlap = overlap

    async def ingest(self, documents: list[IngestDocument]) -> IngestResponse:
        await self.index.ensure_index()
        records: list[dict[str, object]] = []
        for document in documents:
            chunks = chunk_text(
                document.document_id,
                document.text,
                self.chunk_size,
                self.overlap,
            )
            vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
            for chunk, vector in zip(chunks, vectors, strict=True):
                records.append(
                    {
                        "document_id": document.document_id,
                        "chunk_id": chunk.chunk_id,
                        "title": document.title,
                        "text": chunk.text,
                        "category": document.category,
                        "url": document.url,
                        "authors": document.authors,
                        "chunk_index": chunk.chunk_index,
                        self.index.vector_field: vector,
                    }
                )
        indexed = await self.index.index_chunks(records)
        return IngestResponse(
            documents_indexed=len(documents),
            chunks_indexed=indexed,
            index=self.index.index,
        )
