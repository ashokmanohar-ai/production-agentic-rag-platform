from datetime import UTC, datetime
from threading import Lock

from app.knowledge.models import DocumentRecord, DocumentStatus


class DocumentRegistry:
    """Thread-safe in-process registry for Phase 3A; replace with durable DB in production."""

    def __init__(self) -> None:
        self._records: dict[str, DocumentRecord] = {}
        self._hashes: dict[str, str] = {}
        self._lock = Lock()

    def create(
        self,
        document_id: str,
        filename: str,
        content_type: str,
        sha256: str,
        size_bytes: int,
        category: str | None,
    ) -> DocumentRecord:
        now = datetime.now(UTC)
        record = DocumentRecord(
            document_id=document_id,
            filename=filename,
            content_type=content_type,
            sha256=sha256,
            size_bytes=size_bytes,
            status=DocumentStatus.queued,
            category=category,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._records[document_id] = record
            self._hashes[sha256] = document_id
        return record

    def duplicate_for(self, sha256: str) -> str | None:
        with self._lock:
            return self._hashes.get(sha256)

    def get(self, document_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(document_id)

    def list(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda item: item.created_at, reverse=True)

    def update(
        self,
        document_id: str,
        status: DocumentStatus,
        chunks_indexed: int | None = None,
        error: str | None = None,
    ) -> DocumentRecord:
        with self._lock:
            record = self._records[document_id]
            values: dict[str, object] = {
                "status": status,
                "updated_at": datetime.now(UTC),
                "error": error,
            }
            if chunks_indexed is not None:
                values["chunks_indexed"] = chunks_indexed
            updated = record.model_copy(update=values)
            self._records[document_id] = updated
            return updated
