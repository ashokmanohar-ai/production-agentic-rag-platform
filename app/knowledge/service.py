import hashlib
from uuid import uuid4

from fastapi import BackgroundTasks, UploadFile

from app.ingestion.models import IngestDocument
from app.ingestion.service import IngestionService
from app.knowledge.models import DocumentRecord, DocumentStatus, UploadResponse
from app.knowledge.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.knowledge.registry import DocumentRegistry


class KnowledgeService:
    def __init__(
        self,
        registry: DocumentRegistry,
        ingestion: IngestionService,
        max_file_bytes: int,
    ) -> None:
        self.registry = registry
        self.ingestion = ingestion
        self.max_file_bytes = max_file_bytes

    async def queue_upload(
        self,
        upload: UploadFile,
        background_tasks: BackgroundTasks,
        category: str | None = None,
    ) -> UploadResponse:
        filename = upload.filename or "upload"
        suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if suffix not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported file types are PDF, DOCX, TXT and Markdown")
        data = await upload.read(self.max_file_bytes + 1)
        if not data:
            raise ValueError("Uploaded file is empty")
        if len(data) > self.max_file_bytes:
            raise ValueError("Uploaded file exceeds the configured size limit")
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self.registry.duplicate_for(digest)
        if duplicate:
            return UploadResponse(
                document_id=duplicate,
                status=DocumentStatus.duplicate,
                sha256=digest,
                duplicate_of=duplicate,
            )
        document_id = str(uuid4())
        self.registry.create(
            document_id,
            filename,
            upload.content_type or "application/octet-stream",
            digest,
            len(data),
            category,
        )
        background_tasks.add_task(self.process, document_id, filename, data, category)
        return UploadResponse(
            document_id=document_id,
            status=DocumentStatus.queued,
            sha256=digest,
        )

    async def process(
        self,
        document_id: str,
        filename: str,
        data: bytes,
        category: str | None,
    ) -> None:
        try:
            self.registry.update(document_id, DocumentStatus.parsing)
            text = parse_document(filename, data)
            if not text:
                raise ValueError("No extractable text found in document")
            self.registry.update(document_id, DocumentStatus.ingesting)
            response = await self.ingestion.ingest(
                [
                    IngestDocument(
                        document_id=document_id,
                        title=filename,
                        text=text,
                        category=category,
                    )
                ]
            )
            self.registry.update(
                document_id,
                DocumentStatus.indexed,
                chunks_indexed=response.chunks_indexed,
            )
        except Exception as exc:
            self.registry.update(document_id, DocumentStatus.failed, error=str(exc))

    def get(self, document_id: str) -> DocumentRecord | None:
        return self.registry.get(document_id)

    def list(self) -> list[DocumentRecord]:
        return self.registry.list()
