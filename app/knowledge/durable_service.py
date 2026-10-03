import builtins
import hashlib
from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile

from app.ingestion.models import IngestDocument
from app.ingestion.opensearch_index import OpenSearchChunkIndex
from app.ingestion.service import IngestionService
from app.knowledge.durable_models import DurableDocument, DurableUploadResponse, JobRecord
from app.knowledge.parsers import SUPPORTED_EXTENSIONS, parse_document
from app.persistence.entities import DocumentEntity, IngestionJobEntity
from app.persistence.repository import KnowledgeRepository


def document_view(entity: DocumentEntity) -> DurableDocument:
    return DurableDocument(
        document_id=entity.id,
        logical_id=entity.logical_id,
        version=entity.version,
        filename=entity.filename,
        sha256=entity.sha256,
        status=entity.status,
        category=entity.category,
        chunks_indexed=entity.chunks_indexed,
        error=entity.error,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def job_view(entity: IngestionJobEntity) -> JobRecord:
    return JobRecord(
        job_id=entity.id,
        document_id=entity.document_id,
        operation=entity.operation,
        status=entity.status,
        attempts=entity.attempts,
        max_attempts=entity.max_attempts,
        error=entity.error,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


class DurableKnowledgeService:
    def __init__(
        self,
        repository: KnowledgeRepository,
        ingestion: IngestionService,
        index: OpenSearchChunkIndex,
        max_file_bytes: int,
    ) -> None:
        self.repository = repository
        self.ingestion = ingestion
        self.index = index
        self.max_file_bytes = max_file_bytes

    async def upload(
        self, upload: UploadFile, category: str | None = None, tenant_id: str = "default", project_id: str = "default"
    ) -> DurableUploadResponse:
        filename = upload.filename or "upload"
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError("Supported file types are PDF, DOCX, TXT and Markdown")
        data = await upload.read(self.max_file_bytes + 1)
        if not data or len(data) > self.max_file_bytes:
            raise ValueError("Uploaded file is empty or exceeds the configured size limit")
        digest = hashlib.sha256(data).hexdigest()
        duplicate = self.repository.find_by_hash(digest, tenant_id, project_id)
        if duplicate:
            return DurableUploadResponse(
                document_id=duplicate.id,
                logical_id=duplicate.logical_id,
                version=duplicate.version,
                job_id=None,
                status="duplicate",
                duplicate_of=duplicate.id,
            )
        logical_id = str(uuid4())
        entity = self.repository.create_document(
            logical_id,
            filename,
            upload.content_type or "application/octet-stream",
            digest,
            len(data),
            category,
            data,
            tenant_id=tenant_id,
            project_id=project_id,
        )
        job = self.repository.create_job(entity.id)
        return DurableUploadResponse(
            document_id=entity.id,
            logical_id=logical_id,
            version=entity.version,
            job_id=job.id,
            status=entity.status,
        )

    async def process_next(self) -> JobRecord | None:
        job = self.repository.claim_job()
        if not job:
            return None
        document = self.repository.get_document(job.document_id)
        if not document:
            self.repository.finish_job(job.id, False, "Document not found")
            return job_view(self.repository.get_job(job.id) or job)
        try:
            self.repository.update_document(document.id, "parsing")
            text = parse_document(document.filename, document.content)
            if not text:
                raise ValueError("No extractable text found in document")
            self.repository.update_document(document.id, "ingesting")
            result = await self.ingestion.ingest(
                [
                    IngestDocument(
                        document_id=document.id,
                        title=document.filename,
                        text=text,
                        category=document.category,
                        tenant_id=document.tenant_id,
                        project_id=document.project_id,
                    )
                ]
            )
            self.repository.update_document(document.id, "indexed", result.chunks_indexed)
            self.repository.finish_job(job.id, True)
        except Exception as exc:
            self.repository.finish_job(job.id, False, str(exc))
            persisted_job = self.repository.get_job(job.id)
            next_status = "queued" if persisted_job and persisted_job.status == "queued" else "failed"
            self.repository.update_document(document.id, next_status, error=str(exc))
        return job_view(self.repository.get_job(job.id) or job)

    def retry(self, document_id: str, tenant_id: str | None = None, project_id: str | None = None) -> JobRecord:
        document = self.repository.get_document(document_id, tenant_id, project_id)
        if not document:
            raise KeyError(document_id)
        active = self.repository.active_job(document_id)
        if active:
            return job_view(active)
        self.repository.update_document(document_id, "queued", error=None)
        return job_view(self.repository.create_job(document_id, "retry"))

    def reindex(self, document_id: str, tenant_id: str | None = None, project_id: str | None = None) -> JobRecord:
        document = self.repository.get_document(document_id, tenant_id, project_id)
        if not document:
            raise KeyError(document_id)
        active = self.repository.active_job(document_id)
        if active:
            return job_view(active)
        self.repository.update_document(document_id, "queued", error=None)
        return job_view(self.repository.create_job(document_id, "reindex"))

    async def delete(self, document_id: str, tenant_id: str | None = None, project_id: str | None = None) -> bool:
        document = self.repository.get_document(document_id, tenant_id, project_id)
        if not document:
            return False
        await self.index.delete_document(document_id)
        return self.repository.delete_document(document_id)

    def get(self, document_id: str, tenant_id: str | None = None, project_id: str | None = None) -> DurableDocument | None:
        entity = self.repository.get_document(document_id, tenant_id, project_id)
        return document_view(entity) if entity else None

    def list(self, tenant_id: str | None = None, project_id: str | None = None) -> list[DurableDocument]:
        return [document_view(item) for item in self.repository.list_documents(tenant_id, project_id)]

    def versions(self, logical_id: str, tenant_id: str | None = None, project_id: str | None = None) -> builtins.list[DurableDocument]:
        return [document_view(item) for item in self.repository.versions(logical_id, tenant_id, project_id)]

    def job(self, job_id: str, tenant_id: str | None = None, project_id: str | None = None) -> JobRecord | None:
        entity = self.repository.get_job(job_id, tenant_id, project_id)
        return job_view(entity) if entity else None
