from io import BytesIO

import pytest
from fastapi import BackgroundTasks, UploadFile

from app.ingestion.models import IngestResponse
from app.knowledge.models import DocumentStatus
from app.knowledge.registry import DocumentRegistry
from app.knowledge.service import KnowledgeService


class FakeIngestion:
    async def ingest(self, documents):
        return IngestResponse(documents_indexed=1, chunks_indexed=2, index="test")


@pytest.mark.asyncio
async def test_upload_queue_process_and_duplicate() -> None:
    registry = DocumentRegistry()
    service = KnowledgeService(registry, FakeIngestion(), max_file_bytes=1000)
    tasks = BackgroundTasks()
    upload = UploadFile(filename="requirements.txt", file=BytesIO(b"RAG requirements"))
    response = await service.queue_upload(upload, tasks, "requirements")
    assert response.status == DocumentStatus.queued
    await service.process(
        response.document_id,
        "requirements.txt",
        b"RAG requirements",
        "requirements",
    )
    assert service.get(response.document_id).status == DocumentStatus.indexed

    duplicate = UploadFile(filename="copy.txt", file=BytesIO(b"RAG requirements"))
    duplicate_response = await service.queue_upload(duplicate, BackgroundTasks())
    assert duplicate_response.status == DocumentStatus.duplicate
    assert duplicate_response.duplicate_of == response.document_id


@pytest.mark.asyncio
async def test_rejects_unsupported_and_oversized_files() -> None:
    service = KnowledgeService(DocumentRegistry(), FakeIngestion(), max_file_bytes=4)
    with pytest.raises(ValueError, match="Supported"):
        await service.queue_upload(
            UploadFile(filename="bad.exe", file=BytesIO(b"x")),
            BackgroundTasks(),
        )
    with pytest.raises(ValueError, match="size limit"):
        await service.queue_upload(
            UploadFile(filename="large.txt", file=BytesIO(b"12345")),
            BackgroundTasks(),
        )
