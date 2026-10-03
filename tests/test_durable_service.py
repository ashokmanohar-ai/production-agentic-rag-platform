from io import BytesIO

import pytest
from fastapi import UploadFile
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.ingestion.models import IngestResponse
from app.knowledge.durable_service import DurableKnowledgeService
from app.persistence.database import Base
from app.persistence.repository import KnowledgeRepository


class FakeIndex:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete_document(self, document_id: str) -> None:
        self.deleted.append(document_id)


class FakeIngestion:
    index = FakeIndex()

    async def ingest(self, documents):
        return IngestResponse(documents_indexed=1, chunks_indexed=2, index="chunks")


def service() -> DurableKnowledgeService:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    repo = KnowledgeRepository(sessionmaker(bind=engine, expire_on_commit=False, class_=Session))
    ingestion = FakeIngestion()
    return DurableKnowledgeService(repo, ingestion, ingestion.index, 1000)


@pytest.mark.asyncio
async def test_upload_worker_retry_reindex_and_delete() -> None:
    svc = service()
    response = await svc.upload(
        UploadFile(filename="requirements.txt", file=BytesIO(b"RAG requirements")),
        "requirements",
    )
    assert response.job_id
    job = await svc.process_next()
    assert job.status == "completed"
    assert svc.get(response.document_id).status == "indexed"

    retry = svc.retry(response.document_id)
    assert retry.operation == "retry"
    await svc.process_next()

    reindex = svc.reindex(response.document_id)
    assert reindex.operation == "reindex"
    await svc.process_next()
    # Reindex is non-destructive: existing chunks stay available until replacement ingest succeeds.
    assert response.document_id not in svc.index.deleted

    assert await svc.delete(response.document_id)
    assert svc.get(response.document_id) is None


@pytest.mark.asyncio
async def test_duplicate_upload_returns_existing_document() -> None:
    svc = service()
    first = await svc.upload(UploadFile(filename="a.txt", file=BytesIO(b"same")))
    second = await svc.upload(UploadFile(filename="b.txt", file=BytesIO(b"same")))
    assert second.status == "duplicate"
    assert second.duplicate_of == first.document_id


class FailingIngestion:
    index = FakeIndex()

    async def ingest(self, documents):
        raise RuntimeError("temporary outage")


@pytest.mark.asyncio
async def test_transient_failure_requeues_document_until_attempt_limit() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    repo = KnowledgeRepository(sessionmaker(bind=engine, expire_on_commit=False, class_=Session))
    ingestion = FailingIngestion()
    svc = DurableKnowledgeService(repo, ingestion, ingestion.index, 1000)
    response = await svc.upload(UploadFile(filename="retry.txt", file=BytesIO(b"retry me")))
    first = await svc.process_next()
    assert first.status == "queued"
    assert svc.get(response.document_id).status == "queued"
    await svc.process_next()
    final = await svc.process_next()
    assert final.status == "failed"
    assert svc.get(response.document_id).status == "failed"


def test_retry_does_not_create_duplicate_active_job() -> None:
    svc = service()
    document = svc.repository.create_document(
        "logical", "a.txt", "text/plain", "retry-hash", 3, None, b"abc"
    )
    first = svc.retry(document.id)
    second = svc.retry(document.id)
    assert second.job_id == first.job_id
