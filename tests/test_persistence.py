from datetime import UTC, datetime, timedelta

from sqlalchemy import create_engine, update
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.database import Base
from app.persistence.entities import IngestionJobEntity
from app.persistence.repository import KnowledgeRepository


def repository() -> KnowledgeRepository:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    return KnowledgeRepository(sessionmaker(bind=engine, expire_on_commit=False, class_=Session))


def test_document_and_job_are_durable_repository_records() -> None:
    repo = repository()
    document = repo.create_document(
        "logical-1", "a.txt", "text/plain", "abc", 3, None, b"abc"
    )
    job = repo.create_job(document.id)
    assert repo.get_document(document.id).sha256 == "abc"
    claimed = repo.claim_job()
    assert claimed.id == job.id
    assert claimed.status == "processing"
    repo.finish_job(job.id, True)
    assert repo.get_job(job.id).status == "completed"


def test_failed_job_is_requeued_until_attempt_limit() -> None:
    repo = repository()
    document = repo.create_document(
        "logical-1", "a.txt", "text/plain", "def", 3, None, b"abc"
    )
    job = repo.create_job(document.id)
    claimed = repo.claim_job()
    repo.finish_job(claimed.id, False, "temporary")
    assert repo.get_job(job.id).status == "queued"


def test_documents_are_scoped_by_tenant_and_project() -> None:
    repo = repository()
    first = repo.create_document(
        "logical-a", "a.txt", "text/plain", "tenant-a-hash", 3, None, b"abc",
        tenant_id="tenant-a", project_id="project-a",
    )
    repo.create_document(
        "logical-b", "b.txt", "text/plain", "tenant-b-hash", 3, None, b"xyz",
        tenant_id="tenant-b", project_id="project-b",
    )
    assert repo.get_document(first.id, "tenant-a", "project-a") is not None
    assert repo.get_document(first.id, "tenant-b", "project-b") is None
    scoped = repo.list_documents("tenant-a", "project-a")
    assert [item.id for item in scoped] == [first.id]


def test_versions_and_jobs_are_tenant_scoped() -> None:
    repo = repository()
    document = repo.create_document(
        "logical-scope", "scope.txt", "text/plain", "scope-hash", 3, None, b"abc",
        tenant_id="tenant-a", project_id="project-a",
    )
    job = repo.create_job(document.id)
    assert len(repo.versions("logical-scope", "tenant-a", "project-a")) == 1
    assert repo.versions("logical-scope", "tenant-b", "project-b") == []
    assert repo.get_job(job.id, "tenant-a", "project-a") is not None
    assert repo.get_job(job.id, "tenant-b", "project-b") is None


def test_stale_processing_job_is_recovered() -> None:
    repo = repository()
    document = repo.create_document(
        "logical-stale", "stale.txt", "text/plain", "stale-hash", 3, None, b"abc"
    )
    job = repo.create_job(document.id)
    repo.claim_job()
    with repo.sessions() as session:
        session.execute(
            update(IngestionJobEntity)
            .where(IngestionJobEntity.id == job.id)
            .values(updated_at=datetime.now(UTC) - timedelta(minutes=10))
        )
        session.commit()
    assert repo.recover_stale_jobs(60) == 1
    recovered = repo.get_job(job.id)
    assert recovered.status == "queued"
    assert "lease expired" in recovered.error.lower()


def test_active_job_detects_queued_or_processing_work() -> None:
    repo = repository()
    document = repo.create_document(
        "logical-active", "active.txt", "text/plain", "active-hash", 3, None, b"abc"
    )
    job = repo.create_job(document.id)
    assert repo.active_job(document.id).id == job.id
    repo.claim_job()
    assert repo.active_job(document.id).id == job.id
