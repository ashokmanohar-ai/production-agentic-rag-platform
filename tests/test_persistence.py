from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.database import Base
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
