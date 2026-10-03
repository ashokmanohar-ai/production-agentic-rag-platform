from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.persistence.entities import DocumentEntity, IngestionJobEntity


class KnowledgeRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def find_by_hash(self, sha256: str) -> DocumentEntity | None:
        with self.sessions() as session:
            return session.scalar(
                select(DocumentEntity)
                .where(DocumentEntity.sha256 == sha256)
                .order_by(DocumentEntity.version.desc())
            )

    def create_document(
        self,
        logical_id: str,
        filename: str,
        content_type: str,
        sha256: str,
        size_bytes: int,
        category: str | None,
        content: bytes,
        version: int = 1,
    ) -> DocumentEntity:
        entity = DocumentEntity(
            logical_id=logical_id,
            version=version,
            filename=filename,
            content_type=content_type,
            sha256=sha256,
            size_bytes=size_bytes,
            category=category,
            status="queued",
            content=content,
        )
        with self.sessions() as session:
            session.add(entity)
            session.commit()
            session.refresh(entity)
            session.expunge(entity)
        return entity

    def create_job(self, document_id: str, operation: str = "ingest") -> IngestionJobEntity:
        job = IngestionJobEntity(document_id=document_id, operation=operation)
        with self.sessions() as session:
            session.add(job)
            session.commit()
            session.refresh(job)
            session.expunge(job)
        return job

    def get_document(self, document_id: str) -> DocumentEntity | None:
        with self.sessions() as session:
            entity = session.get(DocumentEntity, document_id)
            if entity:
                session.expunge(entity)
            return entity

    def list_documents(self) -> list[DocumentEntity]:
        with self.sessions() as session:
            items = list(
                session.scalars(select(DocumentEntity).order_by(DocumentEntity.created_at.desc()))
            )
            for item in items:
                session.expunge(item)
            return items

    def versions(self, logical_id: str) -> list[DocumentEntity]:
        with self.sessions() as session:
            items = list(
                session.scalars(
                    select(DocumentEntity)
                    .where(DocumentEntity.logical_id == logical_id)
                    .order_by(DocumentEntity.version.desc())
                )
            )
            for item in items:
                session.expunge(item)
            return items

    def update_document(
        self, document_id: str, status: str, chunks: int | None = None, error: str | None = None
    ) -> None:
        with self.sessions() as session:
            entity = session.get(DocumentEntity, document_id)
            if not entity:
                return
            entity.status = status
            entity.error = error
            entity.updated_at = datetime.now(UTC)
            if chunks is not None:
                entity.chunks_indexed = chunks
            session.commit()

    def claim_job(self) -> IngestionJobEntity | None:
        with self.sessions() as session:
            statement = (
                select(IngestionJobEntity)
                .where(IngestionJobEntity.status == "queued")
                .order_by(IngestionJobEntity.created_at)
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            job = session.scalar(statement)
            if not job:
                return None
            job.status = "processing"
            job.attempts += 1
            job.updated_at = datetime.now(UTC)
            session.commit()
            session.refresh(job)
            session.expunge(job)
            return job

    def finish_job(self, job_id: str, success: bool, error: str | None = None) -> None:
        with self.sessions() as session:
            job = session.get(IngestionJobEntity, job_id)
            if not job:
                return
            job.status = "completed" if success else (
                "queued" if job.attempts < job.max_attempts else "failed"
            )
            job.error = error
            job.updated_at = datetime.now(UTC)
            session.commit()

    def get_job(self, job_id: str) -> IngestionJobEntity | None:
        with self.sessions() as session:
            job = session.get(IngestionJobEntity, job_id)
            if job:
                session.expunge(job)
            return job

    def delete_document(self, document_id: str) -> bool:
        with self.sessions() as session:
            entity = session.get(DocumentEntity, document_id)
            if not entity:
                return False
            session.delete(entity)
            session.commit()
            return True
