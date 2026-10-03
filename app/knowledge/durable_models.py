from datetime import datetime

from pydantic import BaseModel


class DurableDocument(BaseModel):
    document_id: str
    logical_id: str
    version: int
    filename: str
    sha256: str
    status: str
    category: str | None
    chunks_indexed: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class JobRecord(BaseModel):
    job_id: str
    document_id: str
    operation: str
    status: str
    attempts: int
    max_attempts: int
    error: str | None
    created_at: datetime
    updated_at: datetime


class DurableUploadResponse(BaseModel):
    document_id: str
    logical_id: str
    version: int
    job_id: str | None
    status: str
    duplicate_of: str | None = None
