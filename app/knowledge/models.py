from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel


class DocumentStatus(StrEnum):
    queued = "queued"
    parsing = "parsing"
    ingesting = "ingesting"
    indexed = "indexed"
    failed = "failed"
    duplicate = "duplicate"


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    content_type: str
    sha256: str
    size_bytes: int
    status: DocumentStatus
    category: str | None = None
    chunks_indexed: int = 0
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class UploadResponse(BaseModel):
    document_id: str
    status: DocumentStatus
    sha256: str
    duplicate_of: str | None = None
