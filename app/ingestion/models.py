from pydantic import BaseModel, Field


class IngestDocument(BaseModel):
    document_id: str = Field(min_length=1, max_length=200)
    title: str = Field(min_length=1, max_length=500)
    text: str = Field(min_length=1)
    category: str | None = Field(default=None, max_length=100)
    url: str | None = None
    authors: list[str] = Field(default_factory=list)


class IngestRequest(BaseModel):
    documents: list[IngestDocument] = Field(min_length=1, max_length=100)


class IngestResponse(BaseModel):
    documents_indexed: int
    chunks_indexed: int
    index: str
