from typing import Literal
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=3, ge=1, le=10)
    use_hybrid: bool = True
    model: str = Field(default="llama3.2:3b", min_length=1, max_length=100)
    categories: list[str] | None = None
    tenant_id: str = Field(default="default", min_length=1, max_length=100)
    project_id: str = Field(default="default", min_length=1, max_length=100)


class SourceItem(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    url: str | None = None
    authors: list[str] = Field(default_factory=list)
    score: float = 0.0
    category: str | None = None


class ReasoningStep(BaseModel):
    step_name: str
    description: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AskResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    reasoning_steps: list[ReasoningStep] = Field(default_factory=list)
    retrieval_attempts: int = 0
    search_mode: Literal["hybrid", "bm25"]
    trace_id: str


class FeedbackRequest(BaseModel):
    trace_id: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    comment: str | None = Field(default=None, max_length=1000)
