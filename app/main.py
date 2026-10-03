from fastapi import Depends, FastAPI
from app.config import Settings, get_settings
from app.models import AskRequest, AskResponse, FeedbackRequest
from app.retrieval.memory import InMemoryRetriever
from app.services.agentic_rag import AgenticRAGService

app = FastAPI(title="Production Agentic RAG Platform", version="1.0.0")


def get_service(settings: Settings = Depends(get_settings)) -> AgenticRAGService:
    retriever = InMemoryRetriever([])
    return AgenticRAGService(retriever, settings.max_retrieval_attempts, settings.guardrail_threshold)


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(request: AskRequest, service: AgenticRAGService = Depends(get_service)) -> AskResponse:
    return await service.ask(request)


@app.post("/api/v1/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, object]:
    return {"success": True, "trace_id": request.trace_id}
