from fastapi import Depends, FastAPI
from opensearchpy import OpenSearch

from app.config import Settings, get_settings
from app.embeddings.ollama import OllamaEmbeddingProvider
from app.ingestion.models import IngestRequest, IngestResponse
from app.ingestion.opensearch_index import OpenSearchChunkIndex
from app.ingestion.service import IngestionService
from app.llm.ollama import OllamaProvider
from app.models import AskRequest, AskResponse, FeedbackRequest
from app.retrieval.opensearch import OpenSearchRetriever
from app.services.agentic_rag import AgenticRAGService

app = FastAPI(title="Production Agentic RAG Platform", version="1.2.0")


def _opensearch(settings: Settings) -> OpenSearch:
    return OpenSearch(hosts=[settings.opensearch_url])


def get_service(settings: Settings = Depends(get_settings)) -> AgenticRAGService:
    retriever = OpenSearchRetriever(
        _opensearch(settings),
        index=settings.opensearch_index,
        neural_model_id=settings.opensearch_neural_model_id,
        vector_field=settings.opensearch_vector_field,
        search_pipeline=settings.opensearch_search_pipeline,
    )
    llm = OllamaProvider(settings.ollama_url)
    return AgenticRAGService(
        retriever, llm, settings.max_retrieval_attempts, settings.guardrail_threshold
    )


def get_ingestion_service(
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    embeddings = OllamaEmbeddingProvider(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    index = OpenSearchChunkIndex(
        _opensearch(settings),
        settings.opensearch_index,
        settings.opensearch_vector_field,
        embeddings.dimensions,
    )
    return IngestionService(
        index,
        embeddings,
        settings.ingestion_chunk_size,
        settings.ingestion_chunk_overlap,
    )


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(
    request: AskRequest, service: AgenticRAGService = Depends(get_service)
) -> AskResponse:
    return await service.ask(request)


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestResponse:
    return await service.ingest(request.documents)


@app.post("/api/v1/feedback")
async def feedback(request: FeedbackRequest) -> dict[str, object]:
    return {"success": True, "trace_id": request.trace_id}
