from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from opensearchpy import OpenSearch
from redis.asyncio import Redis

from app.cache.decorators import CachedLLMProvider, CachedRetriever
from app.cache.redis_cache import RedisCache
from app.config import Settings, get_settings
from app.dashboard import dashboard_html
from app.embeddings.ollama import OllamaEmbeddingProvider
from app.evaluation.history_models import EvaluationComparison
from app.evaluation.models import EvaluationRequest, EvaluationSummary
from app.evaluation.repository import EvaluationRepository
from app.evaluation.runner import EvaluationRunner
from app.ingestion.models import IngestRequest, IngestResponse
from app.ingestion.opensearch_index import OpenSearchChunkIndex
from app.ingestion.service import IngestionService
from app.knowledge.durable_models import DurableDocument, DurableUploadResponse, JobRecord
from app.knowledge.durable_service import DurableKnowledgeService
from app.knowledge.models import DocumentRecord, UploadResponse
from app.knowledge.registry import DocumentRegistry
from app.knowledge.service import KnowledgeService
from app.llm.base import LLMProvider
from app.llm.ollama import OllamaProvider
from app.models import AskRequest, AskResponse, FeedbackRequest
from app.observability.langfuse import LangfuseObservability
from app.persistence.audit import AuditRepository
from app.persistence.database import build_session_factory
from app.persistence.repository import KnowledgeRepository
from app.retrieval.base import Retriever
from app.retrieval.opensearch import OpenSearchRetriever
from app.services.agentic_rag import AgenticRAGService
from app.security import SecurityContext, get_security_context, require_role
from app.runtime import RuntimeDiagnostics

app = FastAPI(title="Production Agentic RAG Platform", version="1.14.0")
document_registry = DocumentRegistry()


def _opensearch(settings: Settings) -> OpenSearch:
    return OpenSearch(hosts=[settings.opensearch_url])




def _observability(settings: Settings) -> LangfuseObservability:
    return LangfuseObservability(
        settings.langfuse_enabled,
        settings.langfuse_host,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
    )


def _cache(settings: Settings) -> RedisCache | None:
    if not settings.cache_enabled:
        return None
    return RedisCache(Redis.from_url(settings.redis_url), settings.cache_ttl_seconds)


def get_service(settings: Settings = Depends(get_settings)) -> AgenticRAGService:
    embeddings = OllamaEmbeddingProvider(
        settings.ollama_url,
        settings.embedding_model,
        settings.embedding_dimensions,
    )
    retriever: Retriever = OpenSearchRetriever(
        _opensearch(settings),
        embeddings=embeddings,
        index=settings.opensearch_index,
        vector_field=settings.opensearch_vector_field,
    )
    llm: LLMProvider = OllamaProvider(settings.ollama_url)
    observability = _observability(settings)
    cache = _cache(settings)
    if cache:
        retriever = CachedRetriever(
            retriever, cache, observability, settings.retrieval_cache_version
        )
        llm = CachedLLMProvider(llm, cache, observability)
    return AgenticRAGService(
        retriever,
        llm,
        settings.max_retrieval_attempts,
        settings.guardrail_threshold,
        observability,
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
        _cache(settings),
    )


def get_knowledge_service(
    ingestion: IngestionService = Depends(get_ingestion_service),
    settings: Settings = Depends(get_settings),
) -> KnowledgeService:
    return KnowledgeService(document_registry, ingestion, settings.max_upload_bytes)


def get_durable_knowledge_service(
    settings: Settings = Depends(get_settings),
) -> DurableKnowledgeService:
    ingestion = get_ingestion_service(settings)
    sessions = build_session_factory(settings.database_url)
    repository = KnowledgeRepository(sessions)
    return DurableKnowledgeService(repository, ingestion, ingestion.index, settings.max_upload_bytes)


@app.get("/api/v1/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/v1/ready")
async def ready(settings: Settings = Depends(get_settings)) -> dict[str, object]:
    diagnostics = RuntimeDiagnostics(settings, build_session_factory(settings.database_url))
    result = await diagnostics.check()
    if result["status"] != "ready":
        raise HTTPException(status_code=503, detail=result)
    return result


@app.post("/api/v1/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    service: AgenticRAGService = Depends(get_service),
    security: SecurityContext = Depends(get_security_context),
) -> AskResponse:
    require_role(security, "reader")
    secured = request.model_copy(update={"tenant_id": security.tenant_id, "project_id": security.project_id})
    return await service.ask(secured)


def get_evaluation_repository(settings: Settings = Depends(get_settings)) -> EvaluationRepository:
    return EvaluationRepository(build_session_factory(settings.database_url))


def get_audit_repository(settings: Settings = Depends(get_settings)) -> AuditRepository:
    return AuditRepository(build_session_factory(settings.database_url))


@app.get("/dashboard", response_class=HTMLResponse)
async def quality_dashboard(
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    security: SecurityContext = Depends(get_security_context),
) -> HTMLResponse:
    require_role(security, "reader")
    return HTMLResponse(dashboard_html(repository.list_runs(30)))


@app.post("/api/v1/evaluations/run", response_model=EvaluationSummary)
async def run_evaluation(
    request: EvaluationRequest,
    service: AgenticRAGService = Depends(get_service),
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    settings: Settings = Depends(get_settings),
    security: SecurityContext = Depends(get_security_context),
) -> EvaluationSummary:
    require_role(security, "contributor")
    judge_llm: LLMProvider | None = OllamaProvider(settings.ollama_url) if request.judge_enabled else None
    summary = await EvaluationRunner(service, judge_llm).run(request)
    repository.save(summary)
    return summary


@app.get("/api/v1/evaluations/history")
async def evaluation_history(
    limit: int = 50,
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    security: SecurityContext = Depends(get_security_context),
) -> list[dict[str, object]]:
    require_role(security, "reader")
    safe_limit = min(max(limit, 1), 200)
    return [
        {
            "run_id": item.id,
            "dataset_name": item.dataset_name,
            "dataset_version": item.dataset_version,
            "pass_rate": item.pass_rate,
            "mean_recall_at_k": item.mean_recall_at_k,
            "mean_ndcg": item.mean_ndcg,
            "mean_answer_relevance": item.mean_answer_relevance,
            "mean_citation_correctness": item.mean_citation_correctness,
            "mean_safety": item.mean_safety,
            "mean_latency_ms": item.mean_latency_ms,
            "regression_gate_passed": item.regression_gate_passed,
            "created_at": item.created_at.isoformat(),
        }
        for item in repository.list_runs(safe_limit)
    ]


@app.get("/api/v1/evaluations/{run_id}", response_model=EvaluationSummary)
async def evaluation_detail(
    run_id: str,
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    security: SecurityContext = Depends(get_security_context),
) -> EvaluationSummary:
    require_role(security, "reader")
    summary = repository.summary(run_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return summary


@app.get("/api/v1/evaluations/compare/{baseline_id}/{current_id}", response_model=EvaluationComparison)
async def compare_evaluations(
    baseline_id: str,
    current_id: str,
    repository: EvaluationRepository = Depends(get_evaluation_repository),
    security: SecurityContext = Depends(get_security_context),
) -> EvaluationComparison:
    require_role(security, "reader")
    comparison = repository.compare(baseline_id, current_id)
    if not comparison:
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    return comparison


@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion_service),
    security: SecurityContext = Depends(get_security_context),
) -> IngestResponse:
    require_role(security, "contributor")
    documents = [item.model_copy(update={"tenant_id": security.tenant_id, "project_id": security.project_id}) for item in request.documents]
    return await service.ingest(documents)


@app.post("/api/v1/documents/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    service: KnowledgeService = Depends(get_knowledge_service),
) -> UploadResponse:
    try:
        return await service.queue_upload(file, background_tasks, category)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/documents", response_model=list[DocumentRecord])
async def list_documents(
    service: KnowledgeService = Depends(get_knowledge_service),
) -> list[DocumentRecord]:
    return service.list()


@app.get("/api/v1/documents/{document_id}", response_model=DocumentRecord)
async def get_document(
    document_id: str,
    service: KnowledgeService = Depends(get_knowledge_service),
) -> DocumentRecord:
    record = service.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return record



@app.post("/api/v1/durable/documents/upload", response_model=DurableUploadResponse)
async def durable_upload_document(
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
    audit: AuditRepository = Depends(get_audit_repository),
) -> DurableUploadResponse:
    require_role(security, "contributor")
    try:
        result = await service.upload(file, category, security.tenant_id, security.project_id)
        audit.record(security, "document.upload", "document", result.document_id)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/v1/durable/documents", response_model=list[DurableDocument])
async def durable_list_documents(
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> list[DurableDocument]:
    require_role(security, "reader")
    return service.list(security.tenant_id, security.project_id)


@app.get("/api/v1/durable/documents/{document_id}", response_model=DurableDocument)
async def durable_get_document(
    document_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> DurableDocument:
    require_role(security, "reader")
    record = service.get(document_id, security.tenant_id, security.project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


@app.get("/api/v1/durable/documents/{document_id}/versions", response_model=list[DurableDocument])
async def durable_document_versions(
    document_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> list[DurableDocument]:
    require_role(security, "reader")
    record = service.get(document_id, security.tenant_id, security.project_id)
    if not record:
        raise HTTPException(status_code=404, detail="Document not found")
    return service.versions(record.logical_id, security.tenant_id, security.project_id)


@app.post("/api/v1/durable/documents/{document_id}/retry", response_model=JobRecord)
async def durable_retry_document(
    document_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> JobRecord:
    require_role(security, "contributor")
    try:
        return service.retry(document_id, security.tenant_id, security.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@app.post("/api/v1/durable/documents/{document_id}/reindex", response_model=JobRecord)
async def durable_reindex_document(
    document_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> JobRecord:
    require_role(security, "contributor")
    try:
        return service.reindex(document_id, security.tenant_id, security.project_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Document not found") from exc


@app.delete("/api/v1/durable/documents/{document_id}")
async def durable_delete_document(
    document_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> dict[str, bool]:
    require_role(security, "admin")
    deleted = await service.delete(document_id, security.tenant_id, security.project_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"deleted": True}


@app.get("/api/v1/durable/jobs/{job_id}", response_model=JobRecord)
async def durable_get_job(
    job_id: str,
    service: DurableKnowledgeService = Depends(get_durable_knowledge_service),
    security: SecurityContext = Depends(get_security_context),
) -> JobRecord:
    require_role(security, "reader")
    job = service.job(job_id, security.tenant_id, security.project_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.post("/api/v1/feedback")
async def feedback(
    request: FeedbackRequest,
    settings: Settings = Depends(get_settings),
    security: SecurityContext = Depends(get_security_context),
) -> dict[str, object]:
    require_role(security, "reader")
    _observability(settings).score(request.trace_id, request.score, request.comment)
    return {"success": True, "trace_id": request.trace_id}
