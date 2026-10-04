import time
from uuid import uuid4

from app.observability.langfuse import LangfuseObservability

from app.agent.context import RuntimeContext
from app.agent.graph import AgenticRAGGraph
from app.agent.state import AgentState
from app.llm.base import LLMProvider
from app.models import AskRequest, AskResponse
from app.retrieval.base import Retriever
from app.observability.metrics import RAG_LATENCY, RAG_REQUESTS, RAG_SOURCES


class AgenticRAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        max_attempts: int = 3,
        guardrail_threshold: int = 70,
        observability: LangfuseObservability | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.guardrail_threshold = guardrail_threshold
        self.graph = AgenticRAGGraph(retriever, llm, max_attempts, guardrail_threshold)
        self.observability = observability or LangfuseObservability(False)

    async def ask(self, request: AskRequest) -> AskResponse:
        trace_id = uuid4().hex
        started = time.perf_counter()
        context = RuntimeContext(
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            model=request.model,
            categories=tuple(request.categories or ()),
            max_retrieval_attempts=self.max_attempts,
            guardrail_threshold=self.guardrail_threshold,
            tenant_id=request.tenant_id,
            project_id=request.project_id,
        )
        initial: AgentState = {
            "original_query": request.query,
            "active_query": request.query,
            "retrieval_attempts": 0,
            "reasoning_steps": [],
            "trace_id": trace_id,
            "runtime_context": context,
        }
        with self.observability.trace(
            trace_id,
            request.query,
            {"model": request.model, "search_mode": "hybrid" if request.use_hybrid else "bm25"},
        ) as span:
            state = await self.graph.compiled.ainvoke(initial)
            if span:
                span.update(
                    output={"answer": state.get("answer", "")},
                    metadata={"retrieval_attempts": state.get("retrieval_attempts", 0)},
                )
        allowed = bool(state.get("allowed", False))
        answer = str(state.get("answer", "")).strip()
        if not allowed:
            answer = "I cannot process that request."
        elif not answer:
            answer = "No sufficiently relevant evidence was found after bounded retrieval attempts."
        sources = state.get("sources", [])
        mode = "hybrid" if request.use_hybrid else "bm25"
        RAG_REQUESTS.labels(mode, "allowed" if allowed else "blocked").inc()
        RAG_LATENCY.labels(mode).observe(time.perf_counter() - started)
        RAG_SOURCES.observe(len(sources))
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=sources,
            reasoning_steps=state.get("reasoning_steps", []),
            retrieval_attempts=state.get("retrieval_attempts", 0),
            search_mode=mode,
            trace_id=trace_id,
        )
