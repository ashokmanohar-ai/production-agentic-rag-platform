from uuid import uuid4
from app.agent.context import RuntimeContext
from app.agent.security import wrap_untrusted_context
from app.agent.sources import extract_sources
from app.models import AskRequest, AskResponse, ReasoningStep
from app.retrieval.base import Retriever


class AgenticRAGService:
    """Bounded adaptive RAG orchestration with explicit runtime configuration."""

    def __init__(self, retriever: Retriever, max_attempts: int = 3, guardrail_threshold: int = 70) -> None:
        self.retriever = retriever
        self.max_attempts = max_attempts
        self.guardrail_threshold = guardrail_threshold

    def _guardrail_score(self, query: str) -> int:
        blocked = ("reveal system prompt", "ignore previous instructions", "steal credentials")
        return 0 if any(term in query.lower() for term in blocked) else 100

    def _rewrite(self, query: str, attempt: int) -> str:
        return f"{query} relevant technical evidence attempt {attempt + 1}"

    async def ask(self, request: AskRequest) -> AskResponse:
        trace_id = str(uuid4())
        context = RuntimeContext(
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            model=request.model,
            categories=tuple(request.categories or ()),
            max_retrieval_attempts=self.max_attempts,
            guardrail_threshold=self.guardrail_threshold,
        )
        steps: list[ReasoningStep] = []
        score = self._guardrail_score(request.query)
        steps.append(ReasoningStep(step_name="guardrail", description="Query safety/scope check", metadata={"score": score}))
        if score < context.guardrail_threshold:
            return AskResponse(query=request.query, answer="I cannot process that request.", reasoning_steps=steps, search_mode="hybrid" if request.use_hybrid else "bm25", trace_id=trace_id)

        active_query = request.query
        documents: list[dict[str, object]] = []
        attempts = 0
        while attempts < context.max_retrieval_attempts:
            attempts += 1
            documents = await self.retriever.search(active_query, context)
            steps.append(ReasoningStep(step_name="retrieve", description="Retrieved candidate evidence", metadata={"attempt": attempts, "count": len(documents), "top_k": context.top_k, "hybrid": context.use_hybrid, "categories": list(context.categories)}))
            if documents:
                break
            if attempts < context.max_retrieval_attempts:
                active_query = self._rewrite(active_query, attempts)
                steps.append(ReasoningStep(step_name="rewrite_query", description="Rewrote query after insufficient evidence", metadata={"attempt": attempts}))

        sources = extract_sources(documents)
        if not documents:
            answer = "No sufficiently relevant evidence was found after bounded retrieval attempts."
        else:
            evidence = wrap_untrusted_context([str(d.get("text", "")) for d in documents])
            # Replace this deterministic local generator with an Ollama/hosted LLM adapter.
            titles = ", ".join(s.title for s in sources) or "retrieved evidence"
            answer = f"Grounded answer prepared from {len(documents)} retrieved passage(s): {titles}."
            steps.append(ReasoningStep(step_name="generate", description="Generated answer from untrusted evidence envelope", metadata={"model": context.model, "context_chars": len(evidence)}))

        return AskResponse(query=request.query, answer=answer, sources=sources, reasoning_steps=steps, retrieval_attempts=attempts, search_mode="hybrid" if request.use_hybrid else "bm25", trace_id=trace_id)
