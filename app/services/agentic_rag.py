from uuid import uuid4

from app.agent.context import RuntimeContext
from app.agent.graph import AgenticRAGGraph
from app.agent.state import AgentState
from app.llm.base import LLMProvider
from app.models import AskRequest, AskResponse
from app.retrieval.base import Retriever


class AgenticRAGService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        max_attempts: int = 3,
        guardrail_threshold: int = 70,
    ) -> None:
        self.max_attempts = max_attempts
        self.guardrail_threshold = guardrail_threshold
        self.graph = AgenticRAGGraph(retriever, llm, max_attempts, guardrail_threshold)

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
        initial: AgentState = {
            "original_query": request.query,
            "active_query": request.query,
            "retrieval_attempts": 0,
            "reasoning_steps": [],
            "trace_id": trace_id,
            "runtime_context": context,
        }
        state = await self.graph.compiled.ainvoke(initial)
        allowed = bool(state.get("allowed", False))
        answer = str(state.get("answer", "")).strip()
        if not allowed:
            answer = "I cannot process that request."
        elif not answer:
            answer = "No sufficiently relevant evidence was found after bounded retrieval attempts."
        return AskResponse(
            query=request.query,
            answer=answer,
            sources=state.get("sources", []),
            reasoning_steps=state.get("reasoning_steps", []),
            retrieval_attempts=state.get("retrieval_attempts", 0),
            search_mode="hybrid" if request.use_hybrid else "bm25",
            trace_id=trace_id,
        )
