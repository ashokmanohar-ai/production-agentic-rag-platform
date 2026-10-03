from langgraph.graph import END, START, StateGraph
from app.agent.state import AgentState
from app.agent.sources import extract_sources
from app.agent.security import wrap_untrusted_context
from app.models import ReasoningStep
from app.retrieval.base import Retriever
from app.llm.base import LLMProvider


class AgenticRAGGraph:
    """Compiled LangGraph workflow: guardrail -> retrieve -> grade -> rewrite/generate."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        max_attempts: int,
        guardrail_threshold: int,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.max_attempts = max_attempts
        self.guardrail_threshold = guardrail_threshold
        builder = StateGraph(AgentState)
        builder.add_node("guardrail", self.guardrail)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("grade_documents", self.grade_documents)
        builder.add_node("rewrite_query", self.rewrite_query)
        builder.add_node("generate", self.generate)
        builder.add_edge(START, "guardrail")
        builder.add_conditional_edges("guardrail", self.route_guardrail, {"retrieve": "retrieve", "end": END})
        builder.add_edge("retrieve", "grade_documents")
        builder.add_conditional_edges(
            "grade_documents",
            self.route_after_grade,
            {"generate": "generate", "rewrite": "rewrite_query", "end": END},
        )
        builder.add_edge("rewrite_query", "retrieve")
        builder.add_edge("generate", END)
        self.compiled = builder.compile()

    async def guardrail(self, state: AgentState) -> dict[str, object]:
        query = state["original_query"]
        blocked = ("reveal system prompt", "ignore previous instructions", "steal credentials")
        score = 0 if any(term in query.lower() for term in blocked) else 100
        steps = list(state.get("reasoning_steps", []))
        steps.append(ReasoningStep(step_name="guardrail", description="Safety/scope gate", metadata={"score": score}))
        return {"allowed": score >= self.guardrail_threshold, "reasoning_steps": steps}

    def route_guardrail(self, state: AgentState) -> str:
        return "retrieve" if state.get("allowed", False) else "end"

    async def retrieve(self, state: AgentState) -> dict[str, object]:
        attempts = state.get("retrieval_attempts", 0) + 1
        docs = await self.retriever.search(state["active_query"], state["runtime_context"])
        steps = list(state.get("reasoning_steps", []))
        steps.append(ReasoningStep(step_name="retrieve", description="Retrieved candidate evidence", metadata={"attempt": attempts, "count": len(docs)}))
        return {"documents": docs, "retrieval_attempts": attempts, "reasoning_steps": steps}

    async def grade_documents(self, state: AgentState) -> dict[str, object]:
        query_terms = {v.lower() for v in state["active_query"].split() if v.strip()}
        relevant = []
        for doc in state.get("documents", []):
            text = str(doc.get("text", "")).lower()
            if any(term in text for term in query_terms):
                relevant.append(doc)
        steps = list(state.get("reasoning_steps", []))
        steps.append(ReasoningStep(step_name="grade_documents", description="Graded evidence relevance", metadata={"relevant": len(relevant)}))
        return {"relevant_documents": relevant, "sources": extract_sources(relevant), "reasoning_steps": steps}

    def route_after_grade(self, state: AgentState) -> str:
        if state.get("relevant_documents"):
            return "generate"
        if state.get("retrieval_attempts", 0) < self.max_attempts:
            return "rewrite"
        return "end"

    async def rewrite_query(self, state: AgentState) -> dict[str, object]:
        prompt = f"Rewrite this search query for better retrieval. Return only the query:\n{state['active_query']}"
        rewritten = await self.llm.generate(prompt, state["runtime_context"].model)
        steps = list(state.get("reasoning_steps", []))
        steps.append(ReasoningStep(step_name="rewrite_query", description="LLM rewrote query after weak retrieval"))
        return {"active_query": rewritten or state["active_query"], "reasoning_steps": steps}

    async def generate(self, state: AgentState) -> dict[str, object]:
        docs = state.get("relevant_documents", [])
        evidence = wrap_untrusted_context([str(d.get("text", "")) for d in docs])
        prompt = (
            "Answer the user using only the supplied evidence. If evidence is insufficient, say so. "
            "Cite evidence using [1], [2], etc.\n\n"
            f"Question: {state['original_query']}\n\n{evidence}"
        )
        answer = await self.llm.generate(prompt, state["runtime_context"].model)
        steps = list(state.get("reasoning_steps", []))
        steps.append(ReasoningStep(step_name="generate", description="Generated evidence-grounded answer", metadata={"model": state["runtime_context"].model}))
        return {"answer": answer, "reasoning_steps": steps}
