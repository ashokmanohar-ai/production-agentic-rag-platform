from typing import TypedDict
from app.agent.context import RuntimeContext
from app.models import ReasoningStep, SourceItem


class AgentState(TypedDict, total=False):
    original_query: str
    active_query: str
    answer: str
    retrieval_attempts: int
    allowed: bool
    documents: list[dict[str, object]]
    relevant_documents: list[dict[str, object]]
    sources: list[SourceItem]
    reasoning_steps: list[ReasoningStep]
    trace_id: str
    runtime_context: RuntimeContext
