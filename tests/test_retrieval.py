import pytest
from app.agent.context import RuntimeContext
from app.retrieval.memory import InMemoryRetriever


@pytest.mark.asyncio
async def test_request_runtime_controls_top_k_mode_and_categories() -> None:
    retriever = InMemoryRetriever([
        {"document_id": "1", "chunk_id": "1a", "title": "AI", "text": "rag evaluation", "category": "cs.AI"},
        {"document_id": "2", "chunk_id": "2a", "title": "IR", "text": "rag retrieval", "category": "cs.IR"},
    ])
    context = RuntimeContext(top_k=1, use_hybrid=False, model="test", categories=("cs.IR",), max_retrieval_attempts=2, guardrail_threshold=70)
    docs = await retriever.search("rag", context)
    assert len(docs) == 1
    assert docs[0]["document_id"] == "2"
    assert docs[0]["search_mode"] == "bm25"
