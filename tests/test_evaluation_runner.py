import pytest

from app.evaluation.models import EvaluationCase, EvaluationDataset, EvaluationRequest
from app.evaluation.runner import EvaluationRunner
from app.retrieval.memory import InMemoryRetriever
from app.services.agentic_rag import AgenticRAGService


class FakeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return "RAG uses retrieval and grounded generation [1]."


@pytest.mark.asyncio
async def test_evaluation_runner_passes_quality_gate() -> None:
    retriever = InMemoryRetriever(
        [{"document_id": "doc-1", "chunk_id": "c1", "title": "RAG", "text": "retrieval grounded generation", "score": 1.0}]
    )
    service = AgenticRAGService(retriever, FakeLLM(), max_attempts=1)
    runner = EvaluationRunner(service)
    request = EvaluationRequest(
        dataset=EvaluationDataset(
            name="smoke",
            version="1.0",
            cases=[
                EvaluationCase(
                    case_id="rag-1",
                    query="Explain retrieval grounded generation",
                    relevant_document_ids=["doc-1"],
                    expected_answer_terms=["retrieval", "grounded", "generation"],
                    top_k=1,
                )
            ],
        )
    )
    result = await runner.run(request)
    assert result.regression_gate_passed is True
    assert result.mean_recall_at_k == 1.0
    assert result.mean_mrr == 1.0
    assert result.mean_ndcg == 1.0
    assert result.results[0].trace_id


class SemanticJudgeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return '{"faithfulness":0.9,"groundedness":0.9,"completeness":0.9,"context_relevance":0.9,"hallucination":0.1,"robustness":0.9}'


@pytest.mark.asyncio
async def test_evaluation_runner_uses_optional_semantic_judge() -> None:
    retriever = InMemoryRetriever(
        [{"document_id": "doc-1", "chunk_id": "c1", "title": "RAG", "text": "retrieval grounded generation", "score": 1.0}]
    )
    service = AgenticRAGService(retriever, FakeLLM(), max_attempts=1)
    request = EvaluationRequest(
        dataset=EvaluationDataset(
            name="semantic", version="1.0",
            cases=[EvaluationCase(case_id="s1", query="RAG retrieval grounded generation", relevant_document_ids=["doc-1"], expected_answer_terms=["retrieval"], top_k=1)],
        ),
        judge_enabled=True,
        judge_model="judge-model",
    )
    result = await EvaluationRunner(service, SemanticJudgeLLM()).run(request)
    assert result.judge_coverage == 1.0
    assert result.mean_faithfulness == 0.9
    assert result.mean_hallucination == 0.1
    assert result.regression_gate_passed is True
