from statistics import mean
from time import perf_counter

from app.evaluation.judge import LLMJudge
from app.evaluation.metrics import (
    answer_term_relevance,
    citation_correctness,
    reciprocal_rank,
    retrieval_metrics,
    safety_score,
    ndcg_at_k,
)
from app.evaluation.models import (
    CaseEvaluation,
    EvaluationRequest,
    EvaluationSummary,
)
from app.llm.base import LLMProvider
from app.models import AskRequest
from app.services.agentic_rag import AgenticRAGService


class EvaluationRunner:
    def __init__(self, service: AgenticRAGService, judge_llm: LLMProvider | None = None) -> None:
        self.service = service
        self.judge_llm = judge_llm

    async def run(self, request: EvaluationRequest) -> EvaluationSummary:
        results: list[CaseEvaluation] = []
        t = request.thresholds
        for case in request.dataset.cases:
            started = perf_counter()
            response = await self.service.ask(
                AskRequest(
                    query=case.query,
                    top_k=case.top_k,
                    use_hybrid=case.use_hybrid,
                    model=case.model,
                    categories=case.categories,
                )
            )
            latency_ms = (perf_counter() - started) * 1000
            retrieved = [source.document_id for source in response.sources]
            relevant = set(case.relevant_document_ids)
            retrieval = retrieval_metrics(retrieved, relevant)
            mrr = reciprocal_rank(retrieved, relevant)
            ndcg = ndcg_at_k(retrieved, relevant)
            relevance = answer_term_relevance(response.answer, case.expected_answer_terms)
            citations = citation_correctness(retrieved, relevant)
            safety = safety_score(response.answer)
            judge = None
            if request.judge_enabled and self.judge_llm and request.judge_model:
                context = "\n".join(source.title for source in response.sources)
                judge = await LLMJudge(self.judge_llm, request.judge_model).evaluate(case.query, response.answer, context)
            semantic_ok = True
            if judge and judge.available:
                semantic_ok = min(judge.faithfulness, judge.groundedness, judge.completeness, judge.context_relevance, judge.robustness) >= t.min_judge_quality
                semantic_ok = semantic_ok and judge.hallucination <= t.max_hallucination
            passed = (
                retrieval.recall_at_k >= t.min_recall_at_k
                and retrieval.precision_at_k >= t.min_precision_at_k
                and mrr >= t.min_mrr
                and ndcg >= t.min_ndcg
                and relevance >= t.min_answer_relevance
                and citations >= t.min_citation_correctness
                and safety >= t.min_safety
                and semantic_ok
            )
            results.append(
                CaseEvaluation(
                    case_id=case.case_id,
                    trace_id=response.trace_id,
                    recall_at_k=retrieval.recall_at_k,
                    precision_at_k=retrieval.precision_at_k,
                    mrr=mrr,
                    ndcg=ndcg,
                    answer_relevance=relevance,
                    citation_correctness=citations,
                    safety=safety,
                    judge_available=bool(judge and judge.available),
                    faithfulness=judge.faithfulness if judge and judge.available else None,
                    groundedness=judge.groundedness if judge and judge.available else None,
                    completeness=judge.completeness if judge and judge.available else None,
                    context_relevance=judge.context_relevance if judge and judge.available else None,
                    hallucination=judge.hallucination if judge and judge.available else None,
                    robustness=judge.robustness if judge and judge.available else None,
                    retrieval_attempts=response.retrieval_attempts,
                    latency_ms=latency_ms,
                    passed=passed,
                )
            )
        count = len(results)
        passed_cases = sum(item.passed for item in results)
        return EvaluationSummary(
            dataset_name=request.dataset.name,
            dataset_version=request.dataset.version,
            cases=count,
            passed_cases=passed_cases,
            pass_rate=passed_cases / count,
            mean_recall_at_k=mean(item.recall_at_k for item in results),
            mean_precision_at_k=mean(item.precision_at_k for item in results),
            mean_mrr=mean(item.mrr for item in results),
            mean_ndcg=mean(item.ndcg for item in results),
            mean_answer_relevance=mean(item.answer_relevance for item in results),
            mean_citation_correctness=mean(item.citation_correctness for item in results),
            mean_safety=mean(item.safety for item in results),
            mean_latency_ms=mean(item.latency_ms for item in results),
            regression_gate_passed=passed_cases == count,
            results=results,
        )
