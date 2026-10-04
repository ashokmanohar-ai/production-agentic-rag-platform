from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from app.evaluation.history_models import EvaluationComparison
from app.evaluation.models import CaseEvaluation, EvaluationSummary
from app.persistence.entities import EvaluationCaseEntity, EvaluationRunEntity


class EvaluationRepository:
    def __init__(self, sessions: sessionmaker[Session]) -> None:
        self.sessions = sessions

    def save(\n        self, summary: EvaluationSummary, tenant_id: str = "default", project_id: str = "default"\n    ) -> str:
        run = EvaluationRunEntity(
            tenant_id=tenant_id,
            project_id=project_id,
            dataset_name=summary.dataset_name,
            dataset_version=summary.dataset_version,
            cases=summary.cases,
            passed_cases=summary.passed_cases,
            pass_rate=summary.pass_rate,
            mean_recall_at_k=summary.mean_recall_at_k,
            mean_precision_at_k=summary.mean_precision_at_k,
            mean_mrr=summary.mean_mrr,
            mean_ndcg=summary.mean_ndcg,
            mean_answer_relevance=summary.mean_answer_relevance,
            mean_citation_correctness=summary.mean_citation_correctness,
            mean_safety=summary.mean_safety,
            mean_latency_ms=summary.mean_latency_ms,
            regression_gate_passed=summary.regression_gate_passed,
        )
        with self.sessions() as session:
            session.add(run)
            session.flush()
            for item in summary.results:
                session.add(EvaluationCaseEntity(
                    run_id=run.id,
                    case_id=item.case_id,
                    trace_id=item.trace_id,
                    recall_at_k=item.recall_at_k,
                    precision_at_k=item.precision_at_k,
                    mrr=item.mrr,
                    ndcg=item.ndcg,
                    answer_relevance=item.answer_relevance,
                    citation_correctness=item.citation_correctness,
                    safety=item.safety,
                    retrieval_attempts=item.retrieval_attempts,
                    latency_ms=item.latency_ms,
                    passed=item.passed,
                ))
            session.commit()
            return run.id

    def list_runs(
        self, limit: int = 50, tenant_id: str | None = None, project_id: str | None = None
    ) -> list[EvaluationRunEntity]:
        with self.sessions() as session:
            statement = select(EvaluationRunEntity)
            if tenant_id is not None:
                statement = statement.where(
                    EvaluationRunEntity.tenant_id == tenant_id,
                    EvaluationRunEntity.project_id == project_id,
                )
            items = list(session.scalars(
                statement.order_by(EvaluationRunEntity.created_at.desc()).limit(limit)
            ))
            for item in items:
                session.expunge(item)
            return items

    def get_run(
        self, run_id: str, tenant_id: str | None = None, project_id: str | None = None
    ) -> EvaluationRunEntity | None:
        with self.sessions() as session:
            item = session.get(EvaluationRunEntity, run_id)
            if item and tenant_id is not None and (
                item.tenant_id != tenant_id or item.project_id != project_id
            ):
                item = None
            if item:
                session.expunge(item)
            return item

    def get_cases(self, run_id: str) -> list[EvaluationCaseEntity]:
        with self.sessions() as session:
            items = list(session.scalars(select(EvaluationCaseEntity).where(EvaluationCaseEntity.run_id == run_id).order_by(EvaluationCaseEntity.case_id)))
            for item in items:
                session.expunge(item)
            return items

    def summary(self, run_id: str) -> EvaluationSummary | None:
        run = self.get_run(run_id)
        if not run:
            return None
        results = [
            CaseEvaluation(
                case_id=item.case_id, trace_id=item.trace_id, recall_at_k=item.recall_at_k,
                precision_at_k=item.precision_at_k, mrr=item.mrr, ndcg=item.ndcg,
                answer_relevance=item.answer_relevance, citation_correctness=item.citation_correctness,
                safety=item.safety, retrieval_attempts=item.retrieval_attempts,
                latency_ms=item.latency_ms, passed=item.passed,
            ) for item in self.get_cases(run_id)
        ]
        return EvaluationSummary(
            dataset_name=run.dataset_name, dataset_version=run.dataset_version, cases=run.cases,
            passed_cases=run.passed_cases, pass_rate=run.pass_rate, mean_recall_at_k=run.mean_recall_at_k,
            mean_precision_at_k=run.mean_precision_at_k, mean_mrr=run.mean_mrr, mean_ndcg=run.mean_ndcg,
            mean_answer_relevance=run.mean_answer_relevance, mean_citation_correctness=run.mean_citation_correctness,
            mean_safety=run.mean_safety, mean_latency_ms=run.mean_latency_ms,
            regression_gate_passed=run.regression_gate_passed, results=results,
        )

    def compare(self, baseline_id: str, current_id: str) -> EvaluationComparison | None:
        baseline = self.get_run(baseline_id)
        current = self.get_run(current_id)
        if not baseline or not current:
            return None
        return EvaluationComparison(
            baseline_run_id=baseline_id, current_run_id=current_id,
            pass_rate_delta=current.pass_rate-baseline.pass_rate,
            recall_delta=current.mean_recall_at_k-baseline.mean_recall_at_k,
            precision_delta=current.mean_precision_at_k-baseline.mean_precision_at_k,
            mrr_delta=current.mean_mrr-baseline.mean_mrr, ndcg_delta=current.mean_ndcg-baseline.mean_ndcg,
            answer_relevance_delta=current.mean_answer_relevance-baseline.mean_answer_relevance,
            citation_delta=current.mean_citation_correctness-baseline.mean_citation_correctness,
            safety_delta=current.mean_safety-baseline.mean_safety,
            latency_ms_delta=current.mean_latency_ms-baseline.mean_latency_ms,
            quality_decreased=current.pass_rate < baseline.pass_rate,
        )
