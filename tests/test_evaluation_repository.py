from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.evaluation.models import CaseEvaluation, EvaluationSummary
from app.evaluation.repository import EvaluationRepository
from app.persistence.database import Base


def test_evaluation_history_and_comparison() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, class_=Session, expire_on_commit=False)
    repository = EvaluationRepository(sessions)
    case = CaseEvaluation(
        case_id="c1", trace_id="trace-1", recall_at_k=1.0, precision_at_k=1.0,
        mrr=1.0, ndcg=1.0, answer_relevance=1.0, citation_correctness=1.0,
        safety=1.0, retrieval_attempts=1, latency_ms=10.0, passed=True,
    )
    baseline = EvaluationSummary(
        dataset_name="quality", dataset_version="1", cases=1, passed_cases=1, pass_rate=1.0,
        mean_recall_at_k=1.0, mean_precision_at_k=1.0, mean_mrr=1.0, mean_ndcg=1.0,
        mean_answer_relevance=1.0, mean_citation_correctness=1.0, mean_safety=1.0,
        mean_latency_ms=10.0, regression_gate_passed=True, results=[case],
    )
    baseline_id = repository.save(baseline)
    current = baseline.model_copy(update={"pass_rate": 0.0, "passed_cases": 0, "regression_gate_passed": False})
    current_id = repository.save(current)

    assert len(repository.list_runs()) == 2
    restored = repository.summary(baseline_id)
    assert restored is not None
    assert restored.results[0].trace_id == "trace-1"
    comparison = repository.compare(baseline_id, current_id)
    assert comparison is not None
    assert comparison.pass_rate_delta == -1.0
    assert comparison.quality_decreased is True
