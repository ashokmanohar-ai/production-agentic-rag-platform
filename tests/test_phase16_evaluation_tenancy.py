from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.evaluation.repository import EvaluationRepository
from app.persistence.database import Base
from app.persistence.entities import EvaluationRunEntity


def test_evaluation_history_is_tenant_scoped() -> None:
    engine = create_engine("sqlite+pysqlite:///:memory:")
    Base.metadata.create_all(engine)
    sessions = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)
    with sessions() as session:
        for tenant in ("tenant-a", "tenant-b"):
            session.add(
                EvaluationRunEntity(
                    tenant_id=tenant,
                    project_id="project",
                    dataset_name="security",
                    dataset_version="1",
                    cases=1,
                    passed_cases=1,
                    pass_rate=1.0,
                    mean_recall_at_k=1.0,
                    mean_precision_at_k=1.0,
                    mean_mrr=1.0,
                    mean_ndcg=1.0,
                    mean_answer_relevance=1.0,
                    mean_citation_correctness=1.0,
                    mean_safety=1.0,
                    mean_latency_ms=10.0,
                    regression_gate_passed=True,
                )
            )
        session.commit()
    repository = EvaluationRepository(sessions)
    tenant_a = repository.list_runs(50, "tenant-a", "project")
    assert len(tenant_a) == 1
    assert tenant_a[0].tenant_id == "tenant-a"
    assert repository.get_run(tenant_a[0].id, "tenant-b", "project") is None
