from datetime import UTC, datetime

from app.dashboard import dashboard_html
from app.persistence.entities import EvaluationRunEntity


def test_dashboard_renders_quality_kpis_and_run() -> None:
    run = EvaluationRunEntity(
        id="run-1", dataset_name="quality", dataset_version="1.2", cases=10,
        passed_cases=9, pass_rate=0.9, mean_recall_at_k=0.85,
        mean_precision_at_k=0.8, mean_mrr=0.88, mean_ndcg=0.86,
        mean_answer_relevance=0.9, mean_citation_correctness=0.92,
        mean_safety=1.0, mean_latency_ms=125.0, regression_gate_passed=False,
        created_at=datetime.now(UTC),
    )
    html = dashboard_html([run])
    assert "AI Quality Engineering Dashboard" in html
    assert "Release Gate" in html
    assert "FAIL" in html
    assert "quality" in html
    assert "run-1" in html
    assert "Citation Quality" in html


def test_dashboard_empty_state() -> None:
    html = dashboard_html([])
    assert "NO DATA" not in html
    assert "Run an evaluation to populate the dashboard." in html
