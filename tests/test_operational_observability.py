from fastapi.testclient import TestClient

from app.main import app
from app.observability.metrics import SLOWindow


client = TestClient(app)


def test_health_returns_correlation_id() -> None:
    response = client.get("/api/v1/health", headers={"X-Correlation-ID": "qe-123"})
    assert response.status_code == 200
    assert response.headers["X-Correlation-ID"] == "qe-123"


def test_metrics_exposes_http_series() -> None:
    client.get("/api/v1/health")
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "rag_http_requests_total" in response.text


def test_slo_window_calculates_error_rate_and_p95() -> None:
    window = SLOWindow()
    window.observe(0.1, False)
    window.observe(0.5, True)
    snapshot = window.snapshot()
    assert snapshot["requests"] == 2
    assert snapshot["errors"] == 1
    assert snapshot["error_rate"] == 0.5
    assert snapshot["p95_latency_ms"] == 500.0
