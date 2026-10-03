from fastapi.testclient import TestClient

from app.main import app
from app.runtime import RuntimeDiagnostics


client = TestClient(app)


def test_ready_contract(monkeypatch) -> None:
    async def ready(self):
        return {
            "status": "ready",
            "dependencies": {
                name: {"status": "ok", "detail": None}
                for name in ("postgres", "redis", "opensearch", "ollama")
            },
        }

    monkeypatch.setattr(RuntimeDiagnostics, "check", ready)
    response = client.get("/api/v1/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_ready_returns_503(monkeypatch) -> None:
    async def not_ready(self):
        return {
            "status": "not_ready",
            "dependencies": {
                "postgres": {"status": "ok", "detail": None},
                "redis": {"status": "ok", "detail": None},
                "opensearch": {"status": "ok", "detail": None},
                "ollama": {"status": "error", "detail": "missing_models:test"},
            },
        }

    monkeypatch.setattr(RuntimeDiagnostics, "check", not_ready)
    response = client.get("/api/v1/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "not_ready"
