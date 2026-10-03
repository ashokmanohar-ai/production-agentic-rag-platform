from fastapi.testclient import TestClient
from app.main import app, get_service
from app.retrieval.memory import InMemoryRetriever
from app.services.agentic_rag import AgenticRAGService


class FakeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return "test answer"


def fake_service() -> AgenticRAGService:
    return AgenticRAGService(InMemoryRetriever([]), FakeLLM(), max_attempts=1)


app.dependency_overrides[get_service] = fake_service
client = TestClient(app)


def test_health() -> None:
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_validation() -> None:
    response = client.post("/api/v1/ask", json={"query": "", "top_k": 99})
    assert response.status_code == 422


def test_ask_contract() -> None:
    response = client.post("/api/v1/ask", json={"query": "RAG evaluation", "use_hybrid": False})
    assert response.status_code == 200
    body = response.json()
    assert body["search_mode"] == "bm25"
    assert "trace_id" in body
