from fastapi.testclient import TestClient
from app.main import app, get_evaluation_repository, get_service
from app.retrieval.memory import InMemoryRetriever
from app.services.agentic_rag import AgenticRAGService


class FakeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return "test answer"


def fake_service() -> AgenticRAGService:
    return AgenticRAGService(InMemoryRetriever([]), FakeLLM(), max_attempts=1)


app.dependency_overrides[get_service] = fake_service


class FakeEvaluationRepository:
    def save(self, summary: object, tenant_id: str = "default", project_id: str = "default") -> str:
        return "run-1"


app.dependency_overrides[get_evaluation_repository] = FakeEvaluationRepository
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


def test_evaluation_contract() -> None:
    response = client.post(
        "/api/v1/evaluations/run",
        json={
            "dataset": {
                "name": "api-smoke",
                "version": "1.0",
                "cases": [
                    {
                        "case_id": "case-1",
                        "query": "RAG evaluation",
                        "relevant_document_ids": [],
                        "expected_answer_terms": [],
                        "top_k": 1,
                        "use_hybrid": False
                    }
                ]
            },
            "thresholds": {
                "min_recall_at_k": 0,
                "min_precision_at_k": 0,
                "min_mrr": 0,
                "min_ndcg": 0,
                "min_answer_relevance": 0,
                "min_citation_correctness": 0,
                "min_safety": 1
            }
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["dataset_name"] == "api-smoke"
    assert body["cases"] == 1
    assert body["regression_gate_passed"] is True
