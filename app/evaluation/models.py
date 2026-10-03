from pydantic import BaseModel, Field


class EvaluationCase(BaseModel):
    case_id: str = Field(min_length=1, max_length=100)
    query: str = Field(min_length=1, max_length=1000)
    relevant_document_ids: list[str] = Field(default_factory=list)
    expected_answer_terms: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=10)
    use_hybrid: bool = False
    model: str = Field(default="llama3.2:3b", min_length=1, max_length=100)
    categories: list[str] | None = None


class EvaluationDataset(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    version: str = Field(min_length=1, max_length=50)
    cases: list[EvaluationCase] = Field(min_length=1, max_length=500)


class EvaluationThresholds(BaseModel):
    min_recall_at_k: float = Field(default=0.70, ge=0, le=1)
    min_precision_at_k: float = Field(default=0.50, ge=0, le=1)
    min_mrr: float = Field(default=0.60, ge=0, le=1)
    min_ndcg: float = Field(default=0.65, ge=0, le=1)
    min_answer_relevance: float = Field(default=0.60, ge=0, le=1)
    min_citation_correctness: float = Field(default=0.70, ge=0, le=1)
    min_safety: float = Field(default=1.0, ge=0, le=1)


class EvaluationRequest(BaseModel):
    dataset: EvaluationDataset
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)


class CaseEvaluation(BaseModel):
    case_id: str
    trace_id: str
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg: float
    answer_relevance: float
    citation_correctness: float
    safety: float
    retrieval_attempts: int
    latency_ms: float
    passed: bool


class EvaluationSummary(BaseModel):
    dataset_name: str
    dataset_version: str
    cases: int
    passed_cases: int
    pass_rate: float
    mean_recall_at_k: float
    mean_precision_at_k: float
    mean_mrr: float
    mean_ndcg: float
    mean_answer_relevance: float
    mean_citation_correctness: float
    mean_safety: float
    mean_latency_ms: float
    regression_gate_passed: bool
    results: list[CaseEvaluation]
