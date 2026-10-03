from pydantic import BaseModel


class EvaluationComparison(BaseModel):
    baseline_run_id: str
    current_run_id: str
    pass_rate_delta: float
    recall_delta: float
    precision_delta: float
    mrr_delta: float
    ndcg_delta: float
    answer_relevance_delta: float
    citation_delta: float
    safety_delta: float
    latency_ms_delta: float
    quality_decreased: bool
