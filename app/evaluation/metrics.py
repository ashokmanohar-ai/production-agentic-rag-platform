from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    precision_at_k: float


def retrieval_metrics(retrieved_ids: list[str], relevant_ids: set[str]) -> RetrievalMetrics:
    if not relevant_ids:
        return RetrievalMetrics(recall_at_k=1.0, precision_at_k=1.0 if not retrieved_ids else 0.0)
    hits = sum(1 for item in retrieved_ids if item in relevant_ids)
    recall = hits / len(relevant_ids)
    precision = hits / len(retrieved_ids) if retrieved_ids else 0.0
    return RetrievalMetrics(recall_at_k=recall, precision_at_k=precision)


def citation_coverage(claim_count: int, cited_claim_count: int) -> float:
    if claim_count <= 0:
        return 1.0
    return min(max(cited_claim_count / claim_count, 0.0), 1.0)
