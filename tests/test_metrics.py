from app.evaluation.metrics import citation_coverage, retrieval_metrics


def test_retrieval_metrics() -> None:
    result = retrieval_metrics(["a", "b", "x"], {"a", "b", "c"})
    assert round(result.recall_at_k, 2) == 0.67
    assert round(result.precision_at_k, 2) == 0.67


def test_citation_coverage() -> None:
    assert citation_coverage(4, 3) == 0.75
    assert citation_coverage(0, 0) == 1.0
