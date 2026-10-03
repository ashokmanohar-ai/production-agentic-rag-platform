from app.observability.langfuse import LangfuseObservability


def test_disabled_observability_is_noop() -> None:
    obs = LangfuseObservability(False)
    with obs.trace("0" * 32, "query", {}) as span:
        assert span is None
    with obs.observation("retrieve", "retriever") as span:
        assert span is None
    obs.score("0" * 32, 1.0, "good")
