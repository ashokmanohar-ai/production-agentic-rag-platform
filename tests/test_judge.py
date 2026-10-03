import pytest

from app.evaluation.judge import LLMJudge


class GoodJudgeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return '{"faithfulness":0.9,"groundedness":0.8,"completeness":0.7,"context_relevance":0.9,"hallucination":0.1,"robustness":1.0}'


class BrokenJudgeLLM:
    async def generate(self, prompt: str, model: str) -> str:
        return "not-json"


@pytest.mark.asyncio
async def test_judge_parses_bounded_scores() -> None:
    result = await LLMJudge(GoodJudgeLLM(), "judge").evaluate("q", "a", "c")
    assert result.available is True
    assert result.faithfulness == 0.9
    assert result.hallucination == 0.1


@pytest.mark.asyncio
async def test_judge_failure_is_non_blocking() -> None:
    result = await LLMJudge(BrokenJudgeLLM(), "judge").evaluate("q", "a", "c")
    assert result.available is False
