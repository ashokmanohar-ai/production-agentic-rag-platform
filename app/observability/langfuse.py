import os
from contextlib import AbstractContextManager, nullcontext
from typing import Any, Literal, cast

from langfuse import get_client


ObservationType = Literal[
    "span", "agent", "tool", "chain", "retriever", "evaluator",
    "guardrail", "generation", "embedding"
]


class LangfuseObservability:
    def __init__(
        self,
        enabled: bool,
        host: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
    ) -> None:
        self.enabled = enabled
        if enabled:
            if host:
                os.environ["LANGFUSE_BASE_URL"] = host
            if public_key:
                os.environ["LANGFUSE_PUBLIC_KEY"] = public_key
            if secret_key:
                os.environ["LANGFUSE_SECRET_KEY"] = secret_key
        self.client = get_client() if enabled else None

    def trace(
        self, trace_id: str, query: str, metadata: dict[str, object]
    ) -> AbstractContextManager[Any]:
        if not self.client:
            return nullcontext()
        return self.client.start_as_current_observation(
            as_type="agent",
            name="agentic-rag",
            input={"query": query},
            metadata=metadata,
            trace_context={"trace_id": trace_id},
        )

    def observation(
        self,
        name: str,
        observation_type: ObservationType,
        input_data: object | None = None,
        model: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> AbstractContextManager[Any]:
        if not self.client:
            return nullcontext()
        manager = self.client.start_as_current_observation(
            as_type=observation_type,
            name=name,
            input=input_data,
            model=model,
            metadata=metadata,
        )
        return cast(AbstractContextManager[Any], manager)

    def score(self, trace_id: str, value: float, comment: str | None) -> None:
        if not self.client:
            return
        self.client.create_score(
            trace_id=trace_id,
            name="user-feedback",
            value=value,
            data_type="NUMERIC",
            comment=comment,
        )
