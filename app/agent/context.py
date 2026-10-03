from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeContext:
    top_k: int
    use_hybrid: bool
    model: str
    categories: tuple[str, ...]
    max_retrieval_attempts: int
    guardrail_threshold: int
    tenant_id: str = "default"
    project_id: str = "default"
