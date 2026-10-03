from app.agent.context import RuntimeContext


def _score(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


class InMemoryRetriever:
    """Small deterministic retriever for local development and tests."""

    def __init__(self, documents: list[dict[str, object]] | None = None) -> None:
        self.documents = documents or []

    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        terms = {term.lower() for term in query.split() if term.strip()}
        candidates: list[dict[str, object]] = []
        for doc in self.documents:
            category = str(doc.get("category", ""))
            if context.categories and category not in context.categories:
                continue
            text = str(doc.get("text", "")).lower()
            score = sum(1 for term in terms if term in text)
            if score:
                item = dict(doc)
                item["score"] = float(score)
                item["search_mode"] = "hybrid" if context.use_hybrid else "bm25"
                candidates.append(item)
        candidates.sort(key=lambda item: _score(item.get("score", 0)), reverse=True)
        return candidates[: context.top_k]
