import httpx


class OllamaEmbeddingProvider:
    def __init__(
        self,
        base_url: str,
        model: str,
        dimensions: int,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._dimensions = dimensions
        self.timeout_seconds = timeout_seconds

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            vectors = response.json().get("embeddings", [])
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding provider returned an unexpected vector count")
        normalized = [[float(value) for value in vector] for vector in vectors]
        for vector in normalized:
            if len(vector) != self._dimensions:
                raise RuntimeError(
                    f"Embedding dimension {len(vector)} does not match configured dimension "
                    f"{self._dimensions}"
                )
        return normalized
