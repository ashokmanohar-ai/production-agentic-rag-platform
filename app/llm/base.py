from typing import Protocol


class LLMProvider(Protocol):
    async def generate(self, prompt: str, model: str) -> str:
        """Generate text using the requested model."""
        ...
