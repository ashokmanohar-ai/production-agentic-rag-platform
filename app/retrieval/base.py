from typing import Protocol
from app.agent.context import RuntimeContext


class Retriever(Protocol):
    async def search(self, query: str, context: RuntimeContext) -> list[dict[str, object]]:
        """Return structured chunks with stable document_id and chunk_id."""
        ...
