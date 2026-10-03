import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    document_id: str
    chunk_id: str
    text: str
    chunk_index: int


def stable_chunk_id(document_id: str, chunk_index: int, text: str) -> str:
    payload = f"{document_id}:{chunk_index}:{text}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]


def chunk_text(
    document_id: str,
    text: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[TextChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[TextChunk] = []
    start = 0
    index = 0
    while start < len(clean):
        end = min(start + chunk_size, len(clean))
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        value = clean[start:end].strip()
        if value:
            chunks.append(
                TextChunk(
                    document_id=document_id,
                    chunk_id=stable_chunk_id(document_id, index, value),
                    text=value,
                    chunk_index=index,
                )
            )
            index += 1
        if end >= len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks
