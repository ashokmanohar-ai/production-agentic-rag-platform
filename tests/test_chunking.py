import pytest

from app.ingestion.chunking import chunk_text


def test_chunk_ids_are_stable() -> None:
    first = chunk_text("doc-1", "alpha beta gamma delta", chunk_size=12, overlap=3)
    second = chunk_text("doc-1", "alpha beta gamma delta", chunk_size=12, overlap=3)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_chunking_has_overlap_and_unique_ids() -> None:
    chunks = chunk_text("doc-1", "one two three four five six seven", chunk_size=15, overlap=4)
    assert len(chunks) > 1
    assert len({item.chunk_id for item in chunks}) == len(chunks)


def test_invalid_overlap_is_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_text("doc-1", "text", chunk_size=10, overlap=10)
