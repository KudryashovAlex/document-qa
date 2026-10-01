import pytest
from pdf_chunker import chunk_text


def test_returns_list_of_non_empty():
    """Чанкер возвращает список непустых чанков."""
    text = ("Первое предложение. Второе предложение. Третье предложение. ") * 10
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert isinstance(chunks, list)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)


def test_small_text_single_chunk():
    """Короткий текст не разбивается и возвращается как один чанк."""
    text = "Короткий текст."
    chunks = chunk_text(text, chunk_size=500, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "Короткий текст."


def test_chunk_size_must_be_greater_than_overlap():
    """chunk_size <= overlap — ошибка конфигурации."""
    with pytest.raises(ValueError):
        chunk_text("текст", chunk_size=100, overlap=100)


def test_overlap_present():
    """Хвост предыдущего чанка присутствует в начале следующего."""
    text = "слово " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 2
    for i in range(len(chunks) - 1):
        tail = chunks[i][-20:]
        assert tail in chunks[i + 1]