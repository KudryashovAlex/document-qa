from pypdf import PdfReader
from typing import List


def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает весь текст из PDF файла."""
    reader = PdfReader(pdf_path)
    full_text = ""

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n\n--- Страница {i + 1} ---\n\n"
            full_text += text

    # Чистим лишние пробелы и переносы
    full_text = full_text.replace("\n", " ")
    full_text = " ".join(full_text.split())

    return full_text


def chunk_text(
        text: str,
        chunk_size: int = 500,
        overlap: int = 100,
) -> List[str]:
    """
    Разбивает текст на чанки с перекрытием.

    Args:
        text: исходный текст
        chunk_size: размер чанка в символах
        overlap: размер перекрытия между чанками

    Returns:
        список чанков
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size должен быть больше overlap")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size

        # Если это последний чанк, берем остаток
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Ищем границу предложения в последних 200 символах чанка
        # чтобы не разрывать предложение на середине
        search_start = max(start, end - 200)
        search_zone = text[search_start:end]

        # Пытаемся найти конец предложения
        best_split = end
        for separator in [". ", "! ", "? ", "。", "!"]:
            last_sep = search_zone.rfind(separator)
            if last_sep != -1:
                best_split = search_start + last_sep + len(separator)
                break

        chunk = text[start:best_split].strip()
        chunks.append(chunk)

        # Следующий чанк начинается с перекрытием
        start = best_split - overlap
        if start <= chunks[-1].__len__() - chunk_size + overlap:
            start = best_split

    return chunks


def chunk_pdf(
        pdf_path: str,
        chunk_size: int = 500,
        overlap: int = 100,
) -> List[dict]:
    """
    Полная функция: читает PDF и разбивает на чанки с метаданными.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, overlap)

    result = []
    for i, chunk in enumerate(chunks):
        result.append({
            "chunk_id": i,
            "text": chunk,
            "length": len(chunk),
            "source": pdf_path,
        })

    return result

def save_chunks_to_json(chunks: List[dict], output_path: str):
    """Сохраняет чанки в JSON файл."""
    import json
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    # Тестируем
    import sys

    pdf_file = sys.argv[1] if len(sys.argv) > 1 else "test_document.pdf"

    print(f"Загружаю PDF: {pdf_file}")
    chunks = chunk_pdf(pdf_file, chunk_size=500, overlap=100)

    print(f"Всего чанков: {len(chunks)}")
    print("-" * 60)
    save_chunks_to_json(chunks, "chunks.json")

    # Выводим первые 3 чанка
    for chunk in chunks[:3]:
        print(f"Чанк #{chunk['chunk_id']} (длина: {chunk['length']})")
        print(chunk["text"][:300] + "...")
        print("-" * 60)