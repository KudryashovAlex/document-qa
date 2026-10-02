import json
import os

import chromadb

from embeddings import get_embedding
from pdf_chunker import chunk_pdf

PDF_PATH = "test_document.pdf"
CHUNKS_FILE = "chunks.json"
DB_PATH = "./chroma_db"
COLLECTION_NAME = "pue_chunks"
CHUNKS_LIMIT = 300  # для эксперимента индексируем не все 3012, а первые 300


def load_chunks(limit: int = CHUNKS_LIMIT) -> list:
    """Загружает чанки из кэша или создает их из PDF."""
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            all_chunks = json.load(f)
    else:
        all_chunks = chunk_pdf(PDF_PATH)
        with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    return all_chunks[:limit]


def build_collection(chunks: list):
    """Прогоняет чанки через bge-m3 и складывает векторы в ChromaDB."""
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # метрика расстояния = косинус
    )

    ids = [f"chunk_{c['chunk_id']}" for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "length": c["length"],
            "chunk_id": c["chunk_id"],
        }
        for c in chunks
    ]

    embeddings = []
    for i, text in enumerate(documents):
        embeddings.append(get_embedding(text))
        if (i + 1) % 50 == 0:
            print(f"Эмбеддингов создано: {i + 1}/{len(documents)}")

    # upsert = вставить или обновить, если ID уже есть
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return collection


def search(collection, query: str, top_k: int = 5) -> list:
    """Семантический поиск: возвращает top_k самых релевантных чанков."""
    query_vec = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=top_k,
    )

    found = []
    for doc, distance, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
    ):
        found.append(
            {
                "text": doc,
                "distance": round(distance, 4),
                "similarity": round(1 - distance, 4),
                "metadata": meta,
            }
        )
    return found


if __name__ == "__main__":
    chunks = load_chunks()
    print(f"Чанков для индексации: {len(chunks)}")

    collection = build_collection(chunks)
    print(f"Векторов в базе: {collection.count()}")
    print("-" * 70)

    query = "На какое напряжение распространяются правила ПУЭ?"
    print(f"Вопрос: {query}\n")

    for item in search(collection, query, top_k=3):
        print(f"[similarity={item['similarity']}] chunk_id={item['metadata']['chunk_id']}")
        print(item["text"][:200])
        print("-" * 70)