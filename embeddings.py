import requests
import math

EMBED_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "bge-m3"


def get_embedding(text: str) -> list:
    """Превращает текст в вектор (эмбеддинг) через локальную модель."""
    response = requests.post(
        EMBED_URL,
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def cosine_similarity(a: list, b: list) -> float:
    """Косинусная близость двух векторов. 1.0 = одинаковый смысл."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


if __name__ == "__main__":
    query = "Как выбрать кабель для электродвигателя мощностью 5 кВт?"

    candidates = [
        "Выбор сечения кабельной линии для питания электродвигателя мощностью 5 кВт",
        "Кабельные линии должны прокладываться с учетом требований ПУЭ",
        "Сегодня хорошая погода, пойдем гулять в парк",
        "Освещение рабочих мест должно соответствовать нормам охраны труда",
        "The use of aluminum wiring is prohibited.",
        "Cables for electric motors are selected in accordance with the table."
    ]

    query_vec = get_embedding(query)
    print(f"Размерность вектора: {len(query_vec)}")
    print("-" * 70)

    results = []
    for text in candidates:
        vec = get_embedding(text)
        score = cosine_similarity(query_vec, vec)
        results.append((score, text))

    # Сортируем по убыванию похожести
    results.sort(key=lambda x: x[0], reverse=True)

    for score, text in results:
        print(f"{score:.4f} | {text}")