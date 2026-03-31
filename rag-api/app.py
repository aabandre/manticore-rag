from fastapi import FastAPI
import requests
import os

app = FastAPI()

MANTICORE = os.getenv("MANTICORE_URL")
OLLAMA = os.getenv("OLLAMA_URL")


def search(query):
    r = requests.post(f"{MANTICORE}/search", json={
        "query": {
            "match": {"*": query}
        },
        "limit": 5
    })
    return r.json()["hits"]["hits"]


def build_context(hits):
    return "\n\n".join([
        h["_source"].get("content", "")
        for h in hits
    ])


def generate_answer(query, context):
    r = requests.post(f"{OLLAMA}/api/chat", json={
        "model": "llama3",
        "messages": [
            {
                "role": "system",
                "content": "Отвечай только по базе знаний. Если нет данных — скажи нет информации."
            },
            {
                "role": "user",
                "content": f"Контекст:\n{context}\n\nВопрос:\n{query}"
            }
        ]
    })
    return r.json()["message"]["content"]


@app.post("/rag")
def rag(data: dict):
    query = data.get("query")

    hits = search(query)
    context = build_context(hits)
    answer = generate_answer(query, context)

    return {
        "answer": answer
    }
