import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.schemas import Question
from app.dependencies import embed_model, faiss_index, documents, get_groq_client, GROQ_MODEL

router = APIRouter()


def search(query: str, top_k: int = 5) -> list:
    embedding = embed_model.encode([query])
    embedding = np.array(embedding).astype("float32")
    distances, indices = faiss_index.search(embedding, top_k)
    return [documents[idx] for idx in indices[0] if 0 <= idx < len(documents)]


def generate_chat_answer(query: str, docs: list) -> str:
    context = "\n\n".join(d.get("text", "").strip() for d in docs if d.get("text")).strip()

    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Tu es un assistant interne d'une entreprise tunisienne spécialisée "
                    "dans les projets environnementaux. "
                    "Réponds uniquement à partir du contexte documentaire fourni. "
                    "Si la question est hors sujet, refuse poliment. "
                    "Max 5 phrases, style clair et professionnel."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXTE :\n{context or 'Aucun document disponible.'}\n\n"
                    f"QUESTION :\n{query}\n\n"
                    "Réponds uniquement à partir du contexte ci-dessus."
                ),
            },
        ],
        temperature=0.2,
        max_tokens=500,
    )
    return response.choices[0].message.content


@router.options("/chat")
async def chat_preflight(request: Request):
    return Response(
        status_code=204,
        headers={
            "Access-Control-Allow-Origin":  "http://localhost:8080",
            "Access-Control-Allow-Methods": "POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age":       "600",
        },
    )


@router.post("/chat")
def chat(q: Question):
    docs   = search(q.question)
    answer = generate_chat_answer(q.question, docs)
    return {
        "question": q.question,
        "answer":   answer,
        "sources":  [d.get("metadata", {}) for d in docs],
    }
