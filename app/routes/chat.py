import numpy as np
from fastapi import APIRouter, Request
from fastapi.responses import Response

from app.schemas import Question
from app.dependencies import embed_model, faiss_index, documents, get_groq_client, GROQ_MODEL

router = APIRouter()


MAX_HISTORY_TURNS = 6  # keep the last N turns for context (3 exchanges)


def search(query: str, top_k: int = 5) -> list:
    embedding = embed_model.encode([query])
    embedding = np.array(embedding).astype("float32")
    distances, indices = faiss_index.search(embedding, top_k)
    return [documents[idx] for idx in indices[0] if 0 <= idx < len(documents)]


def build_search_query(question: str, history: list) -> str:
    """
    Enrich short / vague follow-up questions (e.g. "comme ?") with the
    last user turn so FAISS retrieves relevant documents.
    """
    last_user = ""
    for turn in reversed(history):
        if turn.role == "user":
            last_user = turn.content
            break
    # If the question is short, prepend the previous user question for context
    if len(question.split()) <= 4 and last_user:
        return f"{last_user} {question}".strip()
    return question


def generate_chat_answer(query: str, docs: list, history: list) -> str:
    context = "\n\n".join(d.get("text", "").strip() for d in docs if d.get("text")).strip()

    messages = [
        {
            "role": "system",
            "content": (
                "Tu es un assistant interne d'une entreprise tunisienne spécialisée "
                "dans les projets environnementaux. "
                "Tu disposes de l'historique de la conversation : utilise-le pour "
                "comprendre les questions de suivi (ex. « comme ? », « lesquels ? », "
                "« et en 2024 ? ») en te référant aux échanges précédents. "
                "Réponds en priorité à partir du contexte documentaire fourni. "
                "Si la question est hors sujet, refuse poliment. "
                "Max 5 phrases, style clair et professionnel."
            ),
        },
    ]

    # Replay recent conversation history
    for turn in history[-MAX_HISTORY_TURNS:]:
        role = "assistant" if turn.role == "bot" else "user"
        messages.append({"role": role, "content": turn.content})

    # Current question + freshly retrieved context
    messages.append({
        "role": "user",
        "content": (
            f"CONTEXTE DOCUMENTAIRE :\n{context or 'Aucun document disponible.'}\n\n"
            f"QUESTION :\n{query}\n\n"
            "Réponds en t'appuyant sur le contexte ci-dessus et l'historique de la conversation."
        ),
    })

    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
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
    search_query = build_search_query(q.question, q.history)
    docs   = search(search_query)
    answer = generate_chat_answer(q.question, docs, q.history)
    return {
        "question": q.question,
        "answer":   answer,
        "sources":  [d.get("metadata", {}) for d in docs],
    }
