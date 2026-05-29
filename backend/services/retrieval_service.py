"""
Retrieval-Augmented Generation orchestrator.

Flow:
  1. Embed user query
  2. FAISS top-k vector_ids + similarity scores
  3. SQLite lookup → chunk_text + filename + page
  4. Build augmented prompt
  5. Call Groq LLaMA-3
  6. Persist chat history
  7. Return answer + structured sources
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from backend.database.sqlite_db import db
from backend.vectorstore.faiss_store import store
from backend.services.embedding_service import embedding_service
from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


SYSTEM_PROMPT = """You are a precise enterprise document assistant.
Use the following retrieved context to answer the question accurately.
If the answer cannot be found in the context, say you don't know based on
the provided documents. Cite filenames inline when helpful.

Context:
{context}
"""

USER_PROMPT = "Question: {question}\n\nAnswer:"


@dataclass
class Source:
    filename: str
    page: int
    chunk_preview: str
    similarity: float
    vector_id: int


@dataclass
class RAGAnswer:
    answer: str
    sources: List[Source] = field(default_factory=list)
    model: str = settings.groq_model


def _build_llm() -> ChatGroq:
    api_key = settings.groq_api_key or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to .env or pass via the API."
        )
    return ChatGroq(api_key=api_key, model=settings.groq_model, temperature=0.2)


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for i, ch in enumerate(chunks, start=1):
        lines.append(
            f"[{i}] (source: {ch['filename']}, page {ch['page']})\n{ch['chunk_text']}"
        )
    return "\n\n".join(lines)


def answer_query(question: str, session_id: str = "default", top_k: int | None = None) -> RAGAnswer:
    if store.size == 0:
        raise RuntimeError("No documents indexed yet. Upload a PDF first.")

    k = top_k or settings.top_k

    # 1) Embed query
    qvec = embedding_service.embed_query(question)

    # 2) FAISS search
    hits = store.search(qvec, top_k=k)   # [(vector_id, similarity)]
    if not hits:
        return RAGAnswer(answer="No relevant context was found in the indexed documents.")

    vector_ids = [vid for vid, _ in hits]
    sim_map = {vid: sim for vid, sim in hits}

    # 3) SQLite lookup
    rows = db.fetch_chunks_by_vector_ids(vector_ids)

    # 4) Build prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("user", USER_PROMPT),
    ])
    llm = _build_llm()
    chain = prompt | llm

    context_str = _format_context(rows)
    response = chain.invoke({"context": context_str, "question": question})
    answer_text = getattr(response, "content", str(response))

    # 5) Structure sources
    sources = [
        Source(
            filename=r["filename"],
            page=int(r["page"] or 0),
            chunk_preview=(r["chunk_text"][:280] + ("…" if len(r["chunk_text"]) > 280 else "")),
            similarity=round(sim_map.get(r["vector_id"], 0.0), 4),
            vector_id=r["vector_id"],
        )
        for r in rows
    ]

    # 6) Persist chat
    db.add_chat(session_id=session_id, query=question, response=answer_text)

    return RAGAnswer(answer=answer_text, sources=sources)
