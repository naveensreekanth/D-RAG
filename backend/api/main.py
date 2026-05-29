"""
FastAPI application exposing the Document Intelligence System.

Run from the repo root:
    uvicorn backend.api.main:app --reload --port 8000
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from backend.api.schemas import (
    UploadResponse, QueryRequest, QueryResponse, SourceModel,
    DocumentModel, ChatEntry, StatusResponse,
)
from backend.database.sqlite_db import db
from backend.vectorstore.faiss_store import store
from backend.services.ingestion_service import ingest_pdf
from backend.services.retrieval_service import answer_query
from backend.utils.logger import get_logger

log = get_logger(__name__)

app = FastAPI(
    title="Document Intelligence System",
    description="RAG over PDFs — FAISS + SQLite + LangChain + Groq LLaMA-3.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
@app.get("/health", response_model=StatusResponse, tags=["status"])
def health():
    docs = db.list_documents()
    return StatusResponse(status="ok", documents=len(docs), chunks_indexed=store.size)


# ---------------------------------------------------------------------------
@app.post("/upload", response_model=UploadResponse, tags=["documents"])
async def upload(file: UploadFile = File(..., description="PDF file")):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(400, f"Unsupported content type: {file.content_type}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        result = ingest_pdf(tmp_path, filename=file.filename)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Ingestion failed")
        raise HTTPException(500, f"Ingestion failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    return UploadResponse(
        document_id=result.document_id,
        filename=result.filename,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
        message=f"Indexed {result.chunk_count} chunks across {result.page_count} pages.",
    )


# ---------------------------------------------------------------------------
@app.post("/query", response_model=QueryResponse, tags=["qa"])
def query(req: QueryRequest):
    try:
        result = answer_query(req.question, session_id=req.session_id, top_k=req.top_k)
    except RuntimeError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        log.exception("Query failed")
        raise HTTPException(500, str(e))

    return QueryResponse(
        question=req.question,
        answer=result.answer,
        sources=[SourceModel(**s.__dict__) for s in result.sources],
        model=result.model,
    )


# ---------------------------------------------------------------------------
@app.get("/documents", response_model=list[DocumentModel], tags=["documents"])
def list_documents():
    return [DocumentModel(**d) for d in db.list_documents()]


@app.delete("/documents/{document_id}", tags=["documents"])
def delete_document(document_id: int):
    vector_ids = db.delete_document(document_id)
    store.remove(vector_ids)
    return {"deleted_document_id": document_id, "removed_vectors": len(vector_ids)}


# ---------------------------------------------------------------------------
@app.get("/chat-history", response_model=list[ChatEntry], tags=["chat"])
def chat_history(session_id: Optional[str] = Query(None), limit: int = Query(50, ge=1, le=500)):
    return [ChatEntry(**r) for r in db.get_chat_history(session_id=session_id, limit=limit)]


# ---------------------------------------------------------------------------
@app.delete("/reset", tags=["admin"])
def reset_all():
    db.reset()
    store.reset()
    return {"message": "Knowledge base and chat history cleared."}
