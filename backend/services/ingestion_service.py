"""
End-to-end ingestion pipeline:
  PDF → pages → chunks → embeddings → FAISS + SQLite
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from backend.database.sqlite_db import db
from backend.vectorstore.faiss_store import store
from backend.services.embedding_service import embedding_service
from backend.services.pdf_service import extract_pages, chunk_pages
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class IngestionResult:
    document_id: int
    filename: str
    page_count: int
    chunk_count: int


def ingest_pdf(pdf_path: str, filename: str) -> IngestionResult:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError("No extractable text found in PDF.")

    texts = [c.text for c in chunks]
    vectors = embedding_service.embed_texts(texts)

    # Reserve contiguous vector IDs starting at next free ID
    start_id = db.next_vector_id()
    vector_ids = list(range(start_id, start_id + len(texts)))

    # Persist metadata first so FAISS additions are always reflected in SQLite
    document_id = db.add_document(filename=filename, page_count=len(pages))
    db.add_chunks(document_id, zip(texts, vector_ids, [c.page for c in chunks]))

    # Then add vectors to FAISS using the same IDs
    store.add(vectors, vector_ids)

    log.info("Ingested '%s' → doc_id=%d, %d chunks", filename, document_id, len(chunks))
    return IngestionResult(
        document_id=document_id,
        filename=filename,
        page_count=len(pages),
        chunk_count=len(chunks),
    )
