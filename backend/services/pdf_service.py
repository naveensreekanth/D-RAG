"""
PDF extraction + chunking.

Uses pdfplumber for robust text extraction and LangChain's
RecursiveCharacterTextSplitter for semantically-aware chunking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class Chunk:
    text: str
    page: int  # 1-indexed


_WS = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS.sub(" ", text).strip()


def extract_pages(pdf_path: str) -> List[str]:
    """Return a list of cleaned page strings (empty pages preserved as '')."""
    pages: list[str] = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            try:
                raw = page.extract_text() or ""
            except Exception as e:  # malformed page
                log.warning("Failed to extract a page: %s", e)
                raw = ""
            pages.append(_clean(raw))
    log.info("Extracted %d pages from %s", len(pages), pdf_path)
    return pages


def chunk_pages(pages: List[str]) -> List[Chunk]:
    """Split page text into overlapping chunks, preserving the source page number."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if piece:
                chunks.append(Chunk(text=piece, page=i))
    log.info("Chunked into %d pieces (size=%d, overlap=%d)",
             len(chunks), settings.chunk_size, settings.chunk_overlap)
    return chunks
