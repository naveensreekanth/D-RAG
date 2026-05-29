"""Pydantic request/response models for the FastAPI layer."""
from pydantic import BaseModel, Field
from typing import List, Optional


class UploadResponse(BaseModel):
    document_id: int
    filename: str
    page_count: int
    chunk_count: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: str = Field("default", min_length=1, max_length=64)
    top_k: Optional[int] = Field(None, ge=1, le=20)


class SourceModel(BaseModel):
    filename: str
    page: int
    chunk_preview: str
    similarity: float
    vector_id: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceModel]
    model: str


class DocumentModel(BaseModel):
    document_id: int
    filename: str
    upload_time: str
    page_count: int
    chunk_count: int


class ChatEntry(BaseModel):
    id: int
    session_id: str
    query: str
    response: str
    timestamp: str


class StatusResponse(BaseModel):
    status: str
    documents: int
    chunks_indexed: int
