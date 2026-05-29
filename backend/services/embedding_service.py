"""
Reusable HuggingFace embedding service (all-MiniLM-L6-v2 by default).
Lazy-loads the model the first time it's used.
"""
from __future__ import annotations

from typing import List
import numpy as np

from langchain_huggingface import HuggingFaceEmbeddings

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)


class EmbeddingService:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self._model: HuggingFaceEmbeddings | None = None

    @property
    def model(self) -> HuggingFaceEmbeddings:
        if self._model is None:
            log.info("Loading embedding model: %s", self.model_name)
            self._model = HuggingFaceEmbeddings(model_name=self.model_name)
        return self._model

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        vectors = self.model.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        return np.array(self.model.embed_query(query), dtype=np.float32)


embedding_service = EmbeddingService()
