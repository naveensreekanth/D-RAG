"""
Thin wrapper around a FAISS IndexFlatL2 with persistent integer IDs
that match the `vector_id` column in SQLite.

We deliberately store ONLY vectors here — no text or metadata —
keeping a clean separation of concerns:
  - FAISS  : semantic similarity search
  - SQLite : text + metadata source of truth
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import faiss
import numpy as np

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)
_lock = threading.Lock()


class FaissStore:
    def __init__(self, dim: int = 384, index_dir: Optional[str] = None):
        self.dim = dim
        self.index_dir = index_dir or settings.faiss_index_dir
        os.makedirs(self.index_dir, exist_ok=True)
        self.index_path = os.path.join(self.index_dir, "index.faiss")
        self.index = self._load_or_create()

    # ------------------------------------------------------------------
    def _load_or_create(self) -> faiss.Index:
        if os.path.exists(self.index_path):
            log.info("Loading FAISS index from %s", self.index_path)
            return faiss.read_index(self.index_path)
        log.info("Creating new FAISS index (IndexIDMap over FlatL2, dim=%d)", self.dim)
        return faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))

    def _save(self) -> None:
        faiss.write_index(self.index, self.index_path)

    # ------------------------------------------------------------------
    def add(self, vectors: np.ndarray, ids: list[int]) -> None:
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        with _lock:
            self.index.add_with_ids(vectors, np.array(ids, dtype=np.int64))
            self._save()
        log.info("FAISS: added %d vectors (total=%d)", len(ids), self.index.ntotal)

    def search(self, query_vec: np.ndarray, top_k: int = 4) -> list[tuple[int, float]]:
        if self.index.ntotal == 0:
            return []
        if query_vec.dtype != np.float32:
            query_vec = query_vec.astype(np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
        distances, ids = self.index.search(query_vec, top_k)
        results: list[tuple[int, float]] = []
        for vid, dist in zip(ids[0].tolist(), distances[0].tolist()):
            if vid == -1:
                continue
            # Convert L2 distance to a rough similarity score in [0,1]
            similarity = 1.0 / (1.0 + float(dist))
            results.append((int(vid), similarity))
        return results

    def remove(self, ids: list[int]) -> None:
        if not ids:
            return
        with _lock:
            self.index.remove_ids(np.array(ids, dtype=np.int64))
            self._save()
        log.info("FAISS: removed %d vectors", len(ids))

    def reset(self) -> None:
        with _lock:
            self.index = faiss.IndexIDMap(faiss.IndexFlatL2(self.dim))
            self._save()
        log.info("FAISS index reset.")

    @property
    def size(self) -> int:
        return int(self.index.ntotal)


store = FaissStore()
