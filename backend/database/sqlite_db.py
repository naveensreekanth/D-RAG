"""
SQLite metadata store.

Tables
------
documents     : one row per uploaded PDF
chunks        : one row per text chunk, links to FAISS vector_id
chat_history  : query/response audit log
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime
from typing import Iterable, Optional

from backend.utils.config import settings
from backend.utils.logger import get_logger

log = get_logger(__name__)
_lock = threading.Lock()


SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    filename      TEXT NOT NULL,
    upload_time   TEXT NOT NULL,
    page_count    INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id  INTEGER NOT NULL,
    chunk_text   TEXT NOT NULL,
    vector_id    INTEGER NOT NULL UNIQUE,
    page         INTEGER,
    FOREIGN KEY (document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    query       TEXT NOT NULL,
    response    TEXT NOT NULL,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id);
CREATE INDEX IF NOT EXISTS idx_chunks_document  ON chunks(document_id);
"""


class SQLiteDB:
    def __init__(self, path: Optional[str] = None):
        self.path = path or settings.sqlite_path
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        self._init_schema()

    # ------------------------------------------------------------------
    @contextmanager
    def _conn(self):
        with _lock:
            conn = sqlite3.connect(self.path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys = ON")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as c:
            c.executescript(SCHEMA)
        log.info("SQLite schema ready at %s", self.path)

    # ------------------- documents ------------------------------------
    def add_document(self, filename: str, page_count: int) -> int:
        with self._conn() as c:
            cur = c.execute(
                "INSERT INTO documents (filename, upload_time, page_count) VALUES (?,?,?)",
                (filename, datetime.utcnow().isoformat(), page_count),
            )
            return cur.lastrowid

    def list_documents(self) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT d.*, COUNT(ch.chunk_id) AS chunk_count "
                "FROM documents d LEFT JOIN chunks ch ON ch.document_id = d.document_id "
                "GROUP BY d.document_id ORDER BY d.upload_time DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_document(self, document_id: int) -> list[int]:
        """Delete a document and its chunks. Returns vector_ids that were owned."""
        with self._conn() as c:
            vector_ids = [
                r["vector_id"]
                for r in c.execute(
                    "SELECT vector_id FROM chunks WHERE document_id = ?", (document_id,)
                ).fetchall()
            ]
            c.execute("DELETE FROM documents WHERE document_id = ?", (document_id,))
            return vector_ids

    # ------------------- chunks ---------------------------------------
    def add_chunks(
        self,
        document_id: int,
        chunks: Iterable[tuple[str, int, int]],  # (text, vector_id, page)
    ) -> None:
        with self._conn() as c:
            c.executemany(
                "INSERT INTO chunks (document_id, chunk_text, vector_id, page) VALUES (?,?,?,?)",
                [(document_id, t, v, p) for t, v, p in chunks],
            )

    def fetch_chunks_by_vector_ids(self, vector_ids: list[int]) -> list[dict]:
        if not vector_ids:
            return []
        placeholders = ",".join("?" * len(vector_ids))
        sql = (
            f"SELECT ch.vector_id, ch.chunk_text, ch.page, d.filename, d.document_id "
            f"FROM chunks ch JOIN documents d ON d.document_id = ch.document_id "
            f"WHERE ch.vector_id IN ({placeholders})"
        )
        with self._conn() as c:
            rows = c.execute(sql, vector_ids).fetchall()
        # preserve input order
        by_vid = {r["vector_id"]: dict(r) for r in rows}
        return [by_vid[v] for v in vector_ids if v in by_vid]

    def next_vector_id(self) -> int:
        with self._conn() as c:
            row = c.execute("SELECT COALESCE(MAX(vector_id), -1) AS m FROM chunks").fetchone()
            return int(row["m"]) + 1

    # ------------------- chat history ---------------------------------
    def add_chat(self, session_id: str, query: str, response: str) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO chat_history (session_id, query, response, timestamp) VALUES (?,?,?,?)",
                (session_id, query, response, datetime.utcnow().isoformat()),
            )

    def get_chat_history(self, session_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        with self._conn() as c:
            if session_id:
                rows = c.execute(
                    "SELECT * FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                    (session_id, limit),
                ).fetchall()
            else:
                rows = c.execute(
                    "SELECT * FROM chat_history ORDER BY id DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    # ------------------- maintenance ----------------------------------
    def reset(self) -> None:
        with self._conn() as c:
            c.executescript(
                "DELETE FROM chunks; DELETE FROM documents; DELETE FROM chat_history;"
            )
        log.info("SQLite metadata wiped.")


db = SQLiteDB()
