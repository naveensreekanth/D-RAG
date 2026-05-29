"""
Centralised configuration loader.
Reads environment variables (with .env support) and exposes them as a
typed `settings` singleton.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    sqlite_path: str = os.getenv("SQLITE_PATH", "database/rag_metadata.db")
    faiss_index_dir: str = os.getenv("FAISS_INDEX_DIR", "vectorstore/faiss_index")

    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))


settings = Settings()
