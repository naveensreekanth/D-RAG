# 📄 Document - RAG

A production-style Retrieval-Augmented Generation (RAG) application for
querying enterprise PDF documents with grounded, source-cited answers.

**Stack:** FastAPI · Streamlit · LangChain · FAISS · SQLite · HuggingFace
`all-MiniLM-L6-v2` · Groq LLaMA-3.

---

## ✨ Features

- Upload one or many PDFs and ingest them into a persistent vector index.
- Strict dual-database architecture: **FAISS** for similarity, **SQLite** for
  metadata, chunks, and chat history.
- Semantic retrieval → metadata join → augmented prompt → LLM generation.
- Every answer shows source citations: filename, page, similarity, preview.
- Per-document delete, full knowledge-base reset, top-K tuning, multi-document QA.
- Modular service layout — clean separation between API, services, storage.

---

## 🏗 Architecture

```
              ┌──────────────┐        ┌──────────────────┐
PDF upload ──▶│ pdf_service  │──────▶ │ embedding_svc    │
              └──────────────┘        └────────┬─────────┘
                                               │ vectors
                                               ▼
                              ┌────────────────────────────┐
                              │ FAISS (IndexIDMap+FlatL2)  │  ◀── vector_id
                              └────────────────────────────┘
                                               ▲
                              ┌────────────────┴───────────┐
                              │ SQLite: documents, chunks, │
                              │         chat_history       │
                              └────────────────────────────┘
                                               ▲
                                               │ chunk_text + filename + page
User query ─▶ embed ─▶ FAISS search ─▶ SQLite lookup ─▶ prompt ─▶ Groq LLaMA-3 ─▶ answer + sources
```

---

## 📁 Project Structure

```
rag-docqa-pro/
├── backend/
│   ├── api/                 FastAPI app + Pydantic schemas
│   │   ├── main.py
│   │   └── schemas.py
│   ├── services/            Business logic
│   │   ├── pdf_service.py
│   │   ├── embedding_service.py
│   │   ├── ingestion_service.py
│   │   └── retrieval_service.py
│   ├── database/
│   │   └── sqlite_db.py     SQLite tables + queries
│   ├── vectorstore/
│   │   └── faiss_store.py   FAISS wrapper (persistent)
│   └── utils/
│       ├── config.py
│       └── logger.py
├── frontend/
│   └── streamlit_app.py     Streamlit UI (talks to FastAPI)
├── database/                rag_metadata.db (auto-created)
├── vectorstore/             faiss_index/    (auto-created)
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Setup

```bash
# 1. Clone & enter
cd rag-docqa-pro

# 2. Create venv
python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3. Install
pip install -r requirements.txt

# 4. Configure env
cp .env.example .env
# Edit .env and set GROQ_API_KEY (get one free at https://console.groq.com)

# 5. Run backend (terminal 1)
uvicorn backend.api.main:app --reload --port 8000
#   Swagger UI: http://localhost:8000/docs

# 6. Run frontend (terminal 2)
streamlit run frontend/streamlit_app.py
#   Open: http://localhost:8501
```

---

## 🧠 Workflow

1. **Upload PDF** in the sidebar → backend extracts pages with `pdfplumber`,
   splits into 500-char overlapping chunks, embeds with MiniLM, and stores:
   - the **vectors** in FAISS (keyed by `vector_id`)
   - the **chunk text + filename + page** in SQLite.
2. **Ask a question** → query is embedded → top-K similar `vector_id`s
   returned by FAISS → SQLite resolves them to the original text + source.
3. The retrieved context is injected into a system prompt and sent to
   **Groq LLaMA-3** via LangChain.
4. The answer + structured sources (filename, page, similarity, preview)
   are returned and rendered, and the exchange is persisted to
   `chat_history`.

---

## 🔌 REST API

| Method | Path                         | Purpose                          |
|--------|------------------------------|----------------------------------|
| GET    | `/health`                    | Backend + index status           |
| POST   | `/upload`                    | Upload & ingest a PDF            |
| POST   | `/query`                     | RAG question → answer + sources  |
| GET    | `/documents`                 | List indexed documents           |
| DELETE | `/documents/{id}`            | Delete a document (+ vectors)    |
| GET    | `/chat-history`              | Audit log (filter by session_id) |
| DELETE | `/reset`                     | Wipe knowledge base + history    |

Full interactive docs at `http://localhost:8000/docs`.

---

## 🖼 Screenshots

> _Add screenshots of the Streamlit UI here once running locally._
>
> - `docs/screenshot-chat.png`
> - `docs/screenshot-sidebar.png`

---

## 🛡 Security & Robustness

- Only `.pdf` uploads accepted (MIME + extension check).
- Temp files cleaned in a `finally` block.
- All SQLite writes guarded by a process-level lock.
- FAISS uses persistent `IndexIDMap` so vector IDs stay consistent across runs.
- Secrets read from environment variables only — never hard-coded.

---

## 🔭 Future Improvements

- Per-user authentication and session isolation.
- Hybrid retrieval (BM25 + dense) and re-ranking with a cross-encoder.
- Streaming token-by-token responses from Groq.
- Background ingestion queue for large PDFs.
- Postgres + pgvector for multi-tenant deployments.
- OCR fallback for scanned PDFs (Tesseract).
- Dockerfile + docker-compose for one-command deploys.
