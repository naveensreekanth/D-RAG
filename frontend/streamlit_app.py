"""
Streamlit frontend for the Document Intelligence System.
Talks to the FastAPI backend over HTTP.

Run (with backend already running on :8000):
    streamlit run frontend/streamlit_app.py
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Document Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1.2rem; }
.app-title  { font-size: 1.7rem; font-weight: 700; margin-bottom: 0; }
.app-sub    { color: #6b7280; font-size: 0.92rem; margin-bottom: 1.2rem; }
.src-card   {
  background: #f7f6ff; border-left: 3px solid #6366f1;
  padding: .7rem .9rem; border-radius: 0 6px 6px 0;
  margin-bottom: .55rem; font-size: .85rem; line-height: 1.55;
}
.src-meta   { font-size: .72rem; color: #6366f1; text-transform: uppercase;
              letter-spacing: .04em; font-weight: 600; margin-bottom: .25rem; }
.doc-row    { background: #f8fafc; padding: .55rem .7rem; border-radius: 6px;
              font-size: .82rem; margin-bottom: .35rem; }
.empty-hint { color: #94a3b8; font-size: .85rem; font-style: italic; }
#MainMenu { visibility: hidden; } footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state():
    st.session_state.setdefault("session_id", str(uuid.uuid4())[:8])
    st.session_state.setdefault("chat", [])  # list of {role, content, sources?}
    st.session_state.setdefault("top_k", 4)

_init_state()


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------
def api_health():
    try:
        return requests.get(f"{BACKEND}/health", timeout=4).json()
    except Exception:
        return None

def api_documents():
    try:
        return requests.get(f"{BACKEND}/documents", timeout=5).json()
    except Exception:
        return []

def api_upload(file) -> dict:
    return requests.post(
        f"{BACKEND}/upload",
        files={"file": (file.name, file.getvalue(), "application/pdf")},
        timeout=120,
    ).json()

def api_query(question: str) -> dict:
    return requests.post(
        f"{BACKEND}/query",
        json={
            "question": question,
            "session_id": st.session_state.session_id,
            "top_k": st.session_state.top_k,
        },
        timeout=120,
    ).json()

def api_delete_document(doc_id: int):
    return requests.delete(f"{BACKEND}/documents/{doc_id}", timeout=10).json()

def api_reset():
    return requests.delete(f"{BACKEND}/reset", timeout=10).json()


# ---------------------------------------------------------------------------
# Sidebar — knowledge base management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📚 Knowledge Base")

    health = api_health()
    if health:
        st.success(f"Backend ✓ — {health['documents']} docs · {health['chunks_indexed']} chunks")
    else:
        st.error(f"Backend unreachable at {BACKEND}")

    st.markdown("### Upload PDFs")
    uploads = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                               label_visibility="collapsed")
    if st.button("⚡ Ingest selected", use_container_width=True, disabled=not uploads):
        for f in uploads:
            with st.spinner(f"Indexing {f.name}…"):
                try:
                    res = api_upload(f)
                    if "message" in res:
                        st.success(f"{f.name}: {res['message']}")
                    else:
                        st.error(f"{f.name}: {res}")
                except Exception as e:
                    st.error(f"{f.name}: {e}")

    st.markdown("### Documents")
    docs = api_documents()
    if not docs:
        st.markdown('<div class="empty-hint">No documents yet.</div>', unsafe_allow_html=True)
    for d in docs:
        col1, col2 = st.columns([5, 1])
        col1.markdown(
            f'<div class="doc-row"><b>{d["filename"]}</b><br>'
            f'<span style="color:#64748b;font-size:.75rem;">'
            f'{d["page_count"]} pages · {d["chunk_count"]} chunks · '
            f'{d["upload_time"][:19].replace("T"," ")}</span></div>',
            unsafe_allow_html=True,
        )
        if col2.button("🗑", key=f"del_{d['document_id']}", help="Delete"):
            api_delete_document(d["document_id"])
            st.rerun()

    st.markdown("---")
    st.session_state.top_k = st.slider("Top-K retrieval", 1, 10, st.session_state.top_k)

    if st.button("🧨 Reset knowledge base", use_container_width=True, type="secondary"):
        api_reset()
        st.session_state.chat = []
        st.rerun()

    st.caption(f"Session: `{st.session_state.session_id}`")


# ---------------------------------------------------------------------------
# Main — chat
# ---------------------------------------------------------------------------
st.markdown('<div class="app-title">📄 Document Intelligence</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Ask grounded questions across your indexed PDFs. '
    'Every answer is backed by retrieved source chunks.</div>',
    unsafe_allow_html=True,
)


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"📎 {len(sources)} source chunks"):
        for s in sources:
            st.markdown(
                f'<div class="src-card">'
                f'<div class="src-meta">{s["filename"]} · page {s["page"]} · '
                f'sim {s["similarity"]:.3f}</div>'
                f'{s["chunk_preview"]}'
                f'</div>',
                unsafe_allow_html=True,
            )


# Replay history
for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            render_sources(msg["sources"])


question = st.chat_input("Ask a question about your documents…")
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating…"):
            try:
                res = api_query(question)
                if "answer" in res:
                    st.markdown(res["answer"])
                    render_sources(res.get("sources", []))
                    st.session_state.chat.append({
                        "role": "assistant",
                        "content": res["answer"],
                        "sources": res.get("sources", []),
                    })
                else:
                    err = res.get("detail", str(res))
                    st.error(err)
                    st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {err}"})
            except Exception as e:
                st.error(str(e))
