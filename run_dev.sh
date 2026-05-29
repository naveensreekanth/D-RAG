#!/usr/bin/env bash
# Convenience: start backend + frontend in two background processes.
set -e
uvicorn backend.api.main:app --reload --port 8000 &
BACK=$!
streamlit run frontend/streamlit_app.py
kill $BACK 2>/dev/null || true
