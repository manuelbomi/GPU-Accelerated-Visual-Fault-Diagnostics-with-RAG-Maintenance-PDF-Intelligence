# API

A thin FastAPI layer over the existing `src/query_similar_images.py` and
`src/rag_query.py` pipelines — the "REST API / FastAPI microservice" item
from the main README's roadmap. It doesn't reimplement any vision/RAG
logic; it loads the same models those CLI scripts use once at startup
(`api/models.py`) and calls the same functions per request.

## Prerequisites

Build the embeddings, indices, and classifier first (from the repo root —
see the main [README's Quickstart](../README.md#quickstart)):

```bash
python -m src.extract_embeddings
python -m src.build_faiss_index
python -m src.ingest_pdfs
python -m src.train_classifier
```

## Running

From the repo root (so `data/`, `src/`, and `api/` all resolve):

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | — | Whether the image index, text index, and classifier are loaded |
| POST | `/api/search?top_k=5` | multipart `file` (jpg/png) | Top-`k` visually similar images from `data/raw/`, with similarity scores |
| POST | `/api/diagnose?top_k_text=3` | multipart `file` (jpg/png) | Classifier prediction, similar historical images, retrieved maintenance-PDF passages, related doc names, and (if `OPENAI_API_KEY` is set) an LLM-synthesized remediation summary |

Images referenced in responses are served at `/images/<filename>` (mounted
from `data/raw/`).

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Comma-separated origins allowed to call this API (the frontend's dev server by default) |
| `OPENAI_API_KEY` | unset | Enables LLM synthesis in `/api/diagnose`. **Never commit this** — set it as an environment variable, not in code. |

## A note on ports

Port 8000 is a common default and can already be in use by other local
services (Docker Desktop, other dev servers, etc.). If `/api/health`
requests from the frontend fail with a CORS or connection error but `curl
http://127.0.0.1:8000/api/health` works, something else is likely also
listening on that port — pick a different one (`--port 8010`, etc.) and
update `frontend/.env.local`'s `VITE_API_BASE_URL` to match.
