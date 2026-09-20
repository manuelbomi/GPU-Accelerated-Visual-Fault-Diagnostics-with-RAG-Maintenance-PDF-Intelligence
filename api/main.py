"""FastAPI backend for the React/TypeScript frontend in frontend/.

Wraps the existing src/query_similar_images.py and src/rag_query.py
pipelines as HTTP endpoints -- the "REST API / FastAPI microservice" item
from the README's roadmap -- instead of reimplementing that logic.

Run from the repo root (so `data/`, `src/`, and `api/` all resolve):
    uvicorn api.main:app --reload --port 8000
"""
import os
import shutil
import tempfile
from pathlib import Path

import faiss
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from api.models import store
from api.schemas import (
    DiagnoseResponse,
    HealthResponse,
    RelatedImage,
    RetrievedPassage,
    SearchResponse,
    SimilarImageResult,
)
from src.config import RAW_DIR
from src.query_similar_images import extract_feature
from src.rag_query import rag_query as run_rag_query

ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="Visual Fault Diagnostics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get(
        "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

if RAW_DIR.exists():
    app.mount("/images", StaticFiles(directory=str(RAW_DIR)), name="images")


@app.on_event("startup")
def _load_models():
    store.load_all()


def _save_upload(upload: UploadFile, tmp_dir: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type '{suffix}'. Use one of {sorted(ALLOWED_SUFFIXES)}.")
    dest = Path(tmp_dir) / f"query{suffix}"
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest


def _image_url(name: str) -> str:
    return f"/images/{name}"


@app.get("/api/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        image_search_ready=store.image_search_ready,
        rag_ready=store.rag_ready,
        classifier_ready=store.classifier_ready,
    )


@app.post("/api/search", response_model=SearchResponse)
def search_similar_images(file: UploadFile = File(...), top_k: int = 5):
    if not store.image_search_ready:
        raise HTTPException(
            503,
            "Image index not built yet. Run: python -m src.extract_embeddings "
            "&& python -m src.build_faiss_index",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        query_path = _save_upload(file, tmp_dir)
        qvec = extract_feature(store.image_model, str(query_path))
        faiss.normalize_L2(qvec)
        dists, inds = store.image_index.search(qvec, top_k + 1)  # +1 in case the query itself is in the index

        results = []
        seen = set()
        for dist, idx in zip(dists[0], inds[0]):
            if idx < 0 or idx >= len(store.image_names):
                continue
            name = str(store.image_names[idx])
            if name in seen:
                continue
            seen.add(name)
            results.append(SimilarImageResult(name=name, score=float(dist), url=_image_url(name)))
            if len(results) >= top_k:
                break

    return SearchResponse(query_filename=file.filename or "query", results=results)


@app.post("/api/diagnose", response_model=DiagnoseResponse)
def diagnose(file: UploadFile = File(...), top_k_text: int = 3):
    if not store.rag_ready:
        raise HTTPException(
            503,
            "RAG index not built yet. Run: python -m src.extract_embeddings "
            "&& python -m src.build_faiss_index && python -m src.ingest_pdfs",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        query_path = _save_upload(file, tmp_dir)

        result = run_rag_query(
            str(query_path),
            top_k_text=top_k_text,
            img_model=store.image_model,
            img_index=store.image_index,
            image_names=store.image_names,
            text_index=store.text_index,
            passages=store.text_passages,
            metadata=store.text_metadata,
            text_model=store.text_model,
        )

        predicted_label = None
        predicted_confidence = None
        if store.classifier_ready:
            qvec = extract_feature(store.image_model, str(query_path))
            proba = store.classifier.predict_proba(qvec)[0]
            best = proba.argmax()
            predicted_label = str(store.classifier.classes_[best])
            predicted_confidence = float(proba[best])

    top_images = [
        RelatedImage(name=name, url=_image_url(name))
        for name in result["top_images"]
    ]
    passages = [
        RetrievedPassage(text=text, score=score)
        for text, score in result["retrieved_passages"]
    ]

    return DiagnoseResponse(
        query_filename=file.filename or "query",
        predicted_label=predicted_label,
        predicted_confidence=predicted_confidence,
        top_images=top_images,
        related_docs=result.get("docs", []),
        retrieved_passages=passages,
        synthesis=result.get("synthesis"),
    )
