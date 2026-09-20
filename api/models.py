"""Loads every model/index the API needs exactly once, at process startup,
and hands out the cached instances to request handlers.

rag_query() and query_similar_images() were written as CLI scripts that
reload VGG16, the FAISS indices, and the sentence-transformer on every call
-- fine for a one-shot script, but far too slow per HTTP request. This
module is what lets api/main.py pass pre-loaded models into those same
functions (see the optional arguments added to rag_query() in
src/rag_query.py) instead of eating that reload cost on every request.
"""
import json
from pathlib import Path

import faiss
import numpy as np
from joblib import load as joblib_load
from sentence_transformers import SentenceTransformer

from src.config import (
    CLASSIFIER_FILE,
    EMBEDDINGS_FILE,
    FAISS_DIR,
    FAISS_INDEX_FILE,
    NAMES_FILE,
    RAW_DIR,
)
from src.query_similar_images import load_model as load_image_model

TEXT_INDEX_FILE = FAISS_DIR / "faiss_text_index.faiss"
TEXT_PASSAGES_FILE = FAISS_DIR / "text_passages.npy"
TEXT_METADATA_FILE = FAISS_DIR / "text_metadata.json"


class ModelStore:
    """Populated once by `load_all()` at FastAPI startup; read-only after that."""

    def __init__(self):
        self.image_model = None
        self.image_index = None
        self.image_names = None
        self.text_index = None
        self.text_passages = None
        self.text_metadata = None
        self.text_model = None
        self.classifier = None

    @property
    def image_search_ready(self) -> bool:
        return self.image_index is not None

    @property
    def rag_ready(self) -> bool:
        return self.image_search_ready and self.text_index is not None

    @property
    def classifier_ready(self) -> bool:
        return self.classifier is not None

    def load_all(self):
        if EMBEDDINGS_FILE.exists() and NAMES_FILE.exists() and FAISS_INDEX_FILE.exists():
            self.image_model = load_image_model()
            self.image_index = faiss.read_index(str(FAISS_INDEX_FILE))
            self.image_names = np.load(NAMES_FILE, allow_pickle=True)

        if TEXT_INDEX_FILE.exists() and TEXT_PASSAGES_FILE.exists() and TEXT_METADATA_FILE.exists():
            self.text_index = faiss.read_index(str(TEXT_INDEX_FILE))
            self.text_passages = np.load(TEXT_PASSAGES_FILE, allow_pickle=True)
            with open(TEXT_METADATA_FILE) as f:
                self.text_metadata = json.load(f)
            self.text_model = SentenceTransformer("all-MiniLM-L6-v2")

        if Path(CLASSIFIER_FILE).exists():
            self.classifier = joblib_load(CLASSIFIER_FILE)

    def image_path(self, name: str) -> Path:
        return RAW_DIR / name


store = ModelStore()
