# Enterprise Image Fault Diagnosis & Visual Similarity Search

## Overview
This repository demonstrates an enterprise-ready pipeline to:
- extract image embeddings using a pre-trained VGG16 model,
- index embeddings with FAISS for fast similarity search,
- train a simple classifier on the extracted embeddings to predict fault types,
- provide a Streamlit UI to query similar images and view results.

The intended workflow:
1. Place customer-returned fault images in `data/raw/` and a `labels.csv` mapping (optional).
2. Run `src/extract_embeddings.py` to extract and save embeddings.
3. Run `src/build_faiss_index.py` to create and persist a FAISS index.
4. Run `src/train_classifier.py` to train a classifier on the embeddings + labels.
5. Use `src/query_similar_images.py` or `app/streamlit_visual_search.py` to query the index.

## Repo structure
```
enterprise-image-fault-diagnosis/
│
├─ data/
│   ├─ raw/                     # customer-returned images (fault samples)
│   ├─ processed/               # preprocessed + augmented images (optional)
│   └─ faiss_index/             # saved FAISS index + label mappings
│
├─ notebooks/
│   ├─ 01_explore_dataset.ipynb
│   ├─ 02_train_vgg16_classifier.ipynb
│   └─ 03_build_faiss_index.ipynb
│
├─ src/
│   ├─ config.py
│   ├─ extract_embeddings.py
│   ├─ build_faiss_index.py
│   ├─ query_similar_images.py
│   └─ train_classifier.py
│
├─ app/
│   └─ streamlit_visual_search.py
│
├─ requirements.txt
├─ README.md
└─ LICENSE
```

## Notes
- This project uses `VGG16` from Keras as a feature extractor. You can swap it with modern backbones (EfficientNet, ConvNeXt, ViT).
- `faiss` is used for high-performance nearest-neighbor search. Use `faiss-gpu` for GPU acceleration.
- Embeddings and indices are saved under `data/faiss_index/`.
- The code includes argument parsing so you can adapt to your environment.

See the `src/` scripts for usage examples.


## 🔗 RAG (Retrieval-Augmented Generation) & PDF Knowledgebase

This project now supports:
- Ingesting PDF fault reports into a text FAISS index (script: `src/ingest_pdfs.py`)
- Running a RAG-style retrieval (`src/rag_query.py`) which:
  1. Retrieves top similar images from the image FAISS index.
  2. Maps similar images to labels and related PDFs.
  3. Searches the text FAISS index for relevant passages using SentenceTransformers.
  4. Optionally synthesizes a recommendation using an LLM if `OPENAI_API_KEY` is set.

### Files added
- `data/docs/*.pdf` — example remediation PDFs for fault types.
- `src/ingest_pdfs.py` — extracts text, chunks, embeds, and builds text FAISS index.
- `src/rag_query.py` — orchestration: image -> label -> text retrieval -> optional LLM.

### How to run end-to-end (GPU recommended)
1. Build image embeddings & image FAISS index:
   ```bash
   python src/extract_embeddings.py
   python src/build_faiss_index.py
   ```
2. Ingest PDF docs into the text FAISS:
   ```bash
   python src/ingest_pdfs.py
   ```
3. Run RAG query:
   ```bash
   python src/rag_query.py --query data/raw/crack_0.jpg
   ```
4. (Optional) Set `OPENAI_API_KEY` env var to enable LLM synthesis for summarized remediation steps.

### Architecture Diagram
See `docs_architecture.png` for a visual overview of the system components and flow.
