"""Ingest PDFs from data/docs/, extract text, chunk it, embed with sentence-transformers,
and build a FAISS index for document passages.

Usage:
    python src/ingest_pdfs.py --docs_dir data/docs --out_dir data/faiss_index
"""
import os
import argparse
from pathlib import Path
from pdfminer.high_level import extract_text
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import json
from tqdm import tqdm

def chunk_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

def ingest(docs_dir, out_dir, model_name='all-MiniLM-L6-v2'):
    docs_dir = Path(docs_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(model_name)

    passages = []
    metadata = []
    for pdf in docs_dir.glob('*.pdf'):
        text = extract_text(str(pdf))
        chunks = chunk_text(text)
        for idx, ch in enumerate(chunks):
            passages.append(ch)
            metadata.append({
                'source_pdf': pdf.name,
                'chunk_id': idx
            })

    if len(passages) == 0:
        print('No passages extracted.')
        return

    embeddings = model.encode(passages, convert_to_numpy=True, show_progress_bar=True).astype('float32')
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Save
    faiss.write_index(index, str(out_dir / 'faiss_text_index.faiss'))
    np.save(out_dir / 'text_passages.npy', np.array(passages, dtype=object))
    with open(out_dir / 'text_metadata.json', 'w') as f:
        json.dump(metadata, f)
    print('Saved text index, passages and metadata to', out_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--docs_dir', type=str, default='data/docs')
    parser.add_argument('--out_dir', type=str, default='data/faiss_index')
    args = parser.parse_args()
    ingest(args.docs_dir, args.out_dir)
