"""RAG-style retrieval:
1) Given a query image, retrieve similar images via FAISS image index.
2) From labels (or doc_map), find related documents and query the text FAISS index
3) Return top-k relevant passages, and optionally synthesize an answer via OpenAI (if API key provided as OPENAI_API_KEY)

Usage:
    python src/rag_query.py --query path/to/image.jpg --top_k 3
"""
import os
import argparse
import numpy as np
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from src.query_similar_images import load_model, extract_feature, load_faiss
import json
import pandas as pd

def load_text_index(index_path, passages_path, metadata_path):
    index = faiss.read_index(str(index_path))
    passages = np.load(passages_path, allow_pickle=True)
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return index, passages, metadata

def retrieve_documents_for_labels(labels, doc_map_csv='data/processed/doc_map.csv'):
    df = pd.read_csv(doc_map_csv)
    docs = df[df['label'].isin(labels)]['doc'].unique().tolist()
    return docs

def query_text_index(text_index, passages, model, query_text, top_k=3):
    q_emb = model.encode([query_text], convert_to_numpy=True).astype('float32')
    faiss.normalize_L2(q_emb)
    dists, inds = text_index.search(q_emb, top_k)
    results = []
    for dist, idx in zip(dists[0], inds[0]):
        results.append((passages[idx], float(dist)))
    return results

def rag_query(image_path, top_k_text=3, image_index_path='data/faiss_index/index.faiss', text_index_path='data/faiss_index/faiss_text_index.faiss'):
    # 1) image retrieval
    img_model = load_model()
    qvec = extract_feature(img_model, image_path)
    img_index = load_faiss(image_index_path)
    faiss.normalize_L2(qvec)
    dists, inds = img_index.search(qvec, 6)  # get top 6 images
    names = np.load('data/faiss_index/image_names.npy', allow_pickle=True)
    top_image_names = [names[i] for i in inds[0] if i>=0][:5]

    # 2) infer labels from nearby labeled set (requires labels.csv)
    labels = []
    labels_csv = Path('data/processed/labels.csv')
    if labels_csv.exists():
        df = pd.read_csv(labels_csv)
        name_to_label = dict(zip(df['filename'], df['label']))
        for nm in top_image_names:
            if nm in name_to_label:
                labels.append(name_to_label[nm])
    labels = list(dict.fromkeys(labels))  # unique preserving order

    # 3) map labels to docs
    docs = retrieve_documents_for_labels(labels)

    # 4) load text index and passages
    text_index_path = Path(text_index_path)
    if not text_index_path.exists():
        raise FileNotFoundError('Text FAISS index not found. Run src/ingest_pdfs.py first.')

    text_index, passages, metadata = load_text_index(text_index_path, 'data/faiss_index/text_passages.npy', 'data/faiss_index/text_metadata.json')
    text_model = SentenceTransformer('all-MiniLM-L6-v2')

    # 5) build a query_text from labels and top image names
    query_text = ' '.join(labels + top_image_names)
    results = query_text_index(text_index, passages, text_model, query_text, top_k=top_k_text)

    # 6) Optionally, synthesize with an LLM if OPENAI_API_KEY is set
    synthesis = None
    openai_key = os.environ.get('OPENAI_API_KEY')
    if openai_key:
        try:
            import openai
            openai.api_key = openai_key
            prompt = f"Given the following retrieved passages, summarize recommended remediation steps for the fault query: {query_text}\n\nPassages:\n"
            for p, s in results:
                prompt += f"- {p}\n"
            resp = openai.ChatCompletion.create(model='gpt-4o-mini', messages=[{'role':'user','content':prompt}], temperature=0.0, max_tokens=400)
            synthesis = resp['choices'][0]['message']['content']
        except Exception as e:
            synthesis = f'LLM synthesis failed or OpenAI not configured: {e}'

    return {
        'query_text': query_text,
        'top_images': top_image_names,
        'retrieved_passages': results,
        'synthesis': synthesis
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query', required=True)
    args = parser.parse_args()
    out = rag_query(args.query)
    import pprint
    pprint.pprint(out)
