# Enterprise GPU-Accelerated Visual Fault Diagnostics with RAG & Maintenance PDF Intelligence <sub>(Vision AI + FAISS-GPU + LLM PDF Knowledge Retrieval for Industrial Defect Root-Cause Analysis)</sub>

## Overview


#### This system implements an **end-to-end intelligent fault analysis pipeline** for enterprise and industrial manufacturing environments:

| Layer | Capability |
|---|---|
 Vision | VGG16 / TF-GPU image fault classifier  
 Search | FAISS-GPU similarity search for historical defective parts  
 Knowledge | PDF maintenance instructions + RAG for intelligent troubleshooting  
 AI Assistant | LLM answers using enterprise fault documentation  
 UI | Streamlit fault diagnosis assistant  
 Equipped for | NVIDIA GPUs, CUDA, Enterprise workloads  

Designed for digital factories, predictive maintenance teams, and AI-augmented manufacturing support systems.

---

##  Key Features

 **GPU-accelerated training & inference** (TensorFlow-GPU, FAISS-GPU)  
 **Image fault classification + similar-image retrieval**  
 **PDF fault manual ingestion** → **vector store**  
 **LLM-powered fault explanation & root-cause reasoning**  
 **Production-style modular codebase**  
 **Supports enterprise AI deployment workflow**

---

##  High-Level Architecture

```python
flowchart LR
A[Upload Part Image] --> B[GPU Preprocessing]
B --> C[VGG16 CNN Embedding Extraction]
C --> D[FAISS-GPU Similarity Search]
D --> E[Similar Historical Fault Images + Metadata]

E --> F[PDF Fault Manuals Vector Store]
F --> G[LLM RAG Engine]
G --> H[Root Cause + Mitigation Plan + Steps]

4. (Optional) Set `OPENAI_API_KEY` env var to enable LLM synthesis for summarized remediation steps.
```

### Repository Structure

```python
enterprise-image-fault-diagnosis-rag-gpu/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── faiss_index/
│   └── fault_reports/
│       ├── pdfs/
│       └── vector_store/
│
├── src/
│   ├── config.py
│   ├── extract_embeddings.py
│   ├── train_classifier.py
│   ├── build_faiss_index.py
│   ├── query_similar_images.py
│   ├── pdf_ingest.py
│   ├── rag_query.py
│
├── app/
│   └── streamlit_visual_search.py

```

---

## Quickstart (GPU Required)
#### <ins>Install NVIDIA drivers + CUDA</ins>

```python
sudo apt install nvidia-driver-530 nvidia-cuda-toolkit
```

#### <ins>Install Python deps</ins>
```python
pip install -r requirements.txt
```

#### <ins>Train + Build Index + Build PDF Vector Store</ins>

```python
python src/train_classifier.py
python src/extract_embeddings.py
python src/build_faiss_index.py
python src/pdf_ingest.py
```

#### <insRun Streamlit App</ins>
```python
streamlit run app/streamlit_visual_search.py
```

---

##  Enterprise Use-Cases

| Industry | Use Case |
|----------|----------|
| Automotive | Visual quality inspection + service manual lookups |
| Aerospace | Turbine/airframe defect search + maintenance docs QA |
| Semiconductor | Wafer anomaly classification + tool maintenance PDFs |
| Oil & Gas | Pipeline corrosion recognition + repair SOP linking |
| OEM & Factory QA | AI visual assistant for line-side technicians |

---

## Enterprise-Ready Capabilities

| Capability | Description |
|------------|-------------|
| RAG w/ PDF manuals | Knowledge-augmented troubleshooting |
| GPU inference | Real-time similarity search + CNN embeddings |
| Air-gapped support | No cloud dependency required |
| Scalable | Modular services for MLOps pipelines |
| Extensible | Swap VGG16 → ViT / YOLO / EfficientNet |

---

## Roadmap
| Feature | Status |
|---------|--------|
|  CNN Fault Classifier | Done |
|  FAISS-GPU Similarity Search | Done |
|  PDF Fault Ingest + Embeddings | Done |
|  RAG Query System | Done |
|  REST API / FastAPI microservice | Planned |
|  Helm + EKS GPU deployment | Planned |
|  YOLO defect localization | Planned |

---

## Summary

#### This repository demonstrates a real-world enterprise AI system combining:

- Computer vision defect detection

- GPU-accelerated similarity search

- PDF maintenance manual intelligence

- RAG + LLM technician assistant
- 

<ins>Perfect for</ins>:

- Manufacturing AI innovation teams

- Maintenance automation & smart factory projects

- Industrial ML upskilling & PoC deployments
