# Enterprise Image Fault Diagnosis & Visual Similarity Search <sub>(Vision AI + FAISS-GPU + LLM PDF Knowledge Retrieval for Industrial Defect Root-Cause Analysis)</sub>

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

## ⚙️ Key Features

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
