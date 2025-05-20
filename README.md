# Explainable RAG System

[Streamlit App](https://tamannakumavat-explainable-rag-app-3kkuqs.streamlit.app/)

A simple, explainable Retrieval-Augmented Generation (RAG) tool that helps you understand how document chunks influence the generated response using t-SNE visualizations and adjustable chunking strategies.

---

## Features

### t-SNE Visualization
- Visualize embeddings of:
  - All chunks
  - Retrieved chunks
  - Query and response
- Understand the relationship between input and generated output.

### Interactive Chunking
- Choose between:
  - Fixed-size (by word count)
  - Sentence-based chunking
- Adjust chunking parameters in the UI.

### Retrieval & Generation
- Retrieve relevant chunks using vector embeddings.
- Generate context-aware responses using OpenAI’s GPT models.

### Web Interface
- Upload PDF → Ask a question → Visualize how the system responds.

---

## How to Use

1. Upload a PDF document.
2. Select your chunking method and adjust settings.
3. Enter a query.
4. Review retrieved chunks and generated response.
5. Use the t-SNE plot to interpret embedding relationships.

---

## Installation

### Requirements
- Python 3.8+
- OpenAI API Key

### Setup

```bash
git clone https://github.com/your-username/Interpretable-RAG.git
cd Interpretable-RAG
pip install -r requirements.txt
