# RAG Chatbot — Climate Research

A Retrieval-Augmented Generation (RAG) chatbot that retrieves domain-specific passages from a local knowledge base (PDF/TXT) using FAISS and uses an LLM to generate answers with context.

## Features
- Ingest PDF/TXT files and chunk text
- Create embeddings using sentence-transformers
- Build a FAISS index (local)
- Streamlit web UI showing retrieved contexts and generated answers
- Optionally integrate with OpenAI or local HF model for generation

## Install
1. Clone repo
2. Create Python venv:
   ```
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```
   pip install -r requirements.txt
   ```

## Prepare data
Put your domain PDFs / .txt files into `/data` directory.

## Preprocess & build index
```
python src/preprocess.py
python src/embed_and_index.py
```
This will create `data/chunks.json` and `data/faiss_index/` (index + metadata).

## Run locally (Streamlit)
Set environment variable if using OpenAI:
```
export OPENAI_API_KEY="sk-..."
```
Start Streamlit:
```
streamlit run src/app_streamlit.py
```
Open `http://localhost:8501`.

## Deployment
Recommended: Streamlit Cloud
- Push repo to GitHub
- Create app on Streamlit Cloud and connect to repo
- Set `OPENAI_API_KEY` (if using OpenAI) in Secrets

Alternative: Deploy front-end on Vercel + backend on Render / a simple container — not covered here.
