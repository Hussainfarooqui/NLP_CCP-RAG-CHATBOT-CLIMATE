# src/build_index.py
import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# 🔹 Config
PDF_PATH = "data/NLP.pdf"       # Path to your PDF
INDEX_DIR = "data"              # Folder to save index & chunks
MODEL_NAME = "all-MiniLM-L6-v2" # Sentence Transformer model
CHUNK_SIZE = 400                # Approx words per chunk

# 🔹 Step 1: Load PDF and split into chunks
chunks = []
with pdfplumber.open(PDF_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if not text:
            continue
        words = text.split()
        for i in range(0, len(words), CHUNK_SIZE):
            chunk = " ".join(words[i:i+CHUNK_SIZE])
            chunks.append(chunk)

print(f"✅ Extracted {len(chunks)} chunks from PDF.")

# 🔹 Step 2: Generate embeddings
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(chunks, convert_to_numpy=True)
faiss.normalize_L2(embeddings)
print("✅ Generated embeddings.")

# 🔹 Step 3: Build FAISS index
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # Inner product similarity
index.add(embeddings)
print(f"✅ Built FAISS index with {index.ntotal} vectors.")

# 🔹 Step 4: Save index and chunks
os.makedirs(INDEX_DIR, exist_ok=True)
faiss.write_index(index, os.path.join(INDEX_DIR, "index.faiss"))

with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print(f"✅ Saved index and chunks to '{INDEX_DIR}'.")
