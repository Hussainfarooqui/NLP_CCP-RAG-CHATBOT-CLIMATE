# src/embed_and_index.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

MODEL_NAME = "all-MiniLM-L6-v2"

def build_embeddings(chunks, model_name=MODEL_NAME, index_path="../data/faiss_index"):
    model = SentenceTransformer(model_name)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    dim = embeddings.shape[1]
    os.makedirs(index_path, exist_ok=True)

    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    np.save(os.path.join(index_path, "embeddings.npy"), embeddings)
    with open(os.path.join(index_path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved FAISS index to {index_path}")

if __name__ == "__main__":
    chunks = json.load(open("data/chunks.json", "r", encoding="utf-8"))
    build_embeddings(chunks, index_path="data/faiss_index")

