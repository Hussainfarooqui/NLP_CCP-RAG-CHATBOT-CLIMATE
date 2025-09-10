# src/retriever.py
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

INDEX_DIR = "data"
MODEL_NAME = "all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, index_dir=INDEX_DIR, model_name=MODEL_NAME):
        self.index_dir = index_dir
        self.model = SentenceTransformer(model_name)
        
        # FAISS index load
        self.index = faiss.read_index(os.path.join(index_dir, "index.faiss"))
        
        # chunks.json load instead of chunks.pkl
        with open(os.path.join(index_dir, "chunks.json"), "r", encoding="utf-8") as f:
            self.meta = json.load(f)

    def query(self, q, top_k=5):
        emb = self.model.encode([q], convert_to_numpy=True)
        faiss.normalize_L2(emb)
        D, I = self.index.search(emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0:
                continue
            item = self.meta[int(idx)]
            results.append({"score": float(score), "chunk": item})
        return results

if __name__ == "__main__":
    r = Retriever()
    res = r.query("What are the primary drivers of sea level rise?", top_k=3)
    print(res)
