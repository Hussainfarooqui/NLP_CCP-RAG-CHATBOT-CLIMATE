# src/build_index.py
import os
import json
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss

# 🔹 Paths
DATA_DIR = "data"
PDF_FILE = os.path.join(DATA_DIR, "NLP.pdf")  # your PDF
INDEX_FILE = os.path.join(DATA_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(DATA_DIR, "chunks.json")

# 🔹 Model
MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dimension

# 🔹 Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

def extract_text_chunks(pdf_path, chunk_size=500):
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                # Split text into chunks
                for i in range(0, len(text), chunk_size):
                    chunk = text[i:i+chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
    return chunks

def build_index():
    print("Extracting text from PDF...")
    chunks = extract_text_chunks(PDF_FILE)
    print(f"✅ Extracted {len(chunks)} chunks from PDF.")

    print("Generating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
    index.add(embeddings)

    # Save index
    faiss.write_index(index, INDEX_FILE)
    print(f"✅ FAISS index saved to {INDEX_FILE}")

    # Save chunks metadata
    with open(CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"✅ Chunks saved to {CHUNKS_FILE}")

# 🔹 If run directly, build index
if __name__ == "__main__":
    build_index()
