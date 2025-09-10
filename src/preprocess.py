# src/preprocess.py
import os
import re
import pdfplumber
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt', quiet=True)

CHUNK_SIZE = 450
OVERLAP = 50

def read_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def clean_text(text):
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\[[0-9]+\]', '', text)
    return text.strip()

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = word_tokenize(text)
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if end >= n:
            break
    return chunks

def load_and_process(data_dir):
    docs = []
    for fname in os.listdir(data_dir):
        path = os.path.join(data_dir, fname)
        if fname.lower().endswith(".pdf"):
            raw = read_pdf(path)
        elif fname.lower().endswith(".txt"):
            raw = read_txt(path)
        else:
            continue
        cleaned = clean_text(raw)
        chunks = chunk_text(cleaned)
        for i, c in enumerate(chunks):
            docs.append({
                "doc_id": fname,
                "chunk_id": f"{fname}-{i}",
                "text": c
            })
    return docs

if __name__ == "__main__":
    import json
    docs = load_and_process("data")
    print(f"Processed {len(docs)} chunks.")
    with open("data/chunks.json", "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

