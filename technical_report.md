# Technical Report
## Title: Design, Implementation and Deployment of a Retrieval-Augmented Generation (RAG) Chatbot for Climate Research

### 1. Introduction
Retrieval-Augmented Generation (RAG) combines retrieval from a structured knowledge base with the generative power of large language models. This project builds a RAG chatbot providing accurate, source-attributed answers to climate research questions.

### 2. Problem Definition & Domain
Domain: Climate Change Research. The chatbot must retrieve relevant research snippets from a local collection of PDFs and generate responses that cite the original sources.

### 3. Data Collection & Preprocessing
- Data types: PDFs, text files (scientific reports, IPCC summaries, AR6 chapters, review papers).
- Preprocessing pipeline:
  - PDF extraction with `pdfplumber`.
  - Text cleaning (newline normalization, removing bracket refs).
  - Tokenize using NLTK and chunk into overlapping segments (450 words chunk, 50 overlap) to preserve context across chunk boundaries.
  - Save chunks metadata to `chunks.json`.

### 4. Embeddings & Vector Store
- Used `sentence-transformers` (`all-MiniLM-L6-v2`) for embeddings (balance speed & accuracy).
- Stored embeddings in FAISS (`IndexFlatIP`) after L2-normalization for cosine-similarity search.
- Metadata saved alongside index to map results back to document and chunk IDs.

### 5. Retrieval Pipeline
- Query pipeline:
  - Encode query with same SentenceTransformer.
  - Search FAISS top-K (user-set, default 4).
  - Return chunks with scores and doc IDs.

### 6. Generation & Prompting
- Prompt template: context blocks followed by the user question, a short system header (expert on climate), and an instruction to cite chunk IDs.
- Two generation options:
  - OpenAI API (`text-davinci-003`) — easy, high-quality.
  - Local HF model (e.g., small GPT or T5) — offline but may require GPU.

### 7. UI & UX
- Streamlit front-end with:
  - Textbox for questions.
  - Sidebar controls for top_k and model selection.
  - Display of generated answer and retrieved contexts (with source doc name & excerpt).
  - Session-based history.

### 8. Deployment
- Steps for Streamlit Cloud:
  - Push to GitHub.
  - Configure secrets (OPENAI_API_KEY).
  - Streamlit Cloud will run `streamlit run src/app_streamlit.py`.

### 9. Evaluation
- Suggested metrics:
  - Retrieval: Precision@k
  - Generation: Human evaluation (factuality, completeness, citation correctness)
  - Latency: Average time for retrieval + generation

### 10. Challenges & Solutions
- PDF extraction quirks → use `pdfplumber`.
- Long documents → chunking with overlap.
- Hallucinations in LLMs → include retrieved context and cite; add post-hoc verification.

### 11. Future Work
- Implement cross-encoder reranker
- Add provenance scoring & extractive answer highlighting
- Integrate smaller fine-tuned LLM for domain grounding

### 12. Conclusion
This RAG system offers a practical way to combine retrieval accuracy with generative flexibility, making domain knowledge accessible and transparent.
