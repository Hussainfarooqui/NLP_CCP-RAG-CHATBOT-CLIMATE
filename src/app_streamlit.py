# src/app_streamlit.py
import streamlit as st
from retriever import Retriever
from rag_prompt import build_prompt, generate_with_openai
import os

st.set_page_config(page_title="RAG Chatbot - Climate ", page_icon="🌍")
st.title("RAG Chatbot — Climate Research By Muhammad Hussain Ahmed Farooqui")
st.write("Ask domain-specific climate questions. Retrieved sources are shown for transparency.")

@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

if "history" not in st.session_state:
    st.session_state.history = []

query = st.text_input("Enter your question:", key="query", placeholder="e.g., What drives polar ice melt?")

top_k = st.sidebar.slider("Top K retrieved chunks", 1, 8, 4)

use_openai = st.sidebar.checkbox("Use OpenAI API (recommended)", value=False)
if use_openai and not os.getenv("OPENAI_API_KEY"):
    st.sidebar.error("Set OPENAI_API_KEY in environment to use OpenAI")

if st.button("Ask") and query.strip():
    with st.spinner("Retrieving relevant context..."):
        contexts = retriever.query(query, top_k=top_k)
    prompt = build_prompt(query, contexts)
    with st.spinner("Generating answer..."):
        try:
            if use_openai:
                answer = generate_with_openai(prompt)
            else:
                answer = "OpenAI API not enabled. Showing retrieved context excerpts:\n\n"
                for c in contexts:
                    answer += f"- ({c['chunk']['doc_id']}) {c['chunk']['text'][:350]}...\n\n"
        except Exception as e:
            answer = f"Generation failed: {e}"

    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"q": query, "a": answer, "contexts": contexts})

    # no rerun — just display below



for i, item in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q:** {item['q']}")
    st.markdown(f"**A:** {item['a']}")
    st.markdown("**Retrieved contexts:**")
    for c in item['contexts']:
        st.markdown(f"- Source: `{c['chunk']['doc_id']}` | score: {c['score']:.3f}")
        st.write(c['chunk']['text'][:400] + ("..." if len(c['chunk']['text'])>400 else ""))
    st.markdown("---")

st.sidebar.markdown("**About**\nThis app uses FAISS + sentence-transformers for retrieval. For best results enable an LLM (OpenAI or your local model).")
