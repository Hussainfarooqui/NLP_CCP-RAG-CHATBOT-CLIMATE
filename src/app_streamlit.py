# src/app_streamlit.py
from dotenv import load_dotenv
import os
import streamlit as st
from groq import Groq
from retriever import Retriever
from build_index import build_index  # Import your index builder

# 🔹 Load .env file
load_dotenv()

# 🔹 Get Groq API key from environment or Streamlit secrets
GROQ_API_KEY = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

# 🔹 Streamlit App Title
st.set_page_config(page_title="Climate Chatbot (RAG + Groq)", layout="wide")
st.title("🌍 Climate Chatbot (PDF + Groq LLM)")

# 🔹 Ensure FAISS index exists
INDEX_DIR = "data"
INDEX_FILE = os.path.join(INDEX_DIR, "index.faiss")
CHUNKS_FILE = os.path.join(INDEX_DIR, "chunks.json")

if not os.path.exists(INDEX_FILE) or not os.path.exists(CHUNKS_FILE):
    st.info("Building FAISS index from PDF… This may take a moment.")
    build_index()  # Generates index.faiss + chunks.json
    st.success("Index built successfully!")

# 🔹 Initialize Retriever
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

# 🔹 Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# 🔹 Check API key
if not GROQ_API_KEY:
    st.error("❌ Groq API key not found! Please set it in your .env file or in Streamlit secrets.")
    st.stop()

# 🔹 Function to generate answers using Groq LLM
def generate_answer(query, history):
    # Step 1: Retrieve context from PDF
    results = retriever.query(query, top_k=3)
    context_text = "\n".join([r["chunk"] for r in results])

    # Step 2: Build conversation history
    conversation = ""
    for h in history:
        conversation += f"User: {h['user']}\nAssistant: {h['assistant']}\n"

    # Step 3: Construct prompt
    prompt = f"""
You are a climate research assistant. 
Use the following context from PDF and previous conversation to answer.

Context from PDF:
{context_text}

Conversation so far:
{conversation}

User: {query}
Assistant:
"""

    # Step 4: Call Groq LLM
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    # ✅ Access content correctly
    return response.choices[0].message.content

# 🔹 Initialize Chat History in Session
if "history" not in st.session_state:
    st.session_state.history = []

# 🔹 Chat UI
st.subheader("💬 Ask me anything about Climate (from PDF + Groq)")
user_query = st.text_input("Enter your question:")

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("Ask") and user_query.strip():
        with st.spinner("Thinking..."):
            answer = generate_answer(user_query, st.session_state.history)
            # Save to history
            st.session_state.history.append({"user": user_query, "assistant": answer})

with col2:
    if st.button("🔄 Reset Conversation"):
        st.session_state.history = []
        st.success("Conversation history cleared!")

# 🔹 Display Chat History
if st.session_state.history:
    st.markdown("## 📝 Conversation History")
    for chat in st.session_state.history:
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        st.markdown("---")
