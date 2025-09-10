from dotenv import load_dotenv
import os
# load .env file
load_dotenv()

# get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

import os
import json
import streamlit as st
from groq import Groq
from retriever import Retriever

# 🔹 Streamlit App Title
st.set_page_config(page_title="Climate Chatbot (RAG + Groq)", layout="wide")
st.title("🌍 Climate Chatbot (PDF + Groq LLM)")

# 🔹 Load Groq API Key
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

# 🔹 Initialize Retriever (PDF knowledge base)
@st.cache_resource
def load_retriever():
    return Retriever()

retriever = load_retriever()

# 🔹 Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

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

    return response.choices[0].message["content"]

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
    for i, chat in enumerate(st.session_state.history):
        st.markdown(f"**You:** {chat['user']}")
        st.markdown(f"**Assistant:** {chat['assistant']}")
        st.markdown("---")
