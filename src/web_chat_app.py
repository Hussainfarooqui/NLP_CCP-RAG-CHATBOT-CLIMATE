import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

# Load LLM (OpenAI or any other)
llm = ChatOpenAI(model="gpt-3.5-turbo")

# Simple function: casual intent detect
def is_smalltalk(query):
    smalltalk = ["hi", "hello", "hey", "thanks", "how are you"]
    return any(word in query.lower() for word in smalltalk)

# Web scraping function
def scrape_web(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        # Extract only text
        paragraphs = [p.get_text() for p in soup.find_all("p")]
        return "\n".join(paragraphs[:5])  # first 5 paras
    except Exception as e:
        return f"Error scraping site: {e}"

# Streamlit UI
st.set_page_config(page_title="Chatbot with Web", page_icon="🤖")
st.title("🌐 Web-enabled Chatbot")

query = st.text_input("Ask me something...")

if query:
    if is_smalltalk(query):
        st.success("👋 Hello! How can I help you today?")
    elif query.startswith("http"):  # agar user link de
        text = scrape_web(query)
        prompt = ChatPromptTemplate.from_template("Summarize this webpage:\n\n{text}")
        response = llm.predict(prompt.format(text=text))
        st.write(response)
    else:
        # normal chat
        response = llm.predict(query)
        st.write(response)
