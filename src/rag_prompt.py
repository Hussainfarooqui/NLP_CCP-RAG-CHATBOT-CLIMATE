# src/rag_prompt.py
import os
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

def build_prompt(question, contexts):
    header = "You are an expert assistant on climate change research. Use the provided context to answer the user's question. Reply concisely and cite source chunk IDs.\n\n"
    ctx_text = ""
    for i, c in enumerate(contexts):
        ctx_text += f"[CONTEXT {i+1} | {c['chunk']['doc_id']} | score={c['score']:.3f}]\n{c['chunk']['text']}\n\n"
    prompt = f"{header}Context:\n{ctx_text}\nUser question: {question}\n\nAnswer:"
    return prompt

def generate_with_openai(prompt, max_tokens=300, temperature=0.2):
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set in env.")
    openai.api_key = OPENAI_API_KEY
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=1,
        n=1,
        stop=None
    )
    return resp.choices[0].text.strip()

def generate_with_local_model(prompt, model_name="gpt2", max_tokens=200):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    from retriever import Retriever
    r = Retriever()
    contexts = r.query("What causes sea level rise?", top_k=3)
    p = build_prompt("What causes sea level rise?", contexts)
    print("PROMPT:", p[:500])
