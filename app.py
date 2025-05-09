
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer
import os

# Fix Asyncio Error (for Streamlit compatibility)
import asyncio

try:
    loop = asyncio.get_running_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Set up directory for uploads
UPLOAD_DIR = 'uploads/'
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load Local LLM (GPT-J-6B - Fully Offline)
model_path = './models/GPT-J-6B'
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)

st.set_page_config(page_title='⚖️ Legal RAG Chatbot (Offline)', page_icon='⚖️')
st.title('⚖️ Legal RAG-Based Chatbot (Fully Offline)')
st.write('Upload legal documents and ask any legal question.')

# Upload and Index Legal Documents
docs = st.file_uploader('Upload Legal Documents (PDF, TXT)', accept_multiple_files=True)
legal_texts = []

if docs:
    for doc in docs:
        text = doc.read().decode('utf-8')
        legal_texts.append(text)

    # Build Vector Store with RAG (FAISS)
    vectorizer = TfidfVectorizer().fit_transform(legal_texts)
    faiss_index = faiss.IndexFlatL2(vectorizer.shape[1])
    faiss_index.add(vectorizer.toarray())

def ask_legal_question(question):
    if not docs:
        return "Please upload legal documents to answer questions."

    question_embedding = vectorizer.transform([question]).toarray()
    _, closest_docs = faiss_index.search(question_embedding, 1)
    context = legal_texts[closest_docs[0][0]]
    input_text = f"Context: {context}\\nQuestion: {question}\\nAnswer:"

    inputs = tokenizer(input_text, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_length=200)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    return answer

# User Input
user_question = st.text_input('Enter your legal question here:')
if user_question:
    answer = ask_legal_question(user_question)
    st.write(f'**Answer:** {answer}')
