import os
import streamlit as st
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pdfplumber
import google.generativeai as genai

# --- Configuration ---
load_dotenv()
API_KEYS = [os.getenv("GEMINI_API_KEY_1")]
PDF_PATH = "student-handbook.pdf"
INDEX_FILE = "handbook.index"
CHUNKS_FILE = "handbook_chunks.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
st.set_page_config(page_title="Lasallian Handbook Chatbot", layout="wide")
st.title("📘 Lasallian Handbook Assistant")

# --- Step 1: Extract Chunks from PDF ---
def extract_chunks_from_pdf(pdf_path, max_words=500):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() or "" for page in pdf.pages])
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# --- Step 2: Build or Load Index ---
def build_or_load_index():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        with st.spinner("🔁 Loading cached index..."):
            with open(CHUNKS_FILE, "rb") as f:
                chunks = pickle.load(f)
            index = faiss.read_index(INDEX_FILE)
            return index, chunks
    else:
        with st.spinner("🔍 Processing handbook for the first time..."):
            chunks = extract_chunks_from_pdf(PDF_PATH)
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings))
            faiss.write_index(index, INDEX_FILE)
            with open(CHUNKS_FILE, "wb") as f:
                pickle.dump(chunks, f)
            return index, chunks

# --- Step 3: Retrieve Similar Chunks ---
def retrieve_similar_chunks(question, index, chunks, top_k=3):
    question_vec = embedder.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    return [chunks[i] for i in I[0]]

# --- Step 4: Generate Gemini Response ---
def ask_gemini(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a Lasallian student disciplinary officer. Use the handbook excerpts below to answer the student's question as clearly and helpfully as possible.

--- Handbook Context ---
{context}

--- Student Question ---
{question}
"""
    genai.configure(api_key=API_KEYS[0])
    model = genai.GenerativeModel("gemini-1.5-flash")

    try:
        chat = model.start_chat()
        response = chat.send_message(prompt)
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Error: {e}"

# --- Load Handbook Index ---
if not os.path.exists(PDF_PATH):
    st.warning("⚠️ Please add 'student_handbook.pdf' in the app directory.")
    st.stop()

index, chunks = build_or_load_index()

# --- Chat State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Chat Input ---
user_input = st.chat_input("Ask something about the student handbook...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.spinner("✍️ Thinking..."):
        top_chunks = retrieve_similar_chunks(user_input, index, chunks)
        answer = ask_gemini(user_input, top_chunks)

    st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Display Chat ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
