import os
import streamlit as st
import numpy as np
import faiss
import pickle
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pdfplumber
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor
import asyncio

# --- Setup ---
load_dotenv()
API_KEYS = [os.getenv("GEMINI_API_KEY_1")]
PDF_PATH = "student-handbook.pdf"
INDEX_FILE = "handbook.index"
CHUNKS_FILE = "handbook_chunks.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedder = embedder.to("cpu")
st.set_page_config(
    page_title="ArcherAsks – Lasallian Handbook Q&A",
    page_icon="🏫",
    layout="wide"
)

st.title("📘 Which DLSU rule am I breaking?")

# --- Session State Init ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "out_of_quota" not in st.session_state:
    st.session_state.out_of_quota = False

# --- Chunk Extraction ---
def extract_chunks_from_pdf(pdf_path, max_words=500):
    with pdfplumber.open(pdf_path) as pdf:
        text = " ".join([page.extract_text() or "" for page in pdf.pages])
    words = text.split()
    return [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]

# --- FAISS Index Loader ---
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

async def async_load_index():
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        index, chunks = await loop.run_in_executor(executor, build_or_load_index)
    return index, chunks

def retrieve_similar_chunks(question, index, chunks, top_k=3):
    question_vec = embedder.encode([question])
    D, I = index.search(np.array(question_vec), top_k)
    return [chunks[i] for i in I[0]]

# --- Gemini API ---
def ask_gemini(question, context_chunks):
    context = "\n\n".join(context_chunks)
    prompt = f"""
You are a Lasallian student disciplinary officer.
Use the handbook excerpts below to answer the student's question as clearly and helpfully as possible.
Please use context clues from the handbook to support your answer.
Make the response ghetto or from da hood.

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
        if "quota" in str(e).lower() or "exceeded" in str(e).lower():
            st.session_state.out_of_quota = True
        return f"⚠️ Error: {e}"

# --- PDF Check ---
if not os.path.exists(PDF_PATH):
    st.warning("⚠️ Please add 'student_handbook.pdf' in the app directory.")
    st.stop()

# --- Load index ---
index, chunks = asyncio.run(async_load_index())

# --- Out of Prompts Screen ---
if st.session_state.out_of_quota:
    st.error("🚫 Yall I ran out of prompts :(, (this is cuz im broke (09270251730) :D).")
    st.markdown("Please try again later or contact **hans_lumagui@dlsu.edu.ph** for support.")
    st.stop()

# --- Chat UI ---
if len(st.session_state.messages) == 0:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📚 What are the rules about cheating?"):
            st.session_state.messages.append({"role": "user", "content": "What are the rules about cheating?"})
    with col2:
        if st.button("👕 Dress code violations?"):
            st.session_state.messages.append({"role": "user", "content": "What are the dress code violations?"})

# --- Handle Message (from buttons or chat) ---
user_input = st.chat_input("Ask something about the student handbook...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

if st.session_state.messages and (st.session_state.messages[-1]["role"] == "user"):
    with st.spinner("✍️ Thinking..."):
        q = st.session_state.messages[-1]["content"]
        top_chunks = retrieve_similar_chunks(q, index, chunks)
        answer = ask_gemini(q, top_chunks)
        st.session_state.messages.append({"role": "assistant", "content": answer})

# --- Render Messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Footer & Legal ---
st.markdown("---")
st.markdown("""
<p style='text-align: center; font-size: 0.9em;'>
    🔗 View the source on <a href='https://github.com/bewba/Lasalle-Handbook-Chatbot'>GitHub</a>.<br>
    © 2025 <b>bewba</b>. All rights reserved.
</p>
""", unsafe_allow_html=True)

with st.expander("📜 Terms and Conditions"):
    st.markdown("""
    # Terms and Conditions

    Last updated: May 9, 2025

    Welcome to the Lasallian Handbook Chatbot (“the Service”), developed and maintained by bewba.

    By accessing or using this Service, you agree to the following terms:

    ## 1. Use of the Service
    - This Service is intended to provide helpful responses based on the Lasallian student handbook.
    - It is **not a substitute** for official guidance from school administration or disciplinary officers.

    ## 2. Data Handling
    - No user data is stored or shared. Questions and responses are processed live and are not retained.

    ## 3. Intellectual Property
    - All content generated by this Service is © 2025 Hans Emilio M. Lumagui.
    - You may not copy, redistribute, or use this chatbot’s content for commercial purposes without written permission.

    ## 4. Limitations of Liability
    - The Service is provided “as is” without warranties of any kind.
    - The creator is **not liable** for any actions taken based on the chatbot’s output.

    ## 5. Changes
    - Terms may be updated at any time. Continued use of the Service means you accept the updated terms.

    Questions? Email **hans_lumagui@dlsu.edu.ph**

    _Note from the creator: if this sucks, it's cuz I made this in like 2 hours._
    """)
