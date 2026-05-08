import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image

st.set_page_config(
    page_title="Gesner AI | Your Personal Haitian AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM CSS – FORCE ALL TEXT WHITE ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    /* Force every text element in main area and sidebar to white */
    .stMarkdown, .stTextInput label, .stTextArea label, .stSelectbox label,
    .stFileUploader label, .stButton button, .stCaption, .stMetric label,
    .stExpander, .stExpander summary, .stExpander p, .stExpander div,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer {
        color: #ffffff !important;
    }
    /* Expander header background (semi-transparent) */
    .streamlit-expanderHeader {
        background-color: rgba(15,52,96,0.8) !important;
    }
    /* Button text stays white */
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #ff6b6b !important;
    }
    /* Input fields background */
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #cccccc !important;
    }
    /* Chat messages already have colored backgrounds – keep text white */
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
        text-align: right;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    /* File uploader text */
    .stFileUploader div {
        color: white !important;
    }
    /* Pricing table inside markdown – ensure white */
    table, th, td {
        color: white !important;
        border-color: #e94560 !important;
    }
    th {
        background-color: #0f3460 !important;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOGIN / LOGOUT ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []

def logout():
    st.session_state.authenticated = False
    st.rerun()

def login_page():
    st.markdown("""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 80vh;">
        <div class="login-card" style="background: rgba(15,52,96,0.8); backdrop-filter: blur(12px); border-radius: 30px; padding: 2rem; text-align: center; border: 1px solid #e94560; width: 100%; max-width: 450px; margin: auto;">
            <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
            <div class="login-title" style="color: #ffd966; font-size: 2rem; margin-bottom: 1rem;">Gesner AI</div>
            <p style="color:white;">Enter password to train your personal AI</p>
    """, unsafe_allow_html=True)
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("🔐 Login", use_container_width=True):
        if password == "20082010":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.markdown("</div></div>", unsafe_allow_html=True)

def add_to_training(text):
    if not text.strip():
        return
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    if st.session_state.index is None:
        dim = len(embedding)
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.texts = []
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    st.success(f"✅ Trained: {text[:100]}...")

def load_previous_training():
    if os.path.exists("training_data.json"):
        with open("training_data.json", "r") as f:
            data = json.load(f)
        st.session_state.training_data = data
        if data:
            embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data]
            dim = len(embeddings[0])
            st.session_state.index = faiss.IndexFlatL2(dim)
            st.session_state.index.add(np.array(embeddings))
            st.session_state.texts = [item["text"] for item in data]

def retrieve_relevant_facts(query, k=3):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for idx in indices[0]:
        if idx != -1 and idx < len(st.session_state.texts):
            results.append(st.session_state.texts[idx])
    return results

def generate_response(user_input):
    facts = retrieve_relevant_facts(user_input, k=3)
    if facts:
        context = "\n".join(facts)
        answer = f"Based on what I've learned:\n{context}\n\nTo answer your question: {user_input} – does that help?"
    else:
        answer = f"I don't have specific training on that yet. Please teach me by using the Train section below! Your question: {user_input}"
    return answer

def show_sidebar():
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
    </div>
    <style>
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True)
    st.sidebar.markdown("## **GlobalInternet.py**")
    st.sidebar.markdown("### Gesner AI – Your Personal AI")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Built by Gesner Deslandes** – Coder in Chief")
    st.sidebar.markdown("📞 (509)-47385663")
    st.sidebar.markdown("✉️ deslandes78@gmail.com")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🌐 Website:**")
    st.sidebar.markdown("[https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/](https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/)")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 💰 Pricing")
    st.sidebar.markdown("""
    | License | Price |
    |---------|-------|
    | **Personal** | $49 |
    | **Business** | $299 |
    | **Enterprise** | $999 |
    """)
    if st.sidebar.button("🔓 Logout", use_container_width=True):
        logout()

def main_app():
    show_sidebar()
    load_previous_training()

    st.markdown("<h1 style='text-align:center;'>🧠 Gesner AI – Train Your Personal Haitian AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Teach me through chat, text, images, or files. I learn from everything you share.</p>", unsafe_allow_html=True)

    # --- Chat Interface ---
    st.markdown("## 💬 Chat with Gesner AI")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 You: {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">🤖 Gesner AI: {msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input("Type your message:", key="chat_input")
    if st.button("Send", use_container_width=True):
        if user_input.strip():
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.rerun()

    # --- Training Section (Text) ---
    st.markdown("---")
    st.markdown("## 📚 Train Me (Text)")
    with st.expander("Add a fact or question‑answer pair"):
        training_text = st.text_area("Enter knowledge (e.g., 'Haiti's capital is Port‑au‑Prince')")
        if st.button("Train with this text", use_container_width=True):
            add_to_training(training_text)

    # --- Audio Training (manual transcription) ---
    st.markdown("## 🎤 Train Me with Audio")
    with st.expander("Upload an audio file (you'll need to transcribe manually – or use a service)"):
        audio_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            st.markdown("**After listening, type the transcription below to train me:**")
            transcribed_text = st.text_area("Transcribed text from the audio")
            if st.button("Train with this transcription", use_container_width=True):
                if transcribed_text:
                    add_to_training(transcribed_text)
                else:
                    st.warning("Please enter the transcribed text first.")

    # --- Image Training ---
    st.markdown("## 🖼️ Train Me with Images")
    with st.expander("Upload an image + description"):
        image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        image_description = st.text_area("Describe what this image teaches")
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image", width=200)
            if st.button("Train with this image", use_container_width=True):
                if image_description:
                    add_to_training(image_description)
                else:
                    st.warning("Please add a description to train the AI.")

    # --- File Upload (Text) ---
    st.markdown("## 📄 Train Me with Text Files")
    with st.expander("Upload .txt or .md file"):
        text_file = st.file_uploader("Choose a text file", type=["txt", "md"])
        if text_file is not None:
            content = text_file.read().decode("utf-8")
            st.text_area("File content (preview)", content, height=150)
            if st.button("Train with this file", use_container_width=True):
                add_to_training(content)

    st.markdown("---")
    st.markdown(f"### 📊 Knowledge Base: {len(st.session_state.training_data)} facts trained")
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

    st.markdown('<div class="footer">© GlobalInternet.py – Gesner AI, trained the Haitian way.</div>', unsafe_allow_html=True)

# ---------- ROUTING ----------
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
