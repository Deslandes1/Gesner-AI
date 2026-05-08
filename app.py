import streamlit as st
import streamlit.components.v1 as components
import json
import os
import time
import speech_recognition as sr
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
import io

st.set_page_config(
    page_title="Gesner AI | Your Personal Haitian AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CUSTOM CSS (colorful, Haitian flag inspired) ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
    }
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
    h1, h2, h3 {
        color: #ffd966 !important;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    .training-card {
        background: rgba(0,0,0,0.5);
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ---------- SPINNING GLOBE ----------
def spinning_globe():
    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
    </div>
    <style>
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
    """, unsafe_allow_html=True)

# ---------- LOGIN / LOGOUT ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "training_data" not in st.session_state:
    st.session_state.training_data = []  # list of {"text": str, "embedding": list}
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    # Load a small sentence transformer (will be cached)
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

# ---------- TRAINING FUNCTIONS ----------
def add_to_training(text):
    """Add a text fact to the knowledge base."""
    if not text.strip():
        return
    # Create embedding
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    # Update FAISS index
    if st.session_state.index is None:
        dim = len(embedding)
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.texts = []
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    st.session_state.texts.append(text)
    # Save to disk (for persistence)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    st.success(f"✅ Trained: {text[:100]}...")

def load_previous_training():
    if os.path.exists("training_data.json"):
        with open("training_data.json", "r") as f:
            data = json.load(f)
        st.session_state.training_data = data
        # rebuild index
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
    # Retrieve relevant facts
    facts = retrieve_relevant_facts(user_input, k=3)
    # Simple response generation: if facts exist, use them; else generic.
    if facts:
        context = "\n".join(facts)
        answer = f"Based on what I've learned:\n{context}\n\nTo answer your question: {user_input} – does that help?"
    else:
        answer = f"I don't have specific training on that yet. Please teach me by using the Train section below! Your question: {user_input}"
    return answer

# ---------- TRANSCRIBE AUDIO ----------
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return None

# ---------- ANALYZE IMAGE (simple – will just add description) ----------
def analyze_image(image_file, user_description):
    # For now, just use the user description as training
    return user_description if user_description else "Image uploaded without description."

# ---------- SIDEBAR ----------
def show_sidebar():
    spinning_globe()
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

# ---------- MAIN APP ----------
def main_app():
    show_sidebar()
    load_previous_training()  # load saved facts

    st.markdown("<h1 style='text-align:center;'>🧠 Gesner AI – Train Your Personal Haitian AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Teach me through chat, audio, or media uploads. I learn from everything you share.</p>", unsafe_allow_html=True)

    # --- Chat Interface ---
    st.markdown("## 💬 Chat with Gesner AI")
    # Display conversation history
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

    # --- Audio Training ---
    st.markdown("## 🎤 Train Me with Audio")
    with st.expander("Upload an audio file (speech)"):
        audio_file = st.file_uploader("Choose an audio file (WAV, MP3)", type=["wav", "mp3"])
        if audio_file is not None:
            with st.spinner("Transcribing audio..."):
                text = transcribe_audio(audio_file)
                if text:
                    st.success(f"Transcribed: {text}")
                    add_to_training(text)
                else:
                    st.error("Could not transcribe audio. Try a clearer recording.")

    # --- Image Training ---
    st.markdown("## 🖼️ Train Me with Images")
    with st.expander("Upload an image + description"):
        image_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
        image_description = st.text_area("Describe what this image teaches (optional but recommended)")
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

    # --- Show trained facts count ---
    st.markdown("---")
    st.markdown(f"### 📊 Knowledge Base: {len(st.session_state.training_data)} facts trained")

    # --- Reset conversation button ---
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

    st.markdown('<div class="footer">© GlobalInternet.py – Gesner AI, trained the Haitian way.</div>', unsafe_allow_html=True)

# ---------- ROUTING ----------
if not st.session_state.authenticated:
    login_page()
else:
    main_app()
