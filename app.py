import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import base64
import hashlib

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht"
}

TEXTS = {
    "en": {
        "training_app_title": "🧠 Gesner AI – Training Center",
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 You: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Send",
        "chat_input_placeholder": "Ask me anything...",
        "training_text_title": "📚 Train Me (Text)",
        "text_area_label": "Enter knowledge",
        "train_text_button": "Train",
        "audio_title": "🎤 Train Me with Audio",
        "audio_upload_label": "Upload Audio File",
        "transcription_textarea": "Type the transcription here",
        "train_transcription_button": "Train",
        "training_success": "✅ Trained: {text}...",
        "no_facts_answer": "I don't know that yet. Please teach me!",
        "login_title": "Gesner AI",
        "login_button": "Login",
        "sidebar_company": "GlobalInternet.py"
    },
    "fr": {
        "training_app_title": "🧠 Gesner IA – Centre d'entraînement",
        "chat_title": "💬 Gesner IA Chat",
        "user_prefix": "🧑‍💻 Vous : ",
        "assistant_prefix": "🤖 Gesner IA : ",
        "send_button": "Envoyer",
        "chat_input_placeholder": "Demandez-moi n'importe quoi...",
        "training_text_title": "📚 Entraînez-moi (texte)",
        "text_area_label": "Entrez la connaissance",
        "train_text_button": "Entraîner",
        "audio_title": "🎤 Entraînez-moi avec audio",
        "audio_upload_label": "Fichier audio",
        "transcription_textarea": "Tapez la transcription",
        "train_transcription_button": "Entraîner",
        "training_success": "✅ Entraîné : {text}...",
        "no_facts_answer": "Je ne connais pas encore cela.",
        "login_title": "Gesner IA",
        "login_button": "Se connecter",
        "sidebar_company": "GlobalInternet.py"
    },
    "ht": {
        "training_app_title": "🧠 Gesner AI – Sant Fòmasyon",
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 Ou : ",
        "assistant_prefix": "🤖 Gesner AI : ",
        "send_button": "Voye",
        "chat_input_placeholder": "Pose yon kesyon...",
        "training_text_title": "📚 Antrene m (tèks)",
        "text_area_label": "Antre konesans lan",
        "train_text_button": "Antrene",
        "audio_title": "🎤 Antrene m ak odyo",
        "audio_upload_label": "Chaje fichye odyo",
        "transcription_textarea": "Tape transkripsyon an",
        "train_transcription_button": "Antrene",
        "training_success": "✅ Antrene : {text}...",
        "no_facts_answer": "Mwen poko konnen sa.",
        "login_title": "Gesner AI",
        "login_button": "Konekte",
        "sidebar_company": "GlobalInternet.py"
    }
}

# ---------- CSS ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stMarkdown, h1, h2, h3, p, label { color: #ffffff !important; }
    .stButton button { background-color: #e94560 !important; color: white !important; border-radius: 20px; border: none; }
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; color: white; }
    .user-bubble { background: #e94560; }
    .assistant-bubble { background: #0f3460; border-left: 5px solid #e94560; }
    .speak-btn { background-color: #ffa500; border: none; border-radius: 50%; padding: 8px 12px; cursor: pointer; color: white; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False

# ---------- VOICE HANDLING ----------
VOICE_CACHE_DIR = "voice_cache"
if not os.path.exists(VOICE_CACHE_DIR):
    os.makedirs(VOICE_CACHE_DIR)

def get_voice_filename(text):
    h = hashlib.md5(text.strip().lower().encode()).hexdigest()
    return os.path.join(VOICE_CACHE_DIR, f"{h}.wav")

def save_voice(text, audio_bytes):
    with open(get_voice_filename(text), "wb") as f:
        f.write(audio_bytes)

def get_voice_html(text, key):
    filename = get_voice_filename(text)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            data = base64.b64encode(f.read()).decode()
        return f"""
        <button class="speak-btn" onclick="document.getElementById('audio_{key}').play()">🔊 Jwe Vwa</button>
        <audio id="audio_{key}" src="data:audio/wav;base64,{data}"></audio>
        """
    return ""

# ---------- AI LOGIC ----------
def add_to_training(text, t_dict):
    if not text.strip(): return
    emb = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": emb.tolist()})
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatL2(len(emb))
    st.session_state.index.add(np.array([emb], dtype=np.float32))
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f)
    st.success(t_dict["training_success"].format(text=text[:30]))

def load_training():
    if os.path.exists("training_data.json") and not st.session_state.texts:
        with open("training_data.json", "r") as f:
            data = json.load(f)
            st.session_state.training_data = data
            if data:
                embs = [np.array(i["embedding"], dtype=np.float32) for i in data]
                st.session_state.index = faiss.IndexFlatL2(len(embs[0]))
                st.session_state.index.add(np.array(embs))
                st.session_state.texts = [i["text"] for i in data]

def get_response(user_input):
    if st.session_state.index is None: return TEXTS[st.session_state.language]["no_facts_answer"]
    query_emb = st.session_state.embedding_model.encode([user_input])[0].astype(np.float32).reshape(1, -1)
    D, I = st.session_state.index.search(query_emb, 1)
    return st.session_state.texts[I[0][0]] if I[0][0] != -1 else TEXTS[st.session_state.language]["no_facts_answer"]

# ---------- MAIN UI ----------
if not st.session_state.authenticated:
    st.title("Gesner AI")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == "20082010":
            st.session_state.authenticated = True
            st.rerun()
else:
    load_training()
    lang_name = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
    st.session_state.language = LANGUAGES[lang_name]
    t = TEXTS[st.session_state.language]
    st.session_state.chat_mode = st.sidebar.toggle("Chat Mode", value=st.session_state.chat_mode)
    st.sidebar.write(f"© {t['sidebar_company']}")

    if st.session_state.chat_mode:
        st.title(t["chat_title"])
        # This is the text for your specific voice recording
        intro_text = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
        
        # DISPLAY CHAT
        for i, chat in enumerate(st.session_state.conversation_history):
            # Using .get() prevents KeyError if for some reason keys are missing
            u_msg = chat.get("user_msg", "")
            a_msg = chat.get("ast_msg", "")
            
            st.markdown(f'<div class="chat-bubble user-bubble">{t["user_prefix"]} {u_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble assistant-bubble">{t["assistant_prefix"]} {a_msg} <br><br><i>{intro_text}</i></div>', unsafe_allow_html=True)
            
            # Show the voice button for your specific recording
            st.components.v1.html(get_voice_html(intro_text, i), height=50)

        # INPUT AREA
        with st.form("chat_form", clear_on_submit=True):
            user_in = st.text_input(t["chat_input_placeholder"])
            if st.form_submit_button(t["send_button"]):
                if user_in:
                    ans = get_response(user_in)
                    # We use unique keys 'user_msg' and 'ast_msg'
                    st.session_state.conversation_history.append({"user_msg": user_in, "ast_msg": ans})
                    st.rerun()
    else:
        st.title(t["training_app_title"])
        with st.expander(t["audio_title"]):
            aud = st.file_uploader(t["audio_upload_label"], type=["wav", "mp3"])
            transcript = st.text_area(t["transcription_textarea"])
            if st.button(t["train_transcription_button"]) and aud and transcript:
                save_voice(transcript, aud.read())
                add_to_training(transcript, t)

        with st.expander(t["training_text_title"]):
            txt_in = st.text_area(t["text_area_label"])
            if st.button(t["train_text_button"]):
                add_to_training(txt_in, t)
