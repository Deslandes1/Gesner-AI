import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
import base64
import hashlib

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- LANGUAGES ----------
LANGUAGES = {"English": "en", "Français": "fr", "Kreyòl Ayisyen": "ht"}

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
        "login_message": "Enter password to access Gesner AI",
        "login_button": "Login",
        "wrong_password": "Incorrect password.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Your Personal AI",
        "built_by": "Gesner Deslandes – Coder in Chief",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Website:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licensing",
        "pricing_table": "| License | Price |\n|---|---|\n| Personal | $49 |\n| Business | $299 |",
        "logout_button": "🔓 Logout",
        "toggle_chat_mode": "Chat Mode",
        "record_btn": "🔴 Record",
        "stop_btn": "⏹️ Stop",
        "download_btn": "💾 Download",
        "dict_title": "📖 Dictionaries",
        "dict_add": "Add Entry",
        "voice_training_title": "🎙️ Voice Training"
    },
    "fr": {
        "training_app_title": "🧠 Gesner IA – Centre d'entraînement",
        "chat_title": "💬 Gesner IA Chat",
        "user_prefix": "🧑‍💻 Vous : ",
        "assistant_prefix": "🤖 Gesner IA : ",
        "send_button": "Envoyer",
        "chat_input_placeholder": "Demandez n'importe quoi...",
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
        "login_message": "Entrez le mot de passe",
        "login_button": "Se connecter",
        "wrong_password": "Mot de passe incorrect.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner IA – Votre IA personnelle",
        "built_by": "Gesner Deslandes – Ingénieur en chef",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Site Web:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licence",
        "pricing_table": "| Licence | Prix |\n|---|---|\n| Personnel | 49$ |\n| Entreprise | 299$ |",
        "logout_button": "🔓 Déconnexion",
        "toggle_chat_mode": "Mode Chat",
        "record_btn": "🔴 Enregistrer",
        "stop_btn": "⏹️ Arrêter",
        "download_btn": "💾 Télécharger",
        "dict_title": "📖 Dictionnaires",
        "dict_add": "Ajouter",
        "voice_training_title": "🎙️ Entraînement vocal"
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
        "login_message": "Antre modpas pou konekte",
        "login_button": "Konekte",
        "wrong_password": "Modpas la pa bon.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – AI Pèsonèl ou",
        "built_by": "Gesner Deslandes – Enjenyè anchèf",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitwèb:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Pri",
        "pricing_table": "| Lisans | Pri |\n|---|---|\n| Pèsonèl | $49 |\n| Biznis | $299 |",
        "logout_button": "🔓 Dekonekte",
        "toggle_chat_mode": "Mòd Chat",
        "record_btn": "🔴 Anrejistre",
        "stop_btn": "⏹️ Sispann",
        "download_btn": "💾 Telechaje",
        "dict_title": "📖 Diksyonè",
        "dict_add": "Ajoute",
        "voice_training_title": "🎙️ Fòmasyon vwa"
    }
}

# ---------- CSS ----------
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); }
    .stMarkdown, h1, h2, h3, p, label, .stButton > button { color: #ffffff !important; }
    .stButton button { background-color: #e94560 !important; border-radius: 20px; border: none; font-weight: bold; width: 100%; }
    .chat-bubble { padding: 15px; border-radius: 15px; margin-bottom: 10px; color: white; }
    .user-bubble { background: #e94560; }
    .assistant-bubble { background: #0f3460; border-left: 5px solid #e94560; }
    .speak-btn { background-color: #ffa500; border: none; border-radius: 20px; padding: 5px 15px; cursor: pointer; color: white; font-weight: bold; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "training_data" not in st.session_state: st.session_state.training_data = []
if "conversation_history" not in st.session_state: st.session_state.conversation_history = []
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "language" not in st.session_state: st.session_state.language = "en"
if "chat_mode" not in st.session_state: st.session_state.chat_mode = False

# ---------- VOICE HANDLING ----------
VOICE_CACHE_DIR = "voice_cache"
if not os.path.exists(VOICE_CACHE_DIR): os.makedirs(VOICE_CACHE_DIR)

def get_voice_filename(text):
    h = hashlib.md5(text.strip().lower().encode()).hexdigest()
    return os.path.join(VOICE_CACHE_DIR, f"{h}.wav")

def save_voice(text, audio_bytes):
    with open(get_voice_filename(text), "wb") as f: f.write(audio_bytes)

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
    with open("training_data.json", "w") as f: json.dump(st.session_state.training_data, f)
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
    if st.session_state.index is None or not st.session_state.texts:
        return TEXTS[st.session_state.language]["no_facts_answer"]
    query_emb = st.session_state.embedding_model.encode([user_input])[0].astype(np.float32).reshape(1, -1)
    D, I = st.session_state.index.search(query_emb, 1)
    return st.session_state.texts[I[0][0]] if I[0][0] != -1 else TEXTS[st.session_state.language]["no_facts_answer"]

# ---------- UI LOGIC ----------
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
    
    st.session_state.chat_mode = st.sidebar.toggle(t["toggle_chat_mode"], value=st.session_state.chat_mode)
    
    # Sidebar Info
    st.sidebar.markdown(f"### {t['sidebar_company']}")
    st.sidebar.write(t['sidebar_product'])
    st.sidebar.write(t['built_by'])
    st.sidebar.write(t['phone'])
    st.sidebar.markdown(f"[Website]({t['website_link']})")
    st.sidebar.markdown(t['pricing_table'])
    if st.sidebar.button(t["logout_button"]):
        st.session_state.authenticated = False
        st.rerun()

    if st.session_state.chat_mode:
        st.title(t["chat_title"])
        intro_vwa = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
        
        # FIX: Loop through history using consistent keys to avoid KeyError
        for i, chat in enumerate(st.session_state.conversation_history):
            u_msg = chat.get("user_msg", "")
            a_msg = chat.get("ast_msg", "")
            
            st.markdown(f'<div class="chat-bubble user-bubble">{t["user_prefix"]} {u_msg}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="chat-bubble assistant-bubble">{t["assistant_prefix"]} {a_msg}</div>', unsafe_allow_html=True)
            st.components.v1.html(get_voice_html(intro_vwa, i), height=50)

        with st.form("chat_form", clear_on_submit=True):
            user_in = st.text_input(t["chat_input_placeholder"])
            if st.form_submit_button(t["send_button"]) and user_in:
                ans = get_response(user_in)
                st.session_state.conversation_history.append({"user_msg": user_in, "ast_msg": ans})
                st.rerun()
    else:
        st.title(t["training_app_title"])
        
        with st.expander(t["training_text_title"]):
            txt_in = st.text_area(t["text_area_label"])
            if st.button(t["train_text_button"]): add_to_training(txt_in, t)

        with st.expander(t["audio_title"]):
            aud = st.file_uploader(t["audio_upload_label"], type=["wav", "mp3"])
            transcript = st.text_area(t["transcription_textarea"])
            if st.button(t["train_transcription_button"]) and aud and transcript:
                save_voice(transcript, aud.read())
                add_to_training(transcript, t)
        
        with st.expander(t["dict_title"]):
            word = st.text_input("Word")
            meaning = st.text_input("Meaning")
            if st.button(t["dict_add"]): add_to_training(f"{word}: {meaning}", t)
