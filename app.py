import streamlit as st
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PIL import Image
import requests
import base64
import time
import hashlib

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
        "training_subtitle": "Teach me facts, dictionaries, encyclopedia.",
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 You: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Send",
        "chat_input_placeholder": "Ask me anything...",
        "training_text_title": "📚 Train Me (Text)",
        "expand_text": "Add a fact or question‑answer pair",
        "text_area_label": "Enter knowledge (e.g., 'Haiti's capital is Port‑au‑Prince')",
        "train_text_button": "Train",
        "audio_title": "🎤 Train Me with Audio",
        "expand_audio": "Record or upload audio + transcription",
        "audio_upload_label": "Upload Audio File",
        "transcribe_label": "Transcribed text:",
        "transcription_textarea": "Type the transcription here",
        "train_transcription_button": "Train",
        "record_btn": "🔴 Record",
        "stop_btn": "⏹️ Stop",
        "download_btn": "💾 Download",
        "image_title": "🖼️ Train Me with Images",
        "expand_image": "Upload an image + description",
        "image_upload_label": "Choose an image",
        "image_description_label": "Describe this image",
        "train_image_button": "Train",
        "file_title": "📄 Train Me with Text Files",
        "expand_file": "Upload .txt or .md file",
        "file_upload_label": "Choose a text file",
        "train_file_button": "Train",
        "knowledge_base": "📊 Knowledge Base: {count} facts trained",
        "clear_chat_button": "Clear Chat History",
        "footer": "© GlobalInternet.py – Gesner AI",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Your Personal AI",
        "built_by": "Gesner Deslandes – Coder in Chief",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Website:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licensing",
        "pricing_table": """
| License | Price (one‑time) |
|---------|------------------|
| **Personal** | $49 |
| **Business** | $299 |
| **Enterprise / Source** | $999 |
""",
        "logout_button": "🔓 Logout",
        "no_facts_answer": "I don't know that yet. Please teach me in Training Mode!",
        "with_facts_answer": "{context}",
        "training_success": "✅ Trained: {text}...",
        "warning_no_text": "Please enter some text.",
        "warning_no_transcription": "Please enter the transcribed text first.",
        "warning_no_description": "Please add a description.",
        "file_preview": "File content (preview)",
        "image_caption": "Uploaded Image",
        "login_title": "Gesner AI",
        "login_message": "Enter password to access Gesner AI",
        "login_button": "Login",
        "wrong_password": "Incorrect password.",
        "dict_title": "📖 Dictionaries",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Word",
        "dict_meaning": "Meaning",
        "dict_add": "Add Entry",
        "dict_delete": "Delete",
        "voice_training_title": "🎙️ Voice Training",
        "voice_upload": "Upload voice (WAV/MP3)",
        "voice_transcribed_text": "Text spoken in the audio (exact transcript)",
        "voice_train": "Train voice + text",
        "voice_success": "Voice and text stored!",
        "translation_title": "🌍 Translate & Correct",
        "translation_source_text": "Text to translate (any language)",
        "translate_btn": "Translate to Kreyòl",
        "translation_result": "Translated text (editable)",
        "train_translation_btn": "Train with corrected text",
        "encyclopedia_title": "📚 Encyclopedia",
        "encyclopedia_add": "Add Encyclopedia Entry",
        "encyclopedia_title_field": "Title",
        "encyclopedia_content": "Content",
        "encyclopedia_lang": "Language",
        "encyclopedia_tags": "Tags (comma)",
        "encyclopedia_save": "Save Entry",
        "encyclopedia_list": "Existing Entries",
        "voice_download": "Download Recording",
        "test_title": "🧪 Test Training",
        "test_question": "Ask a question to retrieve exact stored fact",
        "test_button": "Test",
        "test_answer_label": "Stored fact:",
        "test_speak_button": "🔊 Speak",
        "upload_voice_label": "Upload your voice for this exact text",
        "chat_mode_title": "💬 Gesner AI Chat",
        "chat_mode_placeholder": "Ask me anything...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Upload voice for this answer",
        "image_upload_label": "📷 Upload image",
        "image_describe_button": "Describe",
        "image_description_result": "Description:",
        "toggle_chat_mode": "Chat Mode"
    },
    "fr": {
        "training_app_title": "🧠 Gesner IA – Centre d'entraînement",
        "training_subtitle": "Enseignez‑moi des faits, dictionnaires, encyclopédie.",
        "chat_title": "💬 Gesner IA Chat",
        "user_prefix": "🧑‍💻 Vous : ",
        "assistant_prefix": "🤖 Gesner IA : ",
        "send_button": "Envoyer",
        "chat_input_placeholder": "Demandez‑moi n'importe quoi...",
        "training_text_title": "📚 Entraînez‑moi (texte)",
        "expand_text": "Ajouter un fait ou une paire Q/R",
        "text_area_label": "Entrez la connaissance",
        "train_text_button": "Entraîner",
        "audio_title": "🎤 Entraînez‑moi avec audio",
        "expand_audio": "Enregistrez ou téléchargez audio + transcription",
        "audio_upload_label": "Fichier audio",
        "transcribe_label": "Texte transcrit :",
        "transcription_textarea": "Tapez la transcription",
        "train_transcription_button": "Entraîner",
        "record_btn": "🔴 Enregistrer",
        "stop_btn": "⏹️ Arrêter",
        "download_btn": "💾 Télécharger",
        "image_title": "🖼️ Entraînez‑moi avec images",
        "expand_image": "Image + description",
        "image_upload_label": "Choisir une image",
        "image_description_label": "Décrivez cette image",
        "train_image_button": "Entraîner",
        "file_title": "📄 Entraînez‑moi avec fichiers texte",
        "expand_file": "Fichier .txt ou .md",
        "file_upload_label": "Choisir un fichier",
        "train_file_button": "Entraîner",
        "knowledge_base": "📊 Base de connaissances : {count} faits",
        "clear_chat_button": "Effacer l'historique",
        "footer": "© GlobalInternet.py – Gesner IA",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner IA – Votre IA personnelle",
        "built_by": "Gesner Deslandes – Ingénieur en chef",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Site web :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licence",
        "pricing_table": """
| Licence | Prix (unique) |
|---------|---------------|
| **Personnelle** | 49 $ |
| **Entreprise** | 299 $ |
| **Entreprise / Code source** | 999 $ |
""",
        "logout_button": "🔓 Déconnexion",
        "no_facts_answer": "Je ne connais pas encore cela. Enseignez‑moi en mode Entraînement !",
        "with_facts_answer": "{context}",
        "training_success": "✅ Entraîné : {text}...",
        "warning_no_text": "Veuillez saisir du texte.",
        "warning_no_transcription": "Veuillez d'abord saisir le texte transcrit.",
        "warning_no_description": "Veuillez ajouter une description.",
        "file_preview": "Aperçu du fichier",
        "image_caption": "Image téléchargée",
        "login_title": "Gesner IA",
        "login_message": "Entrez le mot de passe pour accéder à Gesner IA",
        "login_button": "Se connecter",
        "wrong_password": "Mot de passe incorrect.",
        "dict_title": "📖 Dictionnaires",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Mot",
        "dict_meaning": "Signification",
        "dict_add": "Ajouter",
        "dict_delete": "Supprimer",
        "voice_training_title": "🎙️ Entraînement vocal",
        "voice_upload": "Télécharger voix (WAV/MP3)",
        "voice_transcribed_text": "Texte parlé dans l'audio",
        "voice_train": "Entraîner voix + texte",
        "voice_success": "Voix et texte enregistrés !",
        "translation_title": "🌍 Traduire et corriger",
        "translation_source_text": "Texte à traduire (n'importe quelle langue)",
        "translate_btn": "Traduire en Kreyòl",
        "translation_result": "Texte traduit (modifiable)",
        "train_translation_btn": "Entraîner avec ce texte",
        "encyclopedia_title": "📚 Encyclopédie",
        "encyclopedia_add": "Ajouter une entrée",
        "encyclopedia_title_field": "Titre",
        "encyclopedia_content": "Contenu",
        "encyclopedia_lang": "Langue",
        "encyclopedia_tags": "Étiquettes (virgules)",
        "encyclopedia_save": "Enregistrer",
        "encyclopedia_list": "Entrées existantes",
        "voice_download": "Télécharger",
        "test_title": "🧪 Tester l'entraînement",
        "test_question": "Posez une question pour voir le fait stocké",
        "test_button": "Tester",
        "test_answer_label": "Fait stocké :",
        "test_speak_button": "🔊 Lire",
        "upload_voice_label": "Téléchargez votre voix pour ce texte exact",
        "chat_mode_title": "💬 Gesner IA Chat",
        "chat_mode_placeholder": "Demandez‑moi n'importe quoi...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Téléchargez votre voix pour cette réponse",
        "image_upload_label": "📷 Télécharger une image",
        "image_describe_button": "Décrire",
        "image_description_result": "Description :",
        "toggle_chat_mode": "Mode Chat"
    },
    "ht": {
        "training_app_title": "🧠 Gesner AI – Sant Fòmasyon",
        "training_subtitle": "Anseye m reyalite, diksyonè, ansiklopedi.",
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 Ou : ",
        "assistant_prefix": "🤖 Gesner AI : ",
        "send_button": "Voye",
        "chat_input_placeholder": "Pose yon kesyon...",
        "training_text_title": "📚 Antrene m (tèks)",
        "expand_text": "Ajoute yon reyalite oswa kesyon/repons",
        "text_area_label": "Antre konesans lan",
        "train_text_button": "Antrene",
        "audio_title": "🎤 Antrene m ak odyo",
        "expand_audio": "Anrejistre oswa chaje odyo + transkripsyon",
        "audio_upload_label": "Chaje fichye odyo",
        "transcribe_label": "Tèks transkri :",
        "transcription_textarea": "Tape transkripsyon an",
        "train_transcription_button": "Antrene",
        "record_btn": "🔴 Anrejistre",
        "stop_btn": "⏹️ Sispann",
        "download_btn": "💾 Telechaje",
        "image_title": "🖼️ Antrene m ak imaj",
        "expand_image": "Imaj + deskripsyon",
        "image_upload_label": "Chwazi yon imaj",
        "image_description_label": "Dekri imaj sa a",
        "train_image_button": "Antrene",
        "file_title": "📄 Antrene m ak fichye tèks",
        "expand_file": "Fichye .txt oswa .md",
        "file_upload_label": "Chwazi yon fichye",
        "train_file_button": "Antrene",
        "knowledge_base": "📊 Baz konesans : {count} reyalite",
        "clear_chat_button": "Efase listorik",
        "footer": "© GlobalInternet.py – Gesner AI",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – AI Pèsonèl ou",
        "built_by": "Gesner Deslandes – Enjenyè anchèf",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitwèb :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Pri",
        "pricing_table": """
| Lisans | Pri (yon fwa) |
|--------|---------------|
| **Pèsonèl** | $49 |
| **Biznis** | $299 |
| **Antrepriz / Kòd sous** | $999 |
""",
        "logout_button": "🔓 Dekonekte",
        "no_facts_answer": "Mwen poko konnen sa. Tanpri anseye m nan Mòd Fòmasyon!",
        "with_facts_answer": "{context}",
        "training_success": "✅ Antrene : {text}...",
        "warning_no_text": "Tanpri antre kèk tèks.",
        "warning_no_transcription": "Tanpri antre tèks transkri an premye.",
        "warning_no_description": "Tanpri ajoute yon deskripsyon.",
        "file_preview": "Aperçu fichye a",
        "image_caption": "Imaj chaje",
        "login_title": "Gesner AI",
        "login_message": "Antre modpas pou konekte",
        "login_button": "Konekte",
        "wrong_password": "Modpas pa bon.",
        "dict_title": "📖 Diksyonè",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Mo",
        "dict_meaning": "Siyifikasyon",
        "dict_add": "Ajoute",
        "dict_delete": "Efase",
        "voice_training_title": "🎙️ Fòmasyon vwa",
        "voice_upload": "Chaje vwa (WAV/MP3)",
        "voice_transcribed_text": "Tèks ki nan odyo a",
        "voice_train": "Antrene vwa + tèks",
        "voice_success": "Vwa ak tèks sove!",
        "translation_title": "🌍 Tradwi epi korije",
        "translation_source_text": "Tèks pou tradwi (nenpòt lang)",
        "translate_btn": "Tradwi an Kreyòl",
        "translation_result": "Tèks tradwi (kapab modifye)",
        "train_translation_btn": "Antrene avèk tèks sa a",
        "encyclopedia_title": "📚 Ansiklopedi",
        "encyclopedia_add": "Ajoute yon antre",
        "encyclopedia_title_field": "Tit",
        "encyclopedia_content": "Kontni",
        "encyclopedia_lang": "Lang",
        "encyclopedia_tags": "Etikèt (vigil)",
        "encyclopedia_save": "Sove",
        "encyclopedia_list": "Antre ki egziste",
        "voice_download": "Telechaje",
        "test_title": "🧪 Tese fòmasyon",
        "test_question": "Pose yon kesyon pou wè reyalite a",
        "test_button": "Tese",
        "test_answer_label": "Reyalite ki sove :",
        "test_speak_button": "🔊 Pwononse",
        "upload_voice_label": "Chaje vwa ou pou tèks egzak sa a",
        "chat_mode_title": "💬 Gesner AI Chat",
        "chat_mode_placeholder": "Pose yon kesyon...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Chaje vwa ou pou repons sa a",
        "image_upload_label": "📷 Chaje yon imaj",
        "image_describe_button": "Dekri",
        "image_description_result": "Deskripsyon :",
        "toggle_chat_mode": "Mòd Chat"
    }
}

# ---------- CSS (forces all text to bright white) ----------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f3460 0%, #1a1a2e 100%);
        border-right: 2px solid #e94560;
    }
    .stMarkdown, .stTextInput label, .stTextArea label, .stSelectbox label,
    .stFileUploader label, .stButton button, .stCaption, .stMetric label,
    .stExpander, .stExpander summary, .stExpander p, .stExpander div,
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer,
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        width: 100%;
        border: none;
    }
    .stButton button:hover {
        background-color: #ff6b6b !important;
        transform: scale(1.02);
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .speak-btn {
        background-color: #ffaa33;
        border: none;
        border-radius: 30px;
        padding: 5px 12px;
        margin-left: 12px;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
    }
    .speak-btn:hover {
        background-color: #ffcc66;
        transform: scale(1.05);
    }
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
    with st.spinner("Loading AI..."):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    st.session_state.index = None
    st.session_state.texts = []
if "language" not in st.session_state:
    st.session_state.language = "en"
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}

# ---------- VOICE CACHE ----------
VOICE_CACHE_DIR = "voice_cache"
if not os.path.exists(VOICE_CACHE_DIR):
    os.makedirs(VOICE_CACHE_DIR)

def get_voice_filename(text):
    norm = text.strip().lower()
    h = hashlib.md5(norm.encode()).hexdigest()
    return os.path.join(VOICE_CACHE_DIR, f"{h}.wav")

def save_voice_for_text(text, audio_bytes):
    filename = get_voice_filename(text)
    with open(filename, "wb") as f:
        f.write(audio_bytes)
    return filename

def get_voice_for_text(text):
    filename = get_voice_filename(text)
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            return f.read()
    return None

def play_voice_button(text, button_label="🔊", key_suffix=""):
    voice_bytes = get_voice_for_text(text)
    if voice_bytes:
        audio_b64 = base64.b64encode(voice_bytes).decode()
        html = f"""
        <button class="speak-btn" id="voiceBtn_{key_suffix}">{button_label}</button>
        <audio id="customAudio_{key_suffix}" src="data:audio/wav;base64,{audio_b64}"></audio>
        <script>
            document.getElementById('voiceBtn_{key_suffix}').onclick = () => document.getElementById('customAudio_{key_suffix}').play();
        </script>
        """
        return html
    else:
        # Fallback to browser TTS
        safe_text = json.dumps(text)
        html = f"""
        <button class="speak-btn" id="ttsBtn_{key_suffix}">{button_label}</button>
        <script>
            document.getElementById('ttsBtn_{key_suffix}').onclick = () => {{
                const utterance = new SpeechSynthesisUtterance({safe_text});
                window.speechSynthesis.speak(utterance);
            }};
        </script>
        """
        return html

# ---------- FUNCTIONS ----------
def add_to_training(text, t):
    if not text.strip():
        return
    embedding = st.session_state.embedding_model.encode([text])[0]
    st.session_state.training_data.append({"text": text, "embedding": embedding.tolist()})
    if st.session_state.index is None:
        st.session_state.index = faiss.IndexFlatL2(len(embedding))
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f)
    st.success(t.get('training_success', "Trained!").format(text=text[:50]))

def load_previous_training():
    if os.path.exists("training_data.json") and not st.session_state.texts:
        with open("training_data.json", "r") as f:
            data = json.load(f)
            st.session_state.training_data = data
            if data:
                embeddings = [np.array(i["embedding"], dtype=np.float32) for i in data]
                st.session_state.index = faiss.IndexFlatL2(len(embeddings[0]))
                st.session_state.index.add(np.array(embeddings))
                st.session_state.texts = [i["text"] for i in data]

def generate_response(user_input):
    if st.session_state.index is None:
        return TEXTS[st.session_state.language]["no_facts_answer"]
    query_emb = st.session_state.embedding_model.encode([user_input])[0].astype(np.float32).reshape(1, -1)
    D, I = st.session_state.index.search(query_emb, 1)
    if I[0][0] != -1:
        return st.session_state.texts[I[0][0]]
    return TEXTS[st.session_state.language]["no_facts_answer"]

# ---------- UI ----------
if not st.session_state.authenticated:
    st.title("Gesner AI Login")
    pw = st.text_input("Password", type="password")
    if st.button("Login"):
        if pw == "20082010":
            st.session_state.authenticated = True
            st.rerun()
else:
    load_previous_training()
    # Sidebar
    lang_sel = st.sidebar.selectbox("Language", list(LANGUAGES.keys()))
    st.session_state.language = LANGUAGES[lang_sel]
    t = TEXTS[st.session_state.language]
    
    st.session_state.chat_mode = st.sidebar.toggle("Chat Mode", value=st.session_state.chat_mode)
    
    if st.session_state.chat_mode:
        st.title(t["chat_title"])
        
        for idx, chat in enumerate(st.session_state.conversation_history):
            st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]} {chat["user"]}</div>', unsafe_allow_html=True)
            
            # THE CORE REQUEST: Append the specific Kreyòl text to every reply
            intro_voice_text = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
            full_reply = f"{chat['assistant']} \n\n 🤖 Gesner AI: {intro_voice_text}"
            
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]} {full_reply}</div>', unsafe_allow_html=True)
            # Play button for the ending voice
            st.components.v1.html(play_voice_button(intro_voice_text, key_suffix=f"voice_{idx}"), height=50)

        with st.form("chat_input", clear_on_submit=True):
            user_in = st.text_input(t["chat_input_placeholder"])
            if st.form_submit_button(t["send_button"]):
                ans = generate_response(user_in)
                st.session_state.conversation_history.append({"user": user_in, "assistant": ans})
                st.rerun()
    else:
        st.title(t["training_app_title"])
        with st.expander(t["audio_title"]):
            up = st.file_uploader(t["audio_upload_label"], type=["wav", "mp3"])
            trans = st.text_area(t["transcription_textarea"])
            if st.button(t["train_transcription_button"]) and up and trans:
                save_voice_for_text(trans.strip(), up.read())
                add_to_training(trans.strip(), t)
        
        with st.expander(t["training_text_title"]):
            txt = st.text_area(t["text_area_label"])
            if st.button(t["train_text_button"]):
                add_to_training(txt, t)
