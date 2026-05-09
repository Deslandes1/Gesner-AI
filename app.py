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
    /* Force all text to bright white */
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
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
    }
    .stExpanderHeader {
        background-color: rgba(15,52,96,0.8) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------- INIT SESSION STATE ----------
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
if "language" not in st.session_state:
    st.session_state.language = "en"
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False

# ---------- ADD MISSING SESSION STATE INITIALIZATIONS ----------
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}
if "audio_transcriptions" not in st.session_state:
    st.session_state.audio_transcriptions = []
if "encyclopedia" not in st.session_state:
    st.session_state.encyclopedia = []

# Voice cache directory
VOICE_CACHE_DIR = "voice_cache"
if not os.path.exists(VOICE_CACHE_DIR):
    os.makedirs(VOICE_CACHE_DIR)

def get_voice_filename(text):
    """Generate a filename from the exact text (normalized)."""
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
    """Return an HTML/JS button that plays the voice if available, else falls back to TTS."""
    voice_bytes = get_voice_for_text(text)
    if voice_bytes:
        audio_b64 = base64.b64encode(voice_bytes).decode()
        mime = "audio/wav"
        html = f"""
        <button class="speak-btn" id="customVoiceBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <audio id="customAudio_{key_suffix}" style="display:none;"></audio>
        <script>
            const audioData_{key_suffix} = "{audio_b64}";
            const binaryStr_{key_suffix} = atob(audioData_{key_suffix});
            const bytes_{key_suffix} = new Uint8Array(binaryStr_{key_suffix}.length);
            for (let i = 0; i < binaryStr_{key_suffix}.length; i++) bytes_{key_suffix}[i] = binaryStr_{key_suffix}.charCodeAt(i);
            const audioBlob_{key_suffix} = new Blob([bytes_{key_suffix}], {{ type: '{mime}' }});
            const audioUrl_{key_suffix} = URL.createObjectURL(audioBlob_{key_suffix});
            const audioEl_{key_suffix} = document.getElementById('customAudio_{key_suffix}');
            audioEl_{key_suffix}.src = audioUrl_{key_suffix};
            document.getElementById('customVoiceBtn_{key_suffix}').onclick = () => audioEl_{key_suffix}.play();
        </script>
        """
        return html
    else:
        safe_text = json.dumps(text)
        html = f"""
        <button class="speak-btn" id="ttsBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <script>
            document.getElementById('ttsBtn_{key_suffix}').onclick = () => {{
                const utterance = new SpeechSynthesisUtterance({safe_text});
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            }};
        </script>
        """
        return html

# ---------- TRAINING FUNCTIONS ----------
def add_to_training(text, t):
    if not text.strip():
        st.warning(t['warning_no_text'])
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
    st.success(t['training_success'].format(text=text[:100]))

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

# Pre‑train intro text
intro_text_ht = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
if intro_text_ht not in st.session_state.texts:
    embedding = st.session_state.embedding_model.encode([intro_text_ht])[0]
    st.session_state.training_data.append({"text": intro_text_ht, "embedding": embedding.tolist()})
    if st.session_state.index is None:
        dim = len(embedding)
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.texts = []
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    st.session_state.texts.append(intro_text_ht)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)

def retrieve_relevant_facts(query, k=1):
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
    facts = retrieve_relevant_facts(user_input, k=1)
    t = TEXTS[st.session_state.language]
    if facts:
        return facts[0]
    else:
        return t["no_facts_answer"]

def login_page():
    t = TEXTS[st.session_state.language]
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 80vh;">
        <div class="login-card" style="background: rgba(15,52,96,0.8); backdrop-filter: blur(12px); border-radius: 30px; padding: 2rem; text-align: center; border: 1px solid #e94560; width: 100%; max-width: 450px; margin: auto;">
            <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
            <div class="login-title" style="color: #ffd966; font-size: 2rem; margin-bottom: 1rem;">Gesner AI</div>
            <p style="color:white;">{t['login_message']}</p>
    """, unsafe_allow_html=True)
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button(t['login_button'], use_container_width=True):
        if password == "20082010":
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error(t['wrong_password'])
    st.markdown("</div></div>", unsafe_allow_html=True)

def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language", lang_names)
    st.session_state.language = LANGUAGES[selected_lang_name]
    t = TEXTS[st.session_state.language]

    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
    </div>
    """, unsafe_allow_html=True)
    st.sidebar.markdown(f"## **{t['sidebar_company']}**")
    st.sidebar.markdown(f"### {t['sidebar_product']}")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**{t['built_by']}**")
    st.sidebar.markdown(t['phone'])
    st.sidebar.markdown(t['email'])
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"{t['website_label']} [{t['website_link']}]({t['website_link']})")
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### {t['pricing_title']}")
    st.sidebar.markdown(t['pricing_table'])
    
    # Toggle Chat Mode
    chat_mode_toggle = st.sidebar.toggle(t['toggle_chat_mode'], value=st.session_state.chat_mode)
    if chat_mode_toggle != st.session_state.chat_mode:
        st.session_state.chat_mode = chat_mode_toggle
        st.rerun()
    
    if st.sidebar.button(t['logout_button'], use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ---------- DICTIONARY MANAGER ----------
def dictionary_manager(t):
    st.markdown(f"## {t['dict_title']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {t['dict_ht']}")
        w = st.text_input(f"{t['dict_word']} (HT)", key="ht_word")
        m = st.text_input(f"{t['dict_meaning']} (HT)", key="ht_meaning")
        if st.button(t['dict_add'], key="add_ht"):
            if w and m:
                st.session_state.dictionaries["ht"][w] = m
                save_dictionaries()
                st.success(f"Added {w}")
                st.rerun()
        for word, meaning in list(st.session_state.dictionaries["ht"].items()):
            st.text(f"{word}: {meaning}")
            if st.button(f"{t['dict_delete']} {word}", key=f"del_ht_{word}"):
                del st.session_state.dictionaries["ht"][word]
                save_dictionaries()
                st.rerun()
    with col2:
        st.markdown(f"### {t['dict_fr']}")
        w = st.text_input(f"{t['dict_word']} (FR)", key="fr_word")
        m = st.text_input(f"{t['dict_meaning']} (FR)", key="fr_meaning")
        if st.button(t['dict_add'], key="add_fr"):
            if w and m:
                st.session_state.dictionaries["fr"][w] = m
                save_dictionaries()
                st.success(f"Added {w}")
                st.rerun()
        for word, meaning in list(st.session_state.dictionaries["fr"].items()):
            st.text(f"{word}: {meaning}")
            if st.button(f"{t['dict_delete']} {word}", key=f"del_fr_{word}"):
                del st.session_state.dictionaries["fr"][word]
                save_dictionaries()
                st.rerun()
    with col3:
        st.markdown(f"### {t['dict_en']}")
        w = st.text_input(f"{t['dict_word']} (EN)", key="en_word")
        m = st.text_input(f"{t['dict_meaning']} (EN)", key="en_meaning")
        if st.button(t['dict_add'], key="add_en"):
            if w and m:
                st.session_state.dictionaries["en"][w] = m
                save_dictionaries()
                st.success(f"Added {w}")
                st.rerun()
        for word, meaning in list(st.session_state.dictionaries["en"].items()):
            st.text(f"{word}: {meaning}")
            if st.button(f"{t['dict_delete']} {word}", key=f"del_en_{word}"):
                del st.session_state.dictionaries["en"][word]
                save_dictionaries()
                st.rerun()

def save_dictionaries():
    with open("dictionaries.json", "w") as f:
        json.dump(st.session_state.dictionaries, f, indent=2)

def save_audio_transcriptions():
    with open("audio_transcriptions.json", "w") as f:
        json.dump(st.session_state.audio_transcriptions, f, indent=2)

def save_encyclopedia():
    with open("encyclopedia.json", "w") as f:
        json.dump(st.session_state.encyclopedia, f, indent=2)

def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    recorder_html = f"""
    <div id="recorder-container">
        <button id="recordBtn" style="background-color:#e94560; border:none; border-radius:30px; padding:8px 16px; color:white;">{t['record_btn']}</button>
        <button id="stopBtn" disabled style="background-color:#555; border:none; border-radius:30px; padding:8px 16px;">{t['stop_btn']}</button>
        <p id="recordingStatus"></p>
        <audio id="audioPlayback" controls style="width:100%; margin-top:10px;"></audio>
        <a id="downloadLink" style="display:block; margin-top:10px; color:#ffaa66;">{t['download_btn']}</a>
    </div>
    <script>
        let mediaRecorder; let audioChunks = [];
        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusP = document.getElementById('recordingStatus');
        const audioPlayback = document.getElementById('audioPlayback');
        const downloadLink = document.getElementById('downloadLink');
        recordBtn.onclick = async () => {{
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => audioChunks.push(event.data);
            mediaRecorder.onstop = () => {{
                const audioBlob = new Blob(audioChunks, {{ type: 'audio/wav' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                audioPlayback.src = audioUrl;
                downloadLink.href = audioUrl;
                downloadLink.download = 'recording.wav';
                downloadLink.style.display = 'block';
                audioChunks = [];
                statusP.innerText = '';
            }};
            mediaRecorder.start();
            recordBtn.disabled = true;
            stopBtn.disabled = false;
            statusP.innerText = 'Recording...';
        }};
        stopBtn.onclick = () => {{
            mediaRecorder.stop();
            recordBtn.disabled = false;
            stopBtn.disabled = true;
            statusP.innerText = 'Stopped. You can download and upload below.';
        }};
    </script>
    """
    st.components.v1.html(recorder_html, height=200)
    st.markdown(f"### 📂 {t['voice_upload']}")
    uploaded_file = st.file_uploader(t['voice_upload'], type=["wav", "mp3"], key="voice_upload")
    transcript = st.text_area(t['voice_transcribed_text'], key="voice_transcript")
    if uploaded_file and transcript.strip():
        if st.button(t['voice_train'], use_container_width=True):
            audio_bytes = uploaded_file.read()
            save_voice_for_text(transcript.strip(), audio_bytes)
            add_to_training(transcript.strip(), t)
            st.success(t['voice_success'])

def translation_correction(t):
    st.markdown(f"## {t['translation_title']}")
    source = st.text_area(t['translation_source_text'], height=100)
    if st.button(t['translate_btn'], use_container_width=True):
        if source.strip():
            url = "https://api.mymemory.translated.net/get"
            params = {"q": source, "langpair": "auto|ht"}
            try:
                r = requests.get(url, params=params, timeout=10)
                res = r.json()
                translated = res.get("responseData", {}).get("translatedText", "")
                if translated:
                    st.session_state.translated = translated
                else:
                    st.warning("Translation failed")
            except Exception as e:
                st.warning(f"Error: {e}")
    if "translated" in st.session_state:
        corrected = st.text_area(t['translation_result'], value=st.session_state.translated, height=100)
        if st.button(t['train_translation_btn'], use_container_width=True):
            if corrected.strip():
                add_to_training(corrected, t)
                st.success("Trained")
            else:
                st.warning(t['warning_no_text'])

def encyclopedia_manager(t):
    st.markdown(f"## {t['encyclopedia_title']}")
    with st.expander(t['encyclopedia_add']):
        title = st.text_input(t['encyclopedia_title_field'])
        content = st.text_area(t['encyclopedia_content'], height=150)
        lang = st.selectbox(t['encyclopedia_lang'], ["English", "Français", "Kreyòl Ayisyen", "Español"])
        tags = st.text_input(t['encyclopedia_tags'])
        if st.button(t['encyclopedia_save'], use_container_width=True):
            if title and content:
                entry = {"title": title, "content": content, "language": lang, "tags": [t.strip() for t in tags.split(",") if t.strip()], "timestamp": time.time()}
                st.session_state.encyclopedia.append(entry)
                save_encyclopedia()
                add_to_training(f"{title}: {content}", t)
                st.success(f"Added {title}")
                st.rerun()
            else:
                st.warning("Title and content required.")
    st.markdown(f"### {t['encyclopedia_list']}")
    for entry in st.session_state.encyclopedia[-10:]:
        with st.expander(f"{entry['title']} ({entry['language']})"):
            st.markdown(f"**Content:** {entry['content']}")
            st.markdown(f"**Tags:** {', '.join(entry['tags'])}")
            if st.button(f"Delete '{entry['title']}'", key=f"del_enc_{entry['timestamp']}"):
                st.session_state.encyclopedia.remove(entry)
                save_encyclopedia()
                st.rerun()

def test_training(t):
    st.markdown(f"## {t['test_title']}")
    q = st.text_input(t['test_question'])
    if st.button(t['test_button'], use_container_width=True):
        if q.strip():
            facts = retrieve_relevant_facts(q, k=1)
            if facts:
                st.session_state.test_answer = facts[0]
            else:
                st.session_state.test_answer = t["no_facts_answer"]
            st.rerun()
    if "test_answer" in st.session_state:
        st.markdown(f"**{t['test_answer_label']}**")
        st.markdown(f'<div style="background:#0f3460; padding:10px; border-radius:12px;">{st.session_state.test_answer}</div>', unsafe_allow_html=True)
        voice_up = st.file_uploader(t['upload_voice_label'], type=["wav", "mp3"], key="test_voice")
        if voice_up:
            save_voice_for_text(st.session_state.test_answer, voice_up.read())
            st.success("Voice saved")
            st.rerun()
        st.components.v1.html(play_voice_button(st.session_state.test_answer, t['test_speak_button'], "test"), height=50)

# ---------- CHAT MODE (clean interface) ----------
def chat_mode_interface():
    t = TEXTS[st.session_state.language]
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['chat_mode_title']}</h1>", unsafe_allow_html=True)
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    for idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                st.components.v1.html(play_voice_button(msg["content"], t['chat_speak_button'], f"chat_{idx}"), height=50)
    
    user_input = st.text_input(t['chat_mode_placeholder'], key="chat_input_new")
    if st.button(t['send_button'], use_container_width=True, key="chat_send_new"):
        if user_input.strip():
            answer = generate_response(user_input)
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"## 🖼️ {t['image_title']}")
    img_file = st.file_uploader(t['image_upload_label'], type=["jpg", "jpeg", "png"], key="chat_image_upload")
    if img_file:
        img = Image.open(img_file)
        st.image(img, caption=t['image_caption'], width=300)
        if st.button(t['image_describe_button'], use_container_width=True):
            description = f"This image shows a {img_file.name}. (Replace with actual image captioning model.)"
            st.session_state.img_description = description
            st.markdown(f"**{t['image_description_result']}** {description}")
            if st.button("Train this description", use_container_width=True):
                add_to_training(description, t)
                st.success("Description added to knowledge.")
            desc_voice = st.file_uploader("Upload voice for this description (WAV/MP3)", type=["wav", "mp3"], key="desc_voice_upload")
            if desc_voice:
                save_voice_for_text(description, desc_voice.read())
                st.success("Voice saved for description")
                st.rerun()
            if "img_description" in st.session_state:
                st.components.v1.html(play_voice_button(description, "🔊 Hear description", "img_desc"), height=50)
    
    if st.button("Clear Chat", use_container_width=True, key="clear_chat_new"):
        st.session_state.chat_messages = []
        st.rerun()

# ---------- TRAINING MODE (full original dashboard) ----------
def training_mode():
    t = TEXTS[st.session_state.language]
    st.markdown(f"<h1 style='text-align:center;'>{t['training_app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t['training_subtitle']}</p>", unsafe_allow_html=True)
    
    st.markdown(f"## {t['chat_title']}")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
    
    user_input = st.text_input(t["chat_input_placeholder"], key="train_chat_input")
    if st.button(t["send_button"], use_container_width=True):
        if user_input.strip():
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.rerun()
    
    st.markdown("---")
    st.markdown(f"## {t['training_text_title']}")
    with st.expander(t["expand_text"]):
        text = st.text_area(t["text_area_label"])
        if st.button(t["train_text_button"], use_container_width=True):
            add_to_training(text, t)
    
    st.markdown(f"## {t['audio_title']}")
    with st.expander(t["expand_audio"]):
        voice_training(t)
    
    st.markdown(f"## {t['image_title']}")
    with st.expander(t["expand_image"]):
        img_file = st.file_uploader(t["image_upload_label"], type=["jpg", "jpeg", "png"])
        desc = st.text_area(t["image_description_label"])
        if img_file:
            st.image(img_file, caption=t['image_caption'], width=200)
            if st.button(t["train_image_button"], use_container_width=True):
                if desc.strip():
                    add_to_training(desc, t)
                else:
                    st.warning(t['warning_no_description'])
    
    st.markdown(f"## {t['file_title']}")
    with st.expander(t["expand_file"]):
        txt_file = st.file_uploader(t["file_upload_label"], type=["txt", "md"])
        if txt_file:
            content = txt_file.read().decode("utf-8")
            st.text_area(t['file_preview'], content, height=150)
            if st.button(t["train_file_button"], use_container_width=True):
                add_to_training(content, t)
    
    st.markdown("---")
    dictionary_manager(t)
    st.markdown("---")
    translation_correction(t)
    st.markdown("---")
    encyclopedia_manager(t)
    st.markdown("---")
    test_training(t)
    
    st.markdown("---")
    st.markdown(f"### {t['knowledge_base'].format(count=len(st.session_state.training_data))}")
    if st.button(t["clear_chat_button"], use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

# ---------- MAIN ----------
def main_app():
    load_previous_training()
    
    if st.session_state.chat_mode:
        chat_mode_interface()
    else:
        training_mode()
    
    t = TEXTS[st.session_state.language]
    st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)

# ---------- ROUTING ----------
if not st.session_state.authenticated:
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.selectbox("🌐 Language", lang_names, key="login_lang")
    st.session_state.language = LANGUAGES[selected_lang_name]
    login_page()
else:
    main_app()
