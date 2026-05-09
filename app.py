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

st.set_page_config(
    page_title="Gesner AI | Your Personal Haitian AI",
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
        "app_title": "🧠 Gesner AI – Train Your Personal Haitian AI",
        "subtitle": "Teach me through chat, text, images, files, voice, dictionaries and encyclopedia.",
        "chat_title": "💬 Chat with Gesner AI",
        "user_prefix": "🧑‍💻 You: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Send",
        "chat_input_placeholder": "Type your message:",
        "training_text_title": "📚 Train Me (Text)",
        "expand_text": "Add a fact or question‑answer pair",
        "text_area_label": "Enter knowledge (e.g., 'Haiti's capital is Port‑au‑Prince')",
        "train_text_button": "Train with this text",
        "audio_title": "🎤 Train Me with Audio",
        "expand_audio": "Record or upload an audio file, then type the transcription to train me.",
        "audio_upload_label": "Upload Audio File",
        "transcribe_label": "Transcribed text (what the audio says):",
        "transcription_textarea": "Type the transcription here",
        "train_transcription_button": "Train with this transcription",
        "record_btn": "🔴 Record Audio",
        "stop_btn": "⏹️ Stop Recording",
        "download_btn": "💾 Download Recording (WAV)",
        "recording_placeholder": "Recording... will appear here",
        "image_title": "🖼️ Train Me with Images",
        "expand_image": "Upload an image + description",
        "image_upload_label": "Choose an image",
        "image_description_label": "Describe what this image teaches",
        "train_image_button": "Train with this image",
        "file_title": "📄 Train Me with Text Files",
        "expand_file": "Upload .txt or .md file",
        "file_upload_label": "Choose a text file",
        "train_file_button": "Train with this file",
        "knowledge_base": "📊 Knowledge Base: {count} facts trained",
        "clear_chat_button": "Clear Chat History",
        "footer": "© GlobalInternet.py – Gesner AI, trained the Haitian way.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Your Personal AI",
        "built_by": "Built by Gesner Deslandes – Coder in Chief",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Website:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Pricing",
        "pricing_table": """
| License | Price |
|---------|-------|
| **Personal** | $49 |
| **Business** | $299 |
| **Enterprise** | $999 |
""",
        "logout_button": "🔓 Logout",
        "no_facts_answer": "I don't have specific training on that yet. Please teach me by using the Train section below! Your question: {question}",
        "with_facts_answer": "{context}",
        "training_success": "✅ Trained: {text}...",
        "warning_no_text": "Please enter some text to train.",
        "warning_no_transcription": "Please enter the transcribed text first.",
        "warning_no_description": "Please add a description to train the AI.",
        "file_preview": "File content (preview)",
        "image_caption": "Uploaded Image",
        "audio_warning": "Audio file uploaded – please type the transcription above.",
        "login_title": "Gesner AI",
        "login_message": "Enter password to train your personal AI",
        "login_button": "Login",
        "wrong_password": "Incorrect password. Access denied.",
        "dict_title": "📖 Dictionaries (Editable)",
        "dict_ht": "Haitian Creole Dictionary",
        "dict_fr": "French Dictionary",
        "dict_en": "English Dictionary",
        "dict_word": "Word",
        "dict_meaning": "Meaning / Translation",
        "dict_add": "Add Entry",
        "dict_edit": "Edit",
        "dict_delete": "Delete",
        "dict_save": "Save Changes",
        "voice_training_title": "🎙️ Voice Training (Teach Me Your Voice)",
        "voice_record": "Record Audio",
        "voice_upload": "Upload Audio File",
        "voice_transcribe": "Transcribe",
        "voice_transcribed_text": "Transcribed Text (edit if needed)",
        "voice_train": "Train with this voice + text",
        "voice_success": "Voice and text trained!",
        "translation_title": "🌍 Translate & Correct Haitian Creole",
        "translation_source_text": "Text to translate (any language)",
        "translation_target": "Haitian Creole",
        "translate_btn": "Translate to Haitian Creole",
        "translation_result": "Translated text (editable)",
        "train_translation_btn": "Train AI with this corrected text",
        "encyclopedia_title": "📚 Encyclopedia",
        "encyclopedia_add": "Add Encyclopedia Entry",
        "encyclopedia_title_field": "Title",
        "encyclopedia_content": "Content",
        "encyclopedia_lang": "Language",
        "encyclopedia_tags": "Tags (comma separated)",
        "encyclopedia_save": "Save Entry",
        "encyclopedia_list": "Existing Entries",
        "voice_download": "Download Recording",
        "test_title": "🧪 Test Your Training",
        "test_question": "Ask a question to see exactly what I learned and hear it spoken:",
        "test_button": "Ask Gesner AI",
        "test_answer_label": "Retrieved Answer (exact text I learned):",
        "test_speak_button": "🔊 Speak Answer",
        "upload_voice_label": "Upload your voice recording of this exact answer (WAV/MP3)"
    },
    "fr": {
        "app_title": "🧠 Gesner IA – Entraînez votre IA personnelle haïtienne",
        "subtitle": "Enseignez‑moi par chat, texte, images, fichiers, voix, dictionnaires et encyclopédie.",
        "chat_title": "💬 Discuter avec Gesner IA",
        "user_prefix": "🧑‍💻 Vous : ",
        "assistant_prefix": "🤖 Gesner IA : ",
        "send_button": "Envoyer",
        "chat_input_placeholder": "Tapez votre message :",
        "training_text_title": "📚 Entraînez‑moi (texte)",
        "expand_text": "Ajoutez un fait ou une paire question‑réponse",
        "text_area_label": "Entrez une connaissance (ex. 'La capitale d'Haïti est Port‑au‑Prince')",
        "train_text_button": "Entraîner avec ce texte",
        "audio_title": "🎤 Entraînez‑moi avec l’audio",
        "expand_audio": "Enregistrez ou téléchargez un fichier audio, puis tapez la transcription.",
        "audio_upload_label": "Télécharger un fichier audio",
        "transcribe_label": "Texte transcrit (ce que dit l’audio) :",
        "transcription_textarea": "Tapez la transcription ici",
        "train_transcription_button": "Entraîner avec cette transcription",
        "record_btn": "🔴 Enregistrer",
        "stop_btn": "⏹️ Arrêter",
        "download_btn": "💾 Télécharger l’enregistrement (WAV)",
        "recording_placeholder": "Enregistrement... apparaîtra ici",
        "image_title": "🖼️ Entraînez‑moi avec des images",
        "expand_image": "Téléchargez une image + description",
        "image_upload_label": "Choisissez une image",
        "image_description_label": "Décrivez ce que cette image enseigne",
        "train_image_button": "Entraîner avec cette image",
        "file_title": "📄 Entraînez‑moi avec des fichiers texte",
        "expand_file": "Téléchargez un fichier .txt ou .md",
        "file_upload_label": "Choisissez un fichier texte",
        "train_file_button": "Entraîner avec ce fichier",
        "knowledge_base": "📊 Base de connaissances : {count} faits appris",
        "clear_chat_button": "Effacer l’historique du chat",
        "footer": "© GlobalInternet.py – Gesner IA, formée à la haïtienne.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner IA – Votre IA personnelle",
        "built_by": "Construit par Gesner Deslandes – Ingénieur en chef",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Site web :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Tarifs",
        "pricing_table": """
| Licence | Prix |
|---------|------|
| **Personnelle** | 49 $ |
| **Entreprise** | 299 $ |
| **Grande entreprise** | 999 $ |
""",
        "logout_button": "🔓 Déconnexion",
        "no_facts_answer": "Je n’ai pas encore d’apprentissage spécifique sur ce sujet. Veuillez m’enseigner en utilisant la section Entraînement ci‑dessous ! Votre question : {question}",
        "with_facts_answer": "{context}",
        "training_success": "✅ Entraîné : {text}...",
        "warning_no_text": "Veuillez saisir du texte à entraîner.",
        "warning_no_transcription": "Veuillez d’abord saisir le texte transcrit.",
        "warning_no_description": "Veuillez ajouter une description pour entraîner l’IA.",
        "file_preview": "Aperçu du fichier",
        "image_caption": "Image téléchargée",
        "audio_warning": "Fichier audio téléchargé – veuillez saisir la transcription ci‑dessus.",
        "login_title": "Gesner IA",
        "login_message": "Entrez le mot de passe pour entraîner votre IA personnelle",
        "login_button": "Se connecter",
        "wrong_password": "Mot de passe incorrect. Accès refusé.",
        "dict_title": "📖 Dictionnaires (éditables)",
        "dict_ht": "Dictionnaire créole haïtien",
        "dict_fr": "Dictionnaire français",
        "dict_en": "Dictionnaire anglais",
        "dict_word": "Mot",
        "dict_meaning": "Signification / Traduction",
        "dict_add": "Ajouter une entrée",
        "dict_edit": "Modifier",
        "dict_delete": "Supprimer",
        "dict_save": "Enregistrer les modifications",
        "voice_training_title": "🎙️ Apprentissage vocal (enseignez‑moi votre voix)",
        "voice_record": "Enregistrer audio",
        "voice_upload": "Télécharger fichier audio",
        "voice_transcribe": "Retranscrire",
        "voice_transcribed_text": "Texte retranscrit (modifiable)",
        "voice_train": "Entraîner avec cette voix + texte",
        "voice_success": "Voix et texte entraînés !",
        "translation_title": "🌍 Traduire et corriger le créole haïtien",
        "translation_source_text": "Texte à traduire (n’importe quelle langue)",
        "translation_target": "Créole haïtien",
        "translate_btn": "Traduire en créole haïtien",
        "translation_result": "Texte traduit (modifiable)",
        "train_translation_btn": "Entraîner l’IA avec ce texte corrigé",
        "encyclopedia_title": "📚 Encyclopédie",
        "encyclopedia_add": "Ajouter une entrée d’encyclopédie",
        "encyclopedia_title_field": "Titre",
        "encyclopedia_content": "Contenu",
        "encyclopedia_lang": "Langue",
        "encyclopedia_tags": "Étiquettes (séparées par des virgules)",
        "encyclopedia_save": "Enregistrer l’entrée",
        "encyclopedia_list": "Entrées existantes",
        "voice_download": "Télécharger l’enregistrement",
        "test_title": "🧪 Tester votre entraînement",
        "test_question": "Posez une question pour voir exactement ce que j’ai appris et l’entendre:",
        "test_button": "Demander à Gesner IA",
        "test_answer_label": "Réponse récupérée (texte exact que j’ai appris) :",
        "test_speak_button": "🔊 Lire la réponse",
        "upload_voice_label": "Téléchargez votre enregistrement vocal de cette réponse exacte (WAV/MP3)"
    },
    "ht": {
        "app_title": "🧠 Gesner AI – Antrene AI Pèsonèl Ayisyen w la",
        "subtitle": "Anseye m atravè chat, tèks, imaj, fichye, vwa, diksyonè ak ansiklopedi.",
        "chat_title": "💬 Pale ak Gesner AI",
        "user_prefix": "🧑‍💻 Ou : ",
        "assistant_prefix": "🤖 Gesner AI : ",
        "send_button": "Voye",
        "chat_input_placeholder": "Tape mesaj ou a :",
        "training_text_title": "📚 Antrene m (tèks)",
        "expand_text": "Ajoute yon reyalite oswa yon kesyon‑repons",
        "text_area_label": "Antre yon konesans (egzanp: 'Kapital Ayiti se Pòtoprens')",
        "train_text_button": "Antrene ak tèks sa a",
        "audio_title": "🎤 Antrene m ak odyo",
        "expand_audio": "Anrejistre oswa chaje yon fichye odyo, epi tape transkripsyon an.",
        "audio_upload_label": "Chaje yon fichye odyo",
        "transcribe_label": "Tèks transkri (sa odyo a di) :",
        "transcription_textarea": "Tape transkripsyon an isit la",
        "train_transcription_button": "Antrene ak transkripsyon sa a",
        "record_btn": "🔴 Anrejistre",
        "stop_btn": "⏹️ Sispann",
        "download_btn": "💾 Telechaje anrejistreman an (WAV)",
        "recording_placeholder": "Anrejistreman... ap parèt isit la",
        "image_title": "🖼️ Antrene m ak imaj",
        "expand_image": "Chaje yon imaj + deskripsyon",
        "image_upload_label": "Chwazi yon imaj",
        "image_description_label": "Dekri sa imaj sa a anseye",
        "train_image_button": "Antrene ak imaj sa a",
        "file_title": "📄 Antrene m ak fichye tèks",
        "expand_file": "Chaje yon fichye .txt oswa .md",
        "file_upload_label": "Chwazi yon fichye tèks",
        "train_file_button": "Antrene ak fichye sa a",
        "knowledge_base": "📊 Baz konesans : {count} reyalite antrene",
        "clear_chat_button": "Efase listorik chat la",
        "footer": "© GlobalInternet.py – Gesner AI, antrene nan fason ayisyen an.",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – AI Pèsonèl ou",
        "built_by": "Konstwi pa Gesner Deslandes – Enjenyè anchèf",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitwèb :",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Pri",
        "pricing_table": """
| Lisans | Pri |
|--------|-----|
| **Pèsonèl** | $49 |
| **Biznis** | $299 |
| **Antrepriz** | $999 |
""",
        "logout_button": "🔓 Dekonekte",
        "no_facts_answer": "Mwen poko genyen antreman espesifik sou sa. Tanpri anseye m nan seksyon Antreman anba a! Kesyon ou a : {question}",
        "with_facts_answer": "{context}",
        "training_success": "✅ Antrene : {text}...",
        "warning_no_text": "Tanpri antre kèk tèks pou antrene.",
        "warning_no_transcription": "Tanpri antre tèks transkri an premye.",
        "warning_no_description": "Tanpri ajoute yon deskripsyon pou antrene AI a.",
        "file_preview": "Aperçu fichye a",
        "image_caption": "Imaj chaje",
        "audio_warning": "Fichye odyo chaje – tanpri tape transkripsyon an pi wo a.",
        "login_title": "Gesner AI",
        "login_message": "Antre modpas pou antrene AI pèsonèl ou",
        "login_button": "Konekte",
        "wrong_password": "Modpas pa bon. Aksè refize.",
        "dict_title": "📖 Diksyonè (kapab modifye)",
        "dict_ht": "Diksyonè Kreyòl Ayisyen",
        "dict_fr": "Diksyonè Franse",
        "dict_en": "Diksyonè Angle",
        "dict_word": "Mo",
        "dict_meaning": "Siyifikasyon / Tradiksyon",
        "dict_add": "Ajoute yon antre",
        "dict_edit": "Modifye",
        "dict_delete": "Efase",
        "dict_save": "Sove chanjman yo",
        "voice_training_title": "🎙️ Antrenman vwa (Anseye m vwa ou)",
        "voice_record": "Anrejistre odyo",
        "voice_upload": "Chaje yon fichye odyo",
        "voice_transcribe": "Transkri",
        "voice_transcribed_text": "Tèks transkri (ou ka modifye l)",
        "voice_train": "Antrene ak vwa sa a + tèks",
        "voice_success": "Vwa ak tèks antrene!",
        "translation_title": "🌍 Tradwi epi korije Kreyòl Ayisyen",
        "translation_source_text": "Tèks pou tradwi (nenpòt lang)",
        "translation_target": "Kreyòl Ayisyen",
        "translate_btn": "Tradwi an Kreyòl Ayisyen",
        "translation_result": "Tèks tradwi (kapab modifye)",
        "train_translation_btn": "Antrene AI ak tèks korije sa a",
        "encyclopedia_title": "📚 Ansiklopedi",
        "encyclopedia_add": "Ajoute yon antre ansiklopedi",
        "encyclopedia_title_field": "Tit",
        "encyclopedia_content": "Kontni",
        "encyclopedia_lang": "Lang",
        "encyclopedia_tags": "Etikèt (separe ak vigil)",
        "encyclopedia_save": "Sove antre a",
        "encyclopedia_list": "Antre ki egziste deja",
        "voice_download": "Telechaje anrejistreman an",
        "test_title": "🧪 Teste fòmasyon ou",
        "test_question": "Pose yon kesyon pou wè egzakteman sa m aprann epi tande l:",
        "test_button": "Mande Gesner AI",
        "test_answer_label": "Repons ki te jwenn (tèks egzak mwen aprann):",
        "test_speak_button": "🔊 Pwononse repons lan",
        "upload_voice_label": "Chaje vwa ou k ap li repons egzak sa a (WAV/MP3)"
    }
}

# ---------- CUSTOM CSS (unchanged) ----------
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
    h1, h2, h3, h4, h5, h6, p, li, div, span, strong, em, .footer {
        color: #ffffff !important;
    }
    .streamlit-expanderHeader {
        background-color: rgba(15,52,96,0.8) !important;
    }
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
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
    }
    .stTextInput input::placeholder, .stTextArea textarea::placeholder {
        color: #cccccc !important;
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
    .stFileUploader div {
        color: white !important;
    }
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
if "language" not in st.session_state:
    st.session_state.language = "en"

# Data structures
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}
    if os.path.exists("dictionaries.json"):
        with open("dictionaries.json", "r") as f:
            st.session_state.dictionaries = json.load(f)
if "audio_transcriptions" not in st.session_state:
    st.session_state.audio_transcriptions = []
    if os.path.exists("audio_transcriptions.json"):
        with open("audio_transcriptions.json", "r") as f:
            st.session_state.audio_transcriptions = json.load(f)
if "encyclopedia" not in st.session_state:
    st.session_state.encyclopedia = []
    if os.path.exists("encyclopedia.json"):
        with open("encyclopedia.json", "r") as f:
            st.session_state.encyclopedia = json.load(f)

def save_dictionaries():
    with open("dictionaries.json", "w") as f:
        json.dump(st.session_state.dictionaries, f, indent=2)

def save_audio_transcriptions():
    with open("audio_transcriptions.json", "w") as f:
        json.dump(st.session_state.audio_transcriptions, f, indent=2)

def save_encyclopedia():
    with open("encyclopedia.json", "w") as f:
        json.dump(st.session_state.encyclopedia, f, indent=2)

def logout():
    st.session_state.authenticated = False
    st.rerun()

def login_page():
    t = TEXTS[st.session_state.language]
    st.markdown(f"""
    <div style="display: flex; justify-content: center; align-items: center; min-height: 80vh;">
        <div class="login-card" style="background: rgba(15,52,96,0.8); backdrop-filter: blur(12px); border-radius: 30px; padding: 2rem; text-align: center; border: 1px solid #e94560; width: 100%; max-width: 450px; margin: auto;">
            <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
            <div class="login-title" style="color: #ffd966; font-size: 2rem; margin-bottom: 1rem;">{t['login_title']}</div>
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

# Pre‑train the introduction text (if not already present)
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
        return t["no_facts_answer"].format(question=user_input)

def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language / Langue / Lang", lang_names)
    st.session_state.language = LANGUAGES[selected_lang_name]
    t = TEXTS[st.session_state.language]

    st.sidebar.markdown("""
    <div style="text-align: center;">
        <div style="font-size:80px; animation:spin 4s linear infinite; display:inline-block;">🌍</div>
    </div>
    <style>
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
    </style>
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
    if st.sidebar.button(t['logout_button'], use_container_width=True):
        logout()

# ---------- DICTIONARY MANAGER ----------
def dictionary_manager(t):
    st.markdown(f"## {t['dict_title']}")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"### {t['dict_ht']}")
        word_ht = st.text_input(f"{t['dict_word']} (HT)", key="dict_ht_word")
        meaning_ht = st.text_input(f"{t['dict_meaning']} (HT)", key="dict_ht_meaning")
        if st.button(t['dict_add'], key="add_ht"):
            if word_ht and meaning_ht:
                st.session_state.dictionaries["ht"][word_ht] = meaning_ht
                save_dictionaries()
                st.success(f"Added {word_ht}")
                st.rerun()
        for w, m in list(st.session_state.dictionaries["ht"].items()):
            st.text(f"{w}: {m}")
            if st.button(f"{t['dict_delete']} {w}", key=f"del_ht_{w}"):
                del st.session_state.dictionaries["ht"][w]
                save_dictionaries()
                st.rerun()
    with col2:
        st.markdown(f"### {t['dict_fr']}")
        word_fr = st.text_input(f"{t['dict_word']} (FR)", key="dict_fr_word")
        meaning_fr = st.text_input(f"{t['dict_meaning']} (FR)", key="dict_fr_meaning")
        if st.button(t['dict_add'], key="add_fr"):
            if word_fr and meaning_fr:
                st.session_state.dictionaries["fr"][word_fr] = meaning_fr
                save_dictionaries()
                st.success(f"Added {word_fr}")
                st.rerun()
        for w, m in list(st.session_state.dictionaries["fr"].items()):
            st.text(f"{w}: {m}")
            if st.button(f"{t['dict_delete']} {w}", key=f"del_fr_{w}"):
                del st.session_state.dictionaries["fr"][w]
                save_dictionaries()
                st.rerun()
    with col3:
        st.markdown(f"### {t['dict_en']}")
        word_en = st.text_input(f"{t['dict_word']} (EN)", key="dict_en_word")
        meaning_en = st.text_input(f"{t['dict_meaning']} (EN)", key="dict_en_meaning")
        if st.button(t['dict_add'], key="add_en"):
            if word_en and meaning_en:
                st.session_state.dictionaries["en"][word_en] = meaning_en
                save_dictionaries()
                st.success(f"Added {word_en}")
                st.rerun()
        for w, m in list(st.session_state.dictionaries["en"].items()):
            st.text(f"{w}: {m}")
            if st.button(f"{t['dict_delete']} {w}", key=f"del_en_{w}"):
                del st.session_state.dictionaries["en"][w]
                save_dictionaries()
                st.rerun()

# ---------- VOICE TRAINING ----------
def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    st.markdown("### 🎙️ Record directly in your browser")
    recorder_html = f"""
    <div id="recorder-container">
        <button id="recordBtn" style="background-color:#e94560; border:none; border-radius:30px; padding:8px 16px; color:white; font-weight:bold; cursor:pointer;">{t['record_btn']}</button>
        <button id="stopBtn" disabled style="background-color:#555; border:none; border-radius:30px; padding:8px 16px; color:white; margin-left:10px; cursor:pointer;">{t['stop_btn']}</button>
        <p id="recordingStatus" style="color:white;"></p>
        <audio id="audioPlayback" controls style="width:100%; margin-top:10px;"></audio>
        <a id="downloadLink" style="display:block; margin-top:10px; color:#ffaa66;">{t['voice_download']}</a>
    </div>
    <script>
        let mediaRecorder;
        let audioChunks = [];

        const recordBtn = document.getElementById('recordBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusP = document.getElementById('recordingStatus');
        const audioPlayback = document.getElementById('audioPlayback');
        const downloadLink = document.getElementById('downloadLink');

        recordBtn.onclick = async () => {{
            const stream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.ondataavailable = event => {{
                audioChunks.push(event.data);
            }};
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
            statusP.innerText = 'Recording stopped. You can download the file and then upload it below.';
        }};
    </script>
    """
    st.components.v1.html(recorder_html, height=200)
    
    st.markdown("### 📂 Or upload an audio file")
    audio_file = st.file_uploader(t['audio_upload_label'], type=["wav", "mp3", "m4a", "ogg", "webm"], key="audio_train")
    if audio_file:
        st.audio(audio_file, format="audio/wav")
        transcribed_text = st.text_area(t['voice_transcribed_text'], height=100, key="audio_transcript")
        if st.button(t['voice_train'], key="train_audio"):
            if transcribed_text.strip():
                add_to_training(transcribed_text, t)
                audio_bytes = audio_file.read()
                audio_filename = f"voice_{int(time.time())}.wav"
                with open(audio_filename, "wb") as f:
                    f.write(audio_bytes)
                st.session_state.audio_transcriptions.append({
                    "file": audio_filename,
                    "text": transcribed_text,
                    "timestamp": time.time()
                })
                save_audio_transcriptions()
                st.success(t['voice_success'])
            else:
                st.warning(t['warning_no_transcription'])

# ---------- TRANSLATION AND CORRECTION ----------
def translation_and_correction(t):
    st.markdown(f"## {t['translation_title']}")
    source_text = st.text_area(t['translation_source_text'], height=100)
    if st.button(t['translate_btn']):
        if source_text.strip():
            url = "https://api.mymemory.translated.net/get"
            params = {"q": source_text, "langpair": f"auto|ht"}
            try:
                response = requests.get(url, params=params, timeout=10)
                result = response.json()
                translated = result.get("responseData", {}).get("translatedText", "")
                if translated:
                    st.session_state.translated_text = translated
                else:
                    st.warning("Translation failed. Please check your internet.")
            except Exception as e:
                st.warning(f"Translation error: {e}")
        else:
            st.warning("Please enter text to translate.")
    if "translated_text" in st.session_state:
        corrected = st.text_area(t['translation_result'], value=st.session_state.translated_text, height=100)
        if st.button(t['train_translation_btn']):
            if corrected.strip():
                add_to_training(corrected, t)
                st.success(f"Trained: {corrected[:100]}...")
            else:
                st.warning(t['warning_no_text'])

# ---------- ENCYCLOPEDIA MANAGER ----------
def encyclopedia_manager(t):
    st.markdown(f"## {t['encyclopedia_title']}")
    with st.expander(t['encyclopedia_add']):
        title = st.text_input(t['encyclopedia_title_field'])
        content = st.text_area(t['encyclopedia_content'], height=150)
        lang = st.selectbox(t['encyclopedia_lang'], ["English", "Français", "Kreyòl Ayisyen", "Español"])
        tags = st.text_input(t['encyclopedia_tags'], help="Comma separated")
        if st.button(t['encyclopedia_save']):
            if title and content:
                entry = {
                    "title": title,
                    "content": content,
                    "language": lang,
                    "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                    "timestamp": time.time()
                }
                st.session_state.encyclopedia.append(entry)
                save_encyclopedia()
                add_to_training(f"{title}: {content}", t)
                st.success(f"Encyclopedia entry '{title}' added and trained!")
                st.rerun()
            else:
                st.warning("Title and content are required.")
    st.markdown(f"### {t['encyclopedia_list']}")
    for entry in st.session_state.encyclopedia[-10:]:
        with st.expander(f"{entry['title']} ({entry['language']})"):
            st.markdown(f"**Content:** {entry['content']}")
            st.markdown(f"**Tags:** {', '.join(entry['tags'])}")
            if st.button(f"Delete '{entry['title']}'", key=f"del_enc_{entry['timestamp']}"):
                st.session_state.encyclopedia.remove(entry)
                save_encyclopedia()
                st.rerun()

# ---------- TEST TRAINING SECTION (with own voice upload) ----------
def test_training(t):
    st.markdown(f"## {t['test_title']}")
    test_question = st.text_input(t['test_question'], key="test_question")
    if st.button(t['test_button'], use_container_width=True, key="test_btn"):
        if test_question.strip():
            facts = retrieve_relevant_facts(test_question, k=1)
            if facts:
                st.session_state.test_answer = facts[0]
            else:
                st.session_state.test_answer = t["no_facts_answer"].format(question=test_question)
            st.rerun()
    if "test_answer" in st.session_state:
        st.markdown(f"**{t['test_answer_label']}**")
        st.markdown(f'<div class="chat-message assistant-message" style="background:#0f3460;">{st.session_state.test_answer}</div>', unsafe_allow_html=True)
        
        # File uploader for user's own voice recording
        uploaded_voice = st.file_uploader(t['upload_voice_label'], type=["wav", "mp3"], key="test_voice_upload")
        if uploaded_voice is not None:
            st.session_state.user_voice_bytes = uploaded_voice.read()
            st.success("Voice recording loaded. Click the button below to play it.")
        
        # Speak button: if custom voice is uploaded, play that; else use speech synthesis
        speak_button_html = f"""
        <button id="speakAnswerBtn" style="background-color:#e94560; border:none; border-radius:30px; padding:8px 16px; color:white; font-weight:bold; cursor:pointer; margin-top:10px;">{t['test_speak_button']}</button>
        <script>
            let userVoiceBlob = null;
            {"userVoiceBlob = new Blob([" + str(st.session_state.get("user_voice_bytes", b"")) + "], {type: 'audio/wav'});" if st.session_state.get("user_voice_bytes") else ""}
            document.getElementById('speakAnswerBtn').onclick = () => {{
                if (userVoiceBlob) {{
                    const audioUrl = URL.createObjectURL(userVoiceBlob);
                    const audio = new Audio(audioUrl);
                    audio.play();
                }} else {{
                    const text = {json.dumps(st.session_state.test_answer)};
                    const utterance = new SpeechSynthesisUtterance(text);
                    window.speechSynthesis.cancel();
                    window.speechSynthesis.speak(utterance);
                }}
            }};
        </script>
        """
        st.components.v1.html(speak_button_html, height=50)

# ---------- MAIN APP ----------
def main_app():
    t = TEXTS[st.session_state.language]
    show_sidebar()
    load_previous_training()

    st.markdown(f"<h1 style='text-align:center;'>{t['app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t['subtitle']}</p>", unsafe_allow_html=True)

    # --- Chat Interface ---
    st.markdown(f"## {t['chat_title']}")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input(t["chat_input_placeholder"], key="chat_input")
    if st.button(t["send_button"], use_container_width=True):
        if user_input.strip():
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            response = generate_response(user_input)
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.rerun()

    # --- Existing Training Sections ---
    st.markdown("---")
    st.markdown(f"## {t['training_text_title']}")
    with st.expander(t["expand_text"]):
        training_text = st.text_area(t["text_area_label"])
        if st.button(t["train_text_button"], use_container_width=True):
            add_to_training(training_text, t)

    st.markdown(f"## {t['audio_title']}")
    with st.expander(t["expand_audio"]):
        voice_training(t)

    st.markdown(f"## {t['image_title']}")
    with st.expander(t["expand_image"]):
        image_file = st.file_uploader(t["image_upload_label"], type=["jpg", "jpeg", "png"])
        image_description = st.text_area(t["image_description_label"])
        if image_file is not None:
            st.image(image_file, caption=t['image_caption'], width=200)
            if st.button(t["train_image_button"], use_container_width=True):
                if image_description:
                    add_to_training(image_description, t)
                else:
                    st.warning(t['warning_no_description'])

    st.markdown(f"## {t['file_title']}")
    with st.expander(t["expand_file"]):
        text_file = st.file_uploader(t["file_upload_label"], type=["txt", "md"])
        if text_file is not None:
            content = text_file.read().decode("utf-8")
            st.text_area(t['file_preview'], content, height=150)
            if st.button(t["train_file_button"], use_container_width=True):
                add_to_training(content, t)

    # --- New Features ---
    st.markdown("---")
    dictionary_manager(t)
    st.markdown("---")
    translation_and_correction(t)
    st.markdown("---")
    encyclopedia_manager(t)
    st.markdown("---")
    test_training(t)

    st.markdown("---")
    st.markdown(f"### {t['knowledge_base'].format(count=len(st.session_state.training_data))}")
    if st.button(t["clear_chat_button"], use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

    st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)

# ---------- ROUTING ----------
if not st.session_state.authenticated:
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.selectbox("🌐 Language / Langue / Lang", lang_names, key="login_lang")
    st.session_state.language = LANGUAGES[selected_lang_name]
    login_page()
else:
    main_app()
