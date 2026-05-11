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
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- UPDATED CSS (no white background, dark theme everywhere) ----------
st.markdown(
    """
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
    /* Dark backgrounds for inputs and selects */
    .stTextInput input, .stTextArea textarea,
    .stSelectbox select, .stSelectbox div[data-baseweb="select"] > div,
    .stSelectbox div[data-baseweb="select"] input,
    .stNumberInput input, .stDateInput input {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    /* Code blocks (for the 32 letters) */
    .stCodeBlock, .stCodeBlock div, pre, code {
        background-color: #0f3460 !important;
        color: #ffffff !important;
        border-radius: 12px;
        padding: 0.5rem;
    }
    /* Expander headers */
    .stExpanderHeader {
        background-color: rgba(15,52,96,0.9) !important;
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
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

# ========================= FULL TEXTS DICTIONARY (all 4 languages) =========================
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
        "voice_training_title": "🎙️ Voice Training (Kreyòl only)",
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
        "upload_voice_label": "Upload your voice for this exact text (Kreyòl only)",
        "chat_mode_title": "💬 Gesner AI Chat",
        "chat_mode_placeholder": "Ask me anything...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Upload voice for this answer",
        "image_upload_label": "📷 Upload image",
        "image_describe_button": "Describe",
        "image_description_result": "Description:",
        "toggle_chat_mode": "Chat Mode",
        "phonics_title": "🔊 Phonics Training (32 Letters)",
        "phonics_example": "Example word/sentence for letter {letter}",
        "phonics_add": "Teach example",
        "manage_facts": "📚 Manage Trained Facts"
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
        "voice_training_title": "🎙️ Entraînement vocal (Kreyòl seulement)",
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
        "upload_voice_label": "Téléchargez votre voix pour ce texte exact (Kreyòl seulement)",
        "chat_mode_title": "💬 Gesner IA Chat",
        "chat_mode_placeholder": "Demandez‑moi n'importe quoi...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Téléchargez votre voix pour cette réponse",
        "image_upload_label": "📷 Télécharger une image",
        "image_describe_button": "Décrire",
        "image_description_result": "Description :",
        "toggle_chat_mode": "Mode Chat",
        "phonics_title": "🔊 Entraînement phonétique (32 lettres)",
        "phonics_example": "Exemple de mot/phrase pour la lettre {letter}",
        "phonics_add": "Enseigner l'exemple",
        "manage_facts": "📚 Gérer les faits appris"
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
        "voice_training_title": "🎙️ Fòmasyon vwa (Kreyòl sèlman)",
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
        "upload_voice_label": "Chaje vwa ou pou tèks egzak sa a (Kreyòl sèlman)",
        "chat_mode_title": "💬 Gesner AI Chat",
        "chat_mode_placeholder": "Pose yon kesyon...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Chaje vwa ou pou repons sa a",
        "image_upload_label": "📷 Chaje yon imaj",
        "image_describe_button": "Dekri",
        "image_description_result": "Deskripsyon :",
        "toggle_chat_mode": "Mòd Chat",
        "phonics_title": "🔊 Fòmasyon Fònètik (32 Let)",
        "phonics_example": "Egzanp mo/fraz pou let {letter}",
        "phonics_add": "Ansègne egzanp",
        "manage_facts": "📚 Jere Reyalite Aprann"
    },
    "es": {
        "training_app_title": "🧠 Gesner AI – Centro de Entrenamiento",
        "training_subtitle": "Enséñame hechos, diccionarios, enciclopedia.",
        "chat_title": "💬 Gesner AI Chat",
        "user_prefix": "🧑‍💻 Tú: ",
        "assistant_prefix": "🤖 Gesner AI: ",
        "send_button": "Enviar",
        "chat_input_placeholder": "Pregúntame cualquier cosa...",
        "training_text_title": "📚 Entréneme (Texto)",
        "expand_text": "Añadir un hecho o par pregunta/respuesta",
        "text_area_label": "Ingrese el conocimiento",
        "train_text_button": "Entrenar",
        "audio_title": "🎤 Entréneme con audio",
        "expand_audio": "Grabar o subir audio + transcripción",
        "audio_upload_label": "Subir archivo de audio",
        "transcribe_label": "Texto transcrito:",
        "transcription_textarea": "Escriba la transcripción",
        "train_transcription_button": "Entrenar",
        "record_btn": "🔴 Grabar",
        "stop_btn": "⏹️ Detener",
        "download_btn": "💾 Descargar",
        "image_title": "🖼️ Entréneme con imágenes",
        "expand_image": "Subir imagen + descripción",
        "image_upload_label": "Elegir una imagen",
        "image_description_label": "Describa esta imagen",
        "train_image_button": "Entrenar",
        "file_title": "📄 Entréneme con archivos de texto",
        "expand_file": "Subir archivo .txt o .md",
        "file_upload_label": "Elegir un archivo",
        "train_file_button": "Entrenar",
        "knowledge_base": "📊 Base de conocimiento: {count} hechos entrenados",
        "clear_chat_button": "Borrar historial",
        "footer": "© GlobalInternet.py – Gesner AI",
        "sidebar_company": "GlobalInternet.py",
        "sidebar_product": "Gesner AI – Tu IA personal",
        "built_by": "Gesner Deslandes – Codificador Jefe",
        "phone": "📞 (509)-47385663",
        "email": "✉️ deslandes78@gmail.com",
        "website_label": "🌐 Sitio web:",
        "website_link": "https://globalinternetsitepy-abh7v6tnmskxxnuplrdcgk.streamlit.app/",
        "pricing_title": "💰 Licencia",
        "pricing_table": """
| Licencia | Precio (único) |
|----------|----------------|
| **Personal** | $49 |
| **Negocios** | $299 |
| **Empresa / Código fuente** | $999 |
""",
        "logout_button": "🔓 Cerrar sesión",
        "no_facts_answer": "No sé eso todavía. ¡Enséñame en el Modo Entrenamiento!",
        "with_facts_answer": "{context}",
        "training_success": "✅ Entrenado: {text}...",
        "warning_no_text": "Por favor ingrese texto.",
        "warning_no_transcription": "Primero ingrese el texto transcrito.",
        "warning_no_description": "Por favor añada una descripción.",
        "file_preview": "Vista previa del archivo",
        "image_caption": "Imagen subida",
        "login_title": "Gesner AI",
        "login_message": "Ingrese la contraseña para acceder a Gesner AI",
        "login_button": "Iniciar sesión",
        "wrong_password": "Contraseña incorrecta.",
        "dict_title": "📖 Diccionarios",
        "dict_ht": "Kreyòl Ayisyen",
        "dict_fr": "Français",
        "dict_en": "English",
        "dict_word": "Palabra",
        "dict_meaning": "Significado",
        "dict_add": "Añadir entrada",
        "dict_delete": "Eliminar",
        "voice_training_title": "🎙️ Entrenamiento de voz (solo Kreyòl)",
        "voice_upload": "Subir voz (WAV/MP3)",
        "voice_transcribed_text": "Texto hablado en el audio",
        "voice_train": "Entrenar voz + texto",
        "voice_success": "¡Voz y texto guardados!",
        "translation_title": "🌍 Traducir y corregir",
        "translation_source_text": "Texto a traducir (cualquier idioma)",
        "translate_btn": "Traducir a Kreyòl",
        "translation_result": "Texto traducido (editable)",
        "train_translation_btn": "Entrenar con este texto",
        "encyclopedia_title": "📚 Enciclopedia",
        "encyclopedia_add": "Añadir entrada",
        "encyclopedia_title_field": "Título",
        "encyclopedia_content": "Contenido",
        "encyclopedia_lang": "Idioma",
        "encyclopedia_tags": "Etiquetas (coma)",
        "encyclopedia_save": "Guardar entrada",
        "encyclopedia_list": "Entradas existentes",
        "voice_download": "Descargar grabación",
        "test_title": "🧪 Prueba de entrenamiento",
        "test_question": "Haz una pregunta para recuperar el hecho almacenado",
        "test_button": "Probar",
        "test_answer_label": "Hecho almacenado:",
        "test_speak_button": "🔊 Hablar",
        "upload_voice_label": "Sube tu voz para este texto exacto (solo Kreyòl)",
        "chat_mode_title": "💬 Gesner AI Chat",
        "chat_mode_placeholder": "Pregúntame cualquier cosa...",
        "chat_speak_button": "🔊",
        "chat_upload_voice": "Sube tu voz para esta respuesta",
        "image_upload_label": "📷 Subir imagen",
        "image_describe_button": "Describir",
        "image_description_result": "Descripción:",
        "toggle_chat_mode": "Modo Chat",
        "phonics_title": "🔊 Entrenamiento Fonético (32 letras)",
        "phonics_example": "Ejemplo de palabra/frase para la letra {letter}",
        "phonics_add": "Enseñar ejemplo",
        "manage_facts": "📚 Gestionar hechos aprendidos"
    }
}
# ====================================================================

# ---------- SESSION STATE INITIALISATION ----------
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
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = False
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}
if "audio_transcriptions" not in st.session_state:
    st.session_state.audio_transcriptions = []
if "encyclopedia" not in st.session_state:
    st.session_state.encyclopedia = []
if "chat_language" not in st.session_state:
    st.session_state.chat_language = "ht"
if "phonics" not in st.session_state:
    st.session_state.phonics = {}
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

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

# ---------- HYBRID RETRIEVAL (FAISS + TF-IDF) ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_facts_hybrid(query, k=3):
    semantic_results = retrieve_relevant_facts(query, k=k, threshold=1.0)
    if not semantic_results:
        semantic_results = []
    keyword_results = []
    if st.session_state.tfidf_vectorizer and st.session_state.tfidf_matrix:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1:
                keyword_results.append(st.session_state.texts[idx])
    combined = list(dict.fromkeys(semantic_results + keyword_results))
    return combined[:k]

def retrieve_relevant_facts(query, k=3, threshold=0.8):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < threshold:
            results.append(st.session_state.texts[idx])
    return results

def french_web_search(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(f"{query} site:fr OR lang:fr", max_results=1))
            if results:
                return results[0]['body']
            else:
                return "Désolé, je n'ai pas trouvé d'information en ligne pour cette question. Pouvez-vous reformuler ?"
    except Exception as e:
        return f"Erreur de recherche : {str(e)}. Veuillez réessayer."

def apply_phonics(text):
    rules = {
        r'qu': 'k',
        r'c([aeiou])': r'k\1',
        r'ç': 's',
        r'é': 'e',
        r'è': 'e',
        r'ê': 'e',
        r'î': 'i',
        r'ô': 'o',
        r'û': 'u',
        r'à': 'a',
        r'ù': 'u',
        r'œ': 'oe',
        r'æ': 'ae',
        r'ph': 'f',
        r'th': 't',
        r'([^aeiouy])y([aeiou])': r'\1i\2',
        r'-tion$': 'syon',
    }
    corrected = text.lower()
    for pattern, repl in rules.items():
        corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)
    return corrected[0].upper() + corrected[1:] if corrected else ""

def generate_answer_from_training(query, target_lang):
    best_facts = retrieve_facts_hybrid(query, k=3)
    if best_facts:
        return best_facts[0], False, None
    if target_lang == "ht":
        corrected = apply_phonics(query)
        if corrected != query:
            return f"Mw te aprann ou ta dwe ekri: '{corrected}'. M ap kontinye aprann.", True, "ht"
        else:
            return "Mwen poko genyen repons lan. Tanpri moutre mwen lè w anseye m nan Sant Fòmasyon.", True, "ht"
    else:
        fallback_map = {
            "en": "I haven't learned that yet. Please teach me in the Training Center.",
            "fr": "Je n'ai pas encore appris cela. Enseignez‑moi dans le Centre d'entraînement.",
            "es": "Todavía no he aprendido eso. Enséñame en el Centro de Entrenamiento."
        }
        return fallback_map.get(target_lang, "I don't know that yet. Please teach me."), True, target_lang

def generate_response(user_input, target_lang):
    return generate_answer_from_training(user_input, target_lang)

def play_voice_button(text, is_fallback, fallback_audio_lang, button_label="🔊", key_suffix=""):
    if is_fallback:
        lang_map = {"en": "en-US", "fr": "fr-FR", "es": "es-ES", "ht": "fr-FR"}
        tts_lang = lang_map.get(fallback_audio_lang, "en-US")
        safe_text = json.dumps(text)
        html = f"""
        <button class="speak-btn" id="ttsBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <script>
            document.getElementById('ttsBtn_{key_suffix}').onclick = () => {{
                const utterance = new SpeechSynthesisUtterance({safe_text});
                utterance.lang = '{tts_lang}';
                window.speechSynthesis.cancel();
                window.speechSynthesis.speak(utterance);
            }};
        </script>
        """
        return html
    else:
        voice_bytes = get_voice_for_text(text)
        if voice_bytes:
            audio_b64 = base64.b64encode(voice_bytes).decode()
            mime = "audio/wav"
            html = f"""
            <button class="speak-btn" id="voiceBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
            <audio id="customAudio_{key_suffix}" style="display:none;"></audio>
            <script>
                (function() {{
                    const audioData = "{audio_b64}";
                    const binaryStr = atob(audioData);
                    const bytes = new Uint8Array(binaryStr.length);
                    for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
                    const audioBlob = new Blob([bytes], {{ type: '{mime}' }});
                    const audioUrl = URL.createObjectURL(audioBlob);
                    const audioEl = document.getElementById('customAudio_{key_suffix}');
                    audioEl.src = audioUrl;
                    document.getElementById('voiceBtn_{key_suffix}').onclick = () => audioEl.play();
                }})();
            </script>
            """
            return html
        else:
            return ""

def add_to_training(text, t):
    if not text.strip():
        st.warning(t['warning_no_text'])
        return False
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
    build_tfidf()
    st.success(t['training_success'].format(text=text[:100]))
    return True

def update_training_item(idx, new_text, t):
    if not new_text.strip():
        st.warning(t['warning_no_text'])
        return False
    embedding = st.session_state.embedding_model.encode([new_text])[0]
    st.session_state.training_data[idx] = {"text": new_text, "embedding": embedding.tolist()}
    st.session_state.texts = [item["text"] for item in st.session_state.training_data]
    if st.session_state.texts:
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        build_tfidf()
    else:
        st.session_state.index = None
        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    st.success(f"✅ Updated: {new_text[:100]}...")
    return True

def delete_training_item(idx):
    st.session_state.training_data.pop(idx)
    st.session_state.texts = [item["text"] for item in st.session_state.training_data]
    if st.session_state.texts:
        embeddings = [np.array(item["embedding"], dtype=np.float32) for item in st.session_state.training_data]
        dim = len(embeddings[0])
        st.session_state.index = faiss.IndexFlatL2(dim)
        st.session_state.index.add(np.array(embeddings))
        build_tfidf()
    else:
        st.session_state.index = None
        st.session_state.tfidf_vectorizer = None
        st.session_state.tfidf_matrix = None
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    st.success(f"🗑️ Deleted")

def load_previous_training():
    if os.path.exists("training_data.json"):
        try:
            with open("training_data.json", "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                st.session_state.training_data = data
                if data:
                    st.session_state.texts = [item["text"] for item in data]
                    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data]
                    dim = len(embeddings[0])
                    st.session_state.index = faiss.IndexFlatL2(dim)
                    st.session_state.index.add(np.array(embeddings))
                    build_tfidf()
        except Exception:
            pass

# Pre‑train intro text
intro_text_ht = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
if intro_text_ht not in [item["text"] for item in st.session_state.training_data]:
    embedding = st.session_state.embedding_model.encode([intro_text_ht])[0]
    st.session_state.training_data.append({"text": intro_text_ht, "embedding": embedding.tolist()})
    st.session_state.texts = [intro_text_ht]
    dim = len(embedding)
    st.session_state.index = faiss.IndexFlatL2(dim)
    st.session_state.index.add(np.array([embedding], dtype=np.float32))
    build_tfidf()
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)

def login_page():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS[ui_lang]
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

def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language", lang_names, key="main_lang_selector")
    selected_lang_code = LANGUAGES[selected_lang_name]
    st.session_state.ui_language = selected_lang_code
    st.session_state.chat_language = selected_lang_code
    t = TEXTS[st.session_state.ui_language]

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
    
    chat_mode_toggle = st.sidebar.toggle(t['toggle_chat_mode'], value=st.session_state.chat_mode, key="chat_mode_toggle")
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

def save_encyclopedia():
    with open("encyclopedia.json", "w") as f:
        json.dump(st.session_state.encyclopedia, f, indent=2)

def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    st.info("🎙️ Upload your voice for Kreyòl phrases. Gesner AI will use your exact voice when answering those sentences.")
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
            try {{
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
            }} catch (err) {{
                statusP.innerText = 'Microphone access denied or error: ' + err.message;
            }}
        }};
        stopBtn.onclick = () => {{
            if (mediaRecorder && mediaRecorder.state === 'recording') {{
                mediaRecorder.stop();
                recordBtn.disabled = false;
                stopBtn.disabled = true;
                statusP.innerText = 'Stopped. Click Download to save file, then upload below.';
            }}
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
            target_lang = st.session_state.chat_language
            answer, is_fallback, fallback_lang = generate_response(q, target_lang)
            st.session_state.test_answer = answer
            st.session_state.test_is_fallback = is_fallback
            st.session_state.test_fallback_lang = fallback_lang
            st.rerun()
    if "test_answer" in st.session_state:
        st.markdown(f"**{t['test_answer_label']}**")
        st.markdown(f'<div style="background:#0f3460; padding:10px; border-radius:12px;">{st.session_state.test_answer}</div>', unsafe_allow_html=True)
        if not st.session_state.test_is_fallback:
            voice_up = st.file_uploader(t['upload_voice_label'], type=["wav", "mp3"], key="test_voice")
            if voice_up:
                save_voice_for_text(st.session_state.test_answer, voice_up.read())
                st.success("Voice saved")
                st.rerun()
        btn_html = play_voice_button(
            st.session_state.test_answer,
            st.session_state.test_is_fallback,
            st.session_state.test_fallback_lang,
            t['test_speak_button'],
            "test"
        )
        if btn_html:
            st.components.v1.html(btn_html, height=50)
        elif not st.session_state.test_is_fallback:
            st.info("No voice recorded for this answer. You can upload your voice above.")

def chat_mode_interface():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS[ui_lang]
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['chat_mode_title']}</h1>", unsafe_allow_html=True)
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    for idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message" style="width:100%;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                btn_html = play_voice_button(
                    msg["content"],
                    msg.get("is_fallback", False),
                    msg.get("fallback_lang"),
                    t['chat_speak_button'],
                    f"chat_{idx}"
                )
                if btn_html:
                    st.components.v1.html(btn_html, height=50)
    user_input = st.text_input(t['chat_mode_placeholder'], key="chat_input_new")
    if st.button(t['send_button'], use_container_width=True, key="chat_send_new"):
        if user_input.strip():
            target_lang = st.session_state.chat_language
            answer, is_fallback, fallback_lang = generate_response(user_input, target_lang)
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "fallback_lang": fallback_lang
            })
            st.rerun()
    if st.button("Clear Chat", use_container_width=True, key="clear_chat_new"):
        st.session_state.chat_messages = []
        st.rerun()

def phonics_training(t):
    st.subheader(t.get("phonics_title", "🔊 Phonics Training (32 Letters)"))
    col1, col2 = st.columns([1,2])
    with col1:
        st.markdown("**32 Letters:**")
        st.code("A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z")
    with col2:
        all_letters = ["A","AN","B","CH","D","E","È","EN","F","G","H","I","J","K","L","M","N","NG","O","Ò","ON","OU","OUN","P","R","S","T","UI","V","W","Y","Z"]
        letter = st.selectbox("Choose a letter", all_letters, key="phonics_letter")
        example = st.text_input(t.get("phonics_example", f"Example word/sentence that starts with '{letter}'"), key="phonics_example_input")
        if st.button(t.get("phonics_add", "Teach example"), key="phonics_add_btn"):
            if example:
                if letter not in st.session_state.phonics:
                    st.session_state.phonics[letter] = []
                st.session_state.phonics[letter].append(example)
                add_to_training(example, t)
                st.success(f"Gesner AI learned '{example}' for letter '{letter}'")
                st.rerun()
    st.subheader("📚 What Gesner AI has learned about phonics")
    if st.session_state.phonics:
        for l, examples in st.session_state.phonics.items():
            with st.expander(f"Letter {l}"):
                for ex in examples:
                    st.write(f"• {ex}")
    else:
        st.info("No phonics examples taught yet.")

def training_mode():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS[ui_lang]
    st.markdown(f"<h1 style='text-align:center;'>{t['training_app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t['training_subtitle']}</p>", unsafe_allow_html=True)

    # Sanitize conversation history
    clean_history = []
    for msg in st.session_state.conversation_history:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            clean_history.append(msg)
    st.session_state.conversation_history = clean_history

    st.markdown(f"## {t['chat_title']}")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)

    user_input = st.text_input(t["chat_input_placeholder"], key="train_chat_input")
    if st.button(t["send_button"], use_container_width=True):
        if user_input.strip():
            target_lang = st.session_state.chat_language
            answer, is_fallback, fallback_lang = generate_response(user_input, target_lang)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
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
    phonics_training(t)
    st.markdown("---")

    # Manage Trained Facts
    st.markdown(f"## {t.get('manage_facts', '📚 Manage Trained Facts')}")
    if not st.session_state.training_data:
        st.info("No facts trained yet. Use the training tools above.")
    else:
        for idx, item in enumerate(st.session_state.training_data):
            original = item["text"]
            with st.expander(f"Fact #{idx+1}: {original[:60]}..."):
                new_text = st.text_area(f"Edit text", value=original, key=f"edit_{idx}")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("✏️ Save", key=f"save_{idx}"):
                        if new_text.strip() and new_text != original:
                            update_training_item(idx, new_text, t)
                            st.rerun()
                        elif not new_text.strip():
                            st.warning("Text cannot be empty.")
                        else:
                            st.info("No changes made.")
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{idx}"):
                        delete_training_item(idx)
                        st.rerun()
                voice_exists = get_voice_for_text(original) is not None
                st.caption("🔊 Voice file exists" if voice_exists else "🔇 No voice file")

    st.markdown("---")
    st.markdown(f"### {t['knowledge_base'].format(count=len(st.session_state.training_data))}")
    if st.button(t["clear_chat_button"], use_container_width=True):
        st.session_state.conversation_history = []
        st.rerun()

def main_app():
    load_previous_training()
    show_sidebar()
    if st.session_state.chat_mode:
        chat_mode_interface()
    else:
        training_mode()
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS[ui_lang]
    st.markdown(f'<div class="footer">{t["footer"]}</div>', unsafe_allow_html=True)

# ---------- ROUTING ----------
if "ui_language" not in st.session_state:
    st.session_state.ui_language = "en"
if "chat_language" not in st.session_state:
    st.session_state.chat_language = "en"

if not st.session_state.authenticated:
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.selectbox("🌐 Language", lang_names, key="login_lang")
    st.session_state.ui_language = LANGUAGES[selected_lang_name]
    st.session_state.chat_language = st.session_state.ui_language
    login_page()
else:
    main_app()
