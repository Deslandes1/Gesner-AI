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

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht"
}

TEXTS = {
    "en": {
        "app_title": "🧠 Gesner AI – Train Your Personal Haitian AI",
        "subtitle": "Teach me through chat, text, images, or files. I learn from everything you share.",
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
        "expand_audio": "Upload an audio file (you'll need to transcribe manually – or use a service)",
        "audio_upload_label": "Choose an audio file",
        "transcribe_label": "After listening, type the transcription below to train me:",
        "transcription_textarea": "Transcribed text from the audio",
        "train_transcription_button": "Train with this transcription",
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
        "with_facts_answer": "Based on what I've learned:\n{context}\n\nTo answer your question: {question} – does that help?",
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
        "wrong_password": "Incorrect password. Access denied."
    },
    "fr": {
        "app_title": "🧠 Gesner IA – Entraînez votre IA personnelle haïtienne",
        "subtitle": "Enseignez‑moi par chat, texte, images ou fichiers. J’apprends de tout ce que vous partagez.",
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
        "expand_audio": "Téléchargez un fichier audio (vous devrez transcrire manuellement – ou utiliser un service)",
        "audio_upload_label": "Choisissez un fichier audio",
        "transcribe_label": "Après avoir écouté, saisissez la transcription ci‑dessous pour m’entraîner :",
        "transcription_textarea": "Texte transcrit à partir de l’audio",
        "train_transcription_button": "Entraîner avec cette transcription",
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
        "with_facts_answer": "D’après ce que j’ai appris :\n{context}\n\nPour répondre à votre question : {question} – cela vous aide‑t‑il ?",
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
        "wrong_password": "Mot de passe incorrect. Accès refusé."
    },
    "ht": {
        "app_title": "🧠 Gesner AI – Antrene AI Pèsonèl Ayisyen w la",
        "subtitle": "Anseye m atravè chat, tèks, imaj oswa fichye. M aprann de tout sa w pataje avè m.",
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
        "expand_audio": "Chaje yon fichye odyo (w ap bezwen transkri a men – oswa itilize yon sèvis)",
        "audio_upload_label": "Chwazi yon fichye odyo",
        "transcribe_label": "Apre w fin koute, ekri transkripsyon an anba a pou antrene m :",
        "transcription_textarea": "Tèks transkri apati odyo a",
        "train_transcription_button": "Antrene ak transkripsyon sa a",
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
        "with_facts_answer": "Dapre sa m te aprann:\n{context}\n\nPou reponn kesyon ou a : {question} – èske sa ede w ?",
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
        "wrong_password": "Modpas pa bon. Aksè refize."
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
    t = TEXTS[st.session_state.language]
    if facts:
        context = "\n".join(facts)
        return t["with_facts_answer"].format(context=context, question=user_input)
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

    # --- Training Section (Text) ---
    st.markdown("---")
    st.markdown(f"## {t['training_text_title']}")
    with st.expander(t["expand_text"]):
        training_text = st.text_area(t["text_area_label"])
        if st.button(t["train_text_button"], use_container_width=True):
            add_to_training(training_text, t)

    # --- Audio Training ---
    st.markdown(f"## {t['audio_title']}")
    with st.expander(t["expand_audio"]):
        audio_file = st.file_uploader(t["audio_upload_label"], type=["wav", "mp3", "m4a"])
        if audio_file is not None:
            st.audio(audio_file, format="audio/wav")
            st.markdown(f"**{t['transcribe_label']}**")
            transcribed_text = st.text_area(t["transcription_textarea"])
            if st.button(t["train_transcription_button"], use_container_width=True):
                if transcribed_text:
                    add_to_training(transcribed_text, t)
                else:
                    st.warning(t['warning_no_transcription'])

    # --- Image Training ---
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

    # --- File Upload (Text) ---
    st.markdown(f"## {t['file_title']}")
    with st.expander(t["expand_file"]):
        text_file = st.file_uploader(t["file_upload_label"], type=["txt", "md"])
        if text_file is not None:
            content = text_file.read().decode("utf-8")
            st.text_area(t['file_preview'], content, height=150)
            if st.button(t["train_file_button"], use_container_width=True):
                add_to_training(content, t)

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
