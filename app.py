# ===== LIGHTWEIGHT – NO HEAVY MODELS, NO QUESTION DROPDOWN =====
import os
import streamlit as st
import json
import numpy as np
import time
import hashlib
import re
import csv
import io
from PIL import Image
import requests
import base64
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Gesner AI", page_icon="🧠", layout="wide")

# ---------- CSS (dark theme - same as before) ----------
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
    .stCodeBlock, .stCodeBlock div, pre, code {
        background-color: #0f3460 !important;
        color: #ffffff !important;
        border-radius: 12px;
        padding: 0.5rem;
    }
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
    .char-picker {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .char-btn {
        background-color: #2a5298;
        border: none;
        border-radius: 20px;
        padding: 5px 12px;
        color: white;
        cursor: pointer;
        font-size: 1rem;
        transition: 0.2s;
        margin-right: 5px;
    }
    .char-btn:hover {
        background-color: #e94560;
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

# ================= FULL TEXTS DICTIONARY (keep your full version) =================
# For brevity, only 'en' is shown here. Replace with your complete TEXTS from original.
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
        "text_area_label": "Enter knowledge",
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
        "pricing_table": "| License | Price (one‑time) |\n|---------|------------------|\n| **Personal** | $49 |\n| **Business** | $299 |\n| **Enterprise / Source** | $999 |\n",
        "logout_button": "🔓 Logout",
        "no_facts_answer": "I don't know that yet. Please teach me in Training Mode!",
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
        "phonics_title": "🔊 Phonics Training (32 Letters)",
        "phonics_example": "Example word/sentence for letter {letter}",
        "phonics_add": "Teach example",
        "manage_facts": "📚 Manage Trained Facts",
        "train_entry_button": "Train AI with this entry",
        "trained_entry_success": "✅ Trained: {word} → {meaning}",
        "bulk_training_title": "🚀 Bulk Training (Fast Import)",
        "bulk_csv_label": "Upload CSV file (columns: question, answer OR one column 'fact')",
        "bulk_json_label": "Upload JSON file (array of strings)",
        "bulk_text_label": "Paste text (one fact per line)",
        "bulk_import_button": "Import All Facts"
    },
    # You must add your full TEXTS for fr, ht, es from your original file.
    # For brevity, only 'en' is shown here.
}
# IMPORTANT: Replace the TEXTS dictionary above with your complete multilingual version.

# ---------- SESSION STATE ----------
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
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
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "public_chat_messages" not in st.session_state:
    st.session_state.public_chat_messages = []

# ---------- API KEY ----------
REQUIRED_API_KEY = "PNL_fJC4L5QNjA0GJbc4N8TzIXBjdfIXfgcLv1yZ8Yc"

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

# ---------- TF‑IDF RETRIEVAL ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_facts_tfidf(query, k=3):
    if st.session_state.tfidf_vectorizer is None or st.session_state.tfidf_matrix is None:
        return []
    q_vec = st.session_state.tfidf_vectorizer.transform([query])
    scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
    top_indices = scores.argsort()[-k:][::-1]
    results = []
    for idx in top_indices:
        if scores[idx] > 0.1:
            results.append(st.session_state.texts[idx])
    return results

# ---------- DIRECT KEYWORD ANSWERS ----------
def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    if re.search(r"konbyen vway[èe]l", q_lower) or "vwayel" in q_lower:
        return "Alfabè kreyòl la gen 8 vwayèl: A, E, È, I, O, Ò, OU, UI."
    if re.search(r"konbyen konsò?n", q_lower):
        return "Alfabè kreyòl la gen 24 konsòn."
    if re.search(r"konbyen l[eè]t", q_lower) or "konbyen let" in q_lower:
        return "Alfabè kreyòl la gen 32 lèt: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    if re.search(r"site tout l[eè]t|site l[eè]t|l[eè]t yo", q_lower):
        return "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    if "kisa alfabè a ye" in q_lower or "kisa alfabè" in q_lower:
        return "Alfabè kreyòl la se 32 let ki reprezante tout son lang lan."
    identity_queries = ["kijan ou rele", "kiyès ou ye", "kisa ou ye", "ki moun ou ye", "what is your name", "who are you"]
    if any(q in q_lower for q in identity_queries):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    creator_queries = ["kiyès ki kreye ou", "ki moun ki fè ou", "who created you", "ki moun ki devlope ou", "kiyès ki te kreye ou"]
    if any(q in q_lower for q in creator_queries):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    if q_lower in ["bonjou", "bonswa", "hello", "hi", "salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

# ---------- LOGICAL REASONING ----------
def reason_about_question(query, lang):
    q = query.lower().strip()
    math_match = re.search(r"(\d+)\s*([\+\-\*\/])\s*(\d+)", q)
    if math_match:
        try:
            a, op, b = int(math_match.group(1)), math_match.group(2), int(math_match.group(3))
            if op == '+': res = a + b
            elif op == '-': res = a - b
            elif op == '*': res = a * b
            elif op == '/': res = a / b
            else: res = None
            if res is not None:
                if lang == "ht": return f"Repons lan se {res}."
                elif lang == "fr": return f"La réponse est {res}."
                elif lang == "es": return f"La respuesta es {res}."
                else: return f"The answer is {res}."
        except: pass
    if "kapital" in q or "capital" in q:
        capitals = {
            "france": "Paris", "ayiti": "Pòtoprens", "haiti": "Port‑au‑Prince",
            "etazini": "Washington, D.C.", "usa": "Washington, D.C.",
            "kanada": "Ottawa", "brezil": "Brasília", "alman": "Bèlen",
            "itali": "Wòm", "espay": "Madrid", "angle": "Londr", "japon": "Tokiyo",
        }
        for country, cap in capitals.items():
            if country in q:
                if lang == "ht": return f"Kapital {country.title()} se {cap}."
                elif lang == "fr": return f"La capitale de {country.title()} est {cap}."
                elif lang == "es": return f"La capital de {country.title()} es {cap}."
                else: return f"The capital of {country.title()} is {cap}."
    if "ki lè li ye" in q or "what time" in q:
        import datetime
        now = datetime.datetime.now().strftime("%H:%M")
        if lang == "ht": return f"Kounye a li {now}."
        elif lang == "fr": return f"Il est {now}."
        elif lang == "es": return f"Son las {now}."
        else: return f"It is {now}."
    return None

# ---------- GENERATE ANSWER (no embedding) ----------
def generate_answer_from_training(query, target_lang):
    direct = direct_keyword_answer(query)
    if direct:
        return direct, False, None
    facts = retrieve_facts_tfidf(query, k=3)
    if facts:
        return facts[0], False, None
    logic = reason_about_question(query, target_lang)
    if logic:
        return logic, False, None
    fallbacks = {
        "en": "I don't have an answer for that yet. Please teach me in the Training Center.",
        "fr": "Je n'ai pas encore de réponse. Veuillez m'enseigner dans le Centre d'entraînement.",
        "ht": "Mwen poko gen repons. Tanpri anseye m nan Sant Fòmasyon.",
        "es": "Todavía no tengo respuesta. Por favor enséñame en el Centro de Entrenamiento."
    }
    return fallbacks.get(target_lang, fallbacks["en"]), True, target_lang

def generate_response(user_input, target_lang):
    with st.spinner("🤔 Gesner AI ap reflechi..."):
        time.sleep(0.2)
        answer, is_fallback, fallback_lang = generate_answer_from_training(user_input, target_lang)
    return answer, is_fallback, fallback_lang

def play_voice_button(text, is_fallback, fallback_audio_lang, button_label="🔊", key_suffix=""):
    if is_fallback:
        lang_map = {"en": "en-US", "fr": "fr-FR", "ht": "fr-FR", "es": "es-ES"}
        tts_lang = lang_map.get(fallback_audio_lang, "en-US")
        safe_text = json.dumps(text)
        html = f"""
        <button class="speak-btn" id="ttsBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; margin-left:12px; cursor:pointer;">{button_label}</button>
        <script>
            (function() {{
                const btn = document.getElementById('ttsBtn_{key_suffix}');
                let utterance = null;
                function speakWithVoice() {{
                    if (utterance) window.speechSynthesis.cancel();
                    utterance = new SpeechSynthesisUtterance({safe_text});
                    utterance.lang = '{tts_lang}';
                    let voices = window.speechSynthesis.getVoices();
                    if (voices.length === 0) {{
                        window.speechSynthesis.onvoiceschanged = function() {{
                            voices = window.speechSynthesis.getVoices();
                            selectBestVoice(voices, utterance);
                            window.speechSynthesis.speak(utterance);
                        }};
                        return;
                    }}
                    selectBestVoice(voices, utterance);
                    window.speechSynthesis.speak(utterance);
                }}
                function selectBestVoice(voices, utterance) {{
                    let langCode = '{tts_lang}';
                    let priorityNames = [];
                    if (langCode === 'fr-FR') priorityNames = ['Google français', 'Microsoft Hortense', 'Microsoft Denis', 'Samantha', 'Thomas'];
                    if (langCode === 'en-US') priorityNames = ['Google US English', 'Microsoft David', 'Microsoft Zira', 'Samantha'];
                    if (langCode === 'es-ES') priorityNames = ['Google español', 'Microsoft Helena', 'Microsoft Pablo', 'Monica'];
                    let selected = null;
                    for (let name of priorityNames) {{
                        selected = voices.find(v => v.lang === langCode && v.name.includes(name));
                        if (selected) break;
                    }}
                    if (!selected) selected = voices.find(v => v.lang === langCode);
                    if (selected) utterance.voice = selected;
                }}
                btn.onclick = speakWithVoice;
            }})();
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

# ---------- TRAINING FUNCTIONS (no embedding) ----------
def add_to_training(text, t):
    if not text.strip():
        st.warning(t['warning_no_text'])
        return False
    st.session_state.training_data.append({"text": text})
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
    st.session_state.training_data[idx] = {"text": new_text}
    st.session_state.texts = [item["text"] for item in st.session_state.training_data]
    if st.session_state.texts:
        build_tfidf()
    else:
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
        build_tfidf()
    else:
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
                st.session_state.texts = [item["text"] for item in data]
                build_tfidf()
        except Exception:
            pass

def ensure_intro_text():
    intro_text_ht = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    if intro_text_ht not in st.session_state.texts:
        st.session_state.training_data.append({"text": intro_text_ht})
        st.session_state.texts.append(intro_text_ht)
        build_tfidf()
        with open("training_data.json", "w") as f:
            json.dump(st.session_state.training_data, f, indent=2)

def character_picker(key_prefix, label="Insert Kreyòl characters:"):
    chars = [
        "e", "è", "E", "È", "o", "ò", "O", "Ò",
        "an", "An", "AN", "en", "En", "EN", "on", "On", "ON", "oun", "Oun", "OUN"
    ]
    st.markdown(f"**{label}**")
    cols = st.columns(len(chars))
    for i, ch in enumerate(chars):
        with cols[i]:
            if st.button(ch, key=f"char_{key_prefix}_{ch}"):
                if key_prefix == "train_text":
                    current = st.session_state.get("train_text", "")
                    st.session_state.train_text = current + ch
                    st.rerun()
                elif key_prefix == "train_chat_input":
                    current = st.session_state.get("train_chat_input", "")
                    st.session_state.train_chat_input = current + ch
                    st.rerun()
                elif key_prefix == "img_desc":
                    current = st.session_state.get("img_desc", "")
                    st.session_state.img_desc = current + ch
                    st.rerun()
                elif key_prefix.startswith("edit_"):
                    idx = key_prefix.split("_")[1]
                    key = f"edit_text_{idx}"
                    current = st.session_state.get(key, "")
                    st.session_state[key] = current + ch
                    st.rerun()

# ---------- DICTIONARY MANAGER (unchanged from your original) ----------
def dictionary_manager(t):
    st.markdown(f"## {t['dict_title']}")
    col1, col2, col3 = st.columns(3)
    
    def display_dict(lang_code, lang_label, dict_data):
        st.markdown(f"### {lang_label}")
        w = st.text_input(f"{t['dict_word']} ({lang_code.upper()})", key=f"{lang_code}_word")
        m = st.text_input(f"{t['dict_meaning']} ({lang_code.upper()})", key=f"{lang_code}_meaning")
        if st.button(t['dict_add'], key=f"add_{lang_code}"):
            if w and m:
                dict_data[w] = m
                save_dictionaries()
                st.success(f"Added {w}")
                st.rerun()
        for word, meaning in list(dict_data.items()):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.text(f"{word}: {meaning}")
            with col_b:
                if st.button(t['train_entry_button'], key=f"train_{lang_code}_{word}"):
                    train_text = f"{word} means {meaning}"
                    if add_to_training(train_text, t):
                        st.success(t['trained_entry_success'].format(word=word, meaning=meaning))
                        st.rerun()
                if st.button(f"{t['dict_delete']}", key=f"del_{lang_code}_{word}"):
                    del dict_data[word]
                    save_dictionaries()
                    st.rerun()
    
    with col1:
        display_dict("ht", t['dict_ht'], st.session_state.dictionaries["ht"])
    with col2:
        display_dict("fr", t['dict_fr'], st.session_state.dictionaries["fr"])
    with col3:
        display_dict("en", t['dict_en'], st.session_state.dictionaries["en"])

def save_dictionaries():
    with open("dictionaries.json", "w") as f:
        json.dump(st.session_state.dictionaries, f, indent=2)

def save_encyclopedia():
    with open("encyclopedia.json", "w") as f:
        json.dump(st.session_state.encyclopedia, f, indent=2)

def voice_training(t):
    st.markdown(f"## {t['voice_training_title']}")
    st.info("🎙️ Upload your voice for Kreyòl phrases.")
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

def bulk_training(t):
    st.markdown(f"## {t['bulk_training_title']}")
    st.info("Import many facts at once.")
    
    def import_facts(facts):
        count = 0
        for fact in facts:
            if fact.strip():
                if add_to_training(fact.strip(), t):
                    count += 1
        st.success(f"Imported {count} facts.")
    
    csv_file = st.file_uploader(t['bulk_csv_label'], type=["csv"], key="bulk_csv")
    if csv_file:
        try:
            content = csv_file.read().decode('utf-8')
            reader = csv.DictReader(io.StringIO(content))
            facts = []
            for row in reader:
                if 'question' in row and 'answer' in row:
                    facts.append(f"{row['question']} → {row['answer']}")
                elif 'fact' in row:
                    facts.append(row['fact'])
                else:
                    first_key = list(row.keys())[0]
                    facts.append(row[first_key])
            if facts:
                st.info(f"Found {len(facts)} facts in CSV. Click import to add them.")
                if st.button(t['bulk_import_button'], key="import_csv"):
                    import_facts(facts)
            else:
                st.warning("No valid facts found in CSV.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    
    json_file = st.file_uploader(t['bulk_json_label'], type=["json"], key="bulk_json")
    if json_file:
        try:
            data = json.load(json_file)
            if isinstance(data, list):
                facts = [str(item) for item in data]
                st.info(f"Found {len(facts)} facts in JSON. Click import to add them.")
                if st.button(t['bulk_import_button'], key="import_json"):
                    import_facts(facts)
            else:
                st.warning("JSON must be an array of strings.")
        except Exception as e:
            st.error(f"Error reading JSON: {e}")
    
    text_facts = st.text_area(t['bulk_text_label'], height=200, key="bulk_text")
    if text_facts.strip():
        lines = [line.strip() for line in text_facts.split('\n') if line.strip()]
        st.info(f"Found {len(lines)} facts in text. Click import to add them.")
        if st.button(t['bulk_import_button'], key="import_text"):
            import_facts(lines)

def training_mode():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS.get(ui_lang, TEXTS["en"])
    st.markdown(f"<h1 style='text-align:center;'>🔒 {t['training_app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t['training_subtitle']}</p>", unsafe_allow_html=True)
    
    # Chat inside training
    st.markdown(f"## {t['chat_title']}")
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">{t["user_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message assistant-message">{t["assistant_prefix"]}{msg["content"]}</div>', unsafe_allow_html=True)
    
    character_picker("train_chat_input", "Insert Kreyòl characters:")
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
        character_picker("train_text", "Insert Kreyòl characters for the fact:")
        text = st.text_area(t["text_area_label"], key="train_text")
        if st.button(t["train_text_button"], use_container_width=True):
            add_to_training(text, t)
    
    st.markdown(f"## {t['audio_title']}")
    with st.expander(t["expand_audio"]):
        voice_training(t)
    
    st.markdown(f"## {t['image_title']}")
    with st.expander(t["expand_image"]):
        img_file = st.file_uploader(t["image_upload_label"], type=["jpg", "jpeg", "png"])
        character_picker("img_desc", "Insert Kreyòl characters for the description:")
        desc = st.text_area(t["image_description_label"], key="img_desc")
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
    bulk_training(t)
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
    
    st.markdown(f"## {t.get('manage_facts', '📚 Manage Trained Facts')}")
    if not st.session_state.training_data:
        st.info("No facts trained yet. Use the training tools above.")
    else:
        for idx, item in enumerate(st.session_state.training_data):
            original = item["text"]
            with st.expander(f"Fact #{idx+1}: {original[:60]}..."):
                character_picker(f"edit_{idx}", "Insert Kreyòl characters:")
                new_text = st.text_area(f"Edit text", value=original, key=f"edit_text_{idx}")
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

def public_chat_interface():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS.get(ui_lang, TEXTS["en"])
    st.markdown("<h1 style='text-align:center; color:#ffd966;'>💬 Gesner AI Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Type your question below. I will answer based on what I've learned.</p>", unsafe_allow_html=True)
    
    # --- NO DROPDOWN (questionary removed) ---
    
    # Chat history
    for idx, msg in enumerate(st.session_state.public_chat_messages):
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
                    "🔊",
                    f"public_{idx}"
                )
                if btn_html:
                    st.components.v1.html(btn_html, height=50)
    
    # Free text input
    user_input = st.text_input(t["chat_input_placeholder"], key="public_chat_input")
    if st.button(t["send_button"], use_container_width=True, key="public_send"):
        if user_input.strip():
            target_lang = st.session_state.chat_language
            answer, is_fallback, fallback_lang = generate_response(user_input, target_lang)
            st.session_state.public_chat_messages.append({"role": "user", "content": user_input})
            st.session_state.public_chat_messages.append({
                "role": "assistant",
                "content": answer,
                "is_fallback": is_fallback,
                "fallback_lang": fallback_lang
            })
            st.rerun()
    
    if st.button("Clear Chat", use_container_width=True, key="public_clear"):
        st.session_state.public_chat_messages = []
        st.rerun()

def show_sidebar():
    lang_names = list(LANGUAGES.keys())
    selected_lang_name = st.sidebar.selectbox("🌐 Language", lang_names, key="main_lang_selector")
    selected_lang_code = LANGUAGES[selected_lang_name]
    st.session_state.ui_language = selected_lang_code
    st.session_state.chat_language = selected_lang_code
    t = TEXTS.get(st.session_state.ui_language, TEXTS["en"])
    
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
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔐 Trainer Access (API Key)")
    if not st.session_state.training_access:
        api_key_input = st.sidebar.text_input("Enter API Key", type="password", key="api_key_input")
        if st.sidebar.button("Unlock Training Center"):
            if api_key_input == REQUIRED_API_KEY:
                st.session_state.training_access = True
                st.sidebar.success("Access granted!")
                st.rerun()
            else:
                st.sidebar.error("Invalid API Key")
    else:
        st.sidebar.success("✅ Training mode active")
        if st.sidebar.button("Lock Training Center"):
            st.session_state.training_access = False
            st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("Reset Public Chat"):
        st.session_state.public_chat_messages = []
        st.rerun()

def main():
    load_previous_training()
    ensure_intro_text()
    show_sidebar()
    
    if st.session_state.training_access:
        mode = st.radio("Select mode", ["💬 Public Chat Mode", "🔧 Training Center"], horizontal=True)
        if mode == "💬 Public Chat Mode":
            public_chat_interface()
        else:
            training_mode()
    else:
        public_chat_interface()
    
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS.get(ui_lang, TEXTS["en"])
    st.markdown(f'<div class="footer">{t["footer"]} | Public chat always free, training protected by API key</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    if "ui_language" not in st.session_state:
        st.session_state.ui_language = "en"
    if "chat_language" not in st.session_state:
        st.session_state.chat_language = "en"
    main()
