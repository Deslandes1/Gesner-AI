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
import csv
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from duckduckgo_search import DDGS

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS (dark theme) ----------
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
    .training-locked {
        background-color: rgba(233,69,96,0.2);
        border-left: 4px solid #e94560;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
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

# ================= FULL TEXTS DICTIONARY =================
# (Keep your existing TEXTS dictionary exactly as before – too long to repeat, but it must be here)
# For brevity in this answer, I assume you have it. In the final code, paste your full TEXTS.
TEXTS = {
    "en": { ... },  # paste your existing TEXTS["en"]
    "fr": { ... },  # paste your existing TEXTS["fr"]
    "ht": { ... },  # paste your existing TEXTS["ht"]
    "es": { ... }   # paste your existing TEXTS["es"]
}

# ---------- SESSION STATE ----------
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
if "training_access" not in st.session_state:
    st.session_state.training_access = False
if "public_chat_messages" not in st.session_state:
    st.session_state.public_chat_messages = []

# ---------- API KEY PROTECTION ----------
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

# ---------- HYBRID RETRIEVAL ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_facts_hybrid(query, k=3):
    semantic_results = retrieve_relevant_facts(query, k=k, threshold=1.2)
    if not semantic_results:
        semantic_results = []
    keyword_results = []
    if st.session_state.tfidf_vectorizer is not None and st.session_state.tfidf_matrix is not None:
        q_vec = st.session_state.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(q_vec, st.session_state.tfidf_matrix).flatten()
        top_indices = scores.argsort()[-k:][::-1]
        for idx in top_indices:
            if scores[idx] > 0.1:
                keyword_results.append(st.session_state.texts[idx])
    combined = list(dict.fromkeys(semantic_results + keyword_results))
    return combined[:k]

def retrieve_relevant_facts(query, k=3, threshold=1.2):
    if st.session_state.index is None or st.session_state.index.ntotal == 0:
        return []
    query_embedding = st.session_state.embedding_model.encode([query])[0].astype(np.float32).reshape(1, -1)
    distances, indices = st.session_state.index.search(query_embedding, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(st.session_state.texts) and distances[0][i] < threshold:
            results.append(st.session_state.texts[idx])
    return results

def apply_phonics(text):
    rules = {
        r'qu': 'k', r'c([aeiou])': r'k\1', r'ç': 's',
        r'é': 'e', r'è': 'e', r'ê': 'e', r'î': 'i', r'ô': 'o', r'û': 'u',
        r'à': 'a', r'ù': 'u', r'œ': 'oe', r'æ': 'ae', r'ph': 'f',
        r'th': 't', r'([^aeiouy])y([aeiou])': r'\1i\2', r'-tion$': 'syon',
    }
    corrected = text.lower()
    for pattern, repl in rules.items():
        corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)
    return corrected[0].upper() + corrected[1:] if corrected else ""

# ---------- DIRECT KEYWORD ANSWERS (FIXED) ----------
def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    
    # Vowels (vwayèl) - handles typos like vwayel, genhen
    if re.search(r"konbyen vway[èe]l", q_lower) or "vwayel" in q_lower:
        return "Alfabè kreyòl la gen 8 vwayèl: A, E, È, I, O, Ò, OU, UI."
    
    # Consonants
    if re.search(r"konbyen konsò?n", q_lower):
        return "Alfabè kreyòl la gen 24 konsòn."
    
    # Total letters
    if re.search(r"konbyen l[eè]t", q_lower) or "konbyen let" in q_lower:
        return "Alfabè kreyòl la gen 32 lèt: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    
    # List all letters
    if re.search(r"site tout l[eè]t|site l[eè]t|l[eè]t yo", q_lower):
        return "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    
    # What is alphabet
    if "kisa alfabè a ye" in q_lower or "kisa alfabè" in q_lower:
        return "Alfabè kreyòl la se 32 let ki reprezante tout son lang lan."
    
    # How to write alphabet
    if "kouman pou ekri alfabet la" in q_lower or "ekri alfabè" in q_lower:
        return "Alfabè kreyòl la ekri: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    
    # Identity
    identity_queries = [
        "kijan ou rele", "kiyès ou ye", "kisa ou ye", "kijan ou rele?",
        "ki moun ou ye", "what is your name", "who are you"
    ]
    if any(q in q_lower for q in identity_queries):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    
    # Creator
    creator_queries = [
        "kiyès ki kreye ou", "ki moun ki fè ou", "who created you",
        "ki moun ki devlope ou", "kiyès ki te kreye ou"
    ]
    if any(q in q_lower for q in creator_queries):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    
    # Greetings
    if q_lower in ["bonjou", "bonswa", "hello", "hi", "salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    
    return None

# ---------- LOGICAL REASONING (fallback) ----------
def reason_about_question(query, lang):
    """Attempt to answer common sense questions not in training."""
    q = query.lower().strip()
    
    # Simple math
    math_match = re.search(r"(\d+)\s*[\+\-\*\/]\s*(\d+)", q)
    if math_match:
        try:
            a, b = int(math_match.group(1)), int(math_match.group(2))
            if '+' in q:
                result = a + b
            elif '-' in q:
                result = a - b
            elif '*' in q or 'x' in q:
                result = a * b
            elif '/' in q:
                result = a / b
            else:
                result = None
            if result is not None:
                if lang == "ht":
                    return f"Repons lan se {result}."
                elif lang == "fr":
                    return f"La réponse est {result}."
                elif lang == "es":
                    return f"La respuesta es {result}."
                else:
                    return f"The answer is {result}."
        except:
            pass
    
    # What is the capital of...?
    capital_match = re.search(r"kapital|capital|capital of|capitale de", q)
    if capital_match:
        # Known capitals (extend as needed)
        capitals = {
            "france": "Paris",
            "ayiti": "Pòtoprens",
            "haiti": "Port‑au‑Prince",
            "etazini": "Washington, D.C.",
            "usa": "Washington, D.C.",
            "kanada": "Ottawa",
            "brezil": "Brasília",
            "alman": "Bèlen",
            "itali": "Wòm",
            "espay": "Madrid",
            "angle": "Londr",
            "japon": "Tokiyo",
        }
        for country, cap in capitals.items():
            if country in q:
                if lang == "ht":
                    return f"Kapital {country.title()} se {cap}."
                elif lang == "fr":
                    return f"La capitale de {country.title()} est {cap}."
                elif lang == "es":
                    return f"La capital de {country.title()} es {cap}."
                else:
                    return f"The capital of {country.title()} is {cap}."
    
    # Time, date, weather (simple)
    if "ki lè li ye" in q or "what time" in q:
        import datetime
        now = datetime.datetime.now().strftime("%H:%M")
        if lang == "ht":
            return f"Kounye a li {now}."
        elif lang == "fr":
            return f"Il est {now}."
        elif lang == "es":
            return f"Son las {now}."
        else:
            return f"It is {now}."
    
    # Default: ask to teach
    return None

# ---------- INTELLIGENT RESPONSE (with thinking) ----------
def generate_answer_from_training(query, target_lang):
    # 1) Direct keyword matches
    direct_answer = direct_keyword_answer(query)
    if direct_answer:
        return direct_answer, False, None
    
    # 2) Retrieve from trained facts
    best_facts = retrieve_facts_hybrid(query, k=3)
    if best_facts:
        return best_facts[0], False, None
    
    # 3) Logic reasoning (simulate thinking)
    reason_answer = reason_about_question(query, target_lang)
    if reason_answer:
        return reason_answer, False, None
    
    # 4) Final fallback – polite request to train
    fallbacks = {
        "en": "I don't have an answer for that yet. Please teach me in the Training Center so I can answer it next time.",
        "fr": "Je n'ai pas encore de réponse. Veuillez m'enseigner dans le Centre d'entraînement.",
        "ht": "Mwen poko gen repons. Tanpri anseye m nan Sant Fòmasyon pou m ka reponn pwochèn fwa.",
        "es": "Todavía no tengo respuesta. Por favor enséñame en el Centro de Entrenamiento."
    }
    return fallbacks.get(target_lang, fallbacks["en"]), True, target_lang

def generate_response(user_input, target_lang):
    # Simulate "thinking" – but we can just call the function
    with st.spinner("🤔 Gesner AI ap reflechi..."):
        time.sleep(0.5)  # tiny delay to show thinking
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

# ---------- TRAINING FUNCTIONS (unchanged from previous working version) ----------
# ... (all your existing add_to_training, update_training_item, delete_training_item,
# load_previous_training, ensure_intro_text, character_picker, etc. must be included here)
# To save space, I'm not repeating them – but you MUST paste them from your previous working app.

# ---------- PUBLIC CHAT MODE (with thinking) ----------
def public_chat_interface():
    ui_lang = st.session_state.get("ui_language", "en")
    t = TEXTS.get(ui_lang, TEXTS["en"])
    st.markdown("<h1 style='text-align:center; color:#ffd966;'>💬 Gesner AI Chat</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>Ask me anything – I will think and answer based on my training.</p>", unsafe_allow_html=True)
    
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

# ---------- SIDEBAR (unchanged) ----------
# ... (keep your existing show_sidebar, training_mode, etc.)

# ---------- MAIN ----------
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
