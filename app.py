import streamlit as st
import json
import os
import numpy as np
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

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ---------- CSS (dark theme) – same as before (keep your full CSS here) ----------
# ... (paste your existing CSS block) ...

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht",
    "Español": "es"
}

# ================= FULL TEXTS DICTIONARY =================
# ... (keep your full TEXTS dictionary) ...

# ---------- SESSION STATE ----------
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
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
if "texts" not in st.session_state:
    st.session_state.texts = []

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

# ---------- TF‑IDF ONLY (no neural model) ----------
def build_tfidf():
    if st.session_state.texts:
        st.session_state.tfidf_vectorizer = TfidfVectorizer(stop_words=None)
        st.session_state.tfidf_matrix = st.session_state.tfidf_vectorizer.fit_transform(st.session_state.texts)

def retrieve_relevant_facts(query, k=3):
    if not st.session_state.texts or st.session_state.tfidf_vectorizer is None:
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
    # Vowels
    if re.search(r"konbyen vway[èe]l", q_lower) or "vwayel" in q_lower:
        return "Alfabè kreyòl la gen 8 vwayèl: A, E, È, I, O, Ò, OU, UI."
    # Consonants
    if re.search(r"konbyen konsò?n", q_lower):
        return "Alfabè kreyòl la gen 24 konsòn."
    # Total letters
    if re.search(r"konbyen l[eè]t", q_lower) or "konbyen let" in q_lower:
        return "Alfabè kreyòl la gen 32 lèt: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    # List letters
    if re.search(r"site tout l[eè]t|site l[eè]t|l[eè]t yo", q_lower):
        return "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    # Identity
    identity_queries = ["kijan ou rele", "kiyès ou ye", "kisa ou ye", "ki moun ou ye", "what is your name", "who are you"]
    if any(q in q_lower for q in identity_queries):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    # Creator
    creator_queries = ["kiyès ki kreye ou", "ki moun ki fè ou", "who created you"]
    if any(q in q_lower for q in creator_queries):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py."
    # Greetings
    if q_lower in ["bonjou", "bonswa", "hello", "hi", "salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

# ---------- FALLBACK LOGIC (simple math, capitals, time) ----------
def reason_about_question(query, lang):
    q = query.lower().strip()
    # Simple math
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
    # Capitals
    if "kapital" in q or "capital" in q:
        capitals = {"france": "Paris", "ayiti": "Pòtoprens", "haiti": "Port‑au‑Prince", "etazini": "Washington, D.C.", "usa": "Washington, D.C."}
        for country, cap in capitals.items():
            if country in q:
                if lang == "ht": return f"Kapital {country.title()} se {cap}."
                elif lang == "fr": return f"La capitale de {country.title()} est {cap}."
                elif lang == "es": return f"La capital de {country.title()} es {cap}."
                else: return f"The capital of {country.title()} is {cap}."
    # Current time
    if "ki lè li ye" in q or "what time" in q:
        import datetime
        now = datetime.datetime.now().strftime("%H:%M")
        if lang == "ht": return f"Kounye a li {now}."
        elif lang == "fr": return f"Il est {now}."
        elif lang == "es": return f"Son las {now}."
        else: return f"It is {now}."
    return None

def generate_answer_from_training(query, target_lang):
    direct = direct_keyword_answer(query)
    if direct:
        return direct, False, None
    facts = retrieve_relevant_facts(query, k=3)
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
    return generate_answer_from_training(user_input, target_lang)

def play_voice_button(text, is_fallback, fallback_audio_lang, button_label="🔊", key_suffix=""):
    # (same as before – keep your existing play_voice_button code)
    # ... (paste your existing audio button function) ...
    return ""

# ---------- TRAINING FUNCTIONS ----------
def add_to_training(text, t):
    if not text.strip():
        st.warning(t['warning_no_text'])
        return False
    st.session_state.training_data.append({"text": text, "embedding": []})  # dummy embedding
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    build_tfidf()
    st.success(t['training_success'].format(text=text[:100]))
    return True

def update_training_item(idx, new_text, t):
    ... # unchanged

def delete_training_item(idx):
    ... # unchanged

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
    intro = "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    if intro not in st.session_state.texts:
        add_to_training(intro, {"warning_no_text": "", "training_success": "Trained: {text}"})

def character_picker(key_prefix, label="Insert Kreyòl characters:"):
    # ... (same as before) ...
    pass

# ---------- ALL OTHER FUNCTIONS (dictionary_manager, voice_training, translation_correction,
#            encyclopedia_manager, test_training, phonics_training, bulk_training,
#            training_mode, public_chat_interface, show_sidebar) are unchanged.
#            Paste them exactly from your previous working app (without SentenceTransformer imports).

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
