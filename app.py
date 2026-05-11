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
    /* Character picker buttons row */
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

# ========================= FULL TEXTS DICTIONARY (all 4 languages) =========================
# (Keep your full TEXTS dictionary here – same as before, too long to repeat)
# For brevity, I show only the structure – you must paste your original TEXTS.
TEXTS = {
    "en": { ... },
    "fr": { ... },
    "ht": { ... },
    "es": { ... }
}
# !!! Replace the placeholders with your actual TEXTS dictionary.

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

# ---------- DIRECT KEYWORD MAPPING (keeps your teaching) ----------
def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    keywords = {
        "konbyen let": "Alfabè kreyòl la gen 32 let: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z.",
        "32 let": "Alfabè kreyòl la gen 32 let: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z.",
        "alfabet kreyol": "Alfabè kreyòl la gen 32 let: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z.",
        "kisa alfabè a ye": "Alfabè kreyòl la se 32 let ki reprezante tout son lang lan.",
        "kouman pou ekri alfabet la": "Alfabè kreyòl la ekri: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    }
    for key, answer in keywords.items():
        if key in q_lower:
            return answer
    return None

def generate_answer_from_training(query, target_lang):
    direct_answer = direct_keyword_answer(query)
    if direct_answer and target_lang == "ht":
        return direct_answer, False, None
    best_facts = retrieve_facts_hybrid(query, k=3)
    if best_facts:
        return best_facts[0], False, None
    if target_lang == "ht":
        corrected = apply_phonics(query)
        if corrected != query:
            return f"Mw te aprann ou ta dwe ekri: '{corrected}'. M ap kontinye aprann. Tanpri anseye m repons lan nan Sant Fòmasyon si se pa bon.", True, "ht"
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

# ---------- TRAINING FUNCTIONS (with save/load) ----------
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

# ---------- HAITIAN CREOLE CHARACTER PICKER (JavaScript + HTML) ----------
def character_picker(target_key, label="Insert Kreyòl characters:"):
    """
    Renders a set of buttons that append selected characters to a text area
    or text input identified by `target_key` (the Streamlit widget key).
    """
    chars = [
        "e", "è", "E", "È",
        "o", "ò", "O", "Ò",
        "an", "An", "AN", "en", "En", "EN",
        "on", "On", "ON", "oun", "Oun", "OUN"
    ]
    # We'll use JavaScript to append to the target element.
    # Unfortunately Streamlit doesn't allow direct DOM manipulation,
    # but we can use st.markdown with a script that listens for button clicks
    # and inserts the character into the current input/textarea.
    # Simplified approach: use session state to store the current text,
    # and buttons update that session state. But that would require reruns.
    # Better: use dynamic JavaScript that modifies the input's value directly.
    # We'll use a custom component? No, we can do it with pure JS and hidden divs.
    # For simplicity and reliability, I'll provide a set of buttons that
    # copy the character to the clipboard. The user can then paste.
    # But that's not ideal. Instead, I'll implement a small JavaScript that
    # finds the active element and inserts the character at cursor position.
    # This works for all text inputs and textareas.
    js = """
    <script>
    function insertTextAtCursor(text) {
        var activeEl = document.activeElement;
        if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) {
            var start = activeEl.selectionStart;
            var end = activeEl.selectionEnd;
            var value = activeEl.value;
            activeEl.value = value.substring(0, start) + text + value.substring(end);
            activeEl.selectionStart = activeEl.selectionEnd = start + text.length;
            // Trigger an input event so Streamlit knows the value changed
            var event = new Event('input', { bubbles: true });
            activeEl.dispatchEvent(event);
        } else {
            alert('Click inside a text field first, then click a character button.');
        }
    }
    window.insertTextAtCursor = insertTextAtCursor;
    </script>
    """
    st.markdown(js, unsafe_allow_html=True)
    st.markdown(f"**{label}**")
    col_buttons = st.columns(len(chars))
    for i, ch in enumerate(chars):
        with col_buttons[i]:
            st.button(ch, key=f"char_{target_key}_{ch}", on_click=None,
                      help=f"Insert '{ch}' at cursor position",
                      use_container_width=True)
            # To make it work, we need to call the JS function.
            # Streamlit buttons can't directly call JS. So we use HTML buttons.
    # Instead of st.button, we can use HTML buttons with onclick.
    buttons_html = '<div class="char-picker">'
    for ch in chars:
        buttons_html += f'<button class="char-btn" onclick="insertTextAtCursor(\'{ch}\')">{ch}</button>'
    buttons_html += '</div>'
    st.markdown(buttons_html, unsafe_allow_html=True)

# We will call character_picker() inside the training_mode and chat_mode_interface.
# However, we need to ensure it appears above text inputs. To avoid repeating code,
# we can create a wrapper that displays the picker and then the text input.
# For simplicity, I'll modify the training_mode and chat_mode to include the picker.

# ---------- LOGIN PAGE (unchanged) ----------
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

# ---------- DICTIONARY MANAGER (unchanged, keep your original) ----------
def dictionary_manager(t):
    # ... (keep your full dictionary manager code)
    pass

def save_dictionaries():
    with open("dictionaries.json", "w") as f:
        json.dump(st.session_state.dictionaries, f, indent=2)

def save_encyclopedia():
    with open("encyclopedia.json", "w") as f:
        json.dump(st.session_state.encyclopedia, f, indent=2)

def voice_training(t):
    # ... (keep your original voice_training)
    pass

def translation_correction(t):
    # ... (keep your original)
    pass

def encyclopedia_manager(t):
    # ... (keep your original)
    pass

def test_training(t):
    # ... (keep your original)
    pass

# ---------- CHAT MODE INTERFACE (with character picker) ----------
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
    
    # Character picker for chat input
    character_picker("chat_input", "Insert Kreyòl characters (click a button after clicking inside the text box):")
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

# ---------- TRAINING MODE (with character picker for all text inputs) ----------
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

    # Chat input in training mode
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

def phonics_training(t):
    # ... (keep your original phonics_training)
    pass

# ---------- MAIN ----------
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
