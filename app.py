import streamlit as st
import json
import os
import numpy as np
import time
import hashlib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Gesner AI – Chat",
    page_icon="🧠",
    layout="wide"
)

# ---------- LIGHTWEIGHT STYLING ----------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    }
    .stMarkdown, .stTextInput label, .stButton button, .stCaption, h1, h2, p {
        color: #ffffff !important;
    }
    .stButton button {
        background-color: #e94560 !important;
        color: white !important;
        border-radius: 30px !important;
        font-weight: bold !important;
        border: none;
    }
    .stButton button:hover {
        background-color: #ff6b6b !important;
    }
    .stTextInput input, .stTextArea textarea {
        background-color: #0f3460 !important;
        color: white !important;
        border-radius: 12px;
        border: 1px solid #e94560;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 20px;
        margin-bottom: 1rem;
    }
    .user-message {
        background: linear-gradient(135deg, #e94560, #ff6b6b);
        color: white;
    }
    .assistant-message {
        background: linear-gradient(135deg, #0f3460, #1a4a7a);
        color: white;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        border-top: 1px solid #e94560;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- SESSION STATE ----------
if "training_data" not in st.session_state:
    st.session_state.training_data = []
if "texts" not in st.session_state:
    st.session_state.texts = []
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []
if "tfidf_vectorizer" not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if "tfidf_matrix" not in st.session_state:
    st.session_state.tfidf_matrix = None

# ---------- VOICE CACHE (lightweight) ----------
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

# ---------- TF‑IDF KNOWLEDGE BASE ----------
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

def add_to_training(text):
    if not text.strip():
        st.warning("Please enter some text.")
        return False
    st.session_state.training_data.append({"text": text, "embedding": []})
    st.session_state.texts.append(text)
    with open("training_data.json", "w") as f:
        json.dump(st.session_state.training_data, f, indent=2)
    build_tfidf()
    st.success(f"✅ Learned: {text[:100]}...")
    return True

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

# ---------- DIRECT KEYWORD ANSWERS & REASONING ----------
def direct_keyword_answer(query):
    q_lower = query.lower().strip()
    if re.search(r"konbyen vway[èe]l|vwayel", q_lower):
        return "Alfabè kreyòl la gen 8 vwayèl: A, E, È, I, O, Ò, OU, UI."
    if re.search(r"konbyen konsòn", q_lower):
        return "Alfabè kreyòl la gen 24 konsòn."
    if re.search(r"konbyen l[eè]t|konbyen let", q_lower):
        return "Alfabè kreyòl la gen 32 lèt: A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    if re.search(r"site tout l[eè]t|site l[eè]t|l[eè]t yo", q_lower):
        return "A, AN, B, CH, D, E, È, EN, F, G, H, I, J, K, L, M, N, NG, O, Ò, ON, OU, OUN, P, R, S, T, UI, V, W, Y, Z."
    identity_queries = ["kijan ou rele", "kiyès ou ye", "kisa ou ye", "ki moun ou ye", "what is your name", "who are you"]
    if any(q in q_lower for q in identity_queries):
        return "Non pa mw se Gesner L’IA, kreyatè mw an se Gesner Deslandes nan GlobalInternet.py."
    creator_queries = ["kiyès ki kreye ou", "ki moun ki fè ou", "who created you", "ki moun ki devlope ou", "kiyès ki te kreye ou"]
    if any(q in q_lower for q in creator_queries):
        return "Mwen te kreye pa Gesner Deslandes, fondatè GlobalInternet.py. Li se yon enjenyè ki renmen edike Ayiti."
    if q_lower in ["bonjou", "bonswa", "hello", "hi", "salut"]:
        return "Bonjou! Kijan ou ye? Mwen la pou reponn kesyon ou."
    return None

def reason_about_question(query):
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
                return f"Repons lan se {res}."
        except: pass
    if "kapital" in q or "capital" in q:
        capitals = {
            "france": "Paris", "ayiti": "Pòtoprens", "haiti": "Port‑au‑Prince",
            "etazini": "Washington, D.C.", "usa": "Washington, D.C.", "kanada": "Ottawa",
            "brezil": "Brasília", "alman": "Bèlen", "itali": "Wòm", "espay": "Madrid",
            "angle": "Londr", "japon": "Tokiyo"
        }
        for country, cap in capitals.items():
            if country in q:
                return f"Kapital {country.title()} se {cap}."
    if "ki lè li ye" in q or "what time" in q:
        import datetime
        now = datetime.datetime.now().strftime("%H:%M")
        return f"Kounye a li {now}."
    return None

def generate_answer(query):
    direct = direct_keyword_answer(query)
    if direct:
        return direct
    facts = retrieve_relevant_facts(query, k=3)
    if facts:
        return facts[0]
    logic = reason_about_question(query)
    if logic:
        return logic
    return "Mwen poko gen repons sa. Tanpri anseye m nan seksyon 'Antrene m' anba a."

# ---------- VOICE PLAY BUTTON (simple) ----------
def play_voice_button(text, key_suffix=""):
    voice_bytes = get_voice_for_text(text)
    if voice_bytes:
        import base64
        audio_b64 = base64.b64encode(voice_bytes).decode()
        html = f"""
        <button id="voiceBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; cursor:pointer;">🔊</button>
        <audio id="customAudio_{key_suffix}" style="display:none;"></audio>
        <script>
            (function() {{
                const audioData = "{audio_b64}";
                const binaryStr = atob(audioData);
                const bytes = new Uint8Array(binaryStr.length);
                for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
                const audioBlob = new Blob([bytes], {{ type: 'audio/wav' }});
                const audioUrl = URL.createObjectURL(audioBlob);
                const audioEl = document.getElementById('customAudio_{key_suffix}');
                audioEl.src = audioUrl;
                document.getElementById('voiceBtn_{key_suffix}').onclick = () => audioEl.play();
            }})();
        </script>
        """
        return html
    else:
        # fallback to browser TTS
        safe_text = json.dumps(text)
        html = f"""
        <button id="ttsBtn_{key_suffix}" style="background-color:#ffaa33; border:none; border-radius:30px; padding:5px 12px; cursor:pointer;">🔊</button>
        <script>
            (function() {{
                const btn = document.getElementById('ttsBtn_{key_suffix}');
                let utterance = null;
                btn.onclick = () => {{
                    if (utterance) window.speechSynthesis.cancel();
                    utterance = new SpeechSynthesisUtterance({safe_text});
                    utterance.lang = 'fr-FR';
                    window.speechSynthesis.speak(utterance);
                }};
            }})();
        </script>
        """
        return html

# ---------- MAIN CHAT INTERFACE ----------
def main():
    load_previous_training()

    st.markdown("<h1 style='text-align:center;'>💬 Gesner AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;'>I learn from what you teach me. Ask me anything!</p>", unsafe_allow_html=True)

    # Display chat history
    for idx, msg in enumerate(st.session_state.conversation_history):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                btn_html = play_voice_button(msg["content"], f"chat_{idx}")
                st.components.v1.html(btn_html, height=50)

    # Chat input
    user_input = st.text_input("Your question:", key="chat_input")
    if st.button("Send", use_container_width=True):
        if user_input.strip():
            answer = generate_answer(user_input)
            st.session_state.conversation_history.append({"role": "user", "content": user_input})
            st.session_state.conversation_history.append({"role": "assistant", "content": answer})
            st.rerun()

    # Training section (lightweight – only plain text)
    with st.expander("📚 Train me with new knowledge"):
        new_fact = st.text_area("Enter a fact, sentence, or question‑answer pair:")
        voice_file = st.file_uploader("Upload your voice for this text (optional)", type=["wav", "mp3"])
        if st.button("Learn this", use_container_width=True):
            if new_fact.strip():
                if voice_file:
                    save_voice_for_text(new_fact.strip(), voice_file.read())
                add_to_training(new_fact.strip())
                st.rerun()
            else:
                st.warning("Please enter some text.")

    # Sidebar – simple info and reset
    with st.sidebar:
        st.markdown("## 🌍 GlobalInternet.py")
        st.markdown("### Gesner AI – Your Personal AI")
        st.markdown("---")
        st.markdown("**Gesner Deslandes – Coder in Chief**")
        st.markdown("📞 (509)-47385663")
        st.markdown("✉️ deslandes78@gmail.com")
        st.markdown("---")
        if st.button("🗑️ Clear chat history", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()
        if st.button("🔥 Reset all knowledge (danger!)", use_container_width=True):
            st.session_state.training_data = []
            st.session_state.texts = []
            st.session_state.conversation_history = []
            if os.path.exists("training_data.json"):
                os.remove("training_data.json")
            import shutil
            if os.path.exists(VOICE_CACHE_DIR):
                shutil.rmtree(VOICE_CACHE_DIR)
                os.makedirs(VOICE_CACHE_DIR)
            build_tfidf()
            st.success("All knowledge erased. Start fresh!")
            time.sleep(1)
            st.rerun()

    st.markdown(f'<div class="footer">© GlobalInternet.py – Gesner AI | Fast, lightweight, always learning</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
