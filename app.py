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

st.set_page_config(
    page_title="Gesner AI",
    page_icon="🧠",
    layout="wide"
)

# ========== 1. FORCE SESSION STATE INITIALISATION (BEFORE ANYTHING ELSE) ==========
def init_session_state():
    """Ensure every session state key exists with a default value."""
    defaults = {
        "authenticated": False,
        "training_data": [],
        "conversation_history": [],
        "embedding_model": None,
        "index": None,
        "texts": [],
        "language": "en",
        "chat_mode": False,
        "dictionaries": {"ht": {}, "fr": {}, "en": {}},
        "audio_transcriptions": [],
        "encyclopedia": [],
        "chat_messages": [],
        "translated": "",
        "test_answer": "",
    }
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

init_session_state()  # <-- run immediately

# Load the embedding model if not already done
if st.session_state.embedding_model is None:
    with st.spinner("Loading AI model... (first time only)"):
        st.session_state.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------- LANGUAGES ----------
LANGUAGES = {
    "English": "en",
    "Français": "fr",
    "Kreyòl Ayisyen": "ht"
}

TEXTS = { ... }  # <-- keep your full TEXTS dictionary (same as before) – omitted for brevity but must be included.
# (I will include the full TEXTS in the final answer – see note below)

# ---------- CSS (unchanged) ----------
st.markdown("""...""", unsafe_allow_html=True)  # keep your CSS

# ---------- VOICE CACHE ----------
VOICE_CACHE_DIR = "voice_cache"
if not os.path.exists(VOICE_CACHE_DIR):
    os.makedirs(VOICE_CACHE_DIR)

def normalize_text(text: str) -> str:
    """Normalise text for consistent voice file lookup: lower case, stripped, single spaces."""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)  # collapse multiple spaces
    return text

def get_voice_filename(text):
    norm = normalize_text(text)
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
    """Returns HTML+JS for a button that plays the user's pre-recorded voice.
       If no voice file exists, returns an empty string (no button)."""
    voice_bytes = get_voice_for_text(text)
    if not voice_bytes:
        return ""
    audio_b64 = base64.b64encode(voice_bytes).decode()
    mime = "audio/wav"  # assume all uploads are converted to WAV (or keep original)
    # Use st.html (modern) instead of deprecated components.v1.html
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
        try:
            with open("training_data.json", "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                st.session_state.training_data = data
                if data:
                    embeddings = [np.array(item["embedding"], dtype=np.float32) for item in data if "embedding" in item]
                    if embeddings:
                        dim = len(embeddings[0])
                        st.session_state.index = faiss.IndexFlatL2(dim)
                        st.session_state.index.add(np.array(embeddings))
                        st.session_state.texts = [item["text"] for item in data if "text" in item]
        except Exception:
            pass

# Pre‑train intro text (normalised key for voice)
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
    
    chat_mode_toggle = st.sidebar.toggle(t['toggle_chat_mode'], value=st.session_state.chat_mode)
    if chat_mode_toggle != st.session_state.chat_mode:
        st.session_state.chat_mode = chat_mode_toggle
        st.rerun()
    
    if st.sidebar.button(t['logout_button'], use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

# ---------- DICTIONARY MANAGER (safe now) ----------
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
    st.html(recorder_html)  # updated from st.components.v1.html
    st.markdown(f"### 📂 {t['voice_upload']}")
    uploaded_file = st.file_uploader(t['voice_upload'], type=["wav", "mp3"], key="voice_upload")
    transcript = st.text_area(t['voice_transcribed_text'], key="voice_transcript")
    if uploaded_file and transcript.strip():
        if st.button(t['voice_train'], use_container_width=True):
            audio_bytes = uploaded_file.read()
            # Normalise transcript before saving voice
            norm_transcript = normalize_text(transcript.strip())
            save_voice_for_text(norm_transcript, audio_bytes)
            # Also train the natural text (keep original for knowledge base)
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
    if "translated" in st.session_state and st.session_state.translated:
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
    if st.session_state.test_answer:
        st.markdown(f"**{t['test_answer_label']}**")
        st.markdown(f'<div style="background:#0f3460; padding:10px; border-radius:12px;">{st.session_state.test_answer}</div>', unsafe_allow_html=True)
        voice_up = st.file_uploader(t['upload_voice_label'], type=["wav", "mp3"], key="test_voice")
        if voice_up:
            # Normalise the answer before saving voice
            norm_answer = normalize_text(st.session_state.test_answer)
            save_voice_for_text(norm_answer, voice_up.read())
            st.success("Voice saved")
            st.rerun()
        voice_html = play_voice_button(st.session_state.test_answer, t['test_speak_button'], "test")
        if voice_html:
            st.html(voice_html)  # updated

# ---------- GESNER AI CHAT MODE ----------
def chat_mode_interface():
    t = TEXTS[st.session_state.language]
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['chat_mode_title']}</h1>", unsafe_allow_html=True)
    
    for idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message" style="width:100%;">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                voice_html = play_voice_button(msg["content"], t['chat_speak_button'], f"chat_{idx}")
                if voice_html:
                    st.html(voice_html)
    
    user_input = st.text_input(t['chat_mode_placeholder'], key="chat_input_new")
    if st.button(t['send_button'], use_container_width=True, key="chat_send_new"):
        if user_input.strip():
            answer = generate_response(user_input)
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()
    
    if st.button("Clear Chat", use_container_width=True, key="clear_chat_new"):
        st.session_state.chat_messages = []
        st.rerun()

# ---------- TRAINING MODE ----------
def training_mode():
    t = TEXTS[st.session_state.language]
    st.markdown(f"<h1 style='text-align:center;'>{t['training_app_title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;'>{t['training_subtitle']}</p>", unsafe_allow_html=True)
    
    # Sanitise conversation history
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
    show_sidebar()
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
