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

# ---------- LANGUAGES (unchanged, omitted for brevity) ----------
# ... (full TEXTS dictionary as in previous version) ...
# For space, we keep the same TEXTS as before.
# In the final code you would include the entire TEXTS dictionary.

# ---------- CSS (unchanged) ----------
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

# ---------- DATA STORAGE INIT ----------
if "dictionaries" not in st.session_state:
    st.session_state.dictionaries = {"ht": {}, "fr": {}, "en": {}}
if "audio_transcriptions" not in st.session_state:
    st.session_state.audio_transcriptions = []
if "encyclopedia" not in st.session_state:
    st.session_state.encyclopedia = []

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

def play_voice_button(text, button_label="🔊", key_suffix=""):
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

# ---------- TRAINING FUNCTIONS (unchanged, omitted for brevity) ----------
# ... (add_to_training, load_previous_training, generate_response, etc.) ...

# For the final code, include all the previous training functions.
# Here we show the new chat_mode_interface with auto‑play.
# Assume all other functions (login_page, show_sidebar, dictionary_manager, voice_training, translation_correction, encyclopedia_manager, test_training, training_mode) are present as before.

# ---------- GESNER AI CHAT MODE (with auto‑voice playback) ----------
def chat_mode_interface():
    t = TEXTS[st.session_state.language]
    st.markdown(f"<h1 style='text-align:center; color:#ffd966;'>{t['chat_mode_title']}</h1>", unsafe_allow_html=True)
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    
    # Display conversation and auto‑play voice for new assistant messages
    for idx, msg in enumerate(st.session_state.chat_messages):
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-message user-message">🧑‍💻 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            col1, col2 = st.columns([10, 1])
            with col1:
                st.markdown(f'<div class="chat-message assistant-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
            with col2:
                st.components.v1.html(play_voice_button(msg["content"], t['chat_speak_button'], f"chat_{idx}"), height=50)
            
            # Auto-play voice for the latest assistant message (if not already played)
            if idx == len(st.session_state.chat_messages) - 1 and not msg.get("played", False):
                voice_bytes = get_voice_for_text(msg["content"])
                if voice_bytes:
                    audio_b64 = base64.b64encode(voice_bytes).decode()
                    mime = "audio/wav"
                    auto_html = f"""
                    <audio id="autoPlayAudio_{idx}" style="display:none;" autoplay></audio>
                    <script>
                        (function() {{
                            const audioData = "{audio_b64}";
                            const binaryStr = atob(audioData);
                            const bytes = new Uint8Array(binaryStr.length);
                            for (let i = 0; i < binaryStr.length; i++) bytes[i] = binaryStr.charCodeAt(i);
                            const audioBlob = new Blob([bytes], {{ type: '{mime}' }});
                            const audioUrl = URL.createObjectURL(audioBlob);
                            const audio = document.getElementById('autoPlayAudio_{idx}');
                            audio.src = audioUrl;
                            audio.play().catch(e => console.log("Autoplay prevented:", e));
                            // Mark message as played to avoid repeated auto‑play on rerun
                            window.parent.postMessage({{type: "markPlayed", idx: {idx}}}, "*");
                        }})();
                    </script>
                    """
                    st.components.v1.html(auto_html, height=0)
                    msg["played"] = True
                else:
                    # Fallback TTS auto‑play (optional, but we skip to avoid unexpected audio)
                    pass
    
    user_input = st.text_input(t['chat_mode_placeholder'], key="chat_input_new")
    if st.button(t['send_button'], use_container_width=True, key="chat_send_new"):
        if user_input.strip():
            answer = generate_response(user_input)
            st.session_state.chat_messages.append({"role": "user", "content": user_input})
            st.session_state.chat_messages.append({"role": "assistant", "content": answer, "played": False})
            st.rerun()
    
    # Image upload section (unchanged)
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
                add_to_training(description, TEXTS[st.session_state.language])
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

# ---------- MAIN APP (unchanged) ----------
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
