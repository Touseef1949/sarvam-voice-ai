#!/usr/bin/env python3
"""
Sarvam Voice AI Toolkit — Conversational Voice Agent + TTS Studio
Streamlit app powered by Sarvam AI (saaras:v3 STT, bulbul:v3 TTS)

Two tools in one:
  🎙️ Voice Agent  — Record voice, transcribe, get AI reply, hear it spoken back
  🔊 TTS Studio  — Convert text to speech with 7 Indian voices side-by-side

Requires a Sarvam AI API key. Get one at https://sarvam.ai
"""

import os
import io
import uuid
import time
import base64
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import requests
import streamlit as st
from sarvamai import AsyncSarvamAI, SarvamAI

# ──────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Sarvam Voice AI Toolkit",
    page_icon="🎙️",
    layout="wide",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
TTS_STREAM_URL = "https://api.sarvam.ai/text-to-speech/stream"

VOICES = [
    {"id": "rahul",   "name": "Rahul",  "gender": "Male",   "desc": "Composed voice that builds trust"},
    {"id": "kavya",   "name": "Kavya",  "gender": "Female", "desc": "Everyday conversational tone"},
    {"id": "ratan",   "name": "Ratan",  "gender": "Male",   "desc": "Sharp articulation for clarity"},
    {"id": "priya",   "name": "Priya",  "gender": "Female", "desc": "Upbeat voice with personality"},
    {"id": "ishita",  "name": "Ishita", "gender": "Female", "desc": "Polished voice for enterprise use"},
    {"id": "shreya",  "name": "Shreya", "gender": "Female", "desc": "Precise pronunciation and enunciation"},
    {"id": "shruti",  "name": "Shruti", "gender": "Female", "desc": "Natural and expressive delivery"},
]

LANGUAGES = {
    "English (India)": "en-IN", "Hindi": "hi-IN", "Telugu": "te-IN",
    "Tamil": "ta-IN", "Kannada": "kn-IN", "Malayalam": "ml-IN",
    "Bengali": "bn-IN", "Marathi": "mr-IN", "Gujarati": "gu-IN",
    "Punjabi": "pa-IN", "Odia": "od-IN",
}

STT_MODES = ["transcribe", "translate", "verbatim", "translit", "codemix"]

CODEC_MIME = {"mp3": "audio/mpeg", "wav": "audio/wav"}


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────
def get_api_key() -> str:
    """Try Streamlit secrets first, then env var, else user must type it."""
    return st.secrets.get("SARVAM_API_KEY", os.getenv("SARVAM_API_KEY", "")).strip()


def safe_get(obj: Any, *keys: str, default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur if cur is not None else default


def run_async(coro):
    try:
        return asyncio.run(coro)
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()


def chunk_text(text: str, max_chars: int = 3500) -> List[str]:
    """Split text into chunks, respecting paragraph and sentence boundaries."""
    if len(text) <= max_chars:
        return [text]
    chunks, current = [], ""
    for para in text.split("\n"):
        if len(current) + len(para) + 1 <= max_chars:
            current += para + "\n"
        else:
            if current.strip():
                chunks.append(current.strip())
                current = ""
            if len(para) > max_chars:
                import re
                for sentence in re.split(r'(?<=[.!?|।])\s+', para):
                    if len(current) + len(sentence) + 1 <= max_chars:
                        current += sentence + " "
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                            current = ""
                        if len(sentence) > max_chars:
                            for word in sentence.split(" "):
                                if len(current) + len(word) + 1 <= max_chars:
                                    current += word + " "
                                else:
                                    if current.strip():
                                        chunks.append(current.strip())
                                    current = word + " "
                        else:
                            current = sentence + " "
            else:
                current = para + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ──────────────────────────────────────────────
# Voice Agent functions
# ──────────────────────────────────────────────
async def transcribe_audio_async(audio_bytes: bytes, api_key: str,
                                  language_code: str, stt_mode: str) -> str:
    client = AsyncSarvamAI(api_subscription_key=api_key)
    encoded = base64.b64encode(audio_bytes).decode("utf-8")
    async with client.speech_to_text_streaming.connect(
        model="saaras:v3", mode=stt_mode, language_code=language_code,
        high_vad_sensitivity=True, flush_signal=True,
    ) as ws:
        await ws.transcribe(audio=encoded, encoding="audio/wav", sample_rate=16000)
        await ws.flush()
        parts: List[str] = []
        timeout_at = time.time() + 30
        async for message in ws:
            msg_type = safe_get(message, "type", default="")
            text = (safe_get(message, "text") or safe_get(message, "data", "text")
                    or safe_get(message, "transcript") or safe_get(message, "data", "transcript"))
            if text:
                parts.append(str(text))
            if msg_type in {"transcript", "final_transcript", "result", "final"} and parts:
                break
            if time.time() > timeout_at:
                break
        return " ".join(p.strip() for p in parts if p and str(p).strip()).strip()


def transcribe_audio(audio_bytes: bytes, api_key: str,
                     language_code: str, stt_mode: str) -> str:
    return run_async(transcribe_audio_async(audio_bytes, api_key, language_code, stt_mode))


def get_chat_reply(api_key: str, messages: List[Dict[str, str]],
                   temperature: float, top_p: float, max_tokens: int) -> str:
    client = SarvamAI(api_subscription_key=api_key)
    response = client.chat.completions(
        messages=messages, temperature=temperature,
        top_p=top_p, max_tokens=max_tokens,
    )
    content = (safe_get(response, "choices", default=[None])[0]
               if safe_get(response, "choices", default=[]) else None)
    answer = (safe_get(content, "message", "content") or safe_get(content, "content")
              or safe_get(response, "message", "content") or str(response))
    return str(answer).strip()


def synthesize_speech(api_key: str, text: str, language_code: str,
                      speaker: str, pace: float) -> bytes:
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text, "target_language_code": language_code,
        "speaker": speaker, "model": "bulbul:v3", "pace": pace,
        "speech_sample_rate": 22050, "output_audio_codec": "mp3",
        "enable_preprocessing": True,
    }
    buf = io.BytesIO()
    with requests.post(TTS_STREAM_URL, headers=headers, json=payload,
                       stream=True, timeout=120) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                buf.write(chunk)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────
def init_state():
    if "va_messages" not in st.session_state:
        st.session_state.va_messages = [
            {"role": "system", "content": "You are a helpful, concise, friendly multilingual voice assistant for Indian users."}
        ]
    for key in ["va_transcript", "va_reply", "va_audio", "va_turn"]:
        if key not in st.session_state:
            st.session_state[key] = "" if key != "va_audio" else None
            if key == "va_turn":
                st.session_state[key] = 0
    if "tts_generated" not in st.session_state:
        st.session_state.tts_generated: Dict[str, dict] = {}


# ──────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────
init_state()

st.title("🎙️ Sarvam Voice AI Toolkit")
st.caption("Voice Agent + TTS Studio — powered by Sarvam AI. Supports 11 Indian languages.")

# ── Sidebar ──
with st.sidebar:
    st.header("⚙️ Settings")
    saved_key = get_api_key()
    api_key = st.text_input(
        "Sarvam API Key",
        value=saved_key,
        type="password",
        placeholder="Paste your Sarvam API key",
        help="Get a key at https://sarvam.ai — not stored on our servers.",
    )
    st.markdown("---")
    st.markdown("**🔗 Resources**")
    st.markdown("- [Get API Key](https://sarvam.ai)")
    st.markdown("- [Sarvam Docs](https://docs.sarvam.ai)")
    st.markdown("- [GitHub Repo](https://github.com/Touseef1949)")
    st.markdown("---")
    st.markdown("<p style='text-align:center; color:#888; font-size:0.85rem;'>🎙️ Built by <a href='https://touseefshaik.com' target='_blank'>Touseef Shaik</a></p>", unsafe_allow_html=True)

if not api_key.strip():
    st.warning("👈 Enter your Sarvam API key in the sidebar to get started.")
    st.info(
        "**Don't have a key?** Get one at [sarvam.ai](https://sarvam.ai). "
        "Sarvam AI offers free credits for new users."
    )
    st.stop()

# ── Tabs ──
tab1, tab2 = st.tabs(["🎙️ Voice Agent", "🔊 TTS Studio"])

# ═══════════════════════════════════════════════
# TAB 1: VOICE AGENT
# ═══════════════════════════════════════════════
with tab1:
    st.subheader("Conversational Voice Agent")
    st.caption("Record your voice or type a message. The agent transcribes, replies, and speaks back.")

    with st.expander("Voice Agent Settings", expanded=False):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            va_stt_mode = st.selectbox("STT Mode", STT_MODES, index=0,
                                        help="translate = English output from any language")
            va_stt_lang = st.selectbox("Input Language", list(LANGUAGES.keys()), index=0)
        with col_b:
            va_tts_lang = st.selectbox("Reply Language", list(LANGUAGES.keys()), index=0)
            va_speaker = st.selectbox("TTS Voice", [v["id"] for v in VOICES], index=5,
                                       format_func=lambda x: next((f"{v['name']}" for v in VOICES if v["id"]==x), x))
        with col_c:
            va_temp = st.slider("Temperature", 0.0, 1.5, 0.5, 0.1)
            va_top_p = st.slider("Top-p", 0.1, 1.0, 1.0, 0.1)
            va_max_tok = st.slider("Max tokens", 64, 2000, 500, 32)
            va_pace = st.slider("Voice Pace", 0.7, 1.5, 1.1, 0.1)

        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.va_messages = [
                {"role": "system", "content": st.session_state.va_messages[0]["content"]}
            ]
            st.session_state.va_transcript = ""
            st.session_state.va_reply = ""
            st.session_state.va_audio = None
            st.session_state.va_turn = 0
            st.rerun()

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 🎤 Voice Input")
        recorded = st.audio_input("Record your message", sample_rate=16000)
        if recorded:
            st.audio(recorded)

        st.markdown("#### ⌨️ Text Input")
        typed = st.text_area("Or type your message", placeholder="Ask something...", key="va_typed")
        cta1, cta2 = st.columns(2)
        with cta1:
            send_text = st.button("Send Text", use_container_width=True)
        with cta2:
            send_voice = st.button("Process Audio", use_container_width=True)

    with col2:
        st.markdown("#### 🧠 Conversation")
        display = [m for m in st.session_state.va_messages if m["role"] != "system"]
        if not display:
            st.info("No conversation yet. Record audio or type a message.")
        else:
            for msg in display:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])

    def process_va_message(user_text: str):
        if not user_text.strip():
            st.warning("No text detected. Please try again.")
            return
        st.session_state.va_transcript = user_text.strip()
        st.session_state.va_messages.append({"role": "user", "content": user_text.strip()})
        with st.spinner("Generating reply..."):
            reply = get_chat_reply(api_key, st.session_state.va_messages, va_temp, va_top_p, va_max_tok)
        st.session_state.va_messages.append({"role": "assistant", "content": reply})
        st.session_state.va_reply = reply
        st.session_state.va_turn += 1
        with st.spinner("Synthesizing speech..."):
            audio = synthesize_speech(api_key, reply, LANGUAGES[va_tts_lang], va_speaker, va_pace)
        st.session_state.va_audio = audio

    if send_text and typed.strip():
        process_va_message(typed)
        st.rerun()

    if send_voice:
        if recorded is None:
            st.warning("Please record audio first.")
        else:
            with st.spinner("Transcribing..."):
                transcript = transcribe_audio(
                    recorded.getvalue(), api_key, LANGUAGES[va_stt_lang], va_stt_mode)
            process_va_message(transcript)
            st.rerun()

    st.divider()
    cl, cr = st.columns([1, 1])
    with cl:
        st.markdown("#### Latest Transcript")
        st.code(st.session_state.va_transcript or "—", language="text")
        st.markdown("#### Latest Reply")
        st.code(st.session_state.va_reply or "—", language="text")
    with cr:
        st.markdown("#### Assistant Audio")
        if st.session_state.va_audio:
            st.audio(st.session_state.va_audio, format="audio/mpeg")
            st.download_button("Download Reply", st.session_state.va_audio,
                               f"sarvam_reply_{uuid.uuid4().hex[:8]}.mp3",
                               "audio/mpeg", use_container_width=True)
        else:
            st.write("—")

# ═══════════════════════════════════════════════
# TAB 2: TTS STUDIO
# ═══════════════════════════════════════════════
with tab2:
    st.subheader("Text-to-Speech Studio")
    st.caption("Convert text to speech with 7 Indian voices. Compare side-by-side.")

    left_col, right_col = st.columns([1.15, 1.0], gap="large")

    with left_col:
        tts_text = st.text_area(
            "Input Text", height=260,
            value="नमस्ते! Sarvam AI में आपका स्वागत है।\n\nहम भारतीय भाषाओं के लिए voice technology बनाते हैं।",
            placeholder="Type or paste text here...",
            label_visibility="collapsed",
        )
        tts_lang_label = st.selectbox("Language", list(LANGUAGES.keys()), index=1)
        tts_lang_code = LANGUAGES[tts_lang_label]
        tts_pace = st.slider("Speed", 0.7, 1.5, 1.1, 0.05, key="tts_pace")
        tts_codec = st.selectbox("Output Format", ["mp3", "wav"], index=0)
        tts_preprocess = st.checkbox("Enable Preprocessing", value=True)

        st.markdown("---")
        col_a1, col_a2 = st.columns(2)
        with col_a1:
            gen_all = st.button("🎵 Generate All Voices", use_container_width=True, type="primary")
        with col_a2:
            if st.button("🗑️ Clear Previews", use_container_width=True):
                st.session_state.tts_generated = {}
                st.rerun()

    with right_col:
        st.markdown("#### 🎧 Voices")
        st.caption("Click Generate to create a preview.")

        tabs = st.tabs(["Conversational", "Enterprise"])
        with tabs[0]:
            for voice in VOICES[:5]:
                vid = voice["id"]
                existing = st.session_state.tts_generated.get(vid)
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2.3, 1.2, 2.5])
                    with c1:
                        tag = "🟦 Male" if voice["gender"] == "Male" else "🟪 Female"
                        st.markdown(f"**{voice['name']}**")
                        st.caption(f"{tag} • {voice['desc']}")
                    with c2:
                        clicked = st.button("Generate" if not existing else "Regen",
                                            key=f"tts_{vid}", use_container_width=True)
                    with c3:
                        if existing:
                            mime = CODEC_MIME.get(existing["codec"], "audio/mpeg")
                            st.audio(existing["bytes"], format=mime)
                            st.download_button("Download", existing["bytes"],
                                               f"{vid}.{existing['codec']}", mime,
                                               key=f"dl_{vid}", use_container_width=True)
                        else:
                            st.caption("No preview yet.")
                    if clicked:
                        if not tts_text.strip():
                            st.error("Enter text first.")
                        else:
                            try:
                                chunks = chunk_text(tts_text)
                                all_bytes = b""
                                with st.spinner(f"Generating {voice['name']}..."):
                                    for chunk in chunks:
                                        audio, _ = _tts_stream(api_key, chunk, tts_lang_code, vid, tts_pace, tts_codec, tts_preprocess)
                                        all_bytes += audio
                                st.session_state.tts_generated[vid] = {
                                    "bytes": all_bytes, "codec": tts_codec,
                                }
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

        with tabs[1]:
            for voice in VOICES[5:]:
                vid = voice["id"]
                existing = st.session_state.tts_generated.get(vid)
                with st.container(border=True):
                    c1, c2, c3 = st.columns([2.3, 1.2, 2.5])
                    with c1:
                        st.markdown(f"**{voice['name']}**")
                        st.caption(f"🟪 Female • {voice['desc']}")
                    with c2:
                        clicked = st.button("Generate" if not existing else "Regen",
                                            key=f"tts_{vid}", use_container_width=True)
                    with c3:
                        if existing:
                            mime = CODEC_MIME.get(existing["codec"], "audio/mpeg")
                            st.audio(existing["bytes"], format=mime)
                            st.download_button("Download", existing["bytes"],
                                               f"{vid}.{existing['codec']}", mime,
                                               key=f"dl_{vid}", use_container_width=True)
                        else:
                            st.caption("No preview yet.")
                    if clicked:
                        if not tts_text.strip():
                            st.error("Enter text first.")
                        else:
                            try:
                                chunks = chunk_text(tts_text)
                                all_bytes = b""
                                with st.spinner(f"Generating {voice['name']}..."):
                                    for chunk in chunks:
                                        audio, _ = _tts_stream(api_key, chunk, tts_lang_code, vid, tts_pace, tts_codec, tts_preprocess)
                                        all_bytes += audio
                                st.session_state.tts_generated[vid] = {"bytes": all_bytes, "codec": tts_codec}
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed: {e}")

    # Generate all voices
    if gen_all:
        if not tts_text.strip():
            st.error("Enter text first.")
        else:
            chunks = chunk_text(tts_text)
            prog = st.progress(0)
            msg = st.empty()
            for idx, voice in enumerate(VOICES, start=1):
                msg.info(f"Generating {idx}/{len(VOICES)}: {voice['name']}")
                try:
                    all_bytes = b""
                    for chunk in chunks:
                        audio, _ = _tts_stream(api_key, chunk, tts_lang_code, voice["id"], tts_pace, tts_codec, tts_preprocess)
                        all_bytes += audio
                    st.session_state.tts_generated[voice["id"]] = {"bytes": all_bytes, "codec": tts_codec}
                except Exception as e:
                    st.warning(f"Skipping {voice['name']}: {e}")
                prog.progress(int((idx / len(VOICES)) * 100))
            msg.success("All voices generated!")
            time.sleep(0.5)
            st.rerun()


def _tts_stream(api_key: str, text: str, lang: str, speaker: str,
                pace: float, codec: str, preprocess: bool) -> Tuple[bytes, str]:
    headers = {"api-subscription-key": api_key, "Content-Type": "application/json"}
    payload = {
        "text": text, "target_language_code": lang, "speaker": speaker,
        "model": "bulbul:v3", "pace": pace, "speech_sample_rate": 22050,
        "output_audio_codec": codec, "enable_preprocessing": preprocess,
    }
    buf = io.BytesIO()
    with requests.post(TTS_STREAM_URL, headers=headers, json=payload,
                       stream=True, timeout=120) as r:
        r.raise_for_status()
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                buf.write(chunk)
    return buf.getvalue(), codec
