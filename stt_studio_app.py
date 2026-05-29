import os
import json
import asyncio
import base64
from typing import Any, Dict, List, Union

import streamlit as st
from sarvamai import AsyncSarvamAI


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Sarvam AI • Streaming STT Studio",
    page_icon="🎙️",
    layout="wide",
)


# -----------------------------
# Helpers
# -----------------------------
LANGUAGE_OPTIONS = {
    "English (India)": "en-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Malayalam": "ml-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Punjabi": "pa-IN",
    "Bengali": "bn-IN",
    "Odia": "od-IN",
}

MODE_OPTIONS = [
    "transcribe",
    "verbatim",
    "translit",
    "codemix",
    "translate",
]


def safe_to_dict(obj: Any) -> Dict[str, Any]:
    """Best-effort conversion of SDK/websocket response to dict."""
    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump()
        except Exception:
            pass

    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        try:
            return dict(obj.__dict__)
        except Exception:
            pass

    return {"raw": str(obj)}


def extract_text(message: Union[Dict[str, Any], Any]) -> str:
    """Extract transcript/translation text from different response shapes."""
    msg = safe_to_dict(message)

    for key in ["text", "transcript", "translation", "output", "result"]:
        value = msg.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    data = msg.get("data")
    if isinstance(data, dict):
        for key in ["text", "transcript", "translation", "output", "result"]:
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return json.dumps(msg, indent=2, ensure_ascii=False)


async def transcribe_with_sarvam(
    api_key: str,
    audio_bytes: bytes,
    mode: str,
    language_code: str,
    sample_rate: int,
    high_vad_sensitivity: bool,
    vad_signals: bool,
    flush_signal: bool,
) -> Dict[str, Any]:
    """
    Send uploaded WAV audio to Sarvam streaming STT and return result.
    """
    audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
    client = AsyncSarvamAI(api_subscription_key=api_key)

    events: List[Dict[str, Any]] = []
    final_text = ""
    final_raw: Dict[str, Any] = {}

    connect_kwargs = {
        "model": "saaras:v3",
        "mode": mode,
        "sample_rate": sample_rate,
        "input_audio_codec": "wav",
        "high_vad_sensitivity": high_vad_sensitivity,
        "vad_signals": vad_signals,
        "flush_signal": flush_signal,
    }

    # For translate mode, docs indicate language code is not required.
    if mode != "translate":
        connect_kwargs["language_code"] = language_code

    async with client.speech_to_text_streaming.connect(**connect_kwargs) as ws:
        if mode == "translate":
            await ws.translate(
                audio=audio_b64,
                encoding="audio/wav",
                sample_rate=sample_rate,
            )
        else:
            await ws.transcribe(
                audio=audio_b64,
                encoding="audio/wav",
                sample_rate=sample_rate,
            )

        if flush_signal:
            await ws.flush()

        if vad_signals:
            async for message in ws:
                msg_dict = safe_to_dict(message)
                events.append(msg_dict)

                msg_type = msg_dict.get("type")
                if msg_type in {"transcript", "translation"}:
                    final_raw = msg_dict
                    final_text = extract_text(msg_dict)
                    break
        else:
            message = await ws.recv()
            final_raw = safe_to_dict(message)
            events.append(final_raw)
            final_text = extract_text(final_raw)

    return {
        "text": final_text,
        "events": events,
        "raw": final_raw,
    }


def main():
    st.title("🎙️ Sarvam AI Streaming STT Studio")
    st.caption("Upload a WAV file, preview it, and transcribe it with a cleaner Streamlit interface.")

    with st.sidebar:
        st.header("Configuration")

        default_api_key = os.getenv("SARVAM_API_KEY", "")
        api_key = st.text_input(
            "Sarvam API Key",
            value=default_api_key,
            type="password",
            placeholder="Paste your Sarvam API key",
        )

        mode = st.selectbox("Mode", MODE_OPTIONS, index=0)
        language_label = st.selectbox("Language", list(LANGUAGE_OPTIONS.keys()), index=0)
        language_code = LANGUAGE_OPTIONS[language_label]

        sample_rate = st.selectbox("Sample Rate", [16000, 8000], index=0)
        high_vad_sensitivity = st.toggle("High VAD Sensitivity", value=True)
        vad_signals = st.toggle("VAD Signals", value=True)
        flush_signal = st.toggle("Flush Signal", value=False)

        st.divider()
        st.markdown(
            """
            **Recommended**
            - Use **WAV**
            - Use **16 kHz**
            - Keep speech clear with minimal background noise
            """
        )

    uploaded_file = st.file_uploader(
        "Upload a WAV file",
        type=["wav"],
        help="Streaming STT works with WAV/raw PCM. This app is set up for WAV uploads.",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Model", "saaras:v3")
    with col_b:
        st.metric("Mode", mode)
    with col_c:
        st.metric("Language", "Auto/Selected" if mode == "translate" else language_code)

    if uploaded_file is not None:
        audio_bytes = uploaded_file.read()

        left, right = st.columns([1, 1])

        with left:
            st.subheader("Audio Preview")
            st.audio(audio_bytes, format="audio/wav")
            st.info(
                f"**File:** {uploaded_file.name}\n\n"
                f"**Size:** {len(audio_bytes) / 1024:.1f} KB"
            )

        with right:
            st.subheader("Transcript Output")
            result_placeholder = st.empty()
            result_placeholder.code("Your transcription will appear here...", language="text")

        st.divider()

        run = st.button("🚀 Start Transcription", use_container_width=True)

        if run:
            if not api_key.strip():
                st.error("Please enter your Sarvam API key in the sidebar.")
                st.stop()

            try:
                with st.spinner("Connecting to Sarvam and processing audio..."):
                    result = asyncio.run(
                        transcribe_with_sarvam(
                            api_key=api_key.strip(),
                            audio_bytes=audio_bytes,
                            mode=mode,
                            language_code=language_code,
                            sample_rate=sample_rate,
                            high_vad_sensitivity=high_vad_sensitivity,
                            vad_signals=vad_signals,
                            flush_signal=flush_signal,
                        )
                    )

                transcript_text = result["text"].strip() if result["text"] else ""
                raw_response = result["raw"]
                event_log = result["events"]

                if transcript_text:
                    result_placeholder.code(transcript_text, language="text")
                    st.success("Transcription completed.")
                else:
                    result_placeholder.code(
                        json.dumps(raw_response, indent=2, ensure_ascii=False),
                        language="json",
                    )
                    st.warning("No plain transcript text was found. Showing raw response instead.")

                tab1, tab2, tab3 = st.tabs(["Transcript", "Raw Response", "Event Log"])

                with tab1:
                    st.text_area(
                        "Transcript",
                        value=transcript_text if transcript_text else "",
                        height=250,
                    )
                    if transcript_text:
                        st.download_button(
                            "Download Transcript (.txt)",
                            data=transcript_text,
                            file_name="sarvam_transcript.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )

                with tab2:
                    st.json(raw_response)

                with tab3:
                    st.json(event_log)

            except Exception as e:
                st.error(f"Error: {e}")

    else:
        st.markdown(
            """
            ### What this app gives you
            - Upload + audio preview
            - Sidebar controls for mode, language, sample rate, and VAD
            - Side-by-side audio and output layout
            - Transcript + raw JSON + event log
            - Download transcript as `.txt`
            """
        )


if __name__ == "__main__":
    main()