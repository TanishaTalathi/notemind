"""Voice notes: speech-to-text with Whisper (faster-whisper / CTranslate2, runs offline on the CPU).

Whisper is a multilingual encoder-decoder transformer trained on 680k hours of speech. It detects the spoken
language itself and handles English, Hindi and code-mixed Hinglish. Set NOTEMIND_WHISPER_MODEL to pick a size:
tiny (75 MB) · base (145 MB, fine for English, poor Hindi) · small (480 MB, default) · medium (1.5 GB, best Hindi).

Whisper often confuses spoken Hindi with Urdu (same spoken language, different script) and then writes Arabic
script. So when it hears Hindi or Urdu we decode as Hindi with a Devanagari prompt, which steers the output script.
"""
import io
import logging
import os
import threading
from functools import lru_cache

log = logging.getLogger("notemind.transcribe")
MODEL_SIZE = os.getenv("NOTEMIND_WHISPER_MODEL", "small")
HINDI_PROMPT = "यह वाक्य हिंदी में देवनागरी लिपि में लिखा गया है।"
_lock = threading.Lock()


@lru_cache(maxsize=1)
def _model():
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        return None
    try:
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
    except Exception as exc:  # pragma: no cover - no internet on first download, low memory...
        log.warning("Could not load Whisper %s: %s", MODEL_SIZE, exc)
        return None


def available() -> bool:
    return _model() is not None


def transcribe(audio: bytes, language: str = "auto") -> dict:
    """Transcribe an audio file (webm/ogg/wav/mp3/m4a...). language: 'auto', 'en' or 'hi'."""
    model = _model()
    if model is None:
        return {"available": False, "text": "", "segments": [],
                "message": "Voice notes need faster-whisper: pip install -r requirements.txt"}
    options = dict(beam_size=5, vad_filter=True,  # skip silence, which also stops Whisper inventing text in pauses
                   condition_on_previous_text=False)
    with _lock:  # one transcription at a time keeps memory use predictable
        lang = None if language == "auto" else language
        segments, info = model.transcribe(io.BytesIO(audio), language=lang,
                                          initial_prompt=HINDI_PROMPT if lang == "hi" else None, **options)
        if lang is None and info.language in ("hi", "ur"):
            # segments are generated lazily, so re-running costs nothing but the language detection
            segments, info = model.transcribe(io.BytesIO(audio), language="hi", initial_prompt=HINDI_PROMPT, **options)
        parts = [{"start": round(s.start, 1), "end": round(s.end, 1), "text": s.text.strip()} for s in segments]
    text = " ".join(p["text"] for p in parts if p["text"]).strip()
    return {"available": True, "text": text, "segments": parts, "language": info.language,
            "language_probability": round(info.language_probability, 2), "duration": round(info.duration, 1),
            "engine": f"whisper-{MODEL_SIZE}",
            "message": "" if text else "No speech was recognised. Speak closer to the microphone and try again."}
