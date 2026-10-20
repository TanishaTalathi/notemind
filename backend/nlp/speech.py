"""Offline text-to-speech with Meta's MMS models (facebook/mms-tts-hin and mms-tts-eng, ~145 MB each).

Used by the Listen button when the browser voice doesn't work (Chrome's online "Google" voices fail on
networks that block Google, and Windows has no Hindi voice by default). Needs `transformers` + `torch`.
"""
import io
import logging
import threading
import wave

from . import models

log = logging.getLogger("notemind.speech")
MAX_CHARS = 600
_lock = threading.Lock()  # the model is not safe to run from several request threads at once


def synthesize(text: str, lang: str = "hi") -> bytes | None:
    """Return a mono 16-bit WAV file for `text`, or None if no speech model is available."""
    text = " ".join(text.split())[:MAX_CHARS]
    if not text:
        return None
    loaded = models.tts_pipeline(lang)
    if not loaded:
        return None
    tokenizer, model = loaded
    try:
        import numpy as np
        import torch
        if lang == "en":
            text = text.lower()  # the English MMS vocabulary is lower-case letters only
        inputs = tokenizer(text, return_tensors="pt")
        if inputs["input_ids"].shape[-1] == 0:
            return None
        with _lock, torch.no_grad():
            audio = model(**inputs).waveform[0].numpy()
    except Exception as exc:
        log.warning("Speech synthesis failed: %s", str(exc)[:160])
        return None
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(model.config.sampling_rate)
        out.writeframes(pcm.tobytes())
    return buffer.getvalue()
