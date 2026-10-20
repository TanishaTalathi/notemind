"""English <-> Hindi translation.

Engine order:
  1. MarianMT (Helsinki-NLP opus-mt) running locally, if `transformers` is installed.
  2. Google Translate through the `deep-translator` package (needs internet, no API key).
  3. MyMemory through `deep-translator` (needs internet, no API key), for when Google
     rate-limits or blocks the network.
If none is available the API returns a clear message instead of crashing.
"""
import logging
from . import models
from .text import split_sentences

log = logging.getLogger("notemind.translate")
MAX_CHUNK_CHARS = 4000
MYMEMORY_CHUNK_CHARS = 450  # MyMemory rejects queries over 500 characters
MYMEMORY_CODES = {"en": "en-US", "hi": "hi-IN"}


def _chunks(text: str, limit: int) -> list[str]:
    """Group sentences into chunks under `limit` characters, keeping paragraph breaks."""
    chunks, current = [], ""
    for paragraph in text.split("\n"):
        sentences = split_sentences(paragraph) or ([paragraph] if paragraph.strip() else [])
        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > limit:
                chunks.append(current)
                current = ""
            current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current)
            current = ""
    return chunks


def _marian(text: str, source: str, target: str) -> str | None:
    loaded = models.translation_pipeline(source, target)
    if not loaded:
        return None
    tokenizer, model = loaded
    sentences = split_sentences(text) or [text]
    out = []
    for i in range(0, len(sentences), 8):
        batch = tokenizer(sentences[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=400)
        generated = model.generate(**batch, max_new_tokens=400, num_beams=4)
        out.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
    return " ".join(out)


def _google(text: str, source: str, target: str) -> str | None:
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        return None
    try:
        translator = GoogleTranslator(source=source, target=target)
        return "\n".join(translator.translate(chunk) or "" for chunk in _chunks(text, MAX_CHUNK_CHARS))
    except Exception as exc:
        log.warning("Google translation failed: %s", str(exc)[:160])
        return None


def _mymemory(text: str, source: str, target: str) -> str | None:
    try:
        from deep_translator import MyMemoryTranslator
    except ImportError:
        return None
    if source not in MYMEMORY_CODES or target not in MYMEMORY_CODES:
        return None
    try:
        translator = MyMemoryTranslator(source=MYMEMORY_CODES[source], target=MYMEMORY_CODES[target])
        return "\n".join(translator.translate(chunk) or "" for chunk in _chunks(text, MYMEMORY_CHUNK_CHARS))
    except Exception as exc:
        log.warning("MyMemory translation failed: %s", str(exc)[:160])
        return None


_cache: dict[tuple[str, str, str], tuple[str, str]] = {}


def translate(text: str, source: str, target: str) -> tuple[str | None, str]:
    """Return (translation, engine_name). translation is None if no engine worked.
    Only successful translations are cached, so a temporary network error is retried next time."""
    text = text.strip()
    if not text or source == target:
        return text, "none"
    key = (text, source, target)
    if key in _cache:
        return _cache[key]
    for name, engine in (("marian-mt", _marian), ("google", _google), ("mymemory", _mymemory)):
        try:
            result = engine(text, source, target)
        except Exception as exc:  # pragma: no cover
            log.warning("%s failed: %s", name, exc)
            result = None
        if result:
            if len(_cache) > 512:
                _cache.clear()
            _cache[key] = (result.strip(), name)
            return _cache[key]
    return None, "unavailable"


def try_translate(text: str, source: str, target: str) -> str | None:
    return translate(text, source, target)[0]
