"""Language identification for English, Hindi (Devanagari) and Hinglish (Roman-script Hindi mixed with English)."""
from .hinglish import hinglish_score
from .text import script_ratio


def detect(text: str) -> dict:
    """Script analysis first (Devanagari share), then Hinglish marker words for Roman-script text."""
    if not text.strip():
        return {"language": "en", "confidence": 0.0, "devanagari_ratio": 0.0, "hinglish_ratio": 0.0}
    dev = script_ratio(text)
    markers, latin_words = hinglish_score(text)
    hing = markers / latin_words if latin_words else 0.0

    if dev >= 0.5:
        language, confidence = "hi", 0.6 + 0.4 * dev
    elif dev >= 0.15 or (markers >= 2 and hing >= 0.12) or (markers >= 1 and latin_words <= 6):
        language, confidence = "hinglish", min(0.99, 0.55 + hing + dev)
    else:
        language, confidence = "en", max(0.55, 1.0 - 4 * hing - dev)
    return {"language": language, "confidence": round(min(confidence, 0.99), 2),
            "devanagari_ratio": round(dev, 2), "hinglish_ratio": round(hing, 2)}


def base_language(text: str) -> str:
    """'hi' if the text is mostly Devanagari, otherwise 'en' (Hinglish is processed as English-script text)."""
    return "hi" if script_ratio(text) >= 0.5 else "en"
