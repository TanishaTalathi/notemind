"""'Was this written by AI?' — an explainable estimate, not proof.

Signals (features) computed for the text:
  * log_ppl      GPT-2 perplexity (log). AI text is very predictable to a language model, so it scores low.
  * ppl_std      how much perplexity varies between sentences ("burstiness"). Humans vary more.
  * sent_len_cv  variation of sentence length (std / mean). Humans mix short and long sentences.
  * ai_phrases   stock phrases typical of chatbots ("it is important to note", "plays a crucial role"...).
  * rare_words   share of words not in the wordfreq dictionary (typos, slang, abbreviations). Humans make typos.
  * shorthand    note-taking marks: ->, &, w/, e.g., etc, !!, ... Humans write notes in shorthand.
  * word_len     average word length.
  * mattr        moving-average type/token ratio (vocabulary variety).
  * first_person I / my / we per 100 words.

A logistic-regression model (trained by benchmarks/train_ai_detector.py on the HC3 human-vs-ChatGPT corpus)
turns the standardised features into a probability. The weights live in data/ai_detector.json, so the model is
transparent: each signal's contribution (weight × z-score) is shown to the user.
GPT-2 only understands English, so Hindi/Hinglish text uses the style-only model and gets low confidence.
"""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
from wordfreq import zipf_frequency

from . import models
from .language import detect
from .text import split_sentences, tokenize

WEIGHTS_FILE = Path(__file__).resolve().parent.parent / "data" / "ai_detector.json"
LM_FEATURES = ["log_ppl", "ppl_std"]
# Em dashes were dropped: in HC3 (2023 ChatGPT) they signalled *human* text, but newer chatbots use them heavily,
# so the learned weight would push modern AI text towards "human".
STYLE_FEATURES = ["sent_len_cv", "ai_phrases", "rare_words", "shorthand", "word_len", "mattr", "first_person"]
ALL_FEATURES = LM_FEATURES + STYLE_FEATURES
MIN_WORDS = 40
MAX_LM_TOKENS = 1024

AI_PHRASES = [
    "it is important to note", "it's important to note", "it is worth noting", "it's worth noting", "plays a crucial role",
    "plays a vital role", "plays a key role", "plays an important role", "a crucial role", "a vital role", "in conclusion",
    "in summary", "overall,", "additionally,", "furthermore,", "moreover,", "delve", "a testament to", "tapestry",
    "in today's", "ever-evolving", "landscape of", "realm of", "navigate the", "navigating the", "seamless", "pivotal",
    "fostering", "foster a", "underscore", "a wide range of", "a variety of", "various factors", "it is essential to",
    "it is crucial to", "key aspects", "overall, ", "there are several", "there are many different", "can be used to",
    "is a type of", "refers to the", "in other words", "as a result,", "ultimately,", "comprehensive", "multifaceted",
    "nuanced", "here's", "here is a", "i hope this helps", "let me know if", "as an ai", "certainly!", "great question",
    # Hindi
    "यह ध्यान देना महत्वपूर्ण है", "ध्यान देने योग्य", "महत्वपूर्ण भूमिका", "निष्कर्षतः", "निष्कर्ष में", "इसके अतिरिक्त",
    "इसके अलावा,", "कुल मिलाकर", "विभिन्न प्रकार",
]
_PHRASE_RE = re.compile("|".join(re.escape(p) for p in sorted(AI_PHRASES, key=len, reverse=True)), re.I)
_SHORTHAND_RE = re.compile(r"->|=>|→|←|&|\bw/|\bb/c\b|\bimp\b|\be\.?g\.?|\bi\.e\.?|\bvs\.?\b|\betc\b|\bapprox\b|\bbcz\b|\bcoz\b|"
                           r"\bu\b|\bur\b|\bpls\b|\bthru\b|\bdef\b|\bimpt\b|!!|\?\?|\.\.\.|\(\?\)|\bQ\d|\bAns\b", re.I)
_FIRST_PERSON_RE = re.compile(r"\b(i|i'm|i've|me|my|mine|we|our|us)\b|\b(मैं|मेरा|मेरी|मेरे|हम|हमारा|हमारी)\b", re.I)


def _rare_word_share(words: list[str], lang: str) -> float:
    candidates = [w for w in words if len(w) > 2 and not w.isdigit() and not w[:1].isupper()]
    if not candidates:
        return 0.0
    code = "hi" if lang == "hi" else "en"
    return sum(zipf_frequency(w.lower(), code) == 0 for w in candidates) / len(candidates)


def _mattr(words: list[str], window: int = 50) -> float:
    lowered = [w.lower() for w in words]
    if len(lowered) <= window:
        return len(set(lowered)) / max(len(lowered), 1)
    ratios = [len(set(lowered[i:i + window])) / window for i in range(0, len(lowered) - window + 1, 5)]
    return float(np.mean(ratios))


def _lm_sentence_scores(text: str, sentences: list[str]) -> list[float] | None:
    """Mean negative log-likelihood per sentence under GPT-2 (one forward pass over the whole text)."""
    loaded = models.detector_lm()
    if not loaded:
        return None
    import torch
    tokenizer, model = loaded
    encoded = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=MAX_LM_TOKENS)
    ids = encoded["input_ids"]
    if ids.shape[1] < 3:
        return None
    with torch.no_grad():
        logits = model(input_ids=ids).logits[0, :-1]
    nll = torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none").numpy()
    starts = [o[0] for o in encoded["offset_mapping"][0].tolist()][1:]
    # map every predicted token to the sentence its first character belongs to
    bounds, cursor = [], 0
    for sentence in sentences:
        at = text.find(sentence, cursor)
        at = cursor if at < 0 else at
        bounds.append((at, at + len(sentence)))
        cursor = at + len(sentence)
    per_sentence = [[] for _ in sentences]
    for start, value in zip(starts, nll):
        for i, (a, b) in enumerate(bounds):
            if a <= start < b:
                per_sentence[i].append(float(value))
                break
    return [float(np.mean(v)) if v else float("nan") for v in per_sentence]


def features(text: str, use_lm: bool = True) -> tuple[dict, list[dict]]:
    """Feature values for the whole text plus per-sentence details."""
    sentences = split_sentences(text) or [text.strip()]
    words = tokenize(text)
    n_words = max(len(words), 1)
    lang = detect(text)["language"]
    lengths = np.array([len(tokenize(s)) for s in sentences], dtype=float)
    feats = {
        "sent_len_cv": float(lengths.std() / lengths.mean()) if len(lengths) > 1 and lengths.mean() else 0.0,
        "ai_phrases": 100 * len(_PHRASE_RE.findall(text)) / n_words,
        "rare_words": _rare_word_share(words, "hi" if lang == "hi" else "en"),
        "shorthand": 100 * len(_SHORTHAND_RE.findall(text)) / n_words,
        "word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "mattr": _mattr(words),
        "first_person": 100 * len(_FIRST_PERSON_RE.findall(text)) / n_words,
    }
    sentence_nll = _lm_sentence_scores(text, sentences) if use_lm and lang == "en" else None
    if sentence_nll is not None:
        valid = [v for v in sentence_nll if not math.isnan(v)]
        feats["log_ppl"] = float(np.mean(valid)) if valid else float("nan")
        feats["ppl_std"] = float(np.std(valid)) if len(valid) > 1 else 0.0
    details = []
    for i, sentence in enumerate(sentences):
        nll = sentence_nll[i] if sentence_nll is not None else float("nan")
        details.append({"text": sentence,
                        "perplexity": None if math.isnan(nll) else round(math.exp(nll), 1),
                        "phrases": sorted({m.group(0).strip(" ,").lower() for m in _PHRASE_RE.finditer(sentence)})})
    return feats, details


@lru_cache(maxsize=1)
def _weights() -> dict | None:
    if not WEIGHTS_FILE.exists():
        return None
    return json.loads(WEIGHTS_FILE.read_text(encoding="utf-8"))


def _contributions(feats: dict, model: dict) -> np.ndarray:
    """weight × z-score for every feature of the model (their sum + intercept is the logit)."""
    x = np.array([feats[n] for n in model["features"]])
    z = np.clip((x - np.array(model["mean"])) / np.array(model["scale"]), -4, 4)  # one extreme value can't dominate
    return np.array(model["coef"]) * z


def probability(feats: dict, model_name: str) -> float:
    """P(AI-written) from precomputed features with the 'full' or 'style' model."""
    model = _weights()[model_name]
    return 1 / (1 + math.exp(-float(_contributions(feats, model).sum() + model["intercept"])))


SIGNAL_INFO = {
    "log_ppl": ("Predictability (GPT-2 perplexity)", "How surprised GPT-2 is by each word. AI text is usually very predictable (low perplexity)."),
    "ppl_std": ("Burstiness", "How much predictability changes from sentence to sentence."),
    "sent_len_cv": ("Sentence-length variety", "How much sentence lengths vary. People tend to mix short and long sentences."),
    "ai_phrases": ("Chatbot phrases", "Stock phrases such as “it is important to note” or “plays a crucial role”."),
    "rare_words": ("Typos and rare words", "Words missing from the dictionary: typos, slang, abbreviations."),
    "shorthand": ("Note-taking shorthand", "Arrows, &, w/, e.g., etc, !!, … — the way people take quick notes."),
    "word_len": ("Average word length", "Average number of letters per word."),
    "mattr": ("Vocabulary variety", "Share of different words in every 50-word window."),
    "first_person": ("Personal voice", "How often I / my / we appear."),
}


def _display(name: str, value: float) -> str:
    if name == "log_ppl":
        return f"perplexity {math.exp(value):.0f}"
    if name == "rare_words":
        return f"{100 * value:.1f}% of words"
    if name in ("ai_phrases", "shorthand", "first_person"):
        return f"{value:.1f} per 100 words"
    if name == "word_len":
        return f"{value:.1f} letters"
    return f"{value:.2f}"


def check_ai(text: str) -> dict:
    text = text.strip()
    weights = _weights()
    lang = detect(text)["language"] if text else "en"
    words = len(tokenize(text))
    base = {"words": words, "language": lang,
            "message": "This is a statistical estimate, not proof. Short, edited or translated text can be misjudged, "
                       "so never use it alone to accuse anyone."}
    if not weights:
        return {**base, "available": False, "ai_probability": None, "label": "Unavailable", "signals": [], "sentences": [],
                "engine": "none", "message": "The detector has not been trained yet: run python -m benchmarks.train_ai_detector"}
    if words < 15:
        return {**base, "available": True, "ai_probability": None, "label": "Too short", "confidence": "none",
                "signals": [], "sentences": [], "engine": "none",
                "message": "Add at least a few sentences (40+ words) so there is enough writing to judge."}

    feats, details = features(text)
    use_lm = "log_ppl" in feats and not math.isnan(feats["log_ppl"]) and "full" in weights
    model = weights["full" if use_lm else "style"]
    names = model["features"]
    x = np.array([feats[n] for n in names])
    contributions = _contributions(feats, model)
    probability = 1 / (1 + math.exp(-float(contributions.sum() + model["intercept"])))

    signals = []
    for name, value, contrib in sorted(zip(names, x, contributions), key=lambda t: -abs(t[2])):
        label, explanation = SIGNAL_INFO[name]
        leans = "ai" if contrib > 0.15 else "human" if contrib < -0.15 else "neutral"
        means = model.get("class_means", {})
        typical = ""
        if name in means.get("human", {}):
            typical = (f"In the training data: human texts {_display(name, means['human'][name])}, "
                       f"AI texts {_display(name, means['ai'][name])}.")
        signals.append({"name": name, "label": label, "value": round(float(value), 4), "display": _display(name, float(value)),
                        "contribution": round(float(contrib), 3), "leans": leans, "explanation": explanation, "typical": typical})

    # flag the most predictable sentences (lowest perplexity) and those with chatbot phrases
    ppls = [d["perplexity"] for d in details if d["perplexity"]]
    cutoff = float(np.percentile(ppls, 30)) if len(ppls) >= 4 else None
    for d in details:
        d["flag"] = bool(d["phrases"]) or (cutoff is not None and d["perplexity"] is not None and d["perplexity"] <= min(cutoff, 25))

    if words < MIN_WORDS or lang != "en":
        confidence = "low"
    else:
        confidence = "high" if abs(probability - 0.5) > 0.35 and words >= 120 else "medium"
    label = "Likely AI-written" if probability >= 0.7 else "Likely human-written" if probability <= 0.3 else "Unclear — could be either"
    engine = "gpt2-perplexity + stylometry (logistic regression)" if use_lm else "stylometry only (logistic regression)"
    note = base["message"]
    if lang != "en":
        note = "GPT-2 only understands English, so this uses writing-style signals only (less reliable). " + note
    return {**base, "available": True, "ai_probability": round(probability, 3), "label": label, "confidence": confidence,
            "signals": signals, "sentences": details, "engine": engine, "message": note,
            "model_accuracy": model.get("test_accuracy")}
