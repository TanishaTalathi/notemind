"""Short-answer checking: how close is the student's answer to the expected one?

Three signals:
  * meaning     cosine similarity of multilingual sentence embeddings (paraphrase-multilingual-MiniLM). Good at
                "same topic", so paraphrases and Hindi answers score well, but it can't tell a claim from its
                opposite: "evaporation is when vapour turns into liquid" is 0.93 similar to the right answer.
  * key terms   important words of the expected answer (not stopwords, not words already in the question),
                matched by a 5-letter stem so "absorbs" matches "absorption". Skipped across languages.
  * entailment  a cross-encoder NLI model (multilingual MiniLM trained on MNLI + XNLI, which includes Hindi)
                reads both texts together:
                  expected ⇒ answer   is the answer supported? (partial answers are supported but incomplete)
                  answer ⇒ expected   does the answer cover everything?
                  contradiction       does the answer say something the expected answer rules out?

  completeness = 0.7 · meaning + 0.2 · key terms + 0.1 · (answer ⇒ expected)
  contradiction ≥ 0.7 → capped as incorrect;  answer not supported (entailment < 0.5) → × 0.7
The weights and thresholds were tuned on data/eval/answer_check.json (see the evaluation page).
Without transformers it falls back to TF-IDF similarity and a negation word check.

Displayed score: ≥ 70 correct · 40-69 partly correct · < 40 incorrect
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

from . import models
from .language import base_language
from .text import STOPWORDS, tokens_lower

SEM_LOW, SEM_HIGH = 0.30, 0.80
W_SEM, W_COVER, W_ENTAIL = 0.7, 0.2, 0.1
CONTRADICTION, SUPPORTED = 0.7, 0.5
RAW_CORRECT, RAW_PARTIAL = 0.55, 0.30  # tuned cut-offs, shown to the user as 70 and 40
CORRECT, PARTIAL = 70, 40
_NOT_DEVANAGARI = r"(?<![ऀ-ॿ])"
NEGATION = re.compile(r"\b(not|no|never|none|cannot|can't|doesn't|don't|isn't|aren't|wasn't|weren't|won't|nahi|nahin)\b|"
                      rf"{_NOT_DEVANAGARI}(नहीं|न|मत|ना)(?![ऀ-ॿ])", re.I)
GENERIC = set("process thing things part parts way ways type types kind lot form forms example called known".split())


def _stem(word: str) -> str:
    return word[:5] if len(word) > 5 else word


def _key_terms(expected: str, question: str) -> list[str]:
    asked = {_stem(w) for w in tokens_lower(question)}
    seen, terms = set(), []
    for w in tokens_lower(expected):
        s = _stem(w)
        if w in STOPWORDS or w in GENERIC or len(w) < 3 or w.isdigit() or s in asked or s in seen:
            continue
        seen.add(s)
        terms.append(w)
    return terms


def _semantic(a: str, b: str) -> tuple[float, str]:
    model = models.embedding_model()
    if model is not None:
        va, vb = model.encode([a, b], normalize_embeddings=True)
        return float(va @ vb), "multilingual-embeddings"
    word = TfidfVectorizer(tokenizer=lambda t: [w for w in tokens_lower(t) if w not in STOPWORDS], lowercase=False, token_pattern=None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    try:
        m = hstack([word.fit_transform([a, b]) * 0.5, char.fit_transform([a, b]) * 0.5]).tocsr()
    except ValueError:
        m = char.fit_transform([a, b])
    # TF-IDF cosines run lower than embedding cosines; stretch them onto a comparable range
    return float(min(1.0, cosine_similarity(m[0], m[1])[0, 0] * 1.4)), "tfidf"


def _nli(premise: str, hypothesis: str) -> dict | None:
    loaded = models.nli_model()
    if not loaded:
        return None
    import torch
    tokenizer, model = loaded
    with torch.no_grad():
        logits = model(**tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256)).logits[0]
    probs = torch.softmax(logits, -1).tolist()
    return {model.config.id2label[i].lower(): p for i, p in enumerate(probs)}


def _display_score(raw: float) -> int:
    """Map the tuned cut-offs (0.30, 0.55) onto the familiar 40 / 70 so the number reads naturally."""
    points = [(0.0, 0), (RAW_PARTIAL, PARTIAL), (RAW_CORRECT, CORRECT), (1.0, 100)]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if raw <= x1:
            return int(round(y0 + (y1 - y0) * (raw - x0) / (x1 - x0)))
    return 100


def check_answer(expected: str, answer: str, question: str = "", mode: str = "combined") -> dict:
    """mode: 'combined' (default), or 'no-nli', 'semantic', 'keywords' (alternatives compared on the evaluation page)."""
    expected, answer = expected.strip(), answer.strip()
    if not answer:
        return {"score": 0, "verdict": "incorrect", "similarity": 0.0, "coverage": 0.0, "matched_terms": [],
                "missing_terms": [], "feedback": "Write an answer first.", "engine": "none"}
    similarity, engine = _semantic(expected, answer)
    semantic = float(np.clip((similarity - SEM_LOW) / (SEM_HIGH - SEM_LOW), 0, 1))

    terms = _key_terms(expected, question)
    cross_lingual = base_language(expected) != base_language(answer)
    answer_stems = {_stem(w) for w in tokens_lower(answer)}
    matched = [t for t in terms if _stem(t) in answer_stems]
    missing = [t for t in terms if _stem(t) not in answer_stems]
    coverage = len(matched) / len(terms) if terms else semantic

    forward = backward = None
    if mode == "combined":
        forward, backward = _nli(expected, answer), _nli(answer, expected)
    contradiction = max(forward["contradiction"], backward["contradiction"]) if forward else None
    supported = forward["entailment"] if forward else None

    if mode == "keywords":
        raw = coverage
    elif mode == "semantic":
        raw = semantic
    else:
        w_cover = 0.0 if cross_lingual else W_COVER  # word matching can't work across scripts
        w_entail = W_ENTAIL if backward else 0.0
        raw = (W_SEM * semantic + w_cover * coverage + w_entail * (backward["entailment"] if backward else 0)) / (W_SEM + w_cover + w_entail)
        if forward:
            if contradiction >= CONTRADICTION:
                raw = min(raw, 0.25)
            elif supported < SUPPORTED:
                raw *= 0.7
    # without NLI, fall back to a simple "not" check (embeddings barely notice negation); it can't compare across languages
    negated = (not forward and mode != "keywords" and not cross_lingual
               and bool(NEGATION.search(expected)) != bool(NEGATION.search(answer)))
    if negated:
        raw = min(raw * 0.4, 0.25)
    score = _display_score(raw)
    verdict = "correct" if score >= CORRECT else "partial" if score >= PARTIAL else "incorrect"
    contradicts = bool(contradiction is not None and contradiction >= CONTRADICTION) or negated

    if contradicts:
        feedback = "Your answer seems to say something different from (or the opposite of) the expected answer."
    elif verdict == "correct":
        feedback = "Great — your answer matches the meaning of the expected answer."
    elif verdict == "partial":
        feedback = "Partly right — what you wrote fits, but it's incomplete"
        feedback += f". Try to mention: {', '.join(missing[:4])}." if missing and not cross_lingual else ". Add more detail."
    else:
        feedback = "This doesn't match the expected answer yet."
        if missing and not cross_lingual:
            feedback += f" Key ideas to include: {', '.join(missing[:4])}."
    if mode == "combined" and forward:
        engine += " + NLI"
    elif mode != "combined":
        engine += f" ({mode})"
    return {"score": score, "verdict": verdict, "similarity": round(similarity, 3), "coverage": round(coverage, 3),
            "entailment": round(supported, 3) if supported is not None else None,
            "contradiction": round(contradiction, 3) if contradiction is not None else None,
            "contradicts": contradicts, "matched_terms": matched, "missing_terms": [] if cross_lingual else missing,
            "feedback": feedback, "engine": engine}
