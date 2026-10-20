"""Summarisation.

Default: extractive TextRank. Sentences become TF-IDF vectors, cosine similarity builds a graph,
and PageRank picks the most central sentences. It needs no model and works for English, Hindi and Hinglish.
Optional: abstractive mT5 (XL-Sum) when NOTEMIND_ABSTRACTIVE=1 and `transformers` is installed.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from . import models
from .text import STOPWORDS, split_sentences, tokens_lower, word_count


def _tokenizer(text: str) -> list[str]:
    return [t for t in tokens_lower(text) if t not in STOPWORDS]


def pagerank(sim: np.ndarray, damping: float = 0.85, iterations: int = 100) -> np.ndarray:
    n = sim.shape[0]
    np.fill_diagonal(sim, 0.0)
    row_sums = sim.sum(axis=1, keepdims=True)
    # sentences with no links spread their vote evenly
    transition = np.divide(sim, row_sums, out=np.full_like(sim, 1.0 / n), where=row_sums > 0)
    scores = np.full(n, 1.0 / n)
    for _ in range(iterations):
        new = (1 - damping) / n + damping * transition.T @ scores
        if np.abs(new - scores).sum() < 1e-6:
            break
        scores = new
    return scores


def rank_sentences(sentences: list[str]) -> np.ndarray:
    if len(sentences) == 1:
        return np.ones(1)
    try:
        vectors = TfidfVectorizer(tokenizer=_tokenizer, lowercase=False, token_pattern=None).fit_transform(sentences)
    except ValueError:  # every word was a stopword
        return np.ones(len(sentences)) / len(sentences)
    sim = (vectors @ vectors.T).toarray()
    scores = pagerank(sim)
    # small bonus for the opening sentence (topic sentence) and penalty for very short ones
    scores[0] *= 1.15
    lengths = np.array([word_count(s) for s in sentences])
    scores *= np.clip(lengths / 8, 0.4, 1.0)
    return scores


def _pick(sentences: list[str], scores: np.ndarray, k: int) -> list[str]:
    top = sorted(np.argsort(-scores)[:k])  # keep original order so the summary reads naturally
    return [sentences[i] for i in top]


def _bullet(sentence: str) -> str:
    sentence = re.sub(r"^(however|therefore|moreover|also|thus|so|and|but)[,\s]+", "", sentence, flags=re.I)
    sentence = sentence.rstrip(".।")
    return "• " + sentence[0].upper() + sentence[1:] if sentence else sentence


def _abstractive(text: str, mode: str) -> str | None:
    loaded = models.summary_pipeline()
    if not loaded:
        return None
    tokenizer, model = loaded
    max_len = {"short": 60, "detailed": 160, "bullets": 120}[mode]
    ids = tokenizer([re.sub(r"\s+", " ", text)], return_tensors="pt", truncation=True, max_length=512)
    out = model.generate(**ids, max_length=max_len, no_repeat_ngram_size=2, num_beams=4)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def summarize(text: str, mode: str = "detailed") -> dict:
    sentences = split_sentences(text)
    original_words = word_count(text)
    if not sentences:
        return {"summary": text.strip(), "original_words": original_words, "summary_words": original_words,
                "engine": "none", "key_sentences": []}

    scores = rank_sentences(sentences)
    n = len(sentences)
    k = {"short": max(1, min(3, round(n * 0.2))), "detailed": max(2, round(n * 0.45)),
         "bullets": max(2, min(7, round(n * 0.4)))}.get(mode, max(2, round(n * 0.45)))
    picked = _pick(sentences, scores, min(k, n))

    engine = "textrank"
    summary = None
    if mode != "bullets" and n > 3:
        try:
            summary = _abstractive(text, mode)
            if summary:
                engine = "mt5-xlsum"
        except Exception:  # pragma: no cover
            summary = None
    if not summary:
        summary = "\n".join(_bullet(s) for s in picked) if mode == "bullets" else " ".join(picked)

    return {"summary": summary, "original_words": original_words, "summary_words": word_count(summary),
            "engine": engine, "key_sentences": picked}
