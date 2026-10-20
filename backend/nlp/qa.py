"""'Ask your notes' — retrieval-based question answering (a small RAG system without a generator).

1. Split notes into sentences (the "documents").
2. Embed the question and the sentences:
     * multilingual sentence-transformer (cross-lingual: Hindi question -> English notes) if available,
     * otherwise TF-IDF over words + character n-grams (robust to spelling and word-form variations).
3. Return the most similar sentence(s) as the answer, with the matched sources, in the question's language.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

from . import models
from .language import base_language
from .summarizer import summarize
from .text import STOPWORDS, split_sentences, tokens_lower
from .translator import try_translate

MAIN_IDEA = re.compile(r"main idea|summary|summari[sz]e|gist|overview|मुख्य|सारांश|saar", re.I)
SIMPLE = re.compile(r"explain simply|simple words|in simple|like i'?m|eli5|आसान|सरल|aasan|saral", re.I)

NOT_FOUND = {"en": "I couldn't find this in your notes. Try rephrasing, or add more detail to your notes.",
             "hi": "यह जानकारी आपके नोट्स में नहीं मिली। प्रश्न को दूसरे तरीके से पूछकर देखें।"}


def _tfidf_scores(question: str, sentences: list[str]) -> np.ndarray:
    docs = sentences + [question]
    word = TfidfVectorizer(tokenizer=lambda t: [w for w in tokens_lower(t) if w not in STOPWORDS],
                           lowercase=False, token_pattern=None)
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), lowercase=True)
    try:
        matrix = hstack([word.fit_transform(docs) * 0.6, char.fit_transform(docs) * 0.4]).tocsr()
    except ValueError:
        matrix = char.fit_transform(docs)
    return cosine_similarity(matrix[-1], matrix[:-1]).ravel()


def _embedding_scores(question: str, sentences: list[str]) -> np.ndarray | None:
    model = models.embedding_model()
    if model is None:
        return None
    vectors = model.encode(sentences + [question], normalize_embeddings=True)
    return vectors[:-1] @ vectors[-1]


def _definition_bonus(question: str, sentences: list[str]) -> np.ndarray:
    """'What is X?' -> sentences that start with 'X is/are' get a boost."""
    m = re.search(r"(?:what|who)\s+(?:is|are|was|were)\s+(?:an?\s+|the\s+)?([\w\s'-]+?)\??$", question.strip(), re.I) \
        or re.search(r"^([\wऀ-ॿ\s]+?)\s+क्या\s+(?:है|हैं)", question.strip())
    bonus = np.zeros(len(sentences))
    if m:
        term = m.group(1).strip().lower()
        for i, s in enumerate(sentences):
            if s.lower().startswith(term) or s.lower().startswith("the " + term):
                bonus[i] = 0.15
    return bonus


def answer_question(text: str, question: str) -> dict:
    question = question.strip()
    q_lang, notes_lang = base_language(question), base_language(text)
    sentences = split_sentences(text)
    if not sentences or not question:
        return {"answer": NOT_FOUND[q_lang], "sources": [], "confidence": 0.0, "engine": "none"}

    if MAIN_IDEA.search(question) or SIMPLE.search(question):
        result = summarize(text, "short")
        answer = result["summary"]
        if SIMPLE.search(question):
            from .glossary import simplify
            answer = simplify(text, answer)
        sources = result["key_sentences"]
        engine = "textrank-summary"
        confidence = 0.9
    else:
        search_q = question
        scores = _embedding_scores(question, sentences)
        engine = "multilingual-embeddings"
        if scores is None:
            engine = "tfidf-retrieval"
            if q_lang != notes_lang:  # TF-IDF can't match across scripts, so translate the question first
                search_q = try_translate(question, q_lang, notes_lang) or question
                engine += "+translated-query"
            scores = _tfidf_scores(search_q, sentences)
        scores = scores + _definition_bonus(search_q, sentences)
        order = np.argsort(-scores)
        top = float(scores[order[0]])
        threshold = 0.25 if engine == "multilingual-embeddings" else 0.08
        if top < threshold:
            return {"answer": NOT_FOUND[q_lang], "sources": [], "confidence": round(top, 2), "engine": engine}
        chosen = [order[0]] + [i for i in order[1:3] if scores[i] >= 0.7 * top]
        chosen = sorted(chosen)
        answer = " ".join(sentences[i] for i in chosen)
        sources = [sentences[i] for i in order[:3] if scores[i] >= 0.5 * top]
        confidence = round(min(top / (0.6 if engine == "multilingual-embeddings" else 0.45), 0.99), 2)

    if q_lang != notes_lang:  # answer in the language the question was asked in
        translated = try_translate(answer, notes_lang, q_lang)
        if translated:
            answer = translated
    return {"answer": answer, "sources": sources, "confidence": confidence, "engine": engine}
