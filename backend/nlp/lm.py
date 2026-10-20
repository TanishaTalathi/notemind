"""Language modelling for smart autocomplete.

A trigram language model with interpolated Kneser-Ney smoothing, written from scratch (no library), trained on
  * a small bundled study corpus (English + Hindi, data/corpus_*.txt), and
  * the student's own notes (weighted higher), so suggestions adapt to the topic being written.

    P_KN(w | u v) = max(c(uvw) - D, 0) / c(uv*)  +  D * N1+(uv*) / c(uv*) * P_KN(w | v)
    P_KN(w | v)   = max(N1+(*vw) - D, 0) / N1+(*v*)  +  D * N1+(v*) / N1+(*v*) * P_cont(w)
    P_cont(w)     = N1+(*w) / N1+(**)

Optional: DistilGPT-2 as a fallback engine for English (NOTEMIND_GPT2=1).
"""
import math
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from . import models
from .text import STOPWORDS, split_sentences, tokenize

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
BOS, EOS = "<s>", "</s>"
DISCOUNT = 0.75
NOTES_WEIGHT = 3  # the user's own notes count three times as much as the background corpus


class KneserNeyTrigramLM:
    def __init__(self, discount: float = DISCOUNT):
        self.d = discount
        self.tri = Counter()            # c(u, v, w)
        self.bi_ctx = Counter()         # c(u, v, *)
        self.tri_types = Counter()      # N1+(u, v, *)
        self.cont_bi = Counter()        # N1+(*, v, w)
        self.cont_bi_ctx = Counter()    # N1+(*, v, *)
        self.bi_types = Counter()       # N1+(v, *)  (distinct continuations of v)
        self.cont_uni = Counter()       # N1+(*, w)
        self.followers = defaultdict(set)   # v -> {w}
        self.followers2 = defaultdict(set)  # (u, v) -> {w}
        self.surface = defaultdict(Counter)
        self.vocab: set[str] = set()
        self.total_bigram_types = 0

    # ---------- training ----------
    @staticmethod
    def sentence_tokens(sentence: str) -> list[str]:
        return [t.lower() for t in tokenize(sentence)]

    def fit(self, weighted_texts: list[tuple[str, int]]) -> "KneserNeyTrigramLM":
        trigram_counts = Counter()
        for text, weight in weighted_texts:
            for sentence in split_sentences(text):
                raw = tokenize(sentence)
                for token in raw:
                    self.surface[token.lower()][token] += 1
                tokens = [BOS, BOS] + [t.lower() for t in raw] + [EOS]
                for i in range(2, len(tokens)):
                    trigram_counts[(tokens[i - 2], tokens[i - 1], tokens[i])] += weight
        for (u, v, w), c in trigram_counts.items():
            self.tri[(u, v, w)] = c
            self.bi_ctx[(u, v)] += c
            self.tri_types[(u, v)] += 1
            self.followers2[(u, v)].add(w)
            self.vocab.add(w)
            self.cont_bi[(v, w)] += 1  # one more distinct left context u for the bigram (v, w)
        for (v, w), n in self.cont_bi.items():
            self.cont_bi_ctx[v] += n
            self.bi_types[v] += 1
            self.followers[v].add(w)
            self.cont_uni[w] += 1
        self.total_bigram_types = len(self.cont_bi)
        return self

    # ---------- probabilities ----------
    def p_unigram(self, w: str) -> float:
        return (self.cont_uni[w] + 0.01) / (self.total_bigram_types + 0.01 * (len(self.vocab) + 1))

    def p_bigram(self, w: str, v: str) -> float:
        denom = self.cont_bi_ctx[v]
        if not denom:
            return self.p_unigram(w)
        first = max(self.cont_bi[(v, w)] - self.d, 0) / denom
        lam = self.d * self.bi_types[v] / denom
        return first + lam * self.p_unigram(w)

    def prob(self, w: str, u: str, v: str) -> float:
        denom = self.bi_ctx[(u, v)]
        if not denom:
            return self.p_bigram(w, v)
        first = max(self.tri[(u, v, w)] - self.d, 0) / denom
        lam = self.d * self.tri_types[(u, v)] / denom
        return first + lam * self.p_bigram(w, v)

    def perplexity(self, text: str) -> float:
        log_sum, n = 0.0, 0
        for sentence in split_sentences(text):
            tokens = [BOS, BOS] + self.sentence_tokens(sentence) + [EOS]
            for i in range(2, len(tokens)):
                log_sum += math.log(max(self.prob(tokens[i], tokens[i - 2], tokens[i - 1]), 1e-12))
                n += 1
        return math.exp(-log_sum / n) if n else float("inf")

    # ---------- generation ----------
    def candidates(self, u: str, v: str) -> set[str]:
        return set(self.followers2.get((u, v), set())) | set(self.followers.get(v, set()))

    def best_next(self, u: str, v: str, prefix: str = "", banned: set | None = None) -> tuple[str | None, float]:
        pool = self.candidates(u, v) if not prefix else {w for w in self.vocab if w.startswith(prefix) and w != prefix}
        best, best_p = None, 0.0
        for w in pool:
            if banned and w in banned:
                continue
            p = self.prob(w, u, v)
            if p > best_p:
                best, best_p = w, p
        return best, best_p

    def display(self, word: str) -> str:
        forms = self.surface.get(word)
        return forms.most_common(1)[0][0] if forms else word

    def complete(self, text: str, max_words: int = 5, min_prob: float = 0.06) -> tuple[str, float]:
        """Return the ghost-text continuation for `text` and its average probability."""
        if not text.strip():
            return "", 0.0
        last_sentence = re.split(r"[.!?।]\s*", text)[-1]
        words = [t.lower() for t in tokenize(last_sentence)]
        partial = ""
        if words and not text[-1].isspace() and re.search(r"[\wऀ-ॿ]$", text):
            partial = words.pop()  # user is in the middle of a word
        if not words and not partial:
            return "", 0.0  # start of a new sentence: too little context to guess well
        context = ([BOS, BOS] + words)[-2:]
        pieces, probs = [], []

        if partial:
            if len(partial) < 2:
                return "", 0.0
            w, p = self.best_next(context[0], context[1], prefix=partial)
            if w:
                pieces.append(self.display(w)[len(partial):])
                probs.append(p)
                context = [context[1], w]
            elif partial in self.vocab:  # the "partial" word is already complete: predict the next one
                context = [context[1], partial]
            else:
                return "", 0.0

        seen = set()
        while len(probs) < max_words:
            w, p = self.best_next(context[0], context[1], banned={BOS})
            if not w or w == EOS:
                if pieces and w == EOS:
                    pieces.append(".")
                break
            if p < min_prob or (context[1], w) in seen:
                break
            seen.add((context[1], w))
            pieces.append(" " + self.display(w))
            probs.append(p)
            context = [context[1], w]

        # don't leave the ghost text hanging on "of", "the", "से" ...
        while len(pieces) > 1 and pieces[-1].strip().lower() in STOPWORDS:
            pieces.pop()
            probs.pop()
        suggestion = "".join(pieces)
        if not partial and suggestion.startswith(" ") and text[-1].isspace():
            suggestion = suggestion[1:]
        avg = sum(probs) / len(probs) if probs else 0.0
        return suggestion, round(avg, 3)


@lru_cache(maxsize=1)
def background_corpus() -> str:
    parts = [p.read_text(encoding="utf-8") for p in sorted(DATA_DIR.glob("corpus_*.txt"))]
    return "\n".join(parts)


@lru_cache(maxsize=32)
def model_for(notes_context: str) -> KneserNeyTrigramLM:
    return KneserNeyTrigramLM().fit([(background_corpus(), 1), (notes_context, NOTES_WEIGHT)])


def _gpt2(text: str) -> str:
    pipe = models.gpt2_pipeline()
    if not pipe or re.search(r"[ऀ-ॿ]", text):
        return ""
    out = pipe(text[-400:], max_new_tokens=8, do_sample=False, pad_token_id=50256)[0]["generated_text"]
    continuation = out[len(text[-400:]):]
    return re.split(r"(?<=[.!?])\s", continuation)[0].rstrip()


def autocomplete(text: str) -> dict:
    # Train on everything except the sentence currently being typed, so we don't simply echo it.
    # The context string is cached, so typing within one sentence re-uses the same model.
    cut = max(text.rfind(". "), text.rfind("। "), text.rfind("\n"))
    notes_context = text[: cut + 1] if cut > 0 else ""
    model = model_for(notes_context)
    suggestion, confidence = model.complete(text)
    engine = "kneser-ney-trigram"
    if not suggestion:
        try:
            suggestion = _gpt2(text)
            engine = "distilgpt2" if suggestion else engine
        except Exception:  # pragma: no cover
            suggestion = ""
    return {"suggestion": suggestion, "confidence": confidence, "engine": engine}
