"""Smart glossary, readability scoring and 'simplify my notes'.

* Difficult terms = key concepts that are rare in everyday language (wordfreq Zipf scale).
* Meanings are taken from definition sentences in the notes ("X is ...", "X वह ... है"),
  then translated so every term has an English and a Hindi meaning.
* Readability: Flesch Reading Ease + Flesch-Kincaid grade for English, a length/rare-word estimate for Hindi.
* Simplify: split long sentences and swap formal words for everyday ones.
"""
import re
from concurrent.futures import ThreadPoolExecutor

from .keywords import extract_keywords, term_pattern, zipf
from .language import base_language
from .text import is_devanagari, split_sentences, tokenize, word_count
from .translator import try_translate

_DEF_EN = re.compile(r"^\s*(?:the\s+|an?\s+)?{term}\w*\s+(?:is|are|refers to|means|is called|is known as)\s+(.+)$", re.I)
_DEF_HI = re.compile(r"^\s*{term}\s+(?:वह|एक|को)?\s*(.+?)\s*(?:है|हैं|कहते हैं|कहलाता है)\s*[।.]?$")

SIMPLER_WORDS = {
    "utilize": "use", "utilise": "use", "approximately": "about", "numerous": "many", "commence": "start",
    "facilitate": "help", "sufficient": "enough", "demonstrate": "show", "obtain": "get", "require": "need",
    "requires": "needs", "consequently": "so", "therefore": "so", "however": "but", "nevertheless": "still",
    "additionally": "also", "furthermore": "also", "individuals": "people", "possess": "have", "possesses": "has",
    "assist": "help", "acquire": "get", "subsequently": "later", "prior to": "before", "in order to": "to",
    "a large number of": "many", "the majority of": "most", "maintain": "keep", "maintains": "keeps",
    "construct": "build", "purchase": "buy", "attempt": "try", "terminate": "end", "indicate": "show",
    "indicates": "shows", "initial": "first", "regarding": "about", "modify": "change", "eliminate": "remove",
    "adequate": "enough", "beneficial": "helpful", "convert": "change", "converts": "changes", "approximately equal": "about the same",
    "is able to": "can", "are able to": "can", "in addition": "also", "due to the fact that": "because",
}
_SPLITTERS = [(r",\s+which\s+", ". This "), (r",\s+while\s+", ". Meanwhile, "), (r";\s+", ". "),
              (r"\s+because\s+", ". This is because "), (r",\s+and\s+", ". Also, "), (r",\s+but\s+", ". But ")]


def difficulty(term: str) -> str:
    z = min(zipf(w) for w in term.lower().split())
    easy, medium = (3.5, 2.5) if is_devanagari(term) else (4.0, 3.0)
    return "Easy" if z >= easy else "Medium" if z >= medium else "Hard"


def _definition(term_key: str, sentences: list[str]) -> tuple[str, str]:
    """Return (meaning, kind) where kind is 'definition' or 'context'."""
    esc = re.escape(term_key)
    pattern = (_DEF_HI if is_devanagari(term_key) else _DEF_EN).pattern.format(term=esc)
    for s in sentences:
        m = re.match(pattern, s, re.I)
        if m:
            meaning = m.group(1).strip().rstrip(".।")
            if meaning:
                return meaning[0].upper() + meaning[1:], "definition"
    for s in sentences:  # no definition: show the term in context
        if term_pattern(term_key).search(s):
            words = s.rstrip(".।").split()
            return " ".join(words[:22]) + ("…" if len(words) > 22 else ""), "context"
    return "", "none"


def _syllables(word: str) -> int:
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups) - (1 if w.endswith("e") and not w.endswith(("le", "ee")) and len(groups) > 1 else 0)
    return max(1, count)


def readability(text: str) -> dict:
    sentences = split_sentences(text) or [text]
    words = tokenize(text)
    if not words:
        return {"score": 0, "grade": "—", "level": "—", "avg_sentence_length": 0}
    wps = len(words) / len(sentences)
    if base_language(text) == "hi":
        rare = sum(1 for w in words if zipf(w) < 3.0) / len(words)
        score = 100 - 2.2 * max(0.0, wps - 8) - 120 * rare
        grade_num = round(4 + wps / 3 + 20 * rare)
    else:
        spw = sum(_syllables(w) for w in words) / len(words)
        score = 206.835 - 1.015 * wps - 84.6 * spw
        grade_num = round(0.39 * wps + 11.8 * spw - 15.59)
    score = int(max(0, min(100, round(score))))
    grade = "College" if grade_num > 12 else f"Grade {max(1, grade_num)}"
    level = "Easy" if score >= 70 else "Moderate" if score >= 50 else "Difficult"
    return {"score": score, "grade": grade, "level": level, "avg_sentence_length": round(wps, 1)}


def _swap_words(sentence: str) -> str:
    for hard, easy in sorted(SIMPLER_WORDS.items(), key=lambda kv: -len(kv[0])):
        def repl(m, easy=easy):
            word = m.group(0)
            return easy[0].upper() + easy[1:] if word[0].isupper() else easy
        sentence = re.sub(rf"\b{re.escape(hard)}\b", repl, sentence, flags=re.I)
    return sentence


def simplify(text: str, base: str | None = None, glossary_terms: list[dict] | None = None) -> str:
    source = base if base is not None else text
    out = []
    for sentence in split_sentences(source):
        if not is_devanagari(sentence):
            sentence = _swap_words(sentence)
            if word_count(sentence) > 18:
                for pattern, replacement in _SPLITTERS:
                    sentence = re.sub(pattern, replacement, sentence, count=1)
        elif word_count(sentence) > 20:
            sentence = re.sub(r"\s+और\s+", "। और ", sentence, count=1)
        out.append(sentence.strip())
    simplified = " ".join(out)
    # explain hard terms in brackets the first time they appear (short meanings only)
    for term in glossary_terms or []:
        meaning = term.get("meaning" if not is_devanagari(term["term"]) else "hindi") or ""
        if term["difficulty"] == "Hard" and 0 < len(meaning.split()) <= 10 and meaning.lower() not in simplified.lower():
            pattern = term_pattern(term["term"].lower())
            simplified = pattern.sub(lambda m: f"{m.group(0)} ({meaning[0].lower() + meaning[1:]})", simplified, count=1)
    return simplified


def build_glossary(text: str, max_terms: int = 10) -> dict:
    sentences = split_sentences(text)
    notes_lang = base_language(text)
    keywords = extract_keywords(text, top_n=max_terms + 6)
    rank = {"Hard": 0, "Medium": 1, "Easy": 2}
    picked = sorted(keywords, key=lambda k: (rank[difficulty(k["key"])], -k["score"]))[:max_terms]

    terms = []
    for kw in picked:
        meaning, kind = _definition(kw["key"], sentences)
        terms.append({"term": kw["word"], "difficulty": difficulty(kw["key"]), "source": kind, "_meaning": meaning})

    other = "hi" if notes_lang == "en" else "en"
    with ThreadPoolExecutor(max_workers=6) as pool:
        translated = list(pool.map(lambda t: try_translate(t["_meaning"], notes_lang, other) if t["_meaning"] else "",
                                   terms))
    for t, tr in zip(terms, translated):
        own = t.pop("_meaning")
        t["meaning"], t["hindi"] = (own, tr or "") if notes_lang == "en" else (tr or "", own)

    terms.sort(key=lambda t: rank[t["difficulty"]])
    read = readability(text)
    return {"terms": terms, "readability": read["score"], "grade": read["grade"], "level": read["level"],
            "avg_sentence_length": read["avg_sentence_length"], "simplified": simplify(text, glossary_terms=terms)}
