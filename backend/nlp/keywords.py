"""Keyword / key-phrase extraction shared by the concept map, quiz and glossary.

Score = (how often a term appears in the notes) x (how rare it is in everyday language).
Rarity comes from the `wordfreq` Zipf scale, which plays the role of a huge background IDF corpus,
so "chlorophyll" outranks "plants" even though it appears less often.
"""
import re
from collections import Counter, defaultdict
from functools import lru_cache

from .text import DEVANAGARI, is_devanagari, is_stopword, split_sentences, tokenize

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    def zipf_frequency(word, lang):  # type: ignore
        return 3.0

_BOUND = rf"A-Za-z0-9{DEVANAGARI}"

# Common verbs found in study notes. Keyword candidates should be concepts (nouns), so these
# and their inflections (absorbs, converted, produces...) are filtered out.
_VERBS = set("""
absorb accept add affect allow apply attach attract become begin belong bind break breathe build burn carry cause
change combine compare conduct connect consist contain continue control convert cool create decrease define depend
describe determine develop digest discover divide dissolve drive eat emit enable end enter establish evaporate
exchange explain fall feed fight find flow follow form freeze generate give grow happen heat help hold increase
influence introduce invent involve join keep lead leave lift link live lose maintain mean measure melt mix move need
occur orbit pass produce protect provide pull push react reach reduce reflect regulate release remain remove repel
represent reproduce require result revolve rise rotate rule see separate show split start stop store study supply
support take transfer transform transport travel trap turn understand vary win write
""".split())


def _is_verb_form(word: str) -> bool:
    for suffix in ("", "s", "es", "ed", "d", "ies"):
        if suffix and not word.endswith(suffix):
            continue
        stem = word[: len(word) - len(suffix)] if suffix else word
        if stem in _VERBS or (suffix == "ies" and stem + "y" in _VERBS):
            return True
    return False


@lru_cache(maxsize=20000)
def zipf(word: str) -> float:
    return zipf_frequency(word, "hi" if is_devanagari(word) else "en")


def rarity(term: str) -> float:
    parts = term.split()
    return sum(max(0.5, 6.5 - zipf(p)) for p in parts) / len(parts)


def _valid(token: str) -> bool:
    if is_stopword(token) or token.isdigit():
        return False
    if is_devanagari(token):
        return len(token) >= 2
    low = token.lower()
    if len(low) < 3:
        return False
    # crude verb/adverb filter for English: "converting", "absorbed", "quickly"
    if _is_verb_form(low):
        return False
    return not (len(low) > 5 and low.endswith(("ing", "ed", "ly")))


def term_pattern(term: str) -> re.Pattern:
    """Regex matching a term as whole words (works for Devanagari too); allows an English plural 's'."""
    body = r"\s+".join(re.escape(p) for p in term.split())
    plural = "" if is_devanagari(term) else "(?:e?s)?"
    return re.compile(rf"(?<![{_BOUND}]){body}{plural}(?![{_BOUND}])", re.IGNORECASE)


def extract_keywords(text: str, top_n: int = 15) -> list[dict]:
    sentences = split_sentences(text)
    counts: Counter = Counter()
    first_pos: dict[str, int] = {}
    surface: dict[str, Counter] = defaultdict(Counter)
    position = 0

    for sentence in sentences:
        tokens = tokenize(sentence)
        run: list[str] = []
        for token in tokens + ["."]:  # sentinel flushes the last run
            if token != "." and _valid(token):
                run.append(token)
                continue
            for n in (1, 2, 3):
                for i in range(len(run) - n + 1):
                    gram = run[i:i + n]
                    key = " ".join(g.lower() for g in gram)
                    counts[key] += 1
                    surface[key][" ".join(gram)] += 1
                    first_pos.setdefault(key, position)
            run = []
            position += 1

    # merge English plurals into the singular form ("plants" -> "plant")
    for key in list(counts):
        for suffix in ("es", "s"):
            if not is_devanagari(key) and key.endswith(suffix) and key[: -len(suffix)] in counts:
                single = key[: -len(suffix)]
                counts[single] += counts.pop(key)
                surface[single].update(surface.pop(key))
                break

    scores: dict[str, float] = {}
    for key, count in counts.items():
        n = len(key.split())
        if n > 1 and count < 2:
            # a one-off phrase still counts if its words never appear on their own and are rare
            # (e.g. "कार्बन डाइऑक्साइड", "light-dependent reactions")
            parts = key.split()
            if not (all(counts.get(p, 0) <= count for p in parts) and rarity(key) >= 2.5):
                continue
        early = 1.15 if first_pos.get(key, 99) <= 1 else 1.0
        scores[key] = count * rarity(key) * (1.0 + 0.35 * (n - 1)) * early

    # drop single words that only ever occur inside an accepted phrase ("dioxide" inside "carbon dioxide")
    for key in [k for k in scores if " " in k]:
        for part in key.split():
            if part in scores and counts[part] <= counts[key]:
                scores.pop(part)
        # drop the 3-gram's contained 2-grams if they never appear on their own
    for key in [k for k in scores if len(k.split()) == 3]:
        for sub in (" ".join(key.split()[:2]), " ".join(key.split()[1:])):
            if sub in scores and counts[sub] <= counts[key]:
                scores.pop(sub)

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], first_pos.get(kv[0], 0)))[:top_n]
    if not ranked:
        return []
    best = ranked[0][1]
    result = []
    for key, score in ranked:
        display = surface[key].most_common(1)[0][0]
        if not is_devanagari(display) and display.islower():
            display = display[0].upper() + display[1:]
        result.append({"word": display, "key": key, "score": round(score / best, 3), "count": counts[key]})
    return result
