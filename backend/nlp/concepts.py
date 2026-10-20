"""Concept map: key concepts become nodes; concepts that appear in the same sentence are linked, and the
words between them (usually the verb, e.g. "absorbs", "produces") become the edge label."""
from itertools import combinations

from .keywords import _is_verb_form, extract_keywords, term_pattern
from .text import HI_STOPWORDS, is_devanagari, split_sentences, tokenize

_ARTICLES = {"the", "a", "an", "its", "their", "this", "these", "that", "it", "and", "while", "which", "also", "then"}
_HI_FILLER = {"और", "तथा", "भी", "ही", "तो", "यह", "वह"}


_PREPS = {"into", "to", "from", "in", "by", "with", "for", "of", "on", "as"}
_BE = {"is", "are", "was", "were", "has", "have"}


def _relation(between: str, hindi: bool) -> str:
    """Turn the words between two concepts into a short edge label, preferring the verb."""
    fallback = "संबंधित" if hindi else "related to"
    words = [w for w in tokenize(between) if w.lower() not in _ARTICLES and w not in _HI_FILLER]
    if not words or len(words) > 5:  # far apart in the sentence: only loosely related
        return fallback
    if hindi:
        words = [w for w in words if w not in HI_STOPWORDS]
        return " ".join(words) if 0 < len(words) <= 2 else fallback
    low = [w.lower() for w in words]
    for i, w in enumerate(low):
        if _is_verb_form(w) or (w.endswith("s") and i == 0 and len(w) > 3 and w not in _PREPS):
            label = [w]
            if i > 0 and low[i - 1] in _BE:
                label.insert(0, low[i - 1])
            if i + 1 < len(low) and low[i + 1] in _PREPS:
                label.append(low[i + 1])
            return " ".join(label)
    if len(low) <= 2:
        return " ".join(low)
    if low[0] in _BE:
        return low[0]
    return fallback


def build_concept_map(text: str, max_nodes: int = 12) -> dict:
    keywords = extract_keywords(text, top_n=max_nodes)
    sentences = split_sentences(text)
    hindi = sum(is_devanagari(k["word"]) for k in keywords) > len(keywords) / 2

    nodes, patterns = [], {}
    for kw in keywords:
        patterns[kw["word"]] = term_pattern(kw["key"])
        mentions = [s for s in sentences if patterns[kw["word"]].search(s)]
        nodes.append({"id": kw["word"], "importance": max(1, round(kw["score"] * 10)), "count": kw["count"],
                      "sentences": mentions[:4]})

    edges: dict[tuple[str, str], dict] = {}
    for sentence in sentences:
        found = []
        for name, pattern in patterns.items():
            m = pattern.search(sentence)
            if m:
                found.append((m.start(), m.end(), name))
        found.sort()
        # link neighbours in reading order with a verb label; link the rest of the sentence weakly
        for (s1, e1, a), (s2, e2, b) in zip(found, found[1:]):
            if a == b or s2 < e1:
                continue
            key = tuple(sorted((a, b)))
            if key not in edges:
                edges[key] = {"source": a, "target": b, "relation": _relation(sentence[e1:s2], hindi), "weight": 0}
            edges[key]["weight"] += 2
        for (_, _, a), (_, _, b) in combinations(found, 2):
            key = tuple(sorted((a, b)))
            if a != b and key in edges:
                edges[key]["weight"] += 1

    # keep the graph readable: strongest links first, at most ~2 per node
    edge_list = sorted(edges.values(), key=lambda e: -e["weight"])[: max(len(nodes) * 2, 4)]
    keyword_scores = [{"word": k["word"], "score": k["score"]} for k in keywords[:10]]
    return {"nodes": nodes, "edges": edge_list, "keywords": keyword_scores}
