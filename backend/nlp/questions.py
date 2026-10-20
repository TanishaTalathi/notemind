"""Question generation from notes (English, Hindi and Hinglish).

1. Rank sentences (TextRank) and key concepts (keyword extractor).
2. Pair important sentences with the concepts they contain.
3. Build questions:
   * MCQ             - blank the concept; distractors are other concepts from the same notes
                       (hard = concepts ranked close to the answer, easy = far away ones)
   * Fill in blank   - same cloze sentence, typed answer
   * Short answer    - "What is X?" for definition sentences, "Why is X important?", else "Explain X"
"""
import random
import re
import zlib

from .keywords import extract_keywords, term_pattern, zipf
from .summarizer import rank_sentences
from .text import is_devanagari, split_sentences

BLANK = "_____"


def _qtype(value: str) -> str:
    v = (value or "").lower()
    if "fill" in v or "blank" in v:
        return "fill"
    if "short" in v:
        return "short"
    if "mix" in v:
        return "mixed"
    return "mcq"


def _short_question(sentence: str, term: str, hindi: bool) -> str:
    if hindi:
        if sentence.startswith(term) and re.search(r"\s(वह|एक)\s", sentence) and re.search(r"(है|हैं)\s*[।.]?$", sentence):
            return f"{term} क्या है?"
        return f"{term} के बारे में अपने नोट्स के आधार पर समझाइए।"
    m = re.match(rf"^\s*(?:the\s+|a\s+|an\s+)?{re.escape(term)}\w*\s+(is|are|refers to|means)\b", sentence, re.I)
    if m:
        verb = m.group(1).lower()
        return f"What {'are' if verb == 'are' else 'is'} {term.lower() if not term.isupper() else term}?"
    if re.search(r"\b(important|essential|necessary)\b", sentence, re.I):
        return f"Why is {term.lower()} important?"
    if re.search(r"\bbecause\b", sentence, re.I):
        return f"According to your notes, why does this happen: {sentence.split(' because')[0].strip()}?"
    return f"Explain the role of {term.lower()} in this topic."


def _distractors(answer: dict, keywords: list[dict], sentence: str, difficulty: str, rng: random.Random) -> list[str]:
    others = [k for k in keywords if k["key"] != answer["key"] and not term_pattern(k["key"]).search(sentence)]
    if difficulty == "hard":  # closest in importance and same number of words -> more confusing
        others.sort(key=lambda k: (abs(k["score"] - answer["score"]), len(k["key"].split()) != len(answer["key"].split())))
        pool = others[:5]
    elif difficulty == "easy":
        pool = sorted(others, key=lambda k: -abs(k["score"] - answer["score"]))[:6]
    else:
        pool = others[:8]
    rng.shuffle(pool)
    return [k["word"] for k in pool[:3]]


def generate_questions(text: str, count: int = 5, difficulty: str = "medium", qtype: str = "MCQ",
                       seed: int | None = None) -> dict:
    sentences = split_sentences(text)
    keywords = extract_keywords(text, top_n=20)
    if not sentences or not keywords:
        return {"questions": [], "available": 0}

    difficulty = (difficulty or "medium").lower()
    kind = _qtype(qtype)
    rng = random.Random(seed if seed is not None else zlib.crc32(text.encode("utf-8")))
    sent_scores = rank_sentences(sentences)
    sent_scores = sent_scores / sent_scores.max()

    # answers should be real concepts, not everyday words like "process" or "body"
    def specific(kw: dict) -> bool:
        return is_devanagari(kw["key"]) or min(zipf(w) for w in kw["key"].split()) <= 4.6
    answerable = [k for k in keywords if specific(k)]
    if len(answerable) < min(count, 4):
        answerable = keywords

    candidates = []
    for si, sentence in enumerate(sentences):
        if len(sentence) > 320:
            continue
        for rank, kw in enumerate(answerable):
            match = term_pattern(kw["key"]).search(sentence)
            if not match:
                continue
            if difficulty == "easy":
                score = sent_scores[si] + kw["score"]
            elif difficulty == "hard":
                score = sent_scores[si] + (1 - kw["score"]) + 0.02 * rank
            else:
                score = sent_scores[si] * kw["score"] + 0.3 * rng.random()
            candidates.append((score, si, kw, match))
    candidates.sort(key=lambda c: -c[0])

    used_sentences, used_terms, questions = set(), {}, []
    for allow_repeat in (False, True):
        for score, si, kw, match in candidates:
            if len(questions) >= count:
                break
            if si in used_sentences or (used_terms.get(kw["key"], 0) >= (2 if allow_repeat else 1)):
                continue
            sentence = sentences[si]
            hindi = is_devanagari(sentence)
            answer_text = match.group(0)
            cloze = sentence[: match.start()] + BLANK + sentence[match.end():]
            this_kind = kind if kind != "mixed" else ("mcq", "fill", "short")[len(questions) % 3]
            explanation = ("आपके नोट्स से: " if hindi else "From your notes: ") + f"“{sentence}”"

            if this_kind == "mcq":
                distractors = _distractors(kw, keywords, sentence, difficulty, rng)
                if len(distractors) < 2:
                    continue
                correct = answer_text if hindi else answer_text[0].upper() + answer_text[1:]
                distractors = [d for d in distractors if d.lower() != correct.lower()]
                options = distractors + [correct]
                rng.shuffle(options)
                prompt = "रिक्त स्थान के लिए सही विकल्प चुनिए:" if hindi else "Choose the term that best completes the sentence:"
                questions.append({"type": "mcq", "question": f"{prompt} {cloze}", "options": options,
                                  "answer": correct, "explanation": explanation})
            elif this_kind == "fill":
                prompt = "रिक्त स्थान भरिए:" if hindi else "Fill in the blank:"
                questions.append({"type": "fill", "question": f"{prompt} {cloze}", "answer": answer_text,
                                  "explanation": explanation})
            else:
                # if the sentence defines another concept ("Chlorophyll is ..."), ask about that one instead
                subject = next((k for k in keywords if term_pattern(k["key"]).match(sentence.lstrip())), kw)
                questions.append({"type": "short", "question": _short_question(sentence, subject["word"], hindi),
                                  "answer": sentence, "explanation": explanation})
            used_sentences.add(si)
            used_terms[kw["key"]] = used_terms.get(kw["key"], 0) + 1
    return {"questions": questions, "available": len({c[1] for c in candidates})}
