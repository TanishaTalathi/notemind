"""Evaluation suites for NoteMind. Results are saved to data/eval/results/<suite>.json and shown on /evaluation.

    python -m benchmarks.run all            # or: translation summarization qa language_id answer_check ai_detection asr

Every suite returns the same shape so the frontend can draw any of them:
    headline  [{label, value, hint}]            big numbers
    tables    [{title, columns, rows}]
    bars      [{title, unit, max, items: [{label, value}]}]
    matrix    {title, labels, values}           confusion matrix (rows = true, columns = predicted)
    notes     [str]
"""
import json
import math
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent.parent / "data"
EVAL = DATA / "eval"
RESULTS = EVAL / "results"
CACHE_FILE = EVAL / "online_translation_cache.json"

SUITE_INFO = {
    "translation": {"title": "Translation", "metric": "BLEU · chrF",
                    "description": "English ⇄ Hindi on 20 study sentences, scored against hand-written reference translations."},
    "summarization": {"title": "Summaries", "metric": "ROUGE-1 · ROUGE-2 · ROUGE-L",
                      "description": "TextRank summaries of 6 passages compared with reference summaries, against a lead-sentences baseline."},
    "qa": {"title": "Ask Notes", "metric": "Accuracy",
           "description": "35 questions on 6 passages, including cross-lingual and unanswerable ones. Embeddings vs TF-IDF retrieval."},
    "language_id": {"title": "Language detection", "metric": "Accuracy · F1",
                    "description": "60 labelled snippets in English, Hindi and Hinglish, including tricky ones."},
    "answer_check": {"title": "Answer checking", "metric": "Accuracy · Spearman ρ",
                     "description": "41 hand-graded student answers (correct / partial / incorrect), including Hindi and negated answers."},
    "ai_detection": {"title": "AI-written check", "metric": "Accuracy · F1 · AUC",
                     "description": "Held-out human vs ChatGPT answers from the HC3 corpus (not seen during training)."},
    "asr": {"title": "Voice notes", "metric": "WER · CER",
            "description": "Whisper on 16 spoken sentences (8 English, 8 Hindi) with known transcripts."},
}


def _load(name: str):
    return json.loads((EVAL / name).read_text(encoding="utf-8"))


def _pct(x: float) -> float:
    return round(100 * x, 1)


def _confusion(labels: list[str], gold: list[str], pred: list[str]) -> list[list[int]]:
    index = {l: i for i, l in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for g, p in zip(gold, pred):
        m[index[g]][index[p]] += 1
    return m


def _prf(labels: list[str], gold: list[str], pred: list[str]) -> list[list]:
    rows = []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append([l, _pct(precision), _pct(recall), _pct(f1), tp + fn])
    return rows


# ---------------------------------------------------------------- translation
def suite_translation() -> dict:
    import sacrebleu
    from nlp import translator
    pairs = _load("translation.json")["pairs"]
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    engines = [("marian-mt", translator._marian), ("mymemory", translator._mymemory), ("google", translator._google)]
    clean = lambda s: re.sub(r"[​-‍]", "", s or "").strip()  # zero-width joiners break word matching
    rows, bleu_bars, chrf_bars, examples, notes = [], [], [], [], []
    for src, tgt in (("en", "hi"), ("hi", "en")):
        sources, refs = [p[src] for p in pairs], [p[tgt] for p in pairs]
        for name, engine in engines:
            outputs = []
            for sentence in sources:
                key = f"{name}|{src}|{tgt}|{sentence}"
                if name != "marian-mt" and key in cache:  # online engines have daily limits, so reuse earlier answers
                    outputs.append(cache[key])
                    continue
                try:
                    out = engine(sentence, src, tgt)
                except Exception:
                    out = None
                if not out:
                    break
                outputs.append(out)
                if name != "marian-mt":
                    cache[key] = out
            if len(outputs) < len(sources):
                rows.append([name, f"{src}→{tgt}", "—", "—", "unavailable (offline, blocked or not installed)"])
                continue
            hyps = [clean(o) for o in outputs]
            bleu = sacrebleu.corpus_bleu(hyps, [refs], tokenize="intl").score
            chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
            rows.append([name, f"{src}→{tgt}", round(bleu, 1), round(chrf, 1), "ok"])
            bleu_bars.append({"label": f"{name} {src}→{tgt}", "value": round(bleu, 1)})
            chrf_bars.append({"label": f"{name} {src}→{tgt}", "value": round(chrf, 1)})
            for i in (0, 10):
                examples.append([f"{name} {src}→{tgt}", sources[i], refs[i], hyps[i]])
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
    ok = [r for r in rows if r[4] == "ok"]
    best = max(ok, key=lambda r: r[3]) if ok else None
    notes.append("BLEU counts matching word sequences (0–100); chrF counts matching character sequences, which suits Hindi's rich word forms better.")
    notes.append("Each sentence has a single reference translation, so a correct translation that uses different words still loses points.")
    if any(r[4] != "ok" for r in rows):
        notes.append("Engines marked unavailable could not be reached from this network (Google is blocked here).")
    return {
        "dataset": f"{len(pairs)} sentence pairs, both directions",
        "headline": [{"label": "Best chrF", "value": best[3] if best else "—", "hint": f"{best[0]} {best[1]}" if best else ""},
                     {"label": "Engines tested", "value": len(ok), "hint": "engine × direction"}],
        "tables": [{"title": "Scores", "columns": ["Engine", "Direction", "BLEU", "chrF", "Status"], "rows": rows},
                   {"title": "Examples", "columns": ["System", "Source", "Reference", "Output"], "rows": examples}],
        "bars": [{"title": "chrF (higher is better)", "unit": "", "max": 100, "items": chrf_bars},
                 {"title": "BLEU (higher is better)", "unit": "", "max": 100, "items": bleu_bars}],
        "notes": notes,
    }


# ---------------------------------------------------------------- summarization
def _ngrams(tokens: list[str], n: int):
    from collections import Counter
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def rouge_n(hyp: list[str], ref: list[str], n: int) -> float:
    h, r = _ngrams(hyp, n), _ngrams(ref, n)
    overlap = sum((h & r).values())
    if not overlap:
        return 0.0
    p, rc = overlap / sum(h.values()), overlap / sum(r.values())
    return 2 * p * rc / (p + rc)


def rouge_l(hyp: list[str], ref: list[str]) -> float:
    if not hyp or not ref:
        return 0.0
    prev = [0] * (len(ref) + 1)
    for h in hyp:
        cur = [0]
        for j, r in enumerate(ref, 1):
            cur.append(prev[j - 1] + 1 if h == r else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if not lcs:
        return 0.0
    p, rc = lcs / len(hyp), lcs / len(ref)
    return 2 * p * rc / (p + rc)


def suite_summarization() -> dict:
    from nlp.summarizer import summarize
    from nlp.text import split_sentences, tokens_lower
    passages = _load("passages.json")["passages"]
    systems: dict[str, list] = {"TextRank (short)": [], "TextRank (detailed)": [], "Lead baseline": []}
    examples = []
    for p in passages:
        ref = tokens_lower(p["summary"])
        short = summarize(p["text"], "short")
        detailed = summarize(p["text"], "detailed")
        k = max(1, len(split_sentences(short["summary"])))
        lead = " ".join(split_sentences(p["text"])[:k])  # classic baseline: the first k sentences
        for name, text in (("TextRank (short)", short["summary"]), ("TextRank (detailed)", detailed["summary"]), ("Lead baseline", lead)):
            hyp = tokens_lower(text)
            systems[name].append((rouge_n(hyp, ref, 1), rouge_n(hyp, ref, 2), rouge_l(hyp, ref), len(hyp)))
        examples.append([p["title"], p["summary"], short["summary"]])
    rows, bars = [], []
    for name, scores in systems.items():
        r1, r2, rl, words = (float(np.mean([s[i] for s in scores])) for i in range(4))
        rows.append([name, _pct(r1), _pct(r2), _pct(rl), round(words)])
        bars.append({"label": name, "value": _pct(rl)})
    textrank, lead = rows[0], rows[2]
    return {
        "dataset": f"{len(passages)} passages (5 English, 1 Hindi) with reference summaries",
        "headline": [{"label": "ROUGE-L (TextRank short)", "value": textrank[3], "hint": f"lead baseline {lead[3]}"},
                     {"label": "ROUGE-1 (TextRank short)", "value": textrank[1], "hint": f"lead baseline {lead[1]}"}],
        "tables": [{"title": "ROUGE F1 (%)", "columns": ["System", "ROUGE-1", "ROUGE-2", "ROUGE-L", "Avg words"], "rows": rows},
                   {"title": "Examples", "columns": ["Passage", "Reference", "TextRank (short)"], "rows": examples}],
        "bars": [{"title": "ROUGE-L F1 (higher is better)", "unit": "%", "max": 100, "items": bars}],
        "notes": ["ROUGE measures word overlap with a reference summary: ROUGE-1 single words, ROUGE-2 word pairs, ROUGE-L the longest common word sequence.",
                  "The lead baseline (just the first sentences) is hard to beat on textbook-style text, because textbooks put the main idea first.",
                  "References are abstractive (re-worded), so an extractive system like TextRank can't reach 100."],
    }


# ---------------------------------------------------------------- QA
@contextmanager
def _without_embeddings():
    from nlp import models
    original = models.embedding_model
    models.embedding_model = lambda: None
    try:
        yield
    finally:
        models.embedding_model = original


def suite_qa() -> dict:
    from nlp import models
    from nlp.language import base_language
    from nlp.qa import answer_question
    passages = _load("passages.json")["passages"]
    engines = ["multilingual-embeddings", "tfidf"] if models.embedding_model() is not None else ["tfidf"]
    rows, bars, errors = [], [], []
    for engine in engines:
        stats = {"answerable": [0, 0], "unanswerable": [0, 0], "cross-lingual": [0, 0]}
        start = time.time()
        for p in passages:
            for item in p["questions"]:
                if engine == "tfidf":
                    with _without_embeddings():
                        result = answer_question(p["text"], item["q"])
                else:
                    result = answer_question(p["text"], item["q"])
                top = result["sources"][0] if result["sources"] else ""
                if item["gold"] is None:
                    ok = not result["sources"]
                    stats["unanswerable"][0] += ok
                    stats["unanswerable"][1] += 1
                else:
                    ok = item["gold"].lower() in top.lower()
                    stats["answerable"][0] += ok
                    stats["answerable"][1] += 1
                    if base_language(item["q"]) != base_language(p["text"]):
                        stats["cross-lingual"][0] += ok
                        stats["cross-lingual"][1] += 1
                if not ok and engine == engines[0]:
                    errors.append([item["q"], item["gold"] or "(not in notes)", top or "(said: not found)"])
        total_ok = stats["answerable"][0] + stats["unanswerable"][0]
        total = stats["answerable"][1] + stats["unanswerable"][1]
        acc = lambda k: _pct(stats[k][0] / stats[k][1]) if stats[k][1] else "—"
        rows.append([engine, _pct(total_ok / total), acc("answerable"), acc("cross-lingual"), acc("unanswerable"),
                     round(1000 * (time.time() - start) / total)])
        bars.append({"label": engine, "value": _pct(total_ok / total)})
    best = max(rows, key=lambda r: r[1])
    return {
        "dataset": f"{sum(len(p['questions']) for p in passages)} questions on {len(passages)} passages",
        "headline": [{"label": "Overall accuracy", "value": f"{best[1]}%", "hint": best[0]},
                     {"label": "Says “not in notes” correctly", "value": f"{best[4]}%", "hint": "unanswerable questions"}],
        "tables": [{"title": "Accuracy (%)", "columns": ["Retriever", "Overall", "Answerable", "Cross-lingual", "Unanswerable", "ms / question"], "rows": rows},
                   {"title": f"Mistakes ({engines[0]})", "columns": ["Question", "Expected sentence", "Top source returned"], "rows": errors[:10]}],
        "bars": [{"title": "Overall accuracy", "unit": "%", "max": 100, "items": bars}],
        "notes": ["Answerable questions count as correct when the top source sentence is the one that holds the answer.",
                  "Unanswerable questions count as correct when the bot says the notes don't contain the answer instead of guessing.",
                  "Cross-lingual = Hindi question on English notes or the reverse. TF-IDF needs to translate the question first; embeddings match across languages directly."],
    }


# ---------------------------------------------------------------- language ID
def suite_language_id() -> dict:
    from nlp.language import detect
    items = _load("language_id.json")["items"]
    labels = ["en", "hi", "hinglish"]
    gold = [i["label"] for i in items]
    pred = [detect(i["text"])["language"] for i in items]
    accuracy = sum(g == p for g, p in zip(gold, pred)) / len(items)
    errors = [[i["text"], g, p] for i, g, p in zip(items, gold, pred) if g != p]
    prf = _prf(labels, gold, pred)
    return {
        "dataset": f"{len(items)} snippets",
        "headline": [{"label": "Accuracy", "value": f"{_pct(accuracy)}%", "hint": f"{len(items) - len(errors)}/{len(items)} correct"},
                     {"label": "Macro F1", "value": f"{round(float(np.mean([r[3] for r in prf])), 1)}%", "hint": "average over the 3 languages"}],
        "tables": [{"title": "Per language (%)", "columns": ["Language", "Precision", "Recall", "F1", "Examples"], "rows": prf},
                   {"title": "Mistakes", "columns": ["Text", "True", "Predicted"], "rows": errors}],
        "matrix": {"title": "Confusion matrix", "labels": labels, "values": _confusion(labels, gold, pred)},
        "bars": [{"title": "F1 per language", "unit": "%", "max": 100, "items": [{"label": r[0], "value": r[3]} for r in prf]}],
        "notes": ["Detection uses the share of Devanagari letters plus a list of Roman-script Hindi marker words (hai, ka, ko, me...).",
                  "Hinglish with very few Hindi function words is the hardest case."],
    }


# ---------------------------------------------------------------- answer checking
def suite_answer_check() -> dict:
    from scipy.stats import spearmanr
    from nlp.answer_check import check_answer
    items = _load("answer_check.json")["items"]
    labels = ["correct", "partial", "incorrect"]
    grade_value = {"correct": 2, "partial": 1, "incorrect": 0}
    gold = [i["grade"] for i in items]
    names = {"combined": "Embeddings + key terms + NLI (used)", "no-nli": "Embeddings + key terms",
             "semantic": "Embeddings only", "keywords": "Key terms only"}
    rows, bars, false_ok, matrix, errors = [], [], [], None, []
    for mode, name in names.items():
        results = [check_answer(i["expected"], i["answer"], i["question"], mode=mode) for i in items]
        pred = [r["verdict"] for r in results]
        scores = [r["score"] for r in results]
        accuracy = sum(g == p for g, p in zip(gold, pred)) / len(items)
        rho = spearmanr(scores, [grade_value[g] for g in gold]).statistic
        wrong = [p for g, p in zip(gold, pred) if g == "incorrect"]
        accepted = sum(p == "correct" for p in wrong) / len(wrong)
        means = {l: round(float(np.mean([s for s, g in zip(scores, gold) if g == l])), 1) for l in labels}
        rows.append([name, _pct(accuracy), _pct(accepted), round(float(rho), 3), means["correct"], means["partial"], means["incorrect"]])
        bars.append({"label": name, "value": _pct(accuracy)})
        false_ok.append({"label": name, "value": _pct(accepted)})
        if mode == "combined":
            matrix = {"title": "Confusion matrix (used method)", "labels": labels, "values": _confusion(labels, gold, pred)}
            errors = [[i["question"], i["answer"], g, f"{r['verdict']} ({r['score']})"]
                      for i, g, r in zip(items, gold, results) if g != r["verdict"]]
    best = rows[0]
    return {
        "dataset": f"{len(items)} graded answers to {len({i['question'] for i in items})} questions",
        "headline": [{"label": "Accuracy (3 classes)", "value": f"{best[1]}%", "hint": "correct / partial / incorrect"},
                     {"label": "Wrong answers marked correct", "value": f"{best[2]}%", "hint": "lower is better"},
                     {"label": "Spearman ρ", "value": best[3], "hint": "score vs human grade"}],
        "tables": [{"title": "Methods", "columns": ["Method", "Accuracy %", "Wrong → “correct” %", "Spearman ρ", "Avg score: correct", "partial", "incorrect"], "rows": rows},
                   {"title": "Disagreements with the human grade (used method)", "columns": ["Question", "Student answer", "Human grade", "Checker"], "rows": errors}],
        "matrix": matrix,
        "bars": [{"title": "Accuracy by method (higher is better)", "unit": "%", "max": 100, "items": bars},
                 {"title": "Wrong answers accepted as correct (lower is better)", "unit": "%", "max": 100, "items": false_ok}],
        "notes": ["Embeddings measure “same topic”, not “same claim”: an answer describing the opposite process can be 0.9 similar. "
                  "The NLI model reads both answers together and spots contradictions, which is why far fewer wrong answers get accepted.",
                  "The weights and thresholds were tuned on this same set (leave-one-out cross-validation gave about 61% accuracy), "
                  "so treat the accuracy as optimistic. Many remaining disagreements are short, vague answers where even teachers might differ.",
                  "Spearman ρ measures whether higher scores go with better human grades (1 = perfect ranking)."],
    }


# ---------------------------------------------------------------- AI detection
def suite_ai_detection(limit: int = 120) -> dict:
    from sklearn.metrics import roc_auc_score
    from nlp import ai_detect
    lines = (EVAL / "ai_detection_test.jsonl").read_text(encoding="utf-8").splitlines()[:limit]
    items = [json.loads(l) for l in lines]
    weights = ai_detect._weights()
    models_used = [m for m in ("full", "style") if m in weights]
    probs = {m: [] for m in models_used}
    ppl = {0: [], 1: []}
    for item in items:
        feats, _ = ai_detect.features(item["text"])
        for m in models_used:
            probs[m].append(ai_detect.probability(feats, m))
        if "log_ppl" in feats:
            ppl[item["label"]].append(math.exp(feats["log_ppl"]))
    gold = [i["label"] for i in items]
    rows, bars, matrix = [], [], None
    for m in models_used:
        pred = [int(p >= 0.5) for p in probs[m]]
        tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
        fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
        fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
        accuracy = sum(g == p for g, p in zip(gold, pred)) / len(gold)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        name = "GPT-2 perplexity + style" if m == "full" else "Style signals only"
        rows.append([name, _pct(accuracy), _pct(precision), _pct(recall), _pct(f1), round(roc_auc_score(gold, probs[m]), 3)])
        bars.append({"label": name, "value": _pct(accuracy)})
        if matrix is None:
            names = {0: "human", 1: "AI"}
            matrix = {"title": f"Confusion matrix ({name})", "labels": ["human", "AI"],
                      "values": _confusion(["human", "AI"], [names[g] for g in gold], [names[p] for p in pred])}
    headline = [{"label": "Accuracy", "value": f"{rows[0][1]}%", "hint": rows[0][0]},
                {"label": "ROC AUC", "value": rows[0][5], "hint": "0.5 = guessing, 1 = perfect"}]
    if ppl[0] and ppl[1]:
        headline.append({"label": "Median perplexity", "value": f"{np.median(ppl[0]):.0f} vs {np.median(ppl[1]):.0f}", "hint": "human vs AI text"})
    return {
        "dataset": f"{len(items)} held-out HC3 answers ({sum(gold)} AI, {len(gold) - sum(gold)} human)",
        "headline": headline,
        "tables": [{"title": "Detector", "columns": ["Model", "Accuracy %", "Precision %", "Recall %", "F1 %", "AUC"], "rows": rows}],
        "matrix": matrix,
        "bars": [{"title": "Accuracy", "unit": "%", "max": 100, "items": bars}],
        "notes": ["“AI” is the positive class: precision = how often an “AI” verdict is right; recall = how much AI text is caught.",
                  "HC3 answers were written by ChatGPT (GPT-3.5) in 2023 and the human answers come from Wikipedia and Q&A sites. "
                  "Newer chatbots, edited AI text and messy student notes are all harder, so expect lower accuracy in real use.",
                  "Detectors like this can wrongly flag careful human writers and non-native English writers. Use the result as a hint, never as proof."],
    }


# ---------------------------------------------------------------- speech recognition
def _normalize_transcript(text: str) -> list[str]:
    text = re.sub(r"[^\w\s]|_", " ", text.lower())  # also strips the danda (।)
    return text.split()


def _edit_distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def suite_asr() -> dict:
    from nlp import transcribe as asr
    if not asr.available():
        return {"dataset": "—", "headline": [{"label": "Status", "value": "unavailable", "hint": "faster-whisper not installed"}],
                "tables": [], "bars": [], "notes": ["Install faster-whisper (pip install -r requirements.txt) to run this suite."]}
    items = _load("asr.json")["items"]
    per_lang = {}
    examples = []
    for item in items:
        result = asr.transcribe((EVAL / "asr" / item["file"]).read_bytes(), "auto")
        ref, hyp = _normalize_transcript(item["text"]), _normalize_transcript(result["text"])
        ref_c, hyp_c = list("".join(ref)), list("".join(hyp))
        s = per_lang.setdefault(item["language"], {"w_err": 0, "w_ref": 0, "c_err": 0, "c_ref": 0, "lang_ok": 0, "n": 0})
        s["w_err"] += _edit_distance(ref, hyp)
        s["w_ref"] += len(ref)
        s["c_err"] += _edit_distance(ref_c, hyp_c)
        s["c_ref"] += len(ref_c)
        s["lang_ok"] += result.get("language") == item["language"]
        s["n"] += 1
        examples.append([item["language"], item["text"], result["text"], result.get("language", "?")])
    rows, bars = [], []
    for lang, s in per_lang.items():
        wer, cer = s["w_err"] / s["w_ref"], s["c_err"] / s["c_ref"]
        rows.append([{"en": "English", "hi": "Hindi"}.get(lang, lang), _pct(wer), _pct(cer), f"{s['lang_ok']}/{s['n']}"])
        bars.append({"label": f"WER {lang}", "value": _pct(wer)})
    return {
        "dataset": f"{len(items)} synthetic spoken sentences · {asr.MODEL_SIZE} model",
        "headline": [{"label": f"WER {r[0]}", "value": f"{r[1]}%", "hint": "lower is better"} for r in rows],
        "tables": [{"title": "Error rates (%)", "columns": ["Language", "WER", "CER", "Language detected correctly"], "rows": rows},
                   {"title": "Transcripts", "columns": ["Lang", "Reference", "Whisper output", "Detected"], "rows": examples}],
        "bars": [{"title": "Word error rate (lower is better)", "unit": "%", "max": 100, "items": bars}],
        "notes": ["WER = (substituted + deleted + inserted words) / words in the reference. CER is the same on characters, which is fairer for Hindi spelling variants.",
                  "The clips are synthetic voices with no background noise, so real classroom recordings will have higher error rates.",
                  f"Set NOTEMIND_WHISPER_MODEL=small for noticeably better Hindi (480 MB download); the current model is “{asr.MODEL_SIZE}”."],
    }


SUITES = {
    "translation": suite_translation, "summarization": suite_summarization, "qa": suite_qa, "language_id": suite_language_id,
    "answer_check": suite_answer_check, "ai_detection": suite_ai_detection, "asr": suite_asr,
}


def run_suite(name: str) -> dict:
    start = time.time()
    result = SUITES[name]()
    result.update({"suite": name, **SUITE_INFO[name], "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "seconds": round(time.time() - start, 1)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def load_results() -> dict:
    if not RESULTS.exists():
        return {}
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in RESULTS.glob("*.json") if p.stem in SUITES}


def main(args: list[str]):
    names = list(SUITES) if not args or args == ["all"] else args
    for name in names:
        result = run_suite(name)
        print(f"\n=== {result['title']} ({result['seconds']}s) — {result['dataset']}")
        for h in result["headline"]:
            print(f"  {h['label']}: {h['value']}  {h.get('hint', '')}")
        if result["tables"]:
            table = result["tables"][0]
            print("  " + " | ".join(table["columns"]))
            for row in table["rows"]:
                print("  " + " | ".join(str(c) for c in row))


if __name__ == "__main__":
    main(sys.argv[1:])
