"""Train the AI-written-text detector (nlp/ai_detect.py) on the HC3 corpus.

HC3 (Guo et al., 2023, "How Close is ChatGPT to Human Experts?", CC-BY-SA 4.0) has questions answered by both
humans and ChatGPT. We use the wiki_csai and open_qa subsets.

    python -m benchmarks.train_ai_detector path/to/folder/with/hc3/jsonl/files

1. Clean the text (HC3 has artifacts that would let a model cheat: literal "\\n" in ChatGPT answers,
   missing spaces after full stops and broken quote characters in human answers).
2. Keep question pairs where both answers have 60+ words; truncate both to ~220 words.
3. Split by question (both answers of a question go to the same split): 70% train, 30% test.
4. Fit two logistic-regression models on standardised features: "full" (GPT-2 perplexity + style) and
   "style" (style only, used for Hindi/Hinglish or without transformers).
5. Save weights + test metrics to data/ai_detector.json and the test texts to data/eval/ai_detection_test.jsonl.
"""
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from nlp import ai_detect

DATA = Path(__file__).resolve().parent.parent / "data"
SUBSETS = ["wiki_csai.jsonl", "open_qa.jsonl"]
PAIRS = 350
MAX_WORDS = 220


def clean(text: str) -> str:
    text = text.replace("\\n", "\n").replace("�", "'")
    text = re.sub(r"([.!?])([A-Z])", r"\1 \2", text)
    text = re.sub(r"[ \t]+([.,!?;:])", r"\1", text)
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def truncate(text: str, max_words: int = MAX_WORDS) -> str:
    """Cut at a sentence boundary near max_words."""
    out, count = [], 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        out.append(sentence)
        count += len(sentence.split())
        if count >= max_words:
            break
    return " ".join(out)


def load_pairs(folder: Path) -> list[dict]:
    pairs = []
    for name in SUBSETS:
        for line in (folder / name).read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if not row.get("human_answers") or not row.get("chatgpt_answers"):
                continue
            human, ai = clean(row["human_answers"][0]), clean(row["chatgpt_answers"][0])
            if len(human.split()) >= 60 and len(ai.split()) >= 60:
                pairs.append({"question": row["question"], "human": truncate(human), "ai": truncate(ai), "source": name})
    return pairs


def fit(rows: list[dict], names: list[str]) -> dict:
    train = [r for r in rows if r["split"] == "train"]
    test = [r for r in rows if r["split"] == "test"]
    X = np.array([[r["features"][n] for n in names] for r in train])
    y = np.array([r["label"] for r in train])
    mean, scale = X.mean(axis=0), X.std(axis=0) + 1e-9
    clf = LogisticRegression(C=0.5, max_iter=2000).fit((X - mean) / scale, y)
    Xt = np.array([[r["features"][n] for n in names] for r in test])
    yt = np.array([r["label"] for r in test])
    zt = np.clip((Xt - mean) / scale, -4, 4)
    prob = 1 / (1 + np.exp(-(zt @ clf.coef_[0] + clf.intercept_[0])))
    pred = (prob >= 0.5).astype(int)
    return {
        "features": names, "mean": mean.round(6).tolist(), "scale": scale.round(6).tolist(),
        "coef": clf.coef_[0].round(6).tolist(), "intercept": round(float(clf.intercept_[0]), 6),
        "test_accuracy": round(accuracy_score(yt, pred), 4), "test_precision": round(precision_score(yt, pred), 4),
        "test_recall": round(recall_score(yt, pred), 4), "test_f1": round(f1_score(yt, pred), 4),
        "test_auc": round(roc_auc_score(yt, prob), 4), "train_size": len(train), "test_size": len(test),
        "class_means": {"human": dict(zip(names, X[y == 0].mean(axis=0).round(4).tolist())),
                        "ai": dict(zip(names, X[y == 1].mean(axis=0).round(4).tolist()))},
    }


def main(folder: str):
    rng = random.Random(7)
    pairs = load_pairs(Path(folder))
    rng.shuffle(pairs)
    pairs = pairs[:PAIRS]
    print(f"{len(pairs)} question pairs")
    rows = []
    start = time.time()
    for i, pair in enumerate(pairs):
        split = "train" if i < int(0.7 * len(pairs)) else "test"
        for label, key in ((0, "human"), (1, "ai")):
            feats, _ = ai_detect.features(pair[key])
            rows.append({"text": pair[key], "label": label, "split": split, "source": pair["source"], "features": feats})
        if (i + 1) % 50 == 0:
            print(f"  features for {i + 1} pairs ({time.time() - start:.0f}s)")
    rows = [r for r in rows if all(not np.isnan(r["features"].get(n, 0.0)) for n in r["features"])]
    weights = {"style": fit(rows, ai_detect.STYLE_FEATURES),
               "dataset": "HC3 (wiki_csai + open_qa), Guo et al. 2023, CC-BY-SA 4.0"}
    if all("log_ppl" in r["features"] for r in rows):
        weights["full"] = fit(rows, ai_detect.ALL_FEATURES)
    (DATA / "ai_detector.json").write_text(json.dumps(weights, indent=1), encoding="utf-8")
    with open(DATA / "eval" / "ai_detection_test.jsonl", "w", encoding="utf-8") as out:
        for r in rows:
            if r["split"] == "test":
                out.write(json.dumps({"text": r["text"], "label": r["label"], "source": r["source"]}, ensure_ascii=False) + "\n")
    for name in ("full", "style"):
        if name in weights:
            m = weights[name]
            print(f"{name:5s}: accuracy {m['test_accuracy']:.3f}  F1 {m['test_f1']:.3f}  AUC {m['test_auc']:.3f}  "
                  f"(train {m['train_size']}, test {m['test_size']})")
            for n, c in sorted(zip(m["features"], m["coef"]), key=lambda t: -abs(t[1])):
                print(f"       {n:13s} {c:+.3f}   human {m['class_means']['human'][n]:.3f}  ai {m['class_means']['ai'][n]:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
