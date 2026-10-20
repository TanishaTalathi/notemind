# NoteMind: Multilingual AI Study Assistant

NoteMind is an NLP web app. You type, paste or speak study notes in **English, Hindi or Hinglish** and get a set of study tools from them, plus an evaluation page that measures how well each tool works.

| # | Feature | NLP technique (backend/nlp/) |
|---|---------|------------------------------|
| 1 | **Language detection** (live badge) | Script analysis (Devanagari ratio) + Hinglish marker lexicon (`language.py`) |
| 2 | **Smart autocomplete** (ghost text, Tab to accept) | **Trigram language model with interpolated Kneser-Ney smoothing**, written from scratch and trained on a study corpus + your own notes (`lm.py`) |
| 3 | **Summarization**: Short / Detailed / Bullets | **TextRank**: TF-IDF sentence graph + PageRank (`summarizer.py`); optional abstractive mT5 |
| 4 | **Translation** English ⇄ Hindi | MarianMT (Helsinki-NLP opus-mt) offline, or Google Translate fallback (`translator.py`) |
| 5 | **Quiz generation**: MCQ, fill-in-the-blank, short answer, flashcards | Keyword-sentence pairing, cloze deletion, difficulty-aware distractors (`questions.py`) |
| 6 | **Ask your notes** chatbot (with sources) | Retrieval QA (mini-RAG): TF-IDF word + char n-grams, or multilingual sentence embeddings, plus cross-lingual querying (`qa.py`) |
| 7 | **Concept map** (interactive graph) + keyword chart | Keyword extraction (frequency × rarity from wordfreq), co-occurrence graph, verb-based relation labels (`keywords.py`, `concepts.py`) |
| 8 | **Smart glossary + readability + "Simplify my notes"** | Rarity-based difficulty, definition-pattern mining, Flesch Reading Ease / Flesch-Kincaid grade, rule-based simplification (`glossary.py`) |
| + | **Hinglish cleanup** | Roman→Devanagari transliteration (lexicon + rules + frequency-guided spelling choice), English words kept, then translated to clean English (`hinglish.py`) |
| 9 | **Voice notes** (record or upload audio) | **Whisper** speech recognition (faster-whisper, offline). Hindi/Urdu script fix: Hindi speech is decoded with a Devanagari prompt (`transcribe.py`) |
| 10 | **Short-answer checking** in the quiz | Multilingual **sentence embeddings** + key-term coverage + a cross-encoder **NLI model** (entailment / contradiction) so opposite answers aren't accepted (`answer_check.py`) |
| 11 | **AI-writing check** (“was this written by AI?”) | **GPT-2 perplexity** and burstiness + stylometric features, combined by **logistic regression** trained on the HC3 human-vs-ChatGPT corpus; every signal's contribution is shown (`ai_detect.py`) |
| 12 | **Evaluation page** (`/evaluation`) | BLEU, chrF, ROUGE, accuracy, precision/recall/F1, AUC, confusion matrices, WER/CER on test sets (`benchmarks/run.py`) |

The page also has text-to-speech ("Listen", with an offline Hindi voice), dark mode, .txt upload, sample notes in all three languages, study stats, and export to .txt or PDF.

## Project structure

```
notemind/
├── backend/                FastAPI + the NLP pipeline
│   ├── main.py             API endpoints
│   ├── nlp/                one module per feature
│   ├── benchmarks/         evaluation suites + detector training script
│   ├── data/               language-model corpus, ai_detector.json (trained weights)
│   │   └── eval/           test sets, speech clips, saved evaluation results
│   ├── tests/test_api.py   14 end-to-end tests
│   ├── requirements.txt    lightweight core (~100 MB)
│   └── requirements-ml.txt optional transformer models
└── frontend/               Next.js + Tailwind + shadcn/ui (started from v0)
    ├── app/page.tsx        the main page
    ├── app/evaluation/     the evaluation page
    ├── components/notemind each results tab
    └── lib/api.ts          API client (+ demo data if backend is off)
```

## How to run

You need **Python 3.10+** and **Node.js 18+**.

**1. Backend** (terminal 1)
```bash
cd backend
python -m venv venv
venv\Scripts\activate            # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/docs to try every endpoint in the browser.

**2. Frontend** (terminal 2)
```bash
cd frontend
npm install          # or: pnpm install
npm run dev
```
Open http://localhost:3000. The header badge should say **Backend connected**. If it says **Demo mode**, the backend isn't running and the page is showing sample data.

**3. Run the tests**
```bash
cd backend
pip install pytest httpx
python -m pytest tests -q
```

The first voice note downloads the Whisper `small` model (480 MB). Set `NOTEMIND_WHISPER_MODEL=base` for a smaller, faster model (fine for English, poor for Hindi).

## Translation and optional transformer models

With just `requirements.txt`, translation uses Google Translate through `deep-translator`. This needs internet but no API key. Everything else runs fully offline.

For **offline, model-based** results, install the transformer extras. This is about 2–3 GB, and the models download on first use:
```bash
pip install -r requirements-ml.txt
```
This turns on:
- **MarianMT** (`Helsinki-NLP/opus-mt-en-hi`, `opus-mt-hi-en`) for translation and Hinglish→English
- **Multilingual sentence embeddings** (`paraphrase-multilingual-MiniLM-L12-v2`) for "Ask your notes". With these, a Hindi question can match English notes directly.

You can also set these environment variables before starting uvicorn:
- `NOTEMIND_ABSTRACTIVE=1` uses mT5 (XL-Sum) abstractive summaries. This is a 2.3 GB download.
- `NOTEMIND_GPT2=1` adds DistilGPT-2 as a second autocomplete engine for English.
- `NOTEMIND_USE_ML=0` forces the lightweight classical pipeline.

`GET /health` shows which engines are active. Each result in the UI also shows a small "Engine:" label.

## API

| Method | Path | Body | Returns |
|---|---|---|---|
| POST | `/detect` | `{text}` | `{language, confidence}` |
| POST | `/autocomplete` | `{text}` | `{suggestion, confidence, engine}` |
| POST | `/summarize` | `{text, mode}` | `{summary, original_words, summary_words, key_sentences}` |
| POST | `/translate` | `{text, source, target}` | `{translation, engine, available}` |
| POST | `/questions` | `{text, count, difficulty, qtype, seed?}` | `{questions:[{type, question, options?, answer, explanation}]}` |
| POST | `/ask` | `{text, question}` | `{answer, sources, confidence}` |
| POST | `/concept-map` | `{text}` | `{nodes, edges, keywords}` |
| POST | `/glossary` | `{text}` | `{terms, readability, grade, simplified}` |
| POST | `/cleanup` | `{text}` | `{devanagari, english, changed}` |
| POST | `/transcribe` | form-data `audio` file + `language` (auto/en/hi) | `{text, segments, language, duration, engine}` |
| POST | `/check-answer` | `{expected, answer, question}` | `{score, verdict, similarity, entailment, contradiction, matched_terms, missing_terms, feedback}` |
| POST | `/ai-check` | `{text}` | `{ai_probability, label, confidence, signals, sentences, message}` |
| POST | `/tts` | `{text, lang: "hi"}` | WAV audio (offline Hindi voice) |
| GET | `/evaluate` | — | saved results of every evaluation suite |
| POST | `/evaluate/{suite}` | — | runs one suite (`translation`, `summarization`, `qa`, `language_id`, `answer_check`, `ai_detection`, `asr`) |

## Evaluation

Open http://localhost:3000/evaluation, or run every suite from the terminal (results are saved to `backend/data/eval/results/`):
```bash
cd backend
python -m benchmarks.run all          # or e.g.: python -m benchmarks.run translation asr
```

Results measured on the development laptop (CPU only):

| Feature | Test data | Result |
|---|---|---|
| AI-writing check | 120 held-out HC3 answers | **97.5% accuracy**, AUC 0.994 (style signals only: 86.7%) |
| Answer checking | 41 hand-graded answers | **70.7% accuracy** (3 classes), **0% wrong answers marked correct** (embeddings only: 38.5%) |
| Language detection | 60 snippets | **96.7% accuracy** |
| Voice notes (Whisper small) | 16 synthetic spoken sentences | **WER 5.6% English**, 38.4% Hindi |
| Translation | 20 sentence pairs | chrF: MyMemory 84, MarianMT 50–59 (Google blocked on this network) |
| Summaries | 6 passages | ROUGE-L 38.5 (TextRank) vs 39.8 (first-sentences baseline) |
| Ask Notes | 35 questions | 69% (TF-IDF), 60% (embeddings); weak on unanswerable and cross-lingual questions |

Caveats worth stating in a report:
- The small test sets in `data/eval/` (translation references, summaries, questions, graded answers, language snippets) were written for this project, not taken from a published benchmark. The Hindi references should ideally be checked by a second Hindi speaker.
- Answer-checking weights were tuned on the same 41 answers; leave-one-out cross-validation gives about 61%, a fairer estimate.
- The AI-writing detector was trained and tested on HC3 (2023 ChatGPT vs Wikipedia/Q&A answers). Newer chatbots, edited AI text and student notes are harder. It is an estimate, never proof.
- Speech clips are synthetic voices (Windows SAPI for English, MMS-TTS for Hindi); real recordings will have higher error rates.

To rebuild the generated data:
```bash
python -m benchmarks.make_asr_audio                         # speech clips
python -m benchmarks.train_ai_detector path/to/hc3_folder   # needs wiki_csai.jsonl and open_qa.jsonl from
                                                            # https://huggingface.co/datasets/Hello-SimpleAI/HC3
```

**Data credit:** `data/eval/ai_detection_test.jsonl` and the detector weights come from HC3, Guo et al. 2023, “How Close is ChatGPT to Human Experts?” (CC-BY-SA 4.0).

## Talking points for your report or viva

- **Language model:** `lm.py` implements interpolated Kneser-Ney by hand: continuation counts, absolute discounting D = 0.75, and back-off weights. `KneserNeyTrigramLM.perplexity()` lets you report perplexity. On the bundled corpus, in-domain sentences score about 8, while random word salad scores over 30,000.
- **Why Kneser-Ney?** It predicts words by how many different contexts they follow, not just by raw frequency. So "Francisco" (which follows only "San") doesn't get over-predicted.
- **Personalisation:** the model is retrained on your notes, weighted 3×, so suggestions use your own topic's vocabulary.
- **Low-resource handling:** Hindi and Hinglish work without big models. The pipeline relies on script detection, frequency lists and transliteration, and transformers are an optional upgrade.
- **Bi-encoder vs cross-encoder:** sentence embeddings (bi-encoder) say whether two answers are about the same thing; an NLI cross-encoder reads them together and can tell agreement from contradiction. In answer checking this cut wrong answers accepted as correct from 38.5% to 0%.
- **Perplexity as a signal:** language models find AI text very predictable. On HC3 the median GPT-2 perplexity was 25 for human text and 9 for ChatGPT text, which alone separates them well.
- **Know your data:** the em-dash feature was dropped from the detector because in HC3 it marked *human* text, while newer chatbots use em dashes heavily. A feature can be accurate on the training data and still wrong for real use.
- **Future work:** fine-tune GPT-2 on a notes dataset, add T5 question generation, use IndicTrans2 for more Indian languages, improve Ask Notes on unanswerable and cross-lingual questions, and record real (not synthetic) speech for testing.
