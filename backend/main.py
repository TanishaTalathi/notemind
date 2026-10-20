"""NoteMind API — multilingual AI study assistant.

Run:  uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import logging
import threading
import time
from typing import Literal

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from benchmarks import run as benchmarks
from nlp import models
from nlp.ai_detect import check_ai
from nlp.answer_check import check_answer
from nlp.concepts import build_concept_map
from nlp.glossary import build_glossary
from nlp.hinglish import to_devanagari, word_gloss_english
from nlp.language import base_language, detect
from nlp.lm import autocomplete as lm_autocomplete
from nlp.qa import answer_question
from nlp.questions import generate_questions
from nlp.speech import synthesize
from nlp.summarizer import summarize as summarize_text
from nlp.transcribe import transcribe
from nlp.translator import translate as translate_text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("notemind")

app = FastAPI(title="NoteMind API", version="1.0.0",
              description="Summaries, translation, quizzes, Q&A, concept maps, glossary, Hinglish cleanup "
                          "and autocomplete for study notes in English, Hindi and Hinglish.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MAX_CHARS = 60_000
MAX_AUDIO_BYTES = 25 * 1024 * 1024
Lang = Literal["en", "hi", "hinglish"]


class TextIn(BaseModel):
    text: str = Field(..., max_length=MAX_CHARS)


class SummarizeIn(TextIn):
    mode: Literal["short", "detailed", "bullets"] = "detailed"


class TranslateIn(TextIn):
    source: Lang = "en"
    target: Lang = "hi"


class SpeakIn(BaseModel):
    text: str = Field(..., max_length=2000)
    lang: Literal["hi", "en"] = "hi"


class CheckAnswerIn(BaseModel):
    expected: str = Field(..., max_length=4000)
    answer: str = Field(..., max_length=4000)
    question: str = Field("", max_length=1000)


class QuestionsIn(TextIn):
    count: int = Field(5, ge=1, le=25)
    difficulty: str = "Medium"
    qtype: str = "MCQ"
    seed: int | None = None


class AskIn(TextIn):
    question: str = Field(..., max_length=1000)


TRANSLATION_HELP = ("Translation is unavailable. Connect to the internet (uses Google Translate or MyMemory), or install the "
                    "offline models with: pip install -r requirements-ml.txt")


@app.on_event("startup")
def warm_speech_model():
    # Load the Hindi voice in the background so the first Listen click doesn't wait ~20 s
    if models.ml_available():
        threading.Thread(target=models.tts_pipeline, args=("hi",), daemon=True).start()


@app.middleware("http")
async def timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.0f}"
    return response


@app.get("/")
def root():
    return {"name": "NoteMind API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "engines": models.status()}


@app.get("/health/translate")
def health_translate():
    sample, engine = translate_text("Plants need sunlight.", "en", "hi")
    return {"working": sample is not None, "engine": engine, "sample": sample or ""}


@app.post("/detect")
def detect_language(body: TextIn):
    return detect(body.text)


@app.post("/autocomplete")
def autocomplete(body: TextIn):
    return lm_autocomplete(body.text[-3000:])


@app.post("/summarize")
def summarize(body: SummarizeIn):
    return summarize_text(body.text, body.mode)


@app.post("/translate")
def translate(body: TranslateIn):
    source = "hi" if body.source == "hi" else "en"
    target = "hi" if body.target == "hi" else "en"
    text = body.text
    if body.source == "hinglish" or (detect(text)["language"] == "hinglish" and source == "en"):
        # Roman-script Hindi: convert to Devanagari first so the translator sees real Hindi
        text = to_devanagari(text)[0]
        source = "hi"
        if target == "hi":
            return {"translation": text, "engine": "hinglish-transliteration", "available": True}
    translation, engine = translate_text(text, source, target)
    if translation is None:
        return {"translation": "", "engine": engine, "available": False, "message": TRANSLATION_HELP}
    return {"translation": translation, "engine": engine, "available": True}


@app.post("/tts")
def tts(body: SpeakIn):
    audio = synthesize(body.text, body.lang)
    if audio is None:
        return JSONResponse(status_code=503, content={
            "available": False,
            "message": "Offline speech needs the ML models: pip install -r requirements-ml.txt"})
    return Response(content=audio, media_type="audio/wav")


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...), language: Literal["auto", "en", "hi"] = Form("auto")):
    data = await audio.read()
    if not data:
        return JSONResponse(status_code=400, content={"available": False, "message": "The recording is empty."})
    if len(data) > MAX_AUDIO_BYTES:
        return JSONResponse(status_code=413, content={"available": False, "message": "Audio is larger than 25 MB."})
    try:
        return transcribe(data, language)
    except Exception as exc:  # undecodable file
        log.warning("Transcription failed: %s", exc)
        return JSONResponse(status_code=400, content={"available": True, "message": "Could not read this audio file."})


@app.post("/check-answer")
def check_short_answer(body: CheckAnswerIn):
    return check_answer(body.expected, body.answer, body.question)


@app.post("/ai-check")
def ai_check(body: TextIn):
    return check_ai(body.text)


@app.get("/evaluate")
def evaluation_results():
    return {"suites": benchmarks.SUITE_INFO, "results": benchmarks.load_results()}


@app.post("/evaluate/{suite}")
def run_evaluation(suite: str):
    if suite not in benchmarks.SUITES:
        return JSONResponse(status_code=404, content={"message": f"Unknown suite. Choose one of: {', '.join(benchmarks.SUITES)}"})
    return benchmarks.run_suite(suite)


@app.post("/questions")
def questions(body: QuestionsIn):
    return generate_questions(body.text, body.count, body.difficulty, body.qtype, body.seed)


@app.post("/ask")
def ask(body: AskIn):
    return answer_question(body.text, body.question)


@app.post("/concept-map")
def concept_map(body: TextIn):
    return build_concept_map(body.text)


@app.post("/glossary")
def glossary(body: TextIn):
    return build_glossary(body.text)


@app.post("/cleanup")
@app.post("/normalize")
def cleanup(body: TextIn):
    info = detect(body.text)
    if info["language"] == "en":
        return {"devanagari": body.text, "english": body.text, "changed": [], "input_language": "en",
                "engine": "none", "message": "Your notes are already in English — nothing to clean up."}
    if base_language(body.text) == "hi":
        devanagari, changed = body.text, []
    else:
        devanagari, changed = to_devanagari(body.text)
    english, engine = translate_text(devanagari, "hi", "en")
    if english is None:
        english, engine = word_gloss_english(body.text), "word-gloss (approximate, offline)"
    return {"devanagari": devanagari, "english": english, "changed": changed,
            "input_language": info["language"], "engine": engine}
