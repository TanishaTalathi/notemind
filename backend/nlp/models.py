"""Optional transformer models (Hugging Face). Every feature works without them; when
`transformers` is installed they are loaded lazily on first use and upgrade the results.

Environment variables:
  NOTEMIND_USE_ML=0          turn all transformer models off even if installed
  NOTEMIND_ABSTRACTIVE=1     use mT5 (XL-Sum) for abstractive summaries (2.3 GB download)
  NOTEMIND_GPT2=1            use DistilGPT-2 as a second autocomplete engine for English
"""
import logging
import os
from functools import lru_cache

log = logging.getLogger("notemind.models")

TRANSLATION_MODELS = {("en", "hi"): "Helsinki-NLP/opus-mt-en-hi", ("hi", "en"): "Helsinki-NLP/opus-mt-hi-en"}
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
GPT2_MODEL = "distilgpt2"
TTS_MODELS = {"hi": "facebook/mms-tts-hin", "en": "facebook/mms-tts-eng"}  # ~145 MB each
DETECTOR_LM = "gpt2"  # perplexity scorer for the AI-written check (~550 MB)
NLI_MODEL = "MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli"  # entailment/contradiction for answer checking (~470 MB)


def _flag(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def ml_available() -> bool:
    if not _flag("NOTEMIND_USE_ML", "1"):
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return True
    except Exception:
        return False


_failed: set[str] = set()


def _safe_load(key: str, loader):
    """Load a model once; remember failures (no internet, low memory) so we fall back quickly."""
    if key in _failed or not ml_available():
        return None
    try:
        return loader()
    except Exception as exc:  # pragma: no cover - depends on the machine
        log.warning("Could not load %s (%s). Falling back to the classical pipeline.", key, exc)
        _failed.add(key)
        return None


@lru_cache(maxsize=2)
def translation_pipeline(source: str, target: str):
    name = TRANSLATION_MODELS.get((source, target))
    if not name:
        return None

    def load():
        from transformers import MarianMTModel, MarianTokenizer
        return MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name)

    return _safe_load(name, load)


@lru_cache(maxsize=1)
def embedding_model():
    def load():
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)

    return _safe_load(EMBEDDING_MODEL, load)


@lru_cache(maxsize=1)
def summary_pipeline():
    if not _flag("NOTEMIND_ABSTRACTIVE", "0"):
        return None

    def load():
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        return AutoTokenizer.from_pretrained(SUMMARY_MODEL, use_fast=False), AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL)

    return _safe_load(SUMMARY_MODEL, load)


@lru_cache(maxsize=1)
def gpt2_pipeline():
    if not _flag("NOTEMIND_GPT2", "0"):
        return None

    def load():
        from transformers import pipeline
        return pipeline("text-generation", model=GPT2_MODEL)

    return _safe_load(GPT2_MODEL, load)


@lru_cache(maxsize=2)
def tts_pipeline(lang: str):
    name = TTS_MODELS.get(lang)
    if not name:
        return None

    def load():
        from transformers import AutoTokenizer, VitsModel
        return AutoTokenizer.from_pretrained(name), VitsModel.from_pretrained(name)

    return _safe_load(name, load)


@lru_cache(maxsize=1)
def detector_lm():
    def load():
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(DETECTOR_LM)
        model.eval()
        return AutoTokenizer.from_pretrained(DETECTOR_LM), model

    return _safe_load(DETECTOR_LM, load)


@lru_cache(maxsize=1)
def nli_model():
    def load():
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model = AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
        model.eval()
        return AutoTokenizer.from_pretrained(NLI_MODEL), model

    return _safe_load(NLI_MODEL, load)


def status() -> dict:
    return {
        "transformers_installed": ml_available(),
        "abstractive_summary": _flag("NOTEMIND_ABSTRACTIVE", "0") and ml_available(),
        "gpt2_autocomplete": _flag("NOTEMIND_GPT2", "0") and ml_available(),
        "failed_models": sorted(_failed),
    }
