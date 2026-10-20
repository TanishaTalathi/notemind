from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

EN = ("Photosynthesis is the process by which green plants convert light energy into chemical energy. "
      "Chlorophyll is the green pigment that absorbs sunlight in the chloroplasts. Plants use carbon dioxide "
      "from the air and water from the soil to produce glucose and oxygen. Glucose provides energy for growth, "
      "while oxygen is released into the atmosphere. Photosynthesis is important because it forms the base of "
      "almost every food chain and maintains the balance of oxygen and carbon dioxide.")
HI = ("प्रकाश संश्लेषण वह प्रक्रिया है जिसमें हरे पौधे प्रकाश ऊर्जा को रासायनिक ऊर्जा में बदलते हैं। "
      "क्लोरोफिल सूर्य के प्रकाश को अवशोषित करता है। पौधे कार्बन डाइऑक्साइड और पानी से ग्लूकोज और ऑक्सीजन बनाते हैं। "
      "ऑक्सीजन वायुमंडल में छोड़ी जाती है।")
HINGLISH = "Photosynthesis me plant light energy ko chemical energy me convert karta hai. Paudhe sunlight ki madad se apna khana banate hain."


def post(path, **body):
    r = client.post(path, json=body)
    assert r.status_code == 200, r.text
    return r.json()


def test_detect():
    assert post("/detect", text=EN)["language"] == "en"
    assert post("/detect", text=HI)["language"] == "hi"
    assert post("/detect", text=HINGLISH)["language"] == "hinglish"


def test_autocomplete_learns_from_notes():
    out = post("/autocomplete", text=EN + " Chlorophyll is the green ")
    assert out["suggestion"].lower().startswith("pigment")


def test_summary_modes():
    short = post("/summarize", text=EN, mode="short")
    detailed = post("/summarize", text=EN, mode="detailed")
    bullets = post("/summarize", text=EN, mode="bullets")
    assert 0 < short["summary_words"] <= detailed["summary_words"] < short["original_words"]
    assert bullets["summary"].startswith("•")


def test_questions_all_types():
    for qtype in ("MCQ", "Fill in the blank", "Short answer"):
        qs = post("/questions", text=EN, count=4, difficulty="Medium", qtype=qtype)["questions"]
        assert qs, qtype
        for q in qs:
            if qtype == "MCQ":
                assert q["answer"] in q["options"] and len(set(q["options"])) == len(q["options"])
            if qtype == "Fill in the blank":
                assert "_____" in q["question"]
    hindi = post("/questions", text=HI, count=3, difficulty="Easy", qtype="MCQ")["questions"]
    assert hindi and hindi[0]["question"].startswith("रिक्त")


def test_ask():
    out = post("/ask", text=EN, question="What is chlorophyll?")
    assert "pigment" in out["answer"] and out["sources"]
    assert "couldn't find" in post("/ask", text=EN, question="Who won the cricket world cup?")["answer"]


def test_concept_map():
    out = post("/concept-map", text=EN)
    ids = {n["id"] for n in out["nodes"]}
    assert {"Photosynthesis", "Chlorophyll", "Glucose"} <= ids
    assert all(e["source"] in ids and e["target"] in ids for e in out["edges"])
    assert any(e["relation"] == "absorbs" for e in out["edges"])


def test_glossary():
    out = post("/glossary", text=EN)
    terms = {t["term"]: t for t in out["terms"]}
    assert terms["Chlorophyll"]["difficulty"] == "Hard"
    assert terms["Chlorophyll"]["meaning"].startswith("The green pigment")
    assert 0 <= out["readability"] <= 100 and out["grade"]


def test_cleanup_hinglish():
    out = post("/cleanup", text=HINGLISH)
    assert "करता है" in out["devanagari"] and "sunlight" in out["devanagari"]
    assert out["english"]


def test_translate_never_crashes():
    out = post("/translate", text=EN, source="en", target="hi")
    assert "available" in out


def test_check_answer():
    expected = "Friction is a force that opposes motion between two surfaces in contact."
    good = post("/check-answer", expected=expected, answer="A force that resists movement when two surfaces rub together.", question="What is friction?")
    wrong = post("/check-answer", expected=expected, answer="Friction is a force that helps objects move faster.", question="What is friction?")
    assert good["verdict"] == "correct" and good["score"] >= 70
    assert wrong["verdict"] == "incorrect" and wrong["score"] < good["score"]
    assert post("/check-answer", expected=expected, answer="", question="")["score"] == 0


def test_ai_check():
    out = post("/ai-check", text=EN)
    assert set(out) >= {"ai_probability", "label", "signals", "sentences", "message"}
    if out["available"]:
        assert 0 <= out["ai_probability"] <= 1 and out["signals"]
    assert post("/ai-check", text="Too short.")["label"] in {"Too short", "Unavailable"}


def test_transcribe_voice_note():
    from pathlib import Path
    clip = Path(__file__).resolve().parent.parent / "data" / "eval" / "asr" / "en_03.wav"
    r = client.post("/transcribe", files={"audio": ("note.wav", clip.read_bytes(), "audio/wav")}, data={"language": "auto"})
    assert r.status_code == 200, r.text
    out = r.json()
    if out["available"]:
        assert "heart" in out["text"].lower() and out["language"] == "en"
    assert client.post("/transcribe", files={"audio": ("empty.webm", b"", "audio/webm")}).status_code == 400


def test_evaluation_results_listed():
    out = client.get("/evaluate").json()
    assert "translation" in out["suites"] and isinstance(out["results"], dict)
    assert client.post("/evaluate/nonsense").status_code == 404


def test_tts_returns_audio_or_clear_message():
    r = client.post("/tts", json={"text": "पौधे धूप से भोजन बनाते हैं।", "lang": "hi"})
    if r.status_code == 200:
        assert r.headers["content-type"] == "audio/wav" and r.content[:4] == b"RIFF"
    else:
        assert r.status_code == 503 and r.json()["available"] is False
