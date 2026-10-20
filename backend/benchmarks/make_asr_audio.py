"""Create the speech-recognition test set: 8 English + 8 Hindi spoken sentences with known text.

    python -m benchmarks.make_asr_audio

English audio uses the Windows built-in voice (System.Speech), Hindi audio uses the offline MMS voice
(nlp/speech.py). Synthetic speech is cleaner than a real classroom recording, so treat the WER as a best case.
"""
import json
import subprocess
import sys
from pathlib import Path

from nlp.speech import synthesize

EVAL = Path(__file__).resolve().parent.parent / "data" / "eval"
OUT = EVAL / "asr"
COUNT = 8


def english_wav(text: str, path: Path) -> bool:
    if sys.platform != "win32":
        return False
    script = (
        "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, "
        "[System.Speech.AudioFormat.AudioChannel]::Mono); "
        f"$s.SetOutputToWaveFile('{path}', $fmt); $s.Rate = 0; $s.Speak($env:NOTEMIND_TTS_TEXT); $s.Dispose()"
    )
    env = {**__import__("os").environ, "NOTEMIND_TTS_TEXT": text}
    return subprocess.run(["powershell", "-NoProfile", "-Command", script], env=env).returncode == 0 and path.exists()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pairs = json.loads((EVAL / "translation.json").read_text(encoding="utf-8"))["pairs"][:COUNT]
    items = []
    for i, pair in enumerate(pairs, 1):
        en_path = OUT / f"en_{i:02d}.wav"
        if english_wav(pair["en"], en_path):
            items.append({"file": en_path.name, "language": "en", "text": pair["en"], "voice": "Windows SAPI (Microsoft David/Zira)"})
        audio = synthesize(pair["hi"], "hi")
        if audio:
            hi_path = OUT / f"hi_{i:02d}.wav"
            hi_path.write_bytes(audio)
            items.append({"file": hi_path.name, "language": "hi", "text": pair["hi"], "voice": "MMS-TTS Hindi (facebook/mms-tts-hin)"})
        print(f"{i}/{len(pairs)}")
    (EVAL / "asr.json").write_text(json.dumps({"description": "Synthetic speech with known transcripts (best-case WER).",
                                               "items": items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(items)} clips written to {OUT}")


if __name__ == "__main__":
    main()
