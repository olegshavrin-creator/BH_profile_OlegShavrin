"""Smoke test of the BS 2.0 analyses on one existing segment of a batch run.
Usage: test_analyses.py SEG_MP4 TIMELINE_JSON SEG_NUMBER
"""
import json
import sys
import time
from pathlib import Path

from bs2.analyses.emotions_text import TextEmotion, dominant
from bs2.analyses.emotions_voice import VoiceEmotion
from bs2.analyses.face_expr import FaceExpression
from bs2.analyses.speech_stats import describe, stats_for, vocabulary
from bs2.translate import translate_sentences
from bs2.longvideo import LongVideoAnalyzer

seg, tl_path, n = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
tl = json.loads(tl_path.read_text(encoding="utf-8"))
t = next(x for x in tl if x["segment"] == n)
text_ru = t["transcript"]
print("segment", n, t["start"], t["end"], "| ru:", text_ru[:100])

t0 = time.time()
en = " ".join(translate_sentences([s for s in text_ru.replace("?", ".").split(". ") if s], "ru", "en"))
print("en:", en[:120], f"({time.time() - t0:.1f}s)")

t0 = time.time(); te = TextEmotion(); pe = te(en)
print("text emotions:", dominant(pe), {k: round(v, 2) for k, v in pe.items()}, f"({time.time() - t0:.1f}s)")

t0 = time.time(); ve = VoiceEmotion()
import subprocess, tempfile
wav = Path(tempfile.mkdtemp()) / "a.wav"
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(seg), "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
print("voice A/D/V:", ve.from_file(str(wav)), f"({time.time() - t0:.1f}s)")

t0 = time.time(); fe = FaceExpression(); fr = fe.from_video(str(seg))
print("face:", fr.get("dominant"), {k: round(v, 2) for k, v in fr["expressions"].items()}, "frames", fr["frames"],
      "head_motion", fr.get("head_motion"), "face_share", fr.get("face_share"), f"({time.time() - t0:.1f}s)")

# speech stats need Whisper chunks: re-transcribe just this segment (quick) to test the code path
an = LongVideoAnalyzer(backend=None, lang="ru")
full, chunks = an.transcribe(seg, Path(tempfile.mkdtemp()))
st = stats_for(chunks, 0, t["end"] - t["start"], lang="ru")
print("speech:", st)
print(describe(st))
print("vocab:", vocabulary(full)[:8])
