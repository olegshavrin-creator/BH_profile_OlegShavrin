"""Synthetic job folders for every generation the page and the PDF must read (numbers only, built in a temp dir: no
names, transcripts or job ids). `make_job(kind, base)` writes result.json and, where the generation has them,
explain/explanation.json and a few tiny key-frame JPEGs, and returns the folder.

The generations (`KINDS`):
  imported20        an imported 2.0 job: two members, a stored 2.0 `narrative`, the «собеседование» label and
                    percentiles, key frames in the old explanation format, and no `input.*` (the upload lay outside)
  v30               a 3.0 job: a stored MBTI section of schema 1 (get_mbti recomputes it on render) and percentiles
  oceanai31         a 3.1 OCEAN-AI job
  mm31_old_frames   a 3.1 AMLAI 1.0 job whose explanation carries the key frames in the old format (magnitudes only,
                    no phrases and no signed effect)
  mm31_captions     a 3.1 AMLAI 1.0 job whose explanation carries the key-frame captions (phrase, expressions and the
                    signed per-frame effect)
  short_clip        a single short clip with no timeline at all

All carry synthetic `analyses` — the means of test_characterization.AN plus three `per_segment` rows: one without the
voice, one whose face has no expressions and one with a speech of 0 words.
"""
from __future__ import annotations

import copy
from pathlib import Path

from PIL import Image

from samples import oceanai31 as _oceanai31, own as _own, rep
from test_characterization import AN

from bs3 import jobfiles
from bs3.norms import TRAIT_KEYS

KINDS = ("imported20", "v30", "oceanai31", "mm31_old_frames", "mm31_captions", "short_clip")

# the key frames of a job of AMLAI 1.0: `key_<i>_frame<N>.jpg` — `i` (1, 2, 3) indexes the per-frame attribution,
# `N` (10, 25, 40) is the frame number in the clip that gives its moment (bs3/frame_phrase.py)
_FRAMES = ("key_01_frame10.jpg", "key_02_frame25.jpg", "key_03_frame40.jpg")

JOB_NAMES = {                              # only synthetic year-2000 ids; the 2.0 and 3.0 jobs keep the old naming
    "imported20": "20000101_000000",
    "v30": "20000102_000000",
    "oceanai31": "20000103_000000_0a1b2c3d",
    "mm31_old_frames": "20000104_000000_1a2b3c4d",
    "mm31_captions": "20000105_000000_5e6f7a8b",
    "short_clip": "20000106_000000_9c0d1e2f",
}


# ---------------------------------------------------------------- the synthetic analyses
def _per_segment() -> list:
    """Three per_segment rows: one without the voice, one whose face has no expressions and one with 0 words of
    speech and an empty transcript («нет речи»); the shape facts.segment_cells and the charts read."""
    return [
        {"start": 0.0, "end": 20.0,                                         # no voice
         "speech": {"words_per_min_speech": 132.0, "pause_share": 0.12, "words": 44, "fillers_per_100": 1.0},
         "face": {"expressions": {"happy": 0.55, "neutral": 0.30, "surprise": 0.15}, "frames": 30},
         "emotions_text": {"joy": 0.5, "neutral": 0.45, "anger": 0.05}, "text_en": "hello there"},
        {"start": 20.0, "end": 40.0,                                        # face without expressions
         "voice": {"arousal": 0.32, "dominance": 0.41, "valence": 0.50},
         "speech": {"words_per_min_speech": 88.0, "pause_share": 0.22, "words": 28, "fillers_per_100": 3.0},
         "face": {"frames": 12},
         "emotions_text": {"neutral": 0.80, "sadness": 0.20}, "text_en": "some more words"},
        {"start": 40.0, "end": 60.0,                                        # 0 words of speech, empty transcript
         "voice": {"arousal": 0.20, "dominance": 0.30, "valence": 0.42},
         "speech": {"words_per_min_speech": 0.0, "pause_share": 0.0, "words": 0},
         "face": {"expressions": {"neutral": 0.90, "fear": 0.10}, "frames": 20},
         "emotions_text": {}, "text_en": ""},
    ]


def _analyses(name: str = "B") -> dict:
    an = copy.deepcopy(AN[name])
    an["per_segment"] = _per_segment()
    return an


def _three_segments(r: dict) -> None:
    """Trim the sample timeline to three segments aligned with the per_segment rows (0–20, 20–40, 40–60)."""
    tl = r["timeline"][:3]
    for i, t in enumerate(tl):
        t["segment"], t["start"], t["end"] = i + 1, 20.0 * i, 20.0 * (i + 1)
    r["timeline"] = tl
    r["segments"] = len(tl)
    r["duration_sec"] = 60.0
    r["representative_segment"] = 1


def _percentiles(r: dict, pct: int) -> None:
    """A pool percentile of each trait, as older Russian jobs carry it (dropped from the view, scores.clean_view)."""
    for k in TRAIT_KEYS:
        t = (r.get("traits") or {}).get(k)
        if isinstance(t, dict):
            t["percentile"] = pct
            t["percentile_ref"] = "pool_ru"


# ---------------------------------------------------------------- the explanation of a job of AMLAI 1.0
def _modalities() -> dict:
    return {"input_x_gradient": {k: {"face": {"share": 0.55, "signed": 0.020},
                                     "audio": {"share": 0.40, "signed": 0.015},
                                     "text": {"share": 0.03, "signed": 0.001},
                                     "behavior": {"share": 0.02, "signed": 0.001}} for k in TRAIT_KEYS}}


def _readable_words() -> dict:
    return {"transcript_words": {k: {"up": [{"word": "work", "ru": "работа", "signed": 0.02}],
                                     "down": [{"word": "late", "ru": "поздно", "signed": -0.01}]}
                                 for k in TRAIT_KEYS}}


def _per_output(signed: bool) -> dict:
    """`per_output[trait]` of the frame attribution; index 1, 2, 3 carry the three key frames (_FRAMES)."""
    out = {}
    for k in TRAIT_KEYS:
        out[k] = {"importance": [0.0, 0.30, 0.20, 0.10]}
        if signed:
            out[k]["signed"] = [0.0, 0.25, -0.15, 0.08]
    return out


def _key_frame_info() -> list:
    return [
        {"file": _FRAMES[0], "frame": 1, "phrase": "улыбается, смотрит в камеру",
         "expressions": [{"label": "happy", "ru": "радость", "share": 0.62},
                         {"label": "neutral", "ru": "нейтрально", "share": 0.21}]},
        {"file": _FRAMES[1], "frame": 2, "phrase": "жестикулирует, наклоняется вперёд",
         "expressions": [{"label": "neutral", "ru": "нейтрально", "share": 0.50},
                         {"label": "surprise", "ru": "удивление", "share": 0.20}]},
        {"file": _FRAMES[2], "frame": 3, "phrase": "спокойное лицо, смотрит в сторону",
         "expressions": [{"label": "neutral", "ru": "нейтрально", "share": 0.70}]},
    ]


def _expl_captions() -> dict:
    """The new format: the key-frame captions (phrase, expressions) and the signed per-frame effect."""
    return {"clip_fps": 25.0, "modalities": _modalities(),
            "frames": {"per_output": _per_output(signed=True),
                       "per_frame_effect": [[],
                                            [{"output": "extraversion", "signed": 0.31},
                                             {"output": "openness", "signed": 0.12}],
                                            [{"output": "agreeableness", "signed": -0.20},
                                             {"output": "conscientiousness", "signed": 0.09}],
                                            [{"output": "emotional_stability", "signed": 0.15}]],
                       "key_frame_info": _key_frame_info(), "key_frame_files": list(_FRAMES),
                       "top_frames_overall": [10, 25, 40]},
            "readable_words": _readable_words(), "readable_words_by": "ollama"}


def _expl_old_frames() -> dict:
    """The old format: magnitudes only, no phrases and no signed effect (a job made before the captions)."""
    return {"modalities": _modalities(),
            "frames": {"per_output": _per_output(signed=False), "key_frame_files": list(_FRAMES)},
            "readable_words": _readable_words(), "readable_words_by": "ollama"}


def _write_frames(job: Path) -> list:
    """Three tiny landscape JPEGs in <job>/explain (three per row in the PDF); returns their paths."""
    ex = job / jobfiles.EXPLAIN_DIR
    ex.mkdir(parents=True, exist_ok=True)
    for f in _FRAMES:
        Image.new("RGB", (96, 72), (90, 90, 90)).save(ex / f, "JPEG")
    return [str(ex / f) for f in _FRAMES]


# ---------------------------------------------------------------- the jobs
def make_job(kind: str, base) -> Path:
    """Write a synthetic job of generation `kind` into `base` and return its folder."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, not {kind!r}")
    job = Path(base) / JOB_NAMES[kind]
    job.mkdir(parents=True)

    if kind == "imported20":
        r = rep("B")                                       # two members, ensemble, OCEAN-AI primary
        _three_segments(r)
        r["analyses"] = _analyses("B")
        r["narrative"] = "Оценки дала система OCEAN-AI по русской речи."     # the stored 2.0 plain-language summary
        r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
        for t in r["timeline"]:
            if isinstance(t.get("scores"), dict):
                t["scores"]["interview"] = 0.40
        _percentiles(r, 55)
        _write_frames(job)                                 # shown as OCEAN-AI but carries AMLAI 1.0 files (guard list)
        r["key_frames"] = [str(job / jobfiles.EXPLAIN_DIR / f) for f in _FRAMES]
        jobfiles.write_json(jobfiles.explanation_path(job), _expl_old_frames())
        # no input.* : an imported 2.0 job kept the upload outside the folder
    elif kind == "v30":
        r = rep("B")
        _three_segments(r)
        r["analyses"] = _analyses("B")
        r["mbti"] = {"schema_version": 1, "type": "ENFJ"}  # schema 1: get_mbti recomputes it on render (guard list)
        _percentiles(r, 60)
    elif kind == "oceanai31":
        r = _oceanai31("B")
        _three_segments(r)
        r["analyses"] = _analyses("B")
    elif kind in ("mm31_old_frames", "mm31_captions"):
        r = _own("B")
        _three_segments(r)
        r["analyses"] = _analyses("B")
        r["media"] = {"fps": 25.0, "file_name": "clip.mp4"}
        r["behavior_description"] = "[0-20 s] The person speaks calmly and gestures."
        r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно и жестикулирует."
        r["behavior_description_ru_by"] = "ollama"
        r["key_frames"] = _write_frames(job)
        (job / "input.mp4").write_bytes(b"\x00" * 32)
        jobfiles.write_json(jobfiles.explanation_path(job),
                            _expl_captions() if kind == "mm31_captions" else _expl_old_frames())
    else:                                                  # short_clip
        r = _oceanai31("B")
        r["timeline"] = []
        r["segments"] = 1
        r["duration_sec"] = 8.0
        r["analyses"] = _analyses("B")

    r["job_dir"] = str(job)
    jobfiles.write_json(job / jobfiles.RESULT, r)
    return job
