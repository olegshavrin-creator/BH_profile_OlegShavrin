"""Aggregation of new jobs (design 6.3): segments the main system did not score stay out of the main scores and the
spread; every timeline entry keeps `variants` and `primary_used`. No models: a fake backend and patched ffmpeg."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from bs3 import longvideo
from bs3.norms import TRAIT_KEYS


def _scores(base: float, interview: float | None = None) -> dict:
    d = {k: round(base + 0.01 * i, 4) for i, k in enumerate(TRAIT_KEYS)}
    if interview is not None:
        d["interview"] = interview
    return d


class FakeBackend:
    """Four 20 s segments. `oceanai_on`: segments (1-based) where OCEAN-AI gives a score; the own model always does."""

    def __init__(self, oceanai_on=(1, 3, 4), primary="oceanai"):
        self.oceanai_on, self.primary, self.calls = set(oceanai_on), primary, 0

    def predict_video(self, video, asr=True, transcript=None):
        self.calls += 1
        i = self.calls
        variants = {"mm": _scores(0.30 + 0.02 * i, interview=0.40 + 0.01 * i)}
        if i in self.oceanai_on:
            variants = {"oceanai": _scores(0.60 + 0.01 * i), **variants}
        primary_used = self.primary if self.primary in variants else None
        if primary_used:
            scores = {k: variants[self.primary][k] for k in TRAIT_KEYS}
        else:        # the ensemble falls back to the mean of what is there (here: the own model alone)
            scores = {k: sum(v[k] for v in variants.values()) / len(variants) for k in TRAIT_KEYS}
        scores["interview"] = variants["mm"]["interview"]
        return {"scores": scores, "variants": variants, "members_used": list(variants), "primary": self.primary,
                "primary_used": primary_used, "behavior_description": ""}


def _run(backend):
    old = (longvideo.video_duration, longvideo.cut_segment)
    longvideo.video_duration = lambda path: 80.0
    longvideo.cut_segment = lambda video, s, e, out: None
    try:
        an = longvideo.LongVideoAnalyzer(backend, lang="ru", device="cpu")
        an.transcribe = lambda video, tmpdir: ("", [])
        with tempfile.TemporaryDirectory() as tmp:
            return an.analyze(Path(tmp) / "input.mp4", Path(tmp) / "segments")
    finally:
        longvideo.video_duration, longvideo.cut_segment = old


def test_main_scores_only_from_main_system_segments():
    res = _run(FakeBackend(oceanai_on=(1, 3, 4)))
    assert res["segments"] == 4 and res["segments_ok"] == 4
    oce = [_scores(0.60 + 0.01 * i) for i in (1, 3, 4)]
    for k in TRAIT_KEYS:
        want = float(np.mean([s[k] for s in oce]))
        assert abs(res["scores"][k] - want) < 1e-9, (k, res["scores"][k], want)
        assert abs(res["scores"][k] - res["variants"]["oceanai"][k]) < 1e-9
        assert abs(res["scores_std"][k] - float(np.std([s[k] for s in oce]))) < 1e-9
    # "interview" comes from the own model on every segment and is not masked
    want_iv = float(np.mean([0.40 + 0.01 * i for i in (1, 2, 3, 4)]))
    assert abs(res["scores"]["interview"] - want_iv) < 1e-9
    assert res["representative_segment"] != 2
    assert res["primary"] == "oceanai"


def test_timeline_keeps_variants_and_primary_used():
    res = _run(FakeBackend(oceanai_on=(1, 3, 4)))
    tl = res["timeline"]
    assert [t["primary_used"] for t in tl] == ["oceanai", None, "oceanai", "oceanai"]
    assert all("mm" in t["variants"] for t in tl)
    assert "oceanai" not in tl[1]["variants"] and "oceanai" in tl[0]["variants"]
    assert tl[1]["scores"] is not None          # the segment itself still carries the fallback numbers


def test_main_system_never_scored_keeps_2_0_behaviour():
    res = _run(FakeBackend(oceanai_on=()))
    mm = [_scores(0.30 + 0.02 * i) for i in (1, 2, 3, 4)]
    for k in TRAIT_KEYS:
        assert abs(res["scores"][k] - float(np.mean([s[k] for s in mm]))) < 1e-9
    assert "oceanai" not in res["variants"]


def test_clean_view_idempotent_on_new_job_shape():
    """A 3.0 job built from this aggregation: clean_view leaves the main scores as they are."""
    from bs3 import scores
    res = _run(FakeBackend(oceanai_on=(1, 3, 4)))
    rep = {"traits": {k: {"score": round(res["scores"][k], 4)} for k in TRAIT_KEYS},
           "model": {"lang": "ru", "primary": "oceanai"},
           "variant_scores": res["variants"], "timeline": res["timeline"],
           "scores_std_across_segments": res["scores_std"]}
    v = scores.clean_view(rep)
    for k in TRAIT_KEYS:
        assert abs(v["traits"][k]["score"] - rep["traits"][k]["score"]) < 1e-4
    assert [t["segment"] for t in v["timeline"] if t["scores"] is None] == [2]
    assert v["view_meta"]["segments_used"] == 3
