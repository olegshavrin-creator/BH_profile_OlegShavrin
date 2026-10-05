"""report.build_report (3.1): the traits and the interview label carry the score only. No FIV2 percentile is computed
any more, whatever the model, the language or the caller (the web pipeline, `bs3 infer`); older jobs keep the
percentiles they carry, and the display reads them (test_view, test_method_notes)."""
from __future__ import annotations

import inspect
from pathlib import Path

from bs3 import norms, report
from bs3.norms import TRAIT_KEYS

PCT_KEYS = {"percentile", "percentile_vs_fiv2", "percentile_ref"}


def _result() -> dict:
    scores = {k: 0.51234 + 0.01 * i for i, k in enumerate(TRAIT_KEYS)}
    scores["interview"] = 0.47891
    return {"scores": scores, "transcript": "", "seconds": 1.0}


def test_no_percentiles_in_traits_or_interview():
    # primary None is the `bs3 infer` path that used to add FIV2 percentiles; the others are the web pipeline's
    for kw in ({}, {"primary": "mm", "selected": "mm"}, {"primary": "oceanai", "selected": "oceanai"}):
        for lang in ("ru", "en"):
            rep = report.build_report("clip.mp4", _result(), backend="mm", corpus="fi", lang=lang, asr_model=None, **kw)
            for k in TRAIT_KEYS:
                assert not PCT_KEYS & set(rep["traits"][k]), (kw, lang, k)
                assert set(rep["traits"][k]) - {"alias"} == {"score", "name_ru"}, (kw, lang, k)
            assert rep["traits"]["openness"]["score"] == 0.5123
            assert rep["traits"]["emotional_stability"]["alias"] == "non-neuroticism"
            assert not PCT_KEYS & set(rep["interview"]), (kw, lang)
            assert rep["interview"]["score"] == 0.4789
            assert set(rep["interview"]) == {"score", "name_ru", "disclaimer"}


def test_no_pool_lang_and_no_norms():
    assert "pool_lang" not in inspect.signature(report.build_report).parameters
    assert not hasattr(report, "FIV2_REF") and not hasattr(report, "percentile")
    assert not any(hasattr(norms, n) for n in ("percentile", "load_norms", "_norms"))
    assert not (Path(norms.__file__).parent / "fiv2_norms.json").exists()
