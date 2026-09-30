"""JSON report for one video, following the TZ schema (section 6). The formatters that used to live here (fmt_secs,
seg_label, mmss_labels, clean_word) are in textfmt.py."""
from __future__ import annotations
import datetime as _dt
from pathlib import Path

from . import MODEL_TITLES, PRODUCT, __version__
from .norms import RU_NAMES, TRAIT_KEYS

DISCLAIMER = ("Apparent personality as perceived by observers (First Impressions V2 style); "
              "probabilistic research estimate, not a psychological assessment or diagnosis.")
INTERVIEW_DISCLAIMER = ("ChaLearn job-interview variable: how often crowd annotators said they would invite the person "
                        "to a job interview after a 15-second first impression. Not a hiring recommendation; "
                        "the dataset authors ask not to use it for decisions affecting people.")


DISCLAIMER_RU = ("Кажущаяся личность, как её воспринимают наблюдатели по первому впечатлению (в духе First Impressions V2); "
                 "вероятностная исследовательская оценка, не психологическая диагностика.")
INTERVIEW_DISCLAIMER_RU = ("Метка «собеседование» ChaLearn: как часто разметчики говорили, что пригласили бы человека на "
                           "собеседование после 15-секундного первого впечатления. Не рекомендация о найме; авторы датасета "
                           "просят не использовать её для решений, влияющих на людей.")


def build_report(video: str | Path, result: dict, *, backend: str, corpus: str, lang: str,
                 asr_model: str | None, modalities=("audio", "video", "text"),
                 primary: str | None = None, selected: str | None = None) -> dict:
    """The traits and the interview label get the score only: no percentile against the FIV2 norms or a group of
    processed videos (change of 2026-09-26; older jobs carry percentiles, which the display reads). `primary`: the
    ensemble member that supplied the main score (None = mean of the members). `selected`: the model the user chose
    for this analysis (BS Profiler 3.1: "oceanai" | "mm"; the only one that ran): recorded as `model.selected` with its
    title in `model.selected_title`, and it is the `primary` unless told otherwise."""
    selected = selected or (primary if primary in MODEL_TITLES else None)
    primary = primary or selected
    own_scale = selected == "oceanai" and lang == "ru"          # MuPTA weights; the own model lives on the FIV2 scale
    traits = {}
    for k in TRAIT_KEYS:
        s = result["scores"][k]
        traits[k] = {"score": round(s, 4), "name_ru": RU_NAMES[k]}
    traits["emotional_stability"]["alias"] = "non-neuroticism"
    extra = {}
    if "interview" in result["scores"]:
        s = result["scores"]["interview"]
        # the label of the own model: like the traits, the score only
        extra["interview"] = {"score": round(s, 4),
                              "name_ru": "впечатление «пригласить на собеседование»", "disclaimer": INTERVIEW_DISCLAIMER}
    if result.get("behavior_description"):
        extra["behavior_description"] = result["behavior_description"]
    return {
        "input": str(video),
        "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "traits": traits,
        **extra,
        "transcript": result.get("transcript", ""),
        "modalities_used": list(modalities),
        "model": {"name": "bs-bigfive", "version": __version__, "product": PRODUCT, "backend": backend, "corpus": corpus,
                  "lang": lang, "asr_model": asr_model, "primary": primary,
                  "selected": selected, "selected_title": MODEL_TITLES.get(selected) if selected else None,
                  "trained_on": ("MuPTA" if own_scale else "FIV2 train") if selected else (
                      "FIV2 train" if corpus == "fi" else "MuPTA"),
                  "scale": ("MuPTA (OCEAN-AI, русская речь)" if primary == "oceanai" and lang == "ru"
                            else "FIV2 (First Impressions V2)")},
        "timings_sec": {"total": result.get("seconds")},
        "disclaimer": DISCLAIMER,
        "disclaimer_ru": DISCLAIMER_RU,
    }
