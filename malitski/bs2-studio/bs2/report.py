"""JSON report for one video, following the TZ schema (section 6)."""
from __future__ import annotations
import datetime as _dt
import re
from pathlib import Path

from . import __version__
from .norms import RU_NAMES, TRAIT_KEYS, percentile

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
FIV2_REF = "train First Impressions V2 (6000 клипов)"


def fmt_secs(x) -> str:
    """Durations for the interface: '45 с' below a minute, '6 мин 5 с' above, '1 ч 02 мин' above an hour."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if x < 60:
        return f"{x:.0f} с" if x >= 10 else f"{x:.1f} с"
    m, s = divmod(int(round(x)), 60)
    if m < 60:
        return f"{m} мин {s:02d} с"
    h, m = divmod(m, 60)
    return f"{h} ч {m:02d} мин"


def clean_word(w: str) -> str:
    """Attributed tokens carry their punctuation ('that.', '[in]'); strip it for display."""
    return str(w).strip(" .,;:!?\"'()[]{}«»—-").strip()


def seg_label(start, end) -> str:
    """Position of a segment in the video, always m:ss ('0:00–0:20', '6:00–6:20'), so a timeline column never mixes
    two formats."""
    s, e = int(round(float(start))), int(round(float(end)))
    return f"{s // 60}:{s % 60:02d}–{e // 60}:{e % 60:02d}"


_SEC_LABEL = re.compile(r"\[(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*(?:с|s)\]")


def mmss_labels(text) -> str:
    """Segment labels inside behaviour descriptions ('[100–120 с]') in the timeline format ('[1:40–2:00]')."""
    return _SEC_LABEL.sub(lambda m: "[" + seg_label(float(m.group(1).replace(",", ".")),
                                                    float(m.group(2).replace(",", "."))) + "]", str(text or ""))


def build_report(video: str | Path, result: dict, *, backend: str, corpus: str, lang: str,
                 asr_model: str | None, modalities=("audio", "video", "text"),
                 pool_lang: str | None = None, primary: str | None = None) -> dict:
    """`pool_lang`: report percentiles against the pool of processed videos of that language (see pool.py) instead
    of the FIV2 train labels; used when the main score is on a scale other than FIV2 (OCEAN-AI MuPTA for Russian).
    `primary`: the ensemble member that supplied the main score (None = mean of the members)."""
    traits = {}
    for k in TRAIT_KEYS:
        s = result["scores"][k]
        t = {"score": round(s, 4), "name_ru": RU_NAMES[k]}
        if pool_lang:
            from . import pool as _pool
            pct, n = _pool.percentile(k, s, pool_lang)
            t["percentile"], t["percentile_ref"] = pct, _pool.label(pool_lang, n)
        else:
            t["percentile"] = t["percentile_vs_fiv2"] = percentile(k, s)
            t["percentile_ref"] = FIV2_REF
        traits[k] = t
    traits["emotional_stability"]["alias"] = "non-neuroticism"
    extra = {}
    if "interview" in result["scores"]:
        s = result["scores"]["interview"]
        pct = percentile("interview", s)
        extra["interview"] = {"score": round(s, 4), "percentile": pct, "percentile_vs_fiv2": pct,
                              "percentile_ref": FIV2_REF + ", своя модель",
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
        "model": {"name": "bs-bigfive", "version": __version__, "backend": backend, "corpus": corpus,
                  "lang": lang, "asr_model": asr_model, "primary": primary,
                  "trained_on": "FIV2 train" if corpus == "fi" else "MuPTA",
                  "scale": ("MuPTA (OCEAN-AI, русская речь)" if primary == "oceanai" and lang == "ru"
                            else "FIV2 (First Impressions V2)")},
        "timings_sec": {"total": result.get("seconds")},
        "disclaimer": DISCLAIMER,
        "disclaimer_ru": DISCLAIMER_RU,
    }
