"""One layer of clean scores for everything BS Profiler 3.1 shows (design 6.1, 5.2; changes of 2026-09-26).

`clean_view(rep)` returns a deep copy of a result.json in the same shape (the original is not changed and nothing is
written), in which
- the view holds ONE model (3.1): the one recorded in result.json (`model.selected` of a 3.1 job, `model.primary` of a
  3.0 or an imported 2.0 job — OCEAN-AI), see `main_system`; the Big Five scores are that model's own
  (`variant_scores[main]`), never a mix of two scales, and `variant_scores` (whole video and per segment) keeps that
  member only, so an older job that carried two members shows one; the label «собеседование» (`interview`, whole
  video and per segment) exists only for AMLAI 1.0 and leaves a view of OCEAN-AI with the other member;
- a segment where the model gave no score has `scores = None` and `no_primary = True` (charts already draw such a
  segment as a gap «нет оценки»), and the spread over segments is recomputed on the remaining ones;
- Russian speech shows the scores themselves only: the percentiles against the pool of processed videos that older
  jobs carry are removed from the view (no group of processed videos is compared with anywhere); an older English job
  keeps the percentile against the First Impressions V2 norms (6000 clips of that dataset, not our videos);
- `view_meta` says which model the view shows (`main_system`, `selected`, `selected_title`) and which segments were
  left out.

Levels and MBTI letters use the absolute score of the system on [0, 1] (the customer's formula: threshold 0.5,
borderline zone |v − 0.5| < 0.15). The five level bands are aligned with that zone, so a value with a confident
letter is never «средний уровень» and a value in the borderline zone always is. The report prints a score with two
decimals, and the bands and letters are decided on exactly that printed value (`shown`): the view keeps the scores
unrounded and they are rounded once, here, so a printed 0.65 never gets two readings and the same score never prints
two ways:
d = v − 0.5: d >= 0.30 high, 0.15 <= d < 0.30 above, |d| < 0.15 mid, −0.30 < d <= −0.15 below, d <= −0.30 low
(v >= 0.80 high, 0.65 <= v < 0.80 above, 0.35 < v < 0.65 mid, 0.20 < v <= 0.35 below, v <= 0.20 low).

The cards of «Ключевые факты» paint their value by the same bands, in three states instead of five: `scale_state`
(neutral / below / above for a 0…1 measurement), `tempo_state` (words per minute, TEMPO_BAND) and `emotion_state`
(the dominant emotion or facial expression, EMO_STATE). The colours themselves live in palette.FACT_VALUE.
"""
from __future__ import annotations

import copy
import math
import statistics

from . import MODEL_TITLES
from .labels import SOURCE_OF
from .norms import RU_NAMES, TRAIT_KEYS

# the band edges, in one place: distance of the score from the middle of the scale 0.5
MIDDLE = 0.5
MID_HALF_WIDTH = 0.15          # = the MBTI borderline zone (config/mbti.json "borderline")
HIGH_DISTANCE = 0.30           # v >= 0.80 high, v <= 0.20 low
LEVELS = ("high", "above", "mid", "below", "low")
LEVELS_RU = {"high": "высокий уровень", "above": "выше среднего", "mid": "средний уровень",
             "below": "ниже среднего", "low": "низкий уровень"}
RELATIVE_KEYS = ("percentile", "percentile_ref", "percentile_vs_fiv2", "position")
FALLBACK_ORDER = ("oceanai", "mm")
# how a report that names no model at all is read (an old CLI report, a report of a backend the page does not offer).
# It is deliberately NOT bs3.DEFAULT_MODEL: the default model is what a new analysis starts with, while this is how an
# old file is attributed, and a report made before 3.1 came from OCEAN-AI. Reading such a file as AMLAI 1.0 would put
# «AMLAI 1.0» and its modalities (face, audio, text, behavior) on a page that never ran it.
READ_FALLBACK = "oceanai"

# Where a measured value of «Ключевые факты» sits, in the three words the card colour says (palette.FACT_VALUE):
# around neutral, below it, above it. The 0…1 scales are read with the bands above, so the page never calls a score
# «средний уровень» in one place and paints it «выше» in another; the speech tempo has a band of its own.
NEUTRAL, BELOW, ABOVE = "neutral", "below", "above"
FACT_STATES = (NEUTRAL, BELOW, ABOVE)
TEMPO_BAND = (100, 160)        # words per minute: the usual range of conversational Russian speech, ends included;
                               # the one band of the tempo, for the card colour and for the words (tempo_state)
# emotions and facial expressions: «нейтрально» is the neutral state, the quiet emotions read as below it and the
# loud ones as above it (the face labels share palette.EMO_ALIAS with the text emotions)
EMO_STATE = {"neutral": NEUTRAL, "sadness": BELOW, "fear": BELOW, "disgust": BELOW,
             "joy": ABOVE, "surprise": ABOVE, "anger": ABOVE}

__all__ = ["clean_view", "segment_ok", "level", "level_phrase", "score_text", "shown", "LEVELS_RU",
           "main_system", "recorded_model", "shown_model", "has_explanations", "data_json", "scale_state",
           "tempo_state", "emotion_state", "FACT_STATES", "TEMPO_BAND", "NEUTRAL", "BELOW", "ABOVE", "READ_FALLBACK",
           "scored", "num"]


def num(x) -> float | None:
    """A finite number from a stored value, else None (None, a bool, text that is no number, NaN, infinity)."""
    if x is None or isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _has_scores(d) -> bool:
    return isinstance(d, dict) and any(num(d.get(k)) is not None for k in TRAIT_KEYS)


def scored(entry) -> dict | None:
    """`entry` (the label «собеседование» of a report, `interview`) when it carries a numeric score, else None: an
    entry without one counts as absent wherever it would be shown (the bars, the key facts, the PDF)."""
    return entry if isinstance(entry, dict) and num(entry.get("score")) is not None else None


def recorded_model(rep: dict) -> str | None:
    """The model result.json names: `model.selected` (3.1), else `model.primary` (3.0 and imported 2.0 jobs:
    OCEAN-AI), else a single-model backend of a CLI report; None when the job names none."""
    model = rep.get("model") or {}
    for key in ("selected", "primary", "backend"):
        if model.get(key) in MODEL_TITLES:
            return model[key]
    return None


def main_system(rep: dict) -> tuple[str, bool]:
    """(the model the view shows, primary_missing): the recorded model (`recorded_model`) when it produced scores, or
    when the job has no per-member scores at all (a single-model report keeps its traits); otherwise the first of
    oceanai, mm that did, with primary_missing = True. A job that names no model is read as OCEAN-AI (READ_FALLBACK:
    3.1 shows one model; there is no mean of two systems any more)."""
    recorded = recorded_model(rep)
    var = rep.get("variant_scores") or {}
    if recorded and (_has_scores(var.get(recorded)) or not any(_has_scores(v) for v in var.values())):
        return recorded, False
    for s in FALLBACK_ORDER:
        if _has_scores(var.get(s)):
            return s, recorded is not None
    return recorded or READ_FALLBACK, recorded is not None


def shown_model(rep: dict) -> str | None:
    """The model a report is shown as ("oceanai" | "mm"), for its title and for what it shows: the model of a clean
    view (`view_meta.main_system`, see main_system), else the model a raw result.json names (recorded_model); None
    when a raw result.json names none."""
    return (rep.get("view_meta") or {}).get("main_system") or recorded_model(rep)


def has_explanations(view: dict) -> bool:
    """Does the report show explanations (key frames, modality shares, words, the behaviour description)? They exist
    for AMLAI 1.0 only (3.1): an OCEAN-AI job shows none, also an older one that carries the explanations of the
    second model of 3.0 (the page and the PDF show one model)."""
    return shown_model(view) == "mm"


def segment_ok(rep: dict, t: dict) -> bool:
    """Did the shown model score this segment? Jobs of 3.x say so in `primary_used`; older ones in `members_used`."""
    main, _ = main_system(rep)
    if "primary_used" in t and main == (rep.get("model") or {}).get("primary"):
        return t.get("primary_used") is not None
    variants = t.get("variants")
    if isinstance(variants, dict) and main in variants:
        return _has_scores(variants.get(main))
    members = t.get("members_used")
    if members is None and variants is None:
        return isinstance(t.get("scores"), dict)      # a job without per-member records: its scores are the model's
    return main in (members or [])


def shown(v) -> float | None:
    """The score as the report prints it and decides on it: rounded once to two decimals (None for a missing one)."""
    x = num(v)
    return None if x is None else round(x, 2)


def level(v) -> str | None:
    """Level band of a score v on [0, 1], decided on its printed value `shown(v)` (None for a missing score); see
    the module docstring."""
    x = shown(v)
    if x is None:
        return None
    d = round(x - MIDDLE, 9)             # as the borderline test of mbti: 0.65 − 0.5 is 0.15, not 0.15000000000000002
    if d >= HIGH_DISTANCE:
        return "high"
    if d >= MID_HALF_WIDTH:
        return "above"
    if d > -MID_HALF_WIDTH:
        return "mid"
    if d > -HIGH_DISTANCE:
        return "below"
    return "low"


def level_phrase(v) -> str | None:
    """«высокий уровень» / «выше среднего» / «средний уровень» / «ниже среднего» / «низкий уровень»."""
    lv = level(v)
    return LEVELS_RU[lv] if lv else None


def scale_state(v) -> str | None:
    """Where a measurement on [0, 1] sits, for the colour of a key-fact value: the level band of `v` read as one of
    three states (mid -> neutral, below/low -> below, above/high -> above); None for a missing value. With the bands
    of this module that is 0.36…0.64 neutral, <= 0.35 below, >= 0.65 above on the printed two decimals."""
    lv = level(v)
    if lv is None:
        return None
    return NEUTRAL if lv == "mid" else (BELOW if lv in ("below", "low") else ABOVE)


def tempo_state(wpm) -> str | None:
    """Speech tempo, decided on the whole words per minute the card prints: TEMPO_BAND is neutral, slower is below,
    faster is above; None when there is no tempo. The one classifier of the tempo (owner, 2026-09-27): the colour of
    the card «Темп речи» (facts.key_facts) and the words «медленный / спокойный / быстрый» of the sentence about the
    manner of speech (analyses_text.tempo_clause) and of the characterization all follow it."""
    x = num(wpm)
    if x is None:
        return None
    n = int(round(x))
    lo, hi = TEMPO_BAND
    return BELOW if n < lo else (ABOVE if n > hi else NEUTRAL)


def emotion_state(key: str) -> str | None:
    """State of a dominant emotion or facial expression (EMO_STATE); None for a label we do not place."""
    from .palette import EMO_ALIAS
    return EMO_STATE.get(EMO_ALIAS.get(key, key))


def score_text(v) -> str | None:
    """The score as the page prints it: two decimals («0.73»), the same number as `shown(v)`."""
    x = shown(v)
    return None if x is None else f"{x:.2f}"


def clean_view(rep: dict) -> dict:
    """Deep copy of `rep` with clean main scores and gaps for segments without the main system (see the module
    docstring). Idempotent: clean_view(clean_view(rep)) == clean_view(rep)."""
    view = copy.deepcopy(rep)
    model = view.get("model") or {}
    lang = "ru" if model.get("lang", "ru") == "ru" else "en"
    main, primary_missing = main_system(view)
    var = view.get("variant_scores") or {}
    traits = view.get("traits")
    if not isinstance(traits, dict):
        traits = view["traits"] = {}

    # 1-2. the scores = the shown model's own means
    if _has_scores(var.get(main)):
        for k in TRAIT_KEYS:
            v = num(var[main].get(k))
            if v is None:
                continue
            t = traits.setdefault(k, {"name_ru": RU_NAMES[k]})
            t["score"] = v                  # unrounded: rounded once, at display (shown)

    # 3. segments without the shown model: gaps (decided before the other members are dropped from the segments)
    timeline = view.get("timeline") or []
    dropped = []
    for t in timeline:
        if not isinstance(t, dict):
            continue
        if not segment_ok(view, t):
            dropped.append(t.get("segment"))
            t["scores"] = None
            t["no_primary"] = True
    kept = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]

    # 3a. one model in the view (3.1): the other members an older job carried are left out, whole video and segments;
    # the label «собеседование» exists only for AMLAI 1.0, so a view of OCEAN-AI drops it too (a job imported from
    # 2.0 / 3.0 carries it from the second model), whole video and per segment
    if isinstance(view.get("variant_scores"), dict):
        view["variant_scores"] = {m: v for m, v in view["variant_scores"].items() if m == main}
    own_label = main == "mm"
    if not own_label:
        view.pop("interview", None)
    for t in timeline:
        if not isinstance(t, dict):
            continue
        if isinstance(t.get("variants"), dict):
            t["variants"] = {m: v for m, v in t["variants"].items() if m == main}
        if not own_label:
            if isinstance(t.get("scores"), dict):
                t["scores"].pop("interview", None)
            for v in (t.get("variants") or {}).values():
                if isinstance(v, dict):
                    v.pop("interview", None)

    # 4. spread over the remaining segments (population SD, as longvideo computes it)
    if dropped:
        std = view.get("scores_std_across_segments")
        keys = list(std) if isinstance(std, dict) and std else list(TRAIT_KEYS)
        new = {}
        for k in keys:
            vals = [x for x in (num(t["scores"].get(k)) for t in kept) if x is not None]
            new[k] = statistics.pstdev(vals) if vals else 0.0
        view["scores_std_across_segments"] = new

    # 5. Russian speech: the scores only — no percentile against the pool of processed videos (older jobs carry one)
    if lang == "ru":
        items = [traits.get(k) for k in TRAIT_KEYS] + [view.get("interview")]
        for t in items:
            if isinstance(t, dict):
                for key in RELATIVE_KEYS:
                    t.pop(key, None)
    else:
        for k in TRAIT_KEYS:
            if isinstance(traits.get(k), dict):
                traits[k].pop("position", None)

    # 6. what the view is made of
    used = [t for t in timeline if isinstance(t, dict) and isinstance(t.get("scores"), dict)]
    view["view_meta"] = {
        "main_system": main, "main_source": SOURCE_OF.get(main, main), "lang": lang,
        "selected": main, "selected_title": MODEL_TITLES.get(main, main),
        "segments_total": len(timeline), "segments_used": len(used),
        "segments_without_primary": dropped, "primary_missing": bool(primary_missing),
    }
    return view


LEGACY_KEYS = ("narrative",)         # the 2.0 plain-language summary; 3.x shows «Как получены оценки» instead


def data_json(rep: dict) -> tuple[dict, bool]:
    """(what the tab «Данные» shows, whether anything was left out): a deep copy of result.json; for Russian speech
    without the percentile keys of step 5 of clean_view (older jobs carry percentiles against the pool of processed
    videos) and without the stored 2.0 `narrative` built from them. The file on disk is not changed."""
    data = copy.deepcopy(rep)
    if (data.get("model") or {}).get("lang") != "ru":
        return data, False
    dropped = False
    traits = data.get("traits") if isinstance(data.get("traits"), dict) else {}
    for t in [traits.get(k) for k in TRAIT_KEYS] + [data.get("interview")]:
        if isinstance(t, dict):
            for key in RELATIVE_KEYS:
                if key in t:
                    t.pop(key)
                    dropped = True
    for key in LEGACY_KEYS:
        if key in data:
            data.pop(key)
            dropped = True
    return data, dropped
