"""Captions under the key frames of AMLAI 1.0 (change request «подписи под ключевыми кадрами», 2026-09-26).

One place builds what the page and the PDF print under every key frame, so the same frame is never described two
ways:

- the moment of the whole video (m:ss, tenths when two frames share a second);
- a short phrase of the video-language model about that single frame (Qwen-VL, stored in explanation.json);
- the two strongest facial expressions of that frame (the same 7 classes as the rest of the report);
- what the frame did to the score, with a direction, from the signed frame attribution
  (mm/explain.frame_attribution: `signed` = sum(grad * embedding) per frame and output).

The page shows one short line and opens the rest on hover (`title` of the figure); the PDF prints the same caption in
two lines clipped to the width of the frame. Jobs made before this change carry neither `signed` nor the phrases:
they fall back to the wording they had («сильнее всего повлиял на оценку …») and to the expression alone.

The prompt of the frame phrase and its validation live in frame_phrase, which the explanation code imports without
this module: a phrase is kept only if it stays a short list of visible things — no verdicts about character or mood,
no digits, no sentence.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# PHRASE_MAX and PHRASE_TRIES are not used here: they stay readable as frame_captions.* (tests/test_frame_captions.py)
from .frame_phrase import PHRASE_MAX, PHRASE_TRIES, _decoded_index, _frame_index, clean_phrase  # noqa: F401
from .norms import RU_SHORT, TRAIT_KEYS
from .segments import representative

# ---------------------------------------------------------------- trait names in the two cases the captions need
# «повысил оценку экстраверсии» (genitive) on the page, «повысил экстраверсию» (accusative) in the narrow PDF cell
TRAIT_GEN = {
    "openness": "открытости опыту",
    "conscientiousness": "добросовестности",
    "extraversion": "экстраверсии",
    "agreeableness": "доброжелательности",
    "emotional_stability": "эмоциональной стабильности",
}
TRAIT_ACC = {
    "openness": "открытость опыту",
    "conscientiousness": "добросовестность",
    "extraversion": "экстраверсию",
    "agreeableness": "доброжелательность",
    "emotional_stability": "эмоциональную стабильность",
}


# ---------------------------------------------------------------- moments of the key frames
def moments(report: dict, frames: Sequence[str], media: dict | None = None,
            explanation: dict | None = None) -> Tuple[List[Optional[float]], bool]:
    """(seconds of every key frame in the whole video or None, whether captions must show tenths).

    Key frames come from the representative segment of a long video, otherwise from the whole video; the file name
    carries the frame number inside that clip, so the frame rate that turns it into a second is the rate of THAT
    clip (`clip_fps`, written by the explanation). On a variable-frame-rate source it differs from the average rate
    of the whole video by enough to move the printed second; jobs made before this change have no `clip_fps` and
    keep the average rate they were rendered with. Without any frame rate there is no moment at all."""
    tl_all = report.get("timeline") or []
    seg = representative(report, among="all")
    fps = float((explanation or {}).get("clip_fps") or (media or {}).get("fps")
                or (report.get("media") or {}).get("fps") or 0)
    start = float(seg["start"]) if seg else 0.0
    ok = fps > 0 and (seg is not None or not tl_all)
    out: List[Optional[float]] = []
    for p in frames:
        n = _decoded_index(p)
        out.append(start + n / fps if (ok and n is not None) else None)
    secs = [int(t) for t in out if t is not None]
    return out, len(set(secs)) < len(secs)


def moment_text(t: float, tenths: bool) -> str:
    """«2:14», or «2:14,3» when two key frames fall into the same second."""
    if not tenths:
        return f"{int(t) // 60}:{int(t) % 60:02d}"
    d = int(t * 10)
    return f"{d // 600}:{d // 10 % 60:02d},{d % 10}"


# ---------------------------------------------------------------- expressions and the signed effect
def _expr_of(info: dict | None) -> List[Tuple[str, float]]:
    """[(Russian class name, share)] of the frame, strongest first (at most two)."""
    out = []
    for e in (info or {}).get("expressions") or []:
        ru, share = e.get("ru"), e.get("share")
        if ru and share is not None:
            out.append((str(ru), float(share)))
    return out[:2]


def expr_text(expr: Sequence[Tuple[str, float]], n: int = 2) -> str:
    """«радость 62%, нейтрально 21%».

    A second class that rounds to 0% is not named at all (a «грусть 0%» in the tooltip reads as a mistake), and
    when a second one is named the first is held at 99% so the two do not add up to 101%."""
    items = [(ru, int(round(share * 100))) for ru, share in list(expr)[:n]]
    items = [it for i, it in enumerate(items) if i == 0 or it[1] > 0]
    if len(items) > 1 and items[0][1] >= 100:
        items[0] = (items[0][0], 99)
    return ", ".join(f"{ru} {p}%" for ru, p in items)


def _signed_pairs(fr: dict, idx: Optional[int]) -> List[Tuple[str, float]]:
    """[(trait, signed)] of one frame, |signed| descending; empty for a job made before the signed attribution."""
    if idx is None:
        return []
    eff = (fr.get("per_frame_effect") or [])
    if idx < len(eff) and eff[idx]:
        pairs = [(e["output"], float(e["signed"])) for e in eff[idx]
                 if isinstance(e, dict) and e.get("output") in TRAIT_GEN and e.get("signed") is not None]
        if pairs:
            return sorted(pairs, key=lambda kv: -abs(kv[1]))
    per = fr.get("per_output") or {}
    pairs = []
    for k in TRAIT_KEYS:
        arr = (per.get(k) or {}).get("signed")
        if isinstance(arr, list) and idx < len(arr):
            pairs.append((k, float(arr[idx])))
    return sorted(pairs, key=lambda kv: -abs(kv[1]))


def _legacy_pairs(fr: dict, idx: Optional[int]) -> List[Tuple[str, float]]:
    """The same ranking from the magnitudes of an old job (no direction is known there)."""
    if idx is None:
        return []
    per = fr.get("per_output") or {}
    pairs = []
    for k in TRAIT_KEYS:
        arr = (per.get(k) or {}).get("importance")
        if isinstance(arr, list) and idx < len(arr):
            pairs.append((k, float(arr[idx])))
    return sorted(pairs, key=lambda kv: -abs(kv[1]))


SECOND_SHARE = 0.5                  # a second trait is named when its |signed| is at least half of the first
TINY_SHARE = 0.2                    # below this share of the strongest key frame the direction is not printed


def effect_text(pairs: Sequence[Tuple[str, float]], top_all: float, *, signed: bool = True, short: bool = False,
                mini: bool = False) -> str:
    """«повысил оценку экстраверсии и доброжелательности» / «повысил экстраверсию» (short, for the PDF cell) /
    «повысил эм. стаб.» (mini, for the narrowest cell — five frames in a row are only 34.9 mm wide, and the full
    name of four of the five traits does not fit there; the direction is what the caption must not lose).

    `pairs` — the traits of this frame ranked by |signed|, `top_all` — the largest |signed| over all key frames."""
    if not pairs:
        return ""
    k1, v1 = pairs[0]
    one = (lambda k: RU_SHORT[k].lower()) if mini else (lambda k: TRAIT_ACC[k])
    short = short or mini
    if not signed:
        if short:
            return f"повлиял на {one(k1)}"
        rest = ""
        if len(pairs) > 1 and abs(pairs[1][1]) >= SECOND_SHARE * abs(v1):
            rest = f" и {TRAIT_GEN[pairs[1][0]]}"
        return f"сильнее всего повлиял на оценку {TRAIT_GEN[k1]}{rest}"
    if top_all > 0 and abs(v1) < TINY_SHARE * top_all:
        return f"повлиял на {one(k1)}" if short else f"заметно повлиял на оценку {TRAIT_GEN[k1]}"
    verb1 = "повысил" if v1 >= 0 else "понизил"
    if short:
        return f"{verb1} {one(k1)}"
    if len(pairs) > 1 and abs(pairs[1][1]) >= SECOND_SHARE * abs(v1):
        k2, v2 = pairs[1]
        verb2 = "повысил" if v2 >= 0 else "понизил"
        if verb2 == verb1:
            return f"{verb1} оценку {TRAIT_GEN[k1]} и {TRAIT_GEN[k2]}"
        return f"{verb1} оценку {TRAIT_GEN[k1]}, {verb2} — {TRAIT_GEN[k2]}"
    return f"{verb1} оценку {TRAIT_GEN[k1]}"


# ---------------------------------------------------------------- the captions themselves
def build(report: dict, frames: Sequence[str], explanation: dict | None, media: dict | None = None) -> List[dict]:
    """One dict per key frame, in the order of `frames`:

    `caption` — the line under the frame on the page («2:14 · улыбается, смотрит в камеру»);
    `tooltip` — the line the page opens on hover («радость 62%, нейтрально 21% · повысил оценку экстраверсии»);
    `expr_line` / `effect_short` — the pieces the PDF joins into its second line;
    `alt` — the text alternative of the image."""
    fr = ((explanation or {}).get("frames") or {})
    info_by_file = {str(i.get("file")): i for i in (fr.get("key_frame_info") or []) if isinstance(i, dict)}
    times, tenths = moments(report, frames, media, explanation)
    idxs = [_frame_index(p) for p in frames]
    signed_ok = bool(fr.get("per_frame_effect")) or any(
        isinstance(((fr.get("per_output") or {}).get(k) or {}).get("signed"), list) for k in TRAIT_KEYS)
    ranked = [(_signed_pairs(fr, i) if signed_ok else _legacy_pairs(fr, i)) for i in idxs]
    top_all = max((abs(r[0][1]) for r in ranked if r), default=0.0)
    out = []
    total = len(frames)
    for n, (p, t, idx, pairs) in enumerate(zip(frames, times, idxs, ranked), 1):
        info = info_by_file.get(Path(p).name) or {}
        phrase = clean_phrase(info.get("phrase")) if info.get("phrase") else None
        expr = _expr_of(info)
        label = moment_text(t, tenths) if t is not None else f"кадр {n}"
        tail = phrase or (expr[0][0] if expr else "")
        caption = f"{label} · {tail}" if tail else label
        eff_long = effect_text(pairs, top_all, signed=signed_ok)
        eff_short = effect_text(pairs, top_all, signed=signed_ok, short=True)
        eff_mini = effect_text(pairs, top_all, signed=signed_ok, mini=True)
        tooltip = " · ".join(x for x in (expr_text(expr), eff_long) if x)
        alt = (f"Ключевой кадр, момент {label}" if t is not None else f"Ключевой кадр {n} из {total}")
        out.append({"path": str(p), "n": n, "total": total, "moment": t, "label": label, "phrase": phrase,
                    "expr": expr, "expr_line": expr_text(expr, 1), "effect": eff_long, "effect_short": eff_short,
                    "effect_mini": eff_mini, "tail": tail, "caption": caption, "tooltip": tooltip,
                    "alt": alt if not tail else f"{alt}: {tail}"})
    return out


def has_tenths(report: dict, frames: Sequence[str], media: dict | None = None,
               explanation: dict | None = None) -> bool:
    return moments(report, frames, media, explanation)[1]


def any_moment(entries: Iterable[dict]) -> bool:
    return any(e.get("moment") is not None for e in entries)


def note_flags(entries: Iterable[dict]) -> Tuple[bool, bool, bool]:
    """(any frame described in a few words, any expression, any effect) — a note promises only what is there.

    A job made before the captions carries neither a phrase nor the expressions, so its frames are labelled with
    the bare moment and the note must not announce a short description or a hover that shows an expression."""
    es = list(entries)
    return (any(e.get("tail") for e in es), any(e.get("expr") for e in es), any(e.get("effect") for e in es))


NOTE_MEDIA = ("page", "pdf")


def note_what(entries: Iterable[dict], tenths: bool, medium: str) -> str:
    """What the note under the key frames says stands under a frame («под кадром — …»): the moment of the video
    (with the tenths of a second when `tenths`) or, when no frame has a moment, its number; «и коротко то, что на нём
    видно» when some frame is described (note_flags). `medium` "page" spells the unit out, «(минуты:секунды…)»,
    "pdf" abbreviates it, «(мин:с…)»."""
    if medium not in NOTE_MEDIA:
        raise ValueError(f"medium must be one of {NOTE_MEDIA}, not {medium!r}")
    es = list(entries)
    if not any_moment(es):
        what = "его номер"
    elif medium == "page":
        what = "момент ролика (минуты:секунды" + (", после запятой — десятые доли секунды" if tenths else "") + ")"
    else:
        what = "момент ролика (мин:с, после запятой — десятые доли секунды)" if tenths else "момент ролика (мин:с)"
    return what + (" и коротко то, что на нём видно" if note_flags(es)[0] else "")


def pdf_second_line(entry: dict) -> List[str]:
    """Candidates for the second PDF line, widest first; the caller keeps the first one that fits the cell.

    The direction of the effect is the last thing to drop (the owner asked for the sign), so the short trait name
    («повысил эм. стаб.») is offered before giving up on the effect and printing the expression alone."""
    expr, eff = entry.get("expr_line") or "", entry.get("effect_short") or ""
    mini = entry.get("effect_mini") or ""
    out = [x for x in (" · ".join(p for p in (expr, eff) if p), eff,
                       " · ".join(p for p in (expr, mini) if p), mini, expr) if x]
    seen, res = set(), []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res
