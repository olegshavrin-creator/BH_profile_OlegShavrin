"""Per-segment data of a result.json or of its clean view (scores.clean_view), shared by the web charts, the PDF and the
texts: the scored segments, the representative one, the rows of the table «Значения по отрезкам», the emotions of one
segment and the behaviour descriptions by segment. Only data: no page, no PDF, no model (numpy inside odd_segments).

The representative segment is the one closest to the mean profile; the explanations and the key frames are built on
it. Each place that shows it keeps its own rule (`representative`): the web chart of the scores marks it among the
scored segments, the PDF only when at least two segments are scored, and the key frames look it up in the whole
timeline.

A data module of the import layers (tests/test_layers.py): it imports only leaves.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from .labels import HEAT_ROWS
from .norms import TRAIT_KEYS
from .palette import EMO_ALIAS
from .textfmt import SEC_LABEL

AMONG = ("scored", "all")


def as_float(v) -> float:
    """A stored number as a float; NaN when it is missing or no number (a chart breaks its line there)."""
    try:
        return math.nan if v is None else float(v)
    except (TypeError, ValueError):
        return math.nan


def scored(view: dict) -> List[dict]:
    """The timeline entries that carry scores, in time order (a skipped segment, or one without a score of the
    model of a clean view, has none)."""
    return [t for t in (view.get("timeline") or []) if t.get("scores")]


def representative(view: dict, among: str = "scored", min_scored: int = 0) -> Optional[dict]:
    """The timeline entry of the representative segment (`representative_segment`, numbered from 1), or None.
    among="scored": looked up among the scored segments; among="all": in the whole timeline, a segment without a
    score included. min_scored: None unless at least that many segments are scored."""
    if among not in AMONG:
        raise ValueError(f"among must be one of {AMONG}, not {among!r}")
    rep_i = view.get("representative_segment")
    segs = scored(view)
    if rep_i is None or len(segs) < min_scored:
        return None
    return next((t for t in (segs if among == "scored" else view.get("timeline") or [])
                 if t.get("segment") == rep_i), None)


def empty_text(r: dict) -> bool:
    """The segment's own transcript is empty: the text-emotion model then answers "neutral 100%", which is no data."""
    return "text_en" in r and not str(r.get("text_en") or "").strip()


def seg_words(r: dict) -> int:
    try:
        return int((r.get("speech") or {}).get("words") or 0)
    except (TypeError, ValueError):
        return 0


def emotion_shares(r: dict, source: str) -> Optional[Dict[str, float]]:
    """{emotion: share} of one segment rescaled to sum 1 (rounded model outputs leave 0.99…), or None = no data. Face
    labels are mapped to the text keys (happy -> joy …); an empty transcript is no data, not «neutral 100%»."""
    d = ((r.get("face") or {}).get("expressions") or {}) if source == "face" else ({} if empty_text(r) else
                                                                                   (r.get("emotions_text") or {}))
    vals = {k: 0.0 for k in HEAT_ROWS}
    for k, v in d.items():
        k, x = EMO_ALIAS.get(k, k), as_float(v)
        if k in vals and not math.isnan(x):
            vals[k] += max(0.0, x)
    total = sum(vals.values())
    if not d or total <= 1e-6:
        return None
    return {k: v / total for k, v in vals.items()}


def dominant_emotion(r: dict, source: str) -> Optional[Tuple[str, float]]:
    """(emotion key, raw share) of the largest raw value, the way the per-segment table picks it; None = no data."""
    d = ((r.get("face") or {}).get("expressions") or {}) if source == "face" else ({} if empty_text(r) else
                                                                                   (r.get("emotions_text") or {}))
    items = [(k, as_float(v)) for k, v in d.items() if not math.isnan(as_float(v))]
    if not items:
        return None
    k, v = max(items, key=lambda kv: kv[1])
    return EMO_ALIAS.get(k, k), v


def segment_rows(report: dict) -> list:
    """[(start, end, timeline entry or None, per_segment entry or None)] in time order, matched by the start."""
    by: dict = {}                       # rounded start -> [timeline entry, per_segment entry, start, end]
    for t in report.get("timeline") or []:
        by.setdefault(int(round(float(t["start"]))), [None, None, float(t["start"]), float(t["end"])])[0] = t
    for r in (report.get("analyses") or {}).get("per_segment") or []:
        by.setdefault(int(round(float(r["start"]))), [None, None, float(r["start"]), float(r["end"])])[1] = r
    return [(s, e, t, r) for _, (t, r, s, e) in sorted(by.items())]


def behavior_by_segment(report: dict) -> list:
    """[(start, end, text)] of behavior_description_ru split on its «[0–20 с]» labels; [] when it has none."""
    text = str(report.get("behavior_description_ru") or "")
    ms = list(SEC_LABEL.finditer(text))
    out = []
    for i, m in enumerate(ms):
        s, e = float(m.group(1).replace(",", ".")), float(m.group(2).replace(",", "."))
        body = text[m.end(): ms[i + 1].start() if i + 1 < len(ms) else len(text)].strip()
        out.append((s, e, body))
    return out


def odd_segments(rep: dict) -> List[tuple]:
    """[(timeline entry, z)] of the segments whose mean of the five scores lies more than 2 standard deviations from
    the other segments (at least 4 scored segments), in time order; z > 0 — the scores are higher."""
    import numpy as np
    tl = scored(rep)
    means = np.array([np.mean([t["scores"][k] for k in TRAIT_KEYS]) for t in tl])
    if len(means) < 4 or means.std() <= 0:
        return []
    z = (means - means.mean()) / means.std()
    return [(t, float(z_)) for t, z_ in zip(tl, z) if abs(z_) > 2.0]
