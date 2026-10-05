"""Shared formatters of BS Profiler 3.1: Russian noun forms for a count, clock times, segment labels, durations,
attributed words, the FIV2 reference and the FIV2 percentile in words. The page, the PDF, the journal and the texts take them from here and keep no
copies of their own, so the same number is never written two ways.

Standard library only, no bs3 import: every layer of the package may use it (tests/test_layers.py).
"""
from __future__ import annotations

import re


def plural_ru(n, one: str, few: str, many: str) -> str:
    """Russian noun form for a count: plural_ru(21, "отрезок", "отрезка", "отрезков") -> "отрезок". A fractional count
    is rounded first."""
    n = abs(int(round(float(n))))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# texts stored in result.json by other modules say «92 слов в минуту», «31 сегментов»: fix the noun form on display
_COUNT_NOUNS = {"слов": ("слово", "слова", "слов"), "сегментов": ("отрезок", "отрезка", "отрезков"),
                "отрезков": ("отрезок", "отрезка", "отрезков")}
_GENITIVE_BEFORE = {"из", "до", "от", "около", "без", "для", "больше", "меньше", "более", "менее", "свыше"}


def fix_counts(text: str) -> str:
    """«92 слов в минуту» -> «92 слова в минуту», «(31 сегментов)» -> «(31 отрезок)». Only where the count is in the
    nominative/accusative; after «из», «до», «больше» … the genitive plural is correct and stays."""
    def repl(m):
        prev, n, word = m.group(1) or "", int(m.group(2)), m.group(3)
        if prev.strip().lower() in _GENITIVE_BEFORE:
            return m.group(0)
        return f"{prev}{n} {plural_ru(n, *_COUNT_NOUNS[word])}"
    return re.sub(r"(\b\w+\s)?(\d+)\s(слов|сегментов|отрезков)\b", repl, text or "")


def clock(sec, hours=None) -> str:
    """A moment of the video: m:ss («10:52»), or h:mm:ss («1:00:00») from an hour on.

    `hours`: None chooses the format by the value itself; True or False forces one format, so a column or an axis
    never mixes two («0:10:40» next to «1:05:00»). The seconds are rounded, the same everywhere: the end 651.8 is
    «10:52» in the segments table of the page, in the PDF and in the chart hover. None counts as 0."""
    x = float(sec or 0)
    s = int(round(x))
    if hours is None:
        hours = s >= 3600
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if hours else f"{s // 60}:{s % 60:02d}"


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


SEC_LABEL = re.compile(r"\[(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*(?:с|s)\]")


def mmss_labels(text) -> str:
    """Segment labels inside behaviour descriptions ('[100–120 с]') in the timeline format ('[1:40–2:00]')."""
    return SEC_LABEL.sub(lambda m: "[" + seg_label(float(m.group(1).replace(",", ".")),
                                                   float(m.group(2).replace(",", "."))) + "]", str(text or ""))


def fiv2_ref_ru(ref: str) -> str:
    """percentile_ref from result.json (genitive, reads after «относительно») without technical English words. Only
    an older English job carries one (the percentile against First Impressions V2, read-only since 3.1)."""
    r = re.sub(r",\s*(?:своя модель|AMLAI 1\.0)\s*$", "", ref or "")
    return r.replace("train First Impressions V2", "обучающей выборки First Impressions V2").replace(
        "train FIV2", "обучающей выборки FIV2")


def _is_fiv2(ref: str | None) -> bool:
    """A percentile against the First Impressions V2 norms (6000 clips of that dataset). Percentiles against the pool
    of processed videos ("пула …") or a group of them ("ref:…") are never shown (change of 2026-09-26)."""
    r = ref or ""
    return "First Impressions V2" in r or "FIV2" in r


def pct_phrase(pct, ref: str | None) -> tuple[str, bool]:
    """(the FIV2 percentile in words, whether a percentile tick may be drawn): «выше, чем у 72% людей в FIV2»;
    ("", False) for anything that is not a FIV2 percentile (Russian speech shows the score only). The score bars of
    the page and of the PDF word the same number the same way."""
    if pct is None or not _is_fiv2(ref):
        return "", False
    p = max(0.0, min(100.0, float(pct)))
    group = "людей в FIV2"
    if 45 <= p <= 55:
        return f"примерно посередине среди {group}", True
    return (f"выше, чем у {p:.0f}% {group}" if p > 50 else f"ниже, чем у {100 - p:.0f}% {group}"), True
