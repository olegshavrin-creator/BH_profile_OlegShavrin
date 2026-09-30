"""bs3/textfmt.py: the shared formatters give exactly what the copies they replaced gave (refactoring plan of 3.1,
stage 9), and the copies are gone.

The old outputs of the four clock copies (mbti_html._mmss, pdf_mbti._mmss, journal._mmss, charts._clock), of
page._clock and of the two h:mm:ss copies in pdf_report._seg and the PDF chart axis were recorded over 0…12000 s in
steps of 0.1 s before the copies were deleted; `clock` reproduced every one of them. The sample points below come from
that table; OLD keeps the old bodies, and every run compares them with `clock` again from 0 to 12000 s. The one copy
that cut the seconds off, page._clock of the page's segments table, rounds since stage 14a like the PDF and the
chart hover (owner, 2026-09-27), so `clock` has no truncating mode any more.
"""
from __future__ import annotations

import inspect

from bs3 import (analyses_text, caveats, characterization, facts, journal, mbti, narrative, report,
                 scores, segments, textfmt)
from bs3.web import charts, mbti_html, page, parts as webparts
from bs3.pdf import appendix as pdf_appendix
from bs3.pdf import build as pdf_build
from bs3.pdf import charts as pdf_charts
from bs3.pdf import document as pdf_document
from bs3.pdf import fmt as pdf_fmt
from bs3.pdf import frames as pdf_frames
from bs3.pdf import mbti_section as pdf_mbti
from bs3.pdf import sections as pdf_sections
from bs3.pdf import widgets as pdf_widgets
from bs3.textfmt import (clean_word, clock, fiv2_ref_ru, fix_counts, fmt_secs, mmss_labels, pct_phrase, plural_ru,
                         seg_label)


def _old_auto(sec) -> str:            # mbti_html._mmss = pdf_mbti._mmss = journal._mmss (charts._clock without None)
    s = int(round(float(sec or 0)))
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60}:{s % 60:02d}"


def _old_hours(t) -> str:              # pdf_report._seg from an hour on, and the PDF chart axis from an hour on
    t = int(round(float(t)))
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}"


def _old_mmss(v) -> str:               # the PDF chart axis below an hour
    return f"{int(round(v)) // 60}:{int(round(v)) % 60:02d}"


OLD = {
    "auto": (_old_auto, lambda v: clock(v)),
    "PDF, h:mm:ss": (_old_hours, lambda v: clock(v, hours=True)),
    "PDF axis, m:ss": (_old_mmss, lambda v: clock(v, hours=False)),
}


def test_plural_ru_follows_the_russian_rule():
    for n in range(2000):
        if n % 10 == 1 and n % 100 != 11:
            want = "one"
        elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
            want = "few"
        else:
            want = "many"
        assert plural_ru(n, "one", "few", "many") == want, n
        assert plural_ru(-n, "one", "few", "many") == want, -n
    assert [plural_ru(n, "a", "b", "c") for n in (1, 11, 21, 111, 2, 12, 22, 5, 0)] == list("acacbcbcc")
    # a fractional count is rounded (the key facts pass a rounded tempo): 21.2 -> 21, 1.6 -> 2, 0.5 -> 0
    assert [plural_ru(x, "a", "b", "c") for x in (21.2, 1.6, 0.5, 1.5)] == ["a", "b", "c", "b"]
    assert plural_ru(3, "отрезок", "отрезка", "отрезков") == "отрезка"


def test_clock_keeps_every_old_call_pattern():
    auto = {0: "0:00", 0.4: "0:00", 0.5: "0:00", 1.5: "0:02", 2.5: "0:02", 59.4: "0:59", 59.5: "1:00", 60: "1:00",
            651.8: "10:52", 3599.4: "59:59", 3599.5: "1:00:00", 3600: "1:00:00", 3661.4: "1:01:01",
            7384.6: "2:03:05", 12000: "3:20:00"}
    for v, want in auto.items():
        assert clock(v) == want, v
    assert clock(None) == "0:00"                                    # the MBTI strip of a segment without a start
    # the page's segments table keeps one format for the whole column and rounds like the PDF (stage 14a: the end
    # 651.8 was «10:51» there and «10:52» in the PDF)
    table_mmss = {59.9: "1:00", 651.8: "10:52", 3599.9: "60:00", 3600: "60:00", 7384.6: "123:05"}
    for v, want in table_mmss.items():
        assert clock(v, hours=False) == want, v
    table_hours = {651.8: "0:10:52", 3661.4: "1:01:01", 7384.6: "2:03:05"}
    for v, want in table_hours.items():
        assert clock(v, hours=True) == want, v
    assert "truncate" not in inspect.signature(clock).parameters
    # the PDF from an hour on (segment labels, chart axis) rounds and always writes hours
    for v, want in {59.5: "0:01:00", 651.8: "0:10:52", 3599.5: "1:00:00", 7384.6: "2:03:05"}.items():
        assert clock(v, hours=True) == want, v
    assert clock(3600, hours=False) == "60:00"


def test_clock_equals_the_old_copies_over_three_hours():
    values = [i / 10 for i in range(0, 120001, 7)] + [3599.5, 3600, 3661.4]
    for name, (old, new) in OLD.items():
        bad = [v for v in values if old(v) != new(v)]
        assert not bad, (name, bad[:5])


def test_moved_formatters_keep_their_output():
    assert [fmt_secs(x) for x in (0, 5, 9.96, 10, 45.4, 59.6, 60, 61, 365, 3599.4, 3600, 3725, 7384, None, "x")] == [
        "0.0 с", "5.0 с", "10.0 с", "10 с", "45 с", "60 с", "1 мин 00 с", "1 мин 01 с", "6 мин 05 с", "59 мин 59 с",
        "1 ч 00 мин", "1 ч 02 мин", "2 ч 03 мин", "None", "x"]
    assert [seg_label(a, b) for a, b in ((0, 20), (19.5, 40.49), (640.0, 651.8), (3590, 3605.5))] == [
        "0:00–0:20", "0:20–0:40", "10:40–10:52", "59:50–60:06"]
    assert mmss_labels("[100–120 с] и [0-20s] и [3590,5–3601 с]") == "[1:40–2:00] и [0:00–0:20] и [59:50–60:01]"
    assert mmss_labels(None) == "" and mmss_labels("нет меток") == "нет меток"
    assert [clean_word(w) for w in ("that.", "[in]", "«слово»", " — ", "hello!?")] == ["that", "in", "слово", "",
                                                                                         "hello"]
    assert [fiv2_ref_ru(r) for r in ("train First Impressions V2", "train FIV2, своя модель", "train FIV2, AMLAI 1.0",
                                     "", None, "FIV2 test")] == [
        "обучающей выборки First Impressions V2", "обучающей выборки FIV2", "обучающей выборки FIV2", "", "",
        "FIV2 test"]
    assert [fix_counts(t) for t in ("92 слов в минуту", "(31 сегментов)", "из 31 сегментов", "до 5 отрезков",
                                    "21 слов", "больше 2 слов", "11 слов и 12 отрезков и 22 сегментов", None)] == [
        "92 слова в минуту", "(31 отрезок)", "из 31 сегментов", "до 5 отрезков", "21 слово", "больше 2 слов",
        "11 слов и 12 отрезков и 22 отрезка", ""]


def test_pct_phrase_words_only_a_fiv2_percentile():
    """The wording of the score bars of the page (webparts._pct_phrase) and of the PDF (pdf_report.pct_phrase, which
    took the first element) before stage 10; both now call textfmt.pct_phrase."""
    fiv2 = "train FIV2"
    assert [pct_phrase(p, fiv2) for p in (0, 30, 44.9, 45, 50, 55, 55.1, 72, 100, 120, -3, "61.5")] == [
        ("ниже, чем у 100% людей в FIV2", True), ("ниже, чем у 70% людей в FIV2", True),
        ("ниже, чем у 55% людей в FIV2", True), ("примерно посередине среди людей в FIV2", True),
        ("примерно посередине среди людей в FIV2", True), ("примерно посередине среди людей в FIV2", True),
        ("выше, чем у 55% людей в FIV2", True), ("выше, чем у 72% людей в FIV2", True),
        ("выше, чем у 100% людей в FIV2", True), ("выше, чем у 100% людей в FIV2", True),
        ("ниже, чем у 100% людей в FIV2", True), ("выше, чем у 62% людей в FIV2", True)]
    assert pct_phrase(72, "train First Impressions V2 (6000 клипов)") == ("выше, чем у 72% людей в FIV2", True)
    for ref in ("ref:ru_prov_2026-09-25", "пула обработанных русских роликов (N=5)", "", None):
        assert pct_phrase(72, ref) == ("", False)
    assert pct_phrase(None, fiv2) == ("", False)


def test_the_copies_are_gone():
    """Each module either takes the formatter from textfmt or does not have the name at all."""
    shared = {"plural_ru": plural_ru, "fix_counts": fix_counts, "fmt_secs": fmt_secs, "seg_label": seg_label,
              "mmss_labels": mmss_labels, "clean_word": clean_word, "clock": clock, "fiv2_ref_ru": fiv2_ref_ru,
              "pct_phrase": pct_phrase}
    for mod in (analyses_text, caveats, characterization, charts, facts, journal, mbti, mbti_html, narrative,
                pdf_charts, pdf_document, pdf_fmt, pdf_frames, pdf_mbti, pdf_build, pdf_appendix, pdf_sections,
                pdf_widgets, scores, segments, page, webparts):
        for name, fn in shared.items():
            assert getattr(mod, name, fn) is fn, f"{mod.__name__}.{name} is a copy"
    old = {caveats: "_plural", characterization: "_plural", mbti_html: "_mmss", pdf_mbti: "_mmss", journal: "_mmss",
           charts: "_clock", page: "_clock", webparts: "_ref_ru", pdf_build: "_ref_ru", pdf_appendix: "_ref_ru"}
    for mod, name in old.items():
        assert not hasattr(mod, name), f"{mod.__name__}.{name}"
    assert not hasattr(webparts, "_pct_phrase") and not hasattr(webparts, "_is_fiv2")
    assert pdf_widgets.pct_phrase is pct_phrase            # the score bars of the PDF (pdf/widgets.py, stage 15)
    assert pdf_fmt.clock is clock and pdf_fmt.seg_label is seg_label       # pdf_report._seg, now pdf/fmt.py
    # the passport and the sections of the main part (pdf/sections.py, stage 16) take the formatters from textfmt
    assert (pdf_sections.fmt_secs, pdf_sections.plural_ru, pdf_sections.fix_counts) == (fmt_secs, plural_ru, fix_counts)
    assert segments.SEC_LABEL is textfmt.SEC_LABEL           # behavior_by_segment (from pdf_report, stage 12)
    for name in ("fmt_secs", "seg_label", "mmss_labels", "clean_word", "_SEC_LABEL"):
        assert not hasattr(report, name), f"report.{name}"
    assert hasattr(report, "build_report") and hasattr(report, "DISCLAIMER_RU")
    assert "plural_ru" not in scores.__all__ and not hasattr(scores, "plural_ru")
    assert textfmt.SEC_LABEL.pattern == r"\[(\d+(?:[.,]\d+)?)\s*[–-]\s*(\d+(?:[.,]\d+)?)\s*(?:с|s)\]"
