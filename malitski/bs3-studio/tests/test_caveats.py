"""Caveats C1–C22 without C18 (design 11, task T13; 3.1: one model): every one has a text, the templates are filled,
word forms agree with the numbers, none says «сегмент», none mentions a group of processed videos, a second opinion
or the mean of two systems (changes of 2026-09-26)."""
from __future__ import annotations

import re

from bs3 import caveats
from bs3.report import DISCLAIMER_RU, INTERVIEW_DISCLAIMER_RU

PLACEHOLDER = re.compile(r"\{[a-zA-Z_]+\}")


def _all_filled() -> dict:
    out = {}
    for code in caveats.CODES:
        if code == "C13":
            out[code] = caveats.c13(7, 33)
            out[code + "-mm"] = caveats.c13(2, 9, "mm")
        else:
            out[code] = caveats.text(code)
    return out


def test_every_caveat_has_a_text():
    assert caveats.CODES == tuple(f"C{i}" for i in range(1, 23) if i != 18)
    assert "C18" not in caveats.TEXTS and set(caveats.TEXTS) == set(caveats.CODES)
    for code, t in _all_filled().items():
        assert isinstance(t, str) and len(t) > 30, code
        assert not PLACEHOLDER.search(t), (code, t)
        assert t.endswith(".") or t.endswith("»."), code


def test_one_model_no_second_opinion():
    """3.1: no caveat speaks of a second opinion, an own model, the mean or the agreement of two systems."""
    for code, t in _all_filled().items():
        low = t.lower()
        for w in ("второе мнение", "второго мнения", "своя модель", "своей модели", "двух систем", "усреднен",
                  "основная система", "основной считается", "mm-psyche", "английск"):
            assert w not in low, (code, w)
    assert "OCEAN-AI (веса MuPTA) или AMLAI 1.0" in caveats.text("C7")
    assert "одной модели" in caveats.text("C7") and "напрямую их сравнивать не следует" in caveats.text("C7")
    assert caveats.text("C6") == caveats.C6 and caveats.text("C7") == caveats.C7


def test_c1_c2_are_the_report_disclaimers():
    assert caveats.text("C1") == DISCLAIMER_RU
    assert caveats.text("C2") == INTERVIEW_DISCLAIMER_RU


def test_no_segment_word_and_no_type_claim():
    for code, t in _all_filled().items():
        assert "сегмент" not in t.lower(), code
        # C3 says the opposite on purpose: «Система не определяет тип личности»
        if "определяет тип" in t:
            assert code == "C3" and "не определяет тип личности" in t


def test_no_reference_group():
    for code, t in _all_filled().items():
        low = t.lower()
        for w in ("опорн", "положени", "процентил", "русских ролик", "роликов, обработанных", "предварительн",
                  "типичн", "плохо согласуются", "из 13", "медиан"):
            assert w not in low, (code, w)


def test_c5_c6_c9_absolute_scale():
    assert "от 0.36 до 0.64" in caveats.text("C5") and "0.5" in caveats.text("C5")
    assert "середина шкалы 0.5" in caveats.text("C6")
    assert "у края шкалы" in caveats.text("C9")
    assert "граница" not in caveats.text("C7")
    assert caveats.text("C20") == ("Модель OCEAN-AI не дала оценок по этому ролику, поэтому характеристика и тип "
                                   "построены по оценкам модели AMLAI 1.0, сохранённым в этом задании.")


def test_c13_word_forms():
    assert caveats.c13(7, 33).startswith("В 7 отрезках из 33 модель OCEAN-AI не дала оценки")
    assert "эти отрезки не вошли" in caveats.c13(7, 33) and "показаны пропусками" in caveats.c13(7, 33)
    one = caveats.c13(1, 18)
    assert one.startswith("В 1 отрезке из 18") and "этот отрезок не вошёл" in one and "показан пропуском" in one
    assert caveats.c13(21, 40).startswith("В 21 отрезке из 40")
    # the model is the one the view shows: the own model by its product name
    assert caveats.c13(2, 9, "mm").startswith("В 2 отрезках из 9 модель AMLAI 1.0 не дала оценки")
    assert caveats.c13(2, 9, "oceanai") == caveats.c13(2, 9)


def test_text_fills_templates_by_hand():
    t = caveats.text("C13", k=2, segments_k="отрезках", n=5, model="OCEAN-AI", these="эти отрезки не вошли",
                     shown="показаны пропусками")
    assert t.startswith("В 2 отрезках из 5 модель OCEAN-AI")
    try:
        caveats.text("C18")
    except KeyError:
        pass
    else:
        raise AssertionError("C18 (the second strip of 3.0) must be gone")
