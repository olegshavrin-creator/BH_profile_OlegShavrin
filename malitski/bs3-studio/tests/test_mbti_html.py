"""The tab «Тип MBTI» (design 10.3, 5.5–5.7; task T18) on the numeric copies of samples A and B: one panel titled by
the model that ran («OCEAN-AI, веса MuPTA» / «AMLAI 1.0»; 3.1: no second panel, no roles, no agreement line, no
signs), neuroticism, the letter strip with gaps and its summary line, C19, the reading guide with the caveats (C7 of
one model), the short emotion paragraph; no «сегмент», no «второе мнение» and nothing about a group of processed
videos anywhere (change of 2026-09-26: letters on the absolute scale, the tracks show the score itself)."""
from __future__ import annotations

import copy
import re

from samples import english, rep

from bs3 import caveats, mbti, scores
from bs3.web import mbti_html

AN_B = {"voice": {"mean": {"arousal": 0.1536, "dominance": 0.2612, "valence": 0.3717}},
        "emotions_text": {"mean": {"joy": 0.0611, "surprise": 0.0049, "neutral": 0.8604, "sadness": 0.0233,
                                   "fear": 0.0036, "anger": 0.0281, "disgust": 0.0185}}}


def _mb(r: dict) -> dict:
    return mbti.get_mbti(r, scores.clean_view(r))


RELATIVE = ("опорн", "положени", "типичн", "русских роликов", "обработанных", "предварительн", "большинства")


def _cells(strip: str, axis: str) -> int:
    return len(re.findall(rf"title='[^']*· {axis}: [A-Z], ", strip))


def _own(name: str) -> dict:
    """The numbers of a sample as a 3.1 job of AMLAI 1.0."""
    r = rep(name)
    r["model"].update({"selected": "mm", "primary": "mm"})
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    return r


def test_types_b():
    h = mbti_html.types_html(_mb(rep("B")))
    for s in ("OCEAN-AI, веса MuPTA</div>",
              "«Наставник»", "С учётом границ: ENFJ (пограничных осей нет)",
              "Нейротизм — средний уровень. В MBTI этой шкалы нет, поэтому он приводится отдельно.",
              "I · интроверсия", "экстраверсия · E", "соответствие шкал r ≈ 0.74",
              "E · умеренно (уверенность 0.46) · экстраверсия 0.73 — выше среднего",
              "F · отчётливо (уверенность 0.74) · доброжелательность 0.87 — высокий уровень",
              "J · умеренно (уверенность 0.52) · добросовестность 0.76 — выше среднего"):
        assert s in h, s
    # the marker of extraversion at the score 0.73 on the track 0…1
    assert "left:calc(73.0% - 7px)" in h
    # one panel: no second opinion, no roles, no agreement line, no signs
    assert h.count("С учётом границ:") == 1 and "ISXX" not in h and "ISTP" not in h
    for bad in ("второе мнение", "основная оценка", "расходится", "совпадает", "Уверенных совпадений",
                "хотя бы у одной из систем", "Своя модель"):
        assert bad not in h, bad
    assert "minmax(min(320px,100%),1fr)" in h                        # the panel keeps its width rule on a phone
    assert "сегмент" not in h
    for w in RELATIVE:
        assert w not in h, w


def test_types_a_and_none():
    h = mbti_html.types_html(_mb(rep("A")))
    assert "«Наставник»" in h and "С учётом границ: XNFJ · возможен INFJ («Советник»)" in h
    assert "X (ближе к E) · на границе · экстраверсия 0.56 — средний уровень" in h
    assert "IXXX" not in h and "ISTJ" not in h
    assert mbti_html.types_html(None) == f"<p style='{mbti_html.TEXT14};margin:0'>{caveats.text('C21')}</p>"


def test_types_own_model():
    """A job of AMLAI 1.0: one panel of the own model; C20 only when OCEAN-AI was recorded and gave nothing."""
    h = mbti_html.types_html(_mb(_own("A")))
    assert "С учётом границ: IXXX · тип не выражен: 3 оси из 4 на границе" in h and h.count("С учётом границ:") == 1
    assert ">AMLAI 1.0</div>" in h and "MuPTA" not in h and "Своя модель" not in h
    # the own model's agreeableness 0.47531: printed 0.48 as on the bars, not 0.47 through a stored 0.475
    assert "X (ближе к T) · на границе · доброжелательность 0.48 — средний уровень" in h
    assert caveats.text("C20") not in h and "XNFJ" not in h
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    h = mbti_html.types_html(_mb(r))
    assert "С учётом границ: ISXX" in h and caveats.text("C20") in h


def test_types_old_english_job():
    h = mbti_html.types_html(_mb(english("B")))
    assert h.count("С учётом границ:") == 1 and "среднее двух систем" not in h
    assert "пороги предварительные" not in h
    assert re.search(r"(отчётливо|умеренно) \(уверенность 0\.\d\d\)", h)   # the confidence, labelled


def test_strip_b():
    s = mbti_html.strip_html(_mb(rep("B")))
    for ax in ("E–I", "S–N", "T–F", "J–P"):
        assert _cells(s, ax) == 26, ax
    assert s.count("нет оценки OCEAN-AI") == 7 * 4
    assert "OCEAN-AI: ENFJ во всех 26 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках." in s
    assert ">OCEAN-AI, веса MuPTA</div>" in s                         # the lane title: the model, no role
    assert caveats.text("C8") in s
    assert "overflow-x:auto" in s and s.count("grid-template-columns:56px repeat(33,26px)") == 1
    assert "сегмент" not in s and "второе мнение" not in s and "Основная система" not in s and "основная оценка" not in s


def test_strip_a_short_and_new_jobs():
    s = mbti_html.strip_html(_mb(rep("A")))
    # the strict letters never change, but no segment has a confident ENFJ: the line says so
    assert ("OCEAN-AI: строгий тип ENFJ во всех 17 отрезках с оценкой; строгие буквы всех четырёх осей "
            "совпадают с итогом во всех отрезках; ось E–I на границе во всех 17 отрезках, S–N — в 7, T–F — в 6, "
            "J–P — в 6.") in s
    assert "ENFJ во всех 17 отрезках с оценкой; все четыре оси" not in s
    r = rep("B")
    r["timeline"] = []
    r["segments"] = 1
    assert caveats.text("C19") in mbti_html.strip_html(_mb(r))
    # a 3.x job with per-segment variants of both members still shows one strip: the recorded model's
    r = rep("B")
    for t in r["timeline"]:
        t["variants"] = {"mm": copy.deepcopy(r["variant_scores"]["mm"])}
        if "oceanai" in t["members_used"]:
            t["variants"]["oceanai"] = copy.deepcopy(t["scores"])
    s = mbti_html.strip_html(_mb(r))
    assert s.count("grid-template-columns:56px") == 1 and "второе мнение" not in s and "ISTP" not in s
    # the same numbers as a job of AMLAI 1.0: the own model's strip on all 33 segments
    r["model"].update({"selected": "mm", "primary": "mm"})
    for t in r["timeline"]:
        t["primary_used"] = "mm"
    s = mbti_html.strip_html(_mb(r))
    assert "AMLAI 1.0: строгий тип ISTP во всех 33 отрезках с оценкой" in s and s.count("grid-template-columns:56px") == 1
    assert ">AMLAI 1.0</div>" in s and s.count("нет оценки AMLAI 1.0") == 0
    assert _cells(s, "E–I") == 33


def test_read():
    h = mbti_html.read_html(_mb(rep("B")))
    assert caveats.MBTI_READ == ("C3", "C4", "C5", "C6", "C7", "C9", "C16") and not hasattr(mbti_html, "READ_CAVEATS")
    for code in caveats.MBTI_READ:
        assert caveats.text(code) in h, code
    for s in ("высокое, r ≈ 0.74", "высокое, r ≈ 0.72", "среднее, r ≈ 0.44", "среднее, r ≈ 0.49", "в MBTI не выражается",
              "McCrae, Costa, 1989"):
        assert s in h, s
    # the same guide for any job (an older English job is read as one model too), and without a section at all
    assert mbti_html.read_html(_mb(english("B"))) == h == mbti_html.read_html()
    for w in RELATIVE + ("двух систем", "Основной считается", "второе мнение", "своя модель", "английск"):
        assert w not in h, w


def test_emo_intro():
    r = rep("B")
    assert mbti_html.emo_intro_html(r) == ""
    r["analyses"] = copy.deepcopy(AN_B)
    h = mbti_html.emo_intro_html(r)
    assert "По содержанию речи" in h and "Голос (модель эмоций в речи, шкала 0…1)" in h and h.count("<p") == 1
