"""«Как получены оценки» and the score bars on the clean view (design 9, 6.2; task T15; change of 2026-09-26: no
reference group, Russian bars show the score only)."""
from __future__ import annotations

from samples import english, rep

from bs3 import labels, narrative, scores, textfmt
from bs3.web import parts as webparts

RELATIVE = ("опорн", "положени", "русских роликов", "обработанных", "большинства", "предварительн", "типичн",
            "на русской речи её", "на русских роликах", "порядок черт")


SECOND = ("второе мнение", "своя модель", "своей модели", "двух систем", "каждой системы", "не усредняются",
          "основная система")


def test_method_notes_sample_b():
    v = scores.clean_view(rep("B"))
    t = narrative.method_notes(v)
    assert t.startswith("Оценки дала система OCEAN-AI на весах MuPTA, обученных на русскоязычных участниках. "
                        "Уровни черт и буквы MBTI считаются по самой оценке модели на шкале от 0 до 1 с серединой 0.5.")
    assert "По ходу ролика (26 отрезков с оценкой OCEAN-AI) оценки устойчивы: разброс не больше ±0.02." in t
    assert "В 7 отрезках из 33 модель OCEAN-AI не дала оценки" in t
    assert "не противоречие в выводах" not in t and "MM-PSYCHE" not in t
    assert "сегмент" not in t
    for w in RELATIVE + SECOND:
        assert w not in t.lower(), w


def test_method_notes_own_model_and_old_english_job():
    a = narrative.method_notes(scores.clean_view(rep("A")))
    assert "(17 отрезков с оценкой OCEAN-AI)" in a and "В 1 отрезке из 18" in a
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    own = narrative.method_notes(scores.clean_view(r))
    assert own.startswith("Оценки дала модель AMLAI 1.0, построенная по рецепту MM-PSYCHE и обученная на First "
                          "Impressions V2; транскрипт русской речи для неё переведён на английский.")
    assert "(33 отрезка с оценкой AMLAI 1.0)" in own and "OCEAN-AI" not in own
    for w in RELATIVE + SECOND:
        assert w not in own.lower(), w
    # OCEAN-AI recorded but gave nothing: C20 instead of the source sentence
    r = rep("B")
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    fell = narrative.method_notes(scores.clean_view(r))
    assert fell.startswith("Модель OCEAN-AI не дала оценок по этому ролику")
    # an older English job is read as OCEAN-AI too; nothing about English speech or FIV2 percentiles (3.1)
    en = narrative.method_notes(scores.clean_view(english("B")))
    assert en.startswith("Оценки дала система OCEAN-AI") and "среднее двух систем" not in en
    assert "процентил" not in en.lower() and "от 0 до 1 с серединой 0.5" in en
    assert "предварительная" not in en
    # C13 names the model of the view
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"][:3]:
        t["members_used"] = []
    for t in r["timeline"][3:]:
        t["members_used"] = ["mm"]
    own = narrative.method_notes(scores.clean_view(r))
    assert "В 3 отрезках из 33 модель AMLAI 1.0 не дала оценки" in own and "OCEAN-AI" not in own


def test_bars_on_clean_view_ru():
    r = rep("B")
    for k in r["traits"]:              # an older job: percentiles against the pool of processed videos
        r["traits"][k].update({"percentile": 80.0, "percentile_ref": "пула обработанных русских роликов (N=5)"})
    v = scores.clean_view(r)
    h = webparts._bar_html(v["traits"], v.get("interview"))
    assert "Экстраверсия</b> <span style='margin-left:auto;text-align:right;font-variant-numeric:tabular-nums'>0.73<" in h
    assert "0.71<" in h                                                        # openness 0.712
    assert webparts.SCALE_NOTE in h
    for w in RELATIVE + ("процентил", "риска", "сейчас их"):
        assert w not in h.lower(), w
    assert "left:calc(" not in h                                               # no percentile tick


def test_bars_english_fiv2_percentile():
    r = english("B")
    for k in r["traits"]:
        r["traits"][k].update({"percentile": 72.0, "percentile_ref": "train First Impressions V2 (6000 клипов)"})
    v = scores.clean_view(r)
    h = webparts._bar_html(v["traits"], v.get("interview"))
    assert "выше, чем у 72% людей в FIV2" in h and "процентиль в First Impressions V2" in h
    assert "Все процентили — относительно обучающей выборки First Impressions V2 (6000 клипов)." in h
    assert webparts.TICK_NOTE in h and "опорн" not in h


def test_no_second_opinion_block():
    """One model in the view (3.1): the framed second-opinion block of 3.0 is gone from the module, and the line of
    «Модель и время обработки» names the one model with its clean scores."""
    assert not hasattr(webparts, "_members_html") and not hasattr(webparts, "SECOND_TITLES")
    assert not hasattr(scores, "gap_sentence") and not hasattr(scores, "SECOND_SCALE_RU")
    line = webparts.model_line(scores.clean_view(rep("B")))
    assert line.startswith("Модель OCEAN-AI, веса MuPTA: открытость опыту 0.71, добросовестность 0.76, экстраверсия 0.73")
    assert line.endswith("эмоциональная стабильность 0.53.")
    r = rep("B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    own = webparts.model_line(scores.clean_view(r))
    assert own.startswith("Модель AMLAI 1.0: открытость опыту 0.") and "MuPTA" not in own and "экстраверсия 0.25" in own
    assert labels.model_title("oceanai") == "OCEAN-AI, веса MuPTA" and labels.model_title("mm") == "AMLAI 1.0"
    assert webparts.MEMBER_TITLES["mm"] == "AMLAI 1.0"


def test_pct_phrases_fiv2_only():
    for ref in ("ref:ru_prov_2026-09-25", "пула обработанных русских роликов (N=5)", ""):
        assert textfmt.pct_phrase(92.9, ref) == ("", False)
    assert textfmt.pct_phrase(None, "train FIV2") == ("", False)
    assert textfmt.pct_phrase(30, "train FIV2") == ("ниже, чем у 70% людей в FIV2", True)
    assert textfmt.pct_phrase(72, "train FIV2") == ("выше, чем у 72% людей в FIV2", True)
