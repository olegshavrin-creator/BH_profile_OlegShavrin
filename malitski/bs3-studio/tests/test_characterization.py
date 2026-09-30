"""«Характеристика личности» (design 8, 13.1 test_characterization; task T14).

The lexicon rules of 8.5, and on synthetic views (ru samples A and B, en, without analyses, without the second system,
all traits in the middle, all high, main system missing): length, forbidden words, scores only in the parentheses of
the trait lines, letters agree with the MBTI section, the closing paragraph, nothing about a group of processed videos
and no comparison of the two systems (change of 2026-09-26), no type name when the type is not expressed,
determinism, and the verbatim golden texts tests/golden/char_A.txt, char_B.txt (task T25).
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from samples import english, rep

from bs3 import characterization as C
from bs3 import mbti, scores
from bs3.norms import TRAIT_KEYS

LEVELS = ("high", "above", "mid", "below", "low")
STOP = ("диагноз", "расстройств", "патолог", "норма", "нормальн", "отклонени", "плохо", "хорошо", "пригод", "рекоменд",
        "тревожн", "является", "склонен", "сегмент")
# nothing about a group of processed videos, no comparison of the two systems (change of 2026-09-26)
RELATIVE = ("опорн", "положени", "типичн", "русских ролик", "обработанных", "большинства", "предварительн",
            "группы сравнения", "вторая система", "второе мнение", "согласие двух систем", "противоположно",
            "системы по отдельности", "проверка на")
PRONOUNS = ("он", "она", "его", "её", "ее", "ему", "ей", "него", "неё", "нему", "ней")
GOLDEN = Path(__file__).resolve().parent / "golden"
# design 8.2: 150-650 words (without the paragraph «Согласие двух систем», removed on 2026-09-26, the two samples give
# about 490-520 words)
MIN_WORDS, MAX_WORDS = 150, 650

# numbers only (voice, speech, face and text-emotion means of the two samples), no identities
AN = {
    "A": {"speech": {"words_per_min_speech": 99.8, "pause_share": 0.166, "words": 498},
          "voice": {"mean": {"arousal": 0.3557, "dominance": 0.3981, "valence": 0.4287}},
          "face": {"mean": {"happy": 0.5067, "surprise": 0.0806, "neutral": 0.1034, "sad": 0.024, "fear": 0.1605,
                            "angry": 0.1049, "disgust": 0.0198}},
          "emotions_text": {"mean": {"joy": 0.2518, "surprise": 0.014, "neutral": 0.6873, "sadness": 0.02,
                                     "fear": 0.0027, "anger": 0.0129, "disgust": 0.0114}}},
    "B": {"speech": {"words_per_min_speech": 61.3, "pause_share": 0.144, "words": 570},
          "voice": {"mean": {"arousal": 0.1536, "dominance": 0.2612, "valence": 0.3717}},
          "face": {"mean": {"happy": 0.0359, "surprise": 0.0196, "neutral": 0.6512, "sad": 0.0188, "fear": 0.2452,
                            "angry": 0.0277, "disgust": 0.0015}},
          "emotions_text": {"mean": {"joy": 0.0611, "surprise": 0.0049, "neutral": 0.8604, "sadness": 0.0233,
                                     "fear": 0.0036, "anger": 0.0281, "disgust": 0.0185}}},
}


def _with_analyses(r: dict, name: str) -> dict:
    r["analyses"] = copy.deepcopy(AN[name])
    return r


def _build(r: dict):
    v = scores.clean_view(r)
    mb = mbti.get_mbti(r, v)
    return v, mb, C.build(v, mb)


def _cases() -> dict:
    out = {"A": _with_analyses(rep("A"), "A"), "B": _with_analyses(rep("B"), "B"),
           "en": _with_analyses(english("B"), "B"), "no_analyses": rep("B")}
    r = _with_analyses(rep("B"), "B")
    del r["variant_scores"]["mm"]
    out["no_second"] = r
    r = _with_analyses(rep("B"), "B")                 # every trait in the middle of the scale: all axes on the border
    for k in TRAIT_KEYS:
        r["variant_scores"]["oceanai"][k] = 0.5
    out["all_mid"] = r
    r = _with_analyses(rep("A"), "A")                 # every trait high on the scale
    for k in TRAIT_KEYS:
        r["variant_scores"]["oceanai"][k] = 0.9
    out["all_high"] = r
    r = _with_analyses(rep("B"), "B")                 # OCEAN-AI gave nothing: the own model becomes the main system
    del r["variant_scores"]["oceanai"]
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    out["primary_missing"] = r
    return out


# --------------------------------------------------------------------------------------------------- lexicon ---

def test_lexicon_complete_and_follows_the_rules():
    lex = C.load_lexicon()
    # 2: proofreading of task T25; 3: position phrase of ES in its lead; 4: absolute scale, no reference group;
    # 5: band edges of the printed score, short phrases per level, «эмоциональная стабильность» as everywhere;
    # 6: one model (3.1): «AMLAI 1.0» instead of «своя модель», no mean of two systems, no English basis
    assert lex["lexicon_version"] == 6
    assert set(lex["templates"]["header"]["source"]) == {"ocean_ai", "own_model"}
    assert set(lex["templates"]["basis"]["system"]) == {"oceanai", "mm"} and "en" not in lex["templates"]["basis"]
    text = json.dumps(lex, ensure_ascii=False).lower()
    for w in ("своя модель", "своей модели", "mm-psyche", "двух систем", "английск"):
        assert w not in text, w
    assert set(lex["levels"]) == set(TRAIT_KEYS)
    for k in TRAIT_KEYS:
        assert set(lex["levels"][k]) == set(LEVELS), k
        assert set(lex["names"][k]) == {"title", "acc", "dat"}, k
        for lv in LEVELS:
            entry = lex["levels"][k][lv]
            where = f"{k}/{lv}"
            assert 2 <= len(entry) <= 4, where
            for s in entry:
                low = s.lower()
                assert s[-1] in ".!?", where
                assert not re.search(r"\d", s), where
                for w in STOP:          # at the start of a word: «проявляется» (openness/high) is not «является»
                    assert not re.search(r"(?<![а-яё])" + w, low), (where, w)
                for w in RELATIVE:
                    assert w not in low, (where, w)
                tokens = re.findall(r"[а-яё]+", low)
                for p in PRONOUNS:
                    assert p not in tokens, (where, p)
            if lv in ("high", "low"):
                assert entry[-1].startswith("В беседе"), where


def test_short_phrases_poles():
    lex = C.load_lexicon()
    assert set(lex["short"]) == {"extraversion", "openness", "agreeableness", "conscientiousness"}
    assert lex["short"]["extraversion"] == {"high": "общительным и энергичным",
                                            "above": "общительным и открытым в контакте",
                                            "below": "сдержанным и спокойным в контакте",
                                            "low": "сдержанным и немногословным"}
    assert lex["short"]["conscientiousness"]["low"] == "свободным и спонтанным в манере"
    for k, d in lex["short"].items():
        assert set(d) == {"high", "above", "below", "low"}, k
    # «скорее» + the above phrase agrees with the trait paragraph (no «мягким в общении» under «выше среднего»)
    assert lex["short"]["agreeableness"]["above"] in lex["levels"]["agreeableness"]["above"][0]
    for d in lex["short"].values():
        for s in d.values():
            for w in STOP:
                assert w not in s


# ------------------------------------------------------------------------------------------ synthetic views ---

def test_every_case_reads_within_the_rules():
    names16 = set(mbti.load_config()["type_names_ru"].values())
    for case, r in _cases().items():
        v, mb, ch = _build(r)
        text = ch.plain()
        n = ch.word_count()
        assert MIN_WORDS <= n <= MAX_WORDS, (case, n)
        low = text.lower()
        for w in ("сегмент", "определяет тип", "диагноз"):
            assert w not in low, (case, w)
        # «ключ: число» in the paragraphs (the header label «тип не выражен: 3 оси из 4 на границе» is fixed by 8.3)
        body_text = "\n".join(p["lead"] + " " + p["text"] for p in ch.paragraphs)
        assert not re.search(r"\b\w+:\s*\d", body_text), (case, re.search(r"\b\w+:\s*\d", body_text).group(0))
        for w in RELATIVE:
            assert w not in low, (case, w)
        assert "согласие двух систем" not in low and ch.paragraph("agreement") is None, case
        for p in ch.paragraphs:
            body = p["lead"] + " " + p["text"]
            if p["key"] == "behavior":
                assert not re.search(r"0\.\d{2}", body), case
                rest = re.sub(r"\d+ (слово|слова|слов) в минуту", "", body)
                assert not re.findall(r"\d+(?![\d%])", rest), (case, rest)       # only shares with %
            elif p["key"].startswith("trait:"):
                # the score itself, two decimals, first in the parentheses, and no other number
                k = p["key"][6:]
                assert p["text"].startswith(f"({v['traits'][k]['score']:.2f}"), (case, k)
                assert len(re.findall(r"0\.\d{2}", body)) == 1, (case, k)
            elif p["key"] == "stability":
                assert f"({v['traits']['emotional_stability']['score']:.2f})" in p["lead"], case
                assert len(re.findall(r"0\.\d{2}", body)) == 1, case
            elif p["key"] == "basis":       # the band edges of the scale, and nothing else
                assert set(re.findall(r"0\.\d{2}", body)) == {"0.80", "0.65", "0.79", "0.36", "0.64", "0.21",
                                                               "0.35", "0.20"}, case
            else:
                assert not re.search(r"0\.\d{2}", body), (case, p["key"])
        assert ch.paragraphs[-1]["key"] == "limits" and ch.paragraphs[-1]["lead"] == "Границы вывода."
        assert ch.paragraphs[0]["key"] == "short" and ch.paragraphs[1]["key"] == "basis"
        # the level word of every trait line is the band of its score
        for k in TRAIT_KEYS:
            para = ch.paragraph(f"trait:{k}") or ch.paragraph("stability")
            assert para["lead"].split(" — ")[1].startswith(scores.level_phrase(v["traits"][k]["score"])), (case, k)
        # letters agree with the MBTI section
        x = mb["x_count"]
        shown = mb["type_strict"] if x <= 2 else mb["type"]
        assert ch.header["letters_text"] == shown, case
        typology = ch.paragraph("mbti")["text"]
        assert mb["type_strict"] in typology, case
        if x:
            assert mb["type"] in typology, case
        for word in re.findall(r"\b[EIXSNTFJP]{4}\b", text):
            allowed = {mb["type"], mb["type_strict"], *mb["alternatives"]}
            assert word in allowed, (case, word)
        if x >= 3:
            assert not any(nm in text for nm in names16), case
        # deterministic
        v2, mb2, ch2 = _build(copy.deepcopy(r))
        assert ch2.plain() == text and ch2.html() == ch.html(), case


def test_sample_b():
    _, mb, ch = _build(_with_analyses(rep("B"), "B"))
    assert ch.header_plain() == "ENFJ «Наставник» · MBTI по OCEAN-AI · нейротизм: средний уровень"
    short = ch.short_plain()
    assert short == ("По первому впечатлению от записи человек выглядит заметно доброжелательным и мягким в общении, а "
                     "также скорее собранным и последовательным. В нотации MBTI это тип ENFJ («Наставник»). Нейротизм — "
                     "средний уровень.")
    assert [p["key"] for p in ch.paragraphs] == [
        "short", "basis", "trait:agreeableness", "trait:conscientiousness", "trait:extraversion", "trait:openness",
        "stability", "mbti", "behavior", "limits"]
    assert "по оценкам модели OCEAN-AI (веса MuPTA) по 26 отрезкам записи" in ch.paragraph("basis")["text"]
    e = ch.paragraph("trait:extraversion")
    assert e["lead"] == "Экстраверсия — выше среднего"
    assert e["text"].startswith("(0.73; в MBTI — буква E, умеренно). ")
    a = ch.paragraph("trait:agreeableness")
    assert a["lead"] == "Доброжелательность — высокий уровень"
    assert a["text"].startswith("(0.87; в MBTI — буква F, отчётливо). ")
    st = ch.paragraph("stability")
    assert st["lead"] == "Эмоциональная стабильность — средний уровень (0.53); нейротизм, соответственно, — средний уровень."
    assert st["text"].startswith("Спокойные моменты чередуются с признаками волнения.")
    typ = ch.paragraph("mbti")["text"]
    assert typ.startswith("В нотации MBTI профиль соответствует типу ENFJ («Наставник»).")
    assert typ.endswith("По ходу записи буквы не менялись: по каждой оси буква совпадает с итоговой во всех 26 "
                        "отрезках с оценкой.")
    beh = ch.paragraph("behavior")["text"]
    assert beh.startswith("Темп речи медленный (61 слово в минуту), паузы умеренные. Голос по модели эмоций в речи "
                          "ровный и спокойный. Выражение лица, которое модель распознаёт чаще всего, — нейтральное (65% "
                          "кадров). По содержанию речь в основном нейтральна по эмоциональной окраске. Сдержанный голос "
                          "и неторопливая речь не вполне согласуются с оценкой экстраверсии — стоит посмотреть запись.")
    lim = ch.paragraph("limits")["text"]
    assert "В 7 отрезках из 33 модель OCEAN-AI не дала оценки" in lim
    assert lim.endswith("а не пересказывают эпизоды этого ролика.")
    assert "ISXX" not in ch.plain() and "ISTP" not in ch.plain()        # the second system is not in the text


def test_sample_a():
    _, mb, ch = _build(_with_analyses(rep("A"), "A"))
    assert ch.header_plain() == ("ENFJ «Наставник» · MBTI по OCEAN-AI · ось E–I на границе · нейротизм: средний "
                                 "уровень")
    assert "В нотации MBTI ближе всего тип ENFJ («Наставник»), ось E–I на границе." in ch.short_plain()
    assert ch.paragraph("mbti")["text"].startswith("В нотации MBTI ближе всего тип ENFJ («Наставник»), но ось E–I на "
                                                   "границе, поэтому точнее записать XNFJ: возможен и тип INFJ "
                                                   "(«Советник»).")
    # the strict letters never change, but the axes are on the border in part of the segments: said plainly
    assert ch.paragraph("mbti")["text"].endswith(
        "По ходу записи строгие буквы не менялись: по каждой оси буква строгого деления совпадает с итоговой во всех "
        "17 отрезках с оценкой. При этом ось E–I на границе во всех 17 отрезках, S–N — в 7, T–F — в 6, J–P — в 6.")
    assert "буквы не менялись: по каждой оси буква совпадает" not in ch.plain()
    assert ch.short_plain().startswith("По первому впечатлению от записи человек выглядит скорее доброжелательным и "
                                       "настроенным на сотрудничество")
    assert [p["key"] for p in ch.paragraphs][2:6] == ["trait:agreeableness", "trait:conscientiousness",
                                                      "trait:openness", "trait:extraversion"]
    for k in ("openness", "agreeableness", "conscientiousness"):
        assert ch.paragraph(f"trait:{k}")["lead"].endswith("выше среднего")
    e = ch.paragraph("trait:extraversion")
    assert e["lead"] == "Экстраверсия — средний уровень"
    assert e["text"] == "(0.56; в MBTI ось E–I на границе). На записи сочетаются активность в контакте и сдержанность."
    assert "Сдержанный голос" not in ch.plain() and "Оживлённый голос" not in ch.plain()
    assert "радость (51% кадров)" in ch.paragraph("behavior")["text"]
    assert "IXXX" not in ch.plain() and "ISTJ" not in ch.plain()        # the second system is not in the text
    assert "В 1 отрезке из 18" in ch.paragraph("limits")["text"]


def test_special_cases():
    cases = _cases()
    _, _, ch = _build(cases["no_analyses"])
    assert ch.paragraph("behavior") is None
    _, _, ch_ns = _build(cases["no_second"])
    _, _, ch_b = _build(_with_analyses(rep("B"), "B"))
    assert ch_ns.plain() == ch_b.plain()                 # the second system does not change the text
    _, mb, ch = _build(cases["all_mid"])
    assert mb["type"] == "XXXX" and len(mb["alternatives"]) == 15
    assert "тип не выражен: 4 оси из 4 на границе" in ch.header_plain() and ch.header["name"] is None
    assert "Тип MBTI по этой записи не выражен (XXXX)." in ch.short_plain()
    assert ch.paragraph("mbti")["text"].startswith("По всем четырём осям значения близки к границе (XXXX)")
    assert "ни одна из четырёх черт, связанных с MBTI, не выделяется: все они на среднем уровне." in ch.short_plain()
    assert "оценки лежат в средней зоне шкалы" in ch.paragraph("mbti")["text"]
    _, mb, ch = _build(cases["all_high"])
    assert mb["type"] == "ENFJ"
    assert all(ch.paragraph(f"trait:{k}")["lead"].endswith("высокий уровень")
               for k in ("openness", "conscientiousness", "extraversion", "agreeableness"))
    assert ch.paragraph("stability")["lead"].endswith("нейротизм, соответственно, — низкий уровень.")
    _, mb, ch = _build(cases["primary_missing"])
    assert mb["source"] == "own_model"
    assert "MBTI по AMLAI 1.0" in ch.header_plain()
    assert "по оценкам модели AMLAI 1.0 по 33 отрезкам записи" in ch.paragraph("basis")["text"]
    assert "Модель OCEAN-AI не дала оценок по этому ролику" in ch.paragraph("limits")["text"]
    assert "своя модель" not in ch.plain().lower() and "MM-PSYCHE" not in ch.plain()
    # a 3.1 job of AMLAI 1.0: its name in the header and the basis, no C20 (nothing was missing)
    r = _with_analyses(rep("B"), "B")
    r["model"].update({"selected": "mm", "primary": "mm"})
    r["variant_scores"] = {"mm": r["variant_scores"]["mm"]}
    for t in r["timeline"]:
        t["members_used"] = ["mm"]
    _, mb, ch = _build(r)
    assert "MBTI по AMLAI 1.0" in ch.header_plain() and "по оценкам модели AMLAI 1.0" in ch.paragraph("basis")["text"]
    assert "OCEAN-AI" not in ch.plain()
    # an older English job is read as OCEAN-AI (3.1: no mean of two systems)
    _, mb, ch = _build(cases["en"])
    assert mb["source"] == "ocean_ai" and "MBTI по OCEAN-AI" in ch.header_plain()
    assert "среднему двух систем" not in ch.plain() and "по оценкам модели OCEAN-AI (веса MuPTA)" in ch.paragraph("basis")["text"]
    assert "от 0 до 1" in ch.paragraph("basis")["text"]


def test_no_big_five():
    r = rep("B")
    r["traits"] = {}
    r["variant_scores"] = {}
    v = scores.clean_view(r)
    mb = mbti.get_mbti(r, v)
    assert mb is None
    ch = C.build(v, mb)
    assert "Тип MBTI не рассчитан: в результате нет оценок Big Five." in ch.plain()
    assert ch.paragraphs[-1]["key"] == "limits"


def test_html_and_pdf_forms():
    _, _, ch = _build(_with_analyses(rep("B"), "B"))
    h = ch.html()
    assert h.count("<p ") == len(ch.paragraphs)
    assert "<b>Коротко.</b>" in h and "font-size:36px" in h
    assert "<b>Согласие двух систем.</b>" not in h and "line-height:1.1'>ENFJ</span>" in h    # no borderline axis
    assert "<script" not in h
    pdf = ch.pdf_paragraphs()
    assert pdf[0]["markdown"].startswith("**Коротко.** По первому впечатлению")
    hd = ch.pdf_header()
    assert hd["letters"] == [("E", False), ("N", False), ("F", False), ("J", False)] and hd["name"] == "Наставник"
    _, _, ch = _build(_with_analyses(rep("A"), "A"))
    h = ch.html()
    assert "<span style='opacity:.55;text-decoration:underline 2px dashed #808080;text-underline-offset:6px'>E</span>NFJ" in h
    assert "border:1px dashed #808080" in h and "ось E–I на границе" in h
    assert ch.pdf_header()["letters"] == [("E", True), ("N", False), ("F", False), ("J", False)]
    assert "Здесь появится характеристика личности" in C.placeholder_html()


def test_golden_texts():
    """Verbatim comparison with tests/golden/char_A.txt, char_B.txt (task T25: saved after the proofreading of
    lexicon_version 2, regenerated for lexicon_version 3, 4 (the absolute scale), 5 (printed band edges) and 6 (one
    model, 3.1) from the same inputs as here; in char_A the tempo of 99.8 words per minute is «спокойный» since the
    one tempo band of stage 14a). A lexicon or template change must regenerate them on purpose."""
    for name in ("A", "B"):
        path = GOLDEN / f"char_{name}.txt"
        assert path.is_file(), f"missing golden text {path.name}"
        _, _, ch = _build(_with_analyses(rep(name), name))
        want = path.read_text(encoding="utf-8")
        got = ch.plain()
        if got != want:
            diff = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
            raise AssertionError(f"{name}: differs from {path.name} at character {diff}: "
                                 f"{got[max(0, diff - 40):diff + 40]!r} vs {want[max(0, diff - 40):diff + 40]!r}")
