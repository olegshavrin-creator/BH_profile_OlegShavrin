"""PDF of BS Profiler 3.1 (design 10.7; tasks T21, T27) on the numeric copies of samples A and B: the plan puts the
section «Тип MBTI» right after the Big Five section, the report builds without charts and media, its text has the
characterization, the MBTI section of one model with the letter strip and «Как получены оценки», no second opinion
and nothing of the 2.0 summary; the appendix «Значения по отрезкам» has the MBTI column with a dash on the segments
without OCEAN-AI. 3.1: the passport names one model («Модель»), the bars are «Оценки по чертам», an OCEAN-AI job has
no section 5 and one line under section 4 instead (narrative.NO_EXPLAIN_RU), a job of AMLAI 1.0 with an explanation
has section 5 named after it."""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from samples import english, own as _own, rep

from bs3 import caveats, characterization, mbti, scores
from bs3.facts import card_item, fact_cards
from bs3.narrative import NO_EXPLAIN_RU
from bs3.norms import TRAIT_KEYS
from bs3.pdf import appendix, build, mbti_section
from bs3.pdf.document import TEXT_W_MM, Report

# words of 3.0 that no report of 3.1 may carry (outside the transcript, which the fixtures do not have)
GONE = ("торое мнение", "своя модель", "своей модели", "Своя модель", "MM-PSYCHE", "ансамбл", "Язык речи", "язык речи",
        "основная оценка", "Основная система", "Участники", "среднее двух систем", "Согласие", "английской речи",
        "для английской", "для русской речи")

# the test PDFs go here and are removed with it, not left behind in /tmp
_TMP = tempfile.TemporaryDirectory(prefix="bs3_pdf_test_")


def _parts(r: dict):
    view = scores.clean_view(r)
    mb = mbti.get_mbti(r, view)
    return view, mb, characterization.build(view, mb)


def _build(r: dict, explanation: dict | None = None) -> tuple[Path, dict]:
    view, mb, ch = _parts(r)
    out = Path(tempfile.mkdtemp(dir=_TMP.name)) / "report.pdf"
    build.build_pdf(view, out, explanation=explanation, mbti=mb, character=ch)
    return out, mb


# the shape of explanation.json that the PDF reads: modality shares per trait (pdf/charts draws no chart here,
# build_pdf gets no chart files) and the readable words
EXPL = {"modalities": {"input_x_gradient": {k: {"face": {"share": 0.6}, "audio": {"share": 0.38},
                                                "text": {"share": 0.01}, "behavior": {"share": 0.01}}
                                            for k in TRAIT_KEYS}},
        "readable_words": {"transcript_words": {k: {"up": [{"word": "work", "ru": "работа", "signed": 0.01}],
                                                    "down": []} for k in TRAIT_KEYS}}}


def _text(path: Path) -> str:
    if not shutil.which("pdftotext"):
        raise unittest.SkipTest("pdftotext (poppler-utils) not installed")
    r = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"], capture_output=True, text=True, check=True)
    return re.sub(r"\s+", " ", r.stdout)


def test_plan_puts_mbti_after_profile():
    view, mb, _ = _parts(rep("B"))
    pdf = Report()
    build._plan(pdf, view, None, [], {}, mb)
    keys = list(pdf.plan)
    assert keys[:2] == ["profile", "mbti"], keys
    assert pdf.plan["mbti"] == 2
    pdf2 = Report()
    build._plan(pdf2, view, None, [], {}, None)            # no type: no section, numbers close up
    assert "mbti" not in pdf2.plan and list(pdf2.plan)[1] != "mbti"


def test_plan_explain_section_for_own_model_only():
    """Section 5 exists only for a job of AMLAI 1.0 with an explanation or key frames (3.1)."""
    view, mb, _ = _parts(rep("B"))
    pdf = Report()
    build._plan(pdf, view, EXPL, ["frame.jpg"], {}, mb)
    assert "explain" not in pdf.plan                            # OCEAN-AI: never, whatever the job carries
    own, mb2, _ = _parts(_own("B"))
    pdf = Report()
    build._plan(pdf, own, EXPL, [], {}, mb2)
    assert "explain" in pdf.plan
    pdf = Report()
    build._plan(pdf, own, None, [], {}, mb2)
    assert "explain" not in pdf.plan


def test_strip_geometry():
    assert mbti_section._strip_geometry(33) == (2, 17, 6.0, 4.8)       # 17 + 16, not 30 + 3
    assert mbti_section._strip_geometry(18)[:3] == (1, 18, 6.0)
    blocks, per, cell, row_h = mbti_section._strip_geometry(90)
    assert (blocks, per) == (3, 30) and abs(cell - (TEXT_W_MM - 18) / 30) < 1e-9 and row_h <= cell


def test_facts_start_with_type_card():
    view, mb, _ = _parts(rep("B"))
    cards, legend = fact_cards(view, mb)
    assert cards[0] == mbti.fact_card(mb)                      # the type card carries no state: never coloured
    assert card_item(cards[0])[3] is None
    assert fact_cards(view, None)[0][0] != cards[0]
    # trimming the tempo note for «Речь в цифрах» keeps the state of the card (tests/test_facts.py: the note)
    trimmed, legend2 = fact_cards(view, mb, speech_cards_follow=True)
    assert [card_item(f)[3] for f in trimmed] == [card_item(f)[3] for f in cards] and legend2 == legend
    # the report prints the cards of facts.fact_cards: _render has no copy of its own
    assert not hasattr(build, "_pdf_facts") and not hasattr(build, "_speech_cards")


def test_segment_types_by_start():
    _, mb, _ = _parts(rep("B"))
    t = mbti_section.segment_types_by_start(mb)
    assert len(t) == 33
    assert sum(1 for v in t.values() if v is None) == 7                 # the segments without OCEAN-AI
    assert {v for v in t.values() if v} == {"ENFJ"}


def test_pdf_builds_and_reads():
    for name in ("A", "B"):
        path, mb = _build(rep(name))
        assert path.exists() and path.stat().st_size > 10_000, name
        text = _text(path)
        for s in ("Характеристика личности", "Коротко.", "Границы вывода.", "Ключевые факты", "Тип MBTI · OCEAN-AI",
                  "Big Five: профиль и оценки", "Как получены оценки", "Оценки по чертам",
                  "2. Тип MBTI (перевод шкал Big Five)", "OCEAN-AI, веса MuPTA: ", "Модель OCEAN-AI, веса MuPTA",
                  "Тип по ходу ролика", "Как читать тип MBTI", "Как читать результаты", "BS Profiler 3.1 · стр.",
                  mb["type"], mb["type_strict"], NO_EXPLAIN_RU[:50]):
            assert s in text, (name, s)
        for code in ("C7", "C8", "C9", "C16", "C11", "C15"):
            assert caveats.text(code)[:60] in text, (name, code)
        assert caveats.text("C2")[:40] not in text                     # OCEAN-AI: no label «собеседование» to explain
        # OCEAN-AI: no section «Что повлияло …»; the numbered sections stop before «Как читать результаты»
        assert "Что повлияло" not in text
        n_read = int(re.search(r"(\d)\. Как читать результаты", text).group(1))
        assert re.search(rf"(?<![\d.]){n_read + 1}\. [А-Я]", text) is None, name
        for bad in ("Краткие выводы", "сегмент", "определяет тип личности", "опорн", "положени", "типичн",
                    "русских роликов", "обработанных системой", "предварительн", "большинства", "Согласие двух систем",
                    "Вторая система", "на русских роликах", "на русской речи её", "порядок черт",
                    "Уверенных совпадений", "хотя бы у одной из систем", "BS Profiler 3.0", "ISXX", "IXXX", "ISTP",
                    "ISTJ", "система OCEAN-AI не дала") + GONE:
            assert bad not in text.replace("не определяет тип личности", ""), (name, bad)
        appx = text.split("Значения по отрезкам", 1)[-1]
        assert " MBTI " in appx, name
    # sample B: the strip summary (design 13.2) and the MBTI column with the types of the segments
    path, mb = _build(rep("B"))
    text = _text(path)
    assert "OCEAN-AI: ENFJ во всех 26 отрезках с оценкой; все четыре оси совпадают с итогом во всех отрезках" in text
    assert "F, отчётливо (0.74)" in text and "E, умеренно (0.46)" in text
    assert "Число в скобках после буквы — уверенность по оси" in text          # the number is labelled
    assert "Согласие" not in text
    appx = text.split("Значения по отрезкам", 1)[-1]
    assert "ENFJ" in appx
    assert "«—» в столбцах Big Five и MBTI — модель OCEAN-AI не дала оценки отрезка" in appx
    assert "MBTI — тип отрезка, X — ось на границе" in appx


def test_pdf_own_model():
    """A job of AMLAI 1.0 (3.1): the PDF names its type and no OCEAN-AI type; no second opinion; without an
    explanation there is no section 5 and no OCEAN-AI note either; with one, section 5 is named after the model."""
    path, mb = _build(_own("B"))
    assert path.exists() and mb["source"] == "own_model" and mb["type"] == "ISXX"
    text = _text(path)
    assert "ISXX" in text and "ISTP" in text and "ENFJ" not in text
    assert "AMLAI 1.0: " in text and "Модель AMLAI 1.0" in text
    # OCEAN-AI and its weights are named once, in C7 of «Как читать тип MBTI» (both models on purpose)
    rest = text.replace("OCEAN-AI (веса MuPTA) или AMLAI 1.0", "")
    assert "MuPTA" not in rest and "OCEAN-AI" not in rest
    assert NO_EXPLAIN_RU[:50] not in text and "Что повлияло" not in text
    assert caveats.text("C20")[:40] not in text
    assert "Обучающие данные First Impressions V2" in text
    for bad in GONE:
        assert bad not in text.replace("по рецепту MM-PSYCHE", ""), bad
    path, mb = _build(_own("B"), explanation=EXPL)
    text = _text(path)
    # numbered like the rest (the fixtures carry no analyses, so it is section 3 here, 5 in a full report)
    assert re.search(r"\d\. Что повлияло на оценку модели AMLAI 1\.0", text)
    assert "Объяснения построены для модели AMLAI 1.0 по " in text        # «по отрезку …» with a timeline chart
    assert "Слова, на которые откликнулась модель" in text and NO_EXPLAIN_RU[:50] not in text
    for bad in GONE:
        assert bad not in text.replace("по рецепту MM-PSYCHE", ""), bad


def test_pdf_oceanai_ignores_explanation():
    """An older OCEAN-AI job that carries the explanation and the behaviour description of the second model of 3.0:
    no section 5, no appendix «Описание поведения», the one-line note."""
    r = rep("B")
    r["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."
    path, mb = _build(r, explanation=EXPL)
    text = _text(path)
    assert "Что повлияло" not in text and NO_EXPLAIN_RU[:50] in text
    assert "Слова, на которые откликнулась модель" not in text and "Описание поведения" not in text
    own = _own("B")
    own["behavior_description_ru"] = "[0–20 с] Человек говорит спокойно."
    text = _text(_build(own)[0])
    # 33 segments, one description: the appendix of the notable segments
    assert "Приложение В. Описание поведения" in text


def test_pdf_old_english_job():
    path, mb = _build(english("B"))
    assert path.exists() and mb["source"] == "ocean_ai"
    text = _text(path)
    assert "среднее двух систем" not in text and "торое мнение" not in text
    assert "пороги предварительные" not in text
    assert "Модель OCEAN-AI, веса MuPTA" in text
    for bad in GONE:
        assert bad not in text, bad


def test_analysis_rows_one_model():
    """Appendix А: the model of the view, no language row; «Модальности» names what the model looked at — the
    recorded modalities of a 3.1 job, or those of the shown model when an older job recorded member names instead."""
    view = scores.clean_view(rep("B"))
    view["modalities_used"] = ["oceanai", "mm"]                     # a job of 3.0
    rows = dict(appendix._analysis_rows(view))
    assert rows["Модель"] == "OCEAN-AI, веса MuPTA" and rows["Модальности"] == "голос, видео, речь"
    assert rows["Обучающие данные"] == "MuPTA (русская речь)" and "Язык речи" not in rows and "Система" not in rows
    view["modalities_used"] = ["audio", "video", "text"]           # a job of 3.1
    assert dict(appendix._analysis_rows(view))["Модальности"] == "голос, видео, речь"
    own = dict(appendix._analysis_rows(scores.clean_view(_own("B"))))    # ["mm"] recorded
    assert own["Модель"] == "AMLAI 1.0" and own["Модальности"] == "лицо, голос, речь, описание поведения"
    assert own["Обучающие данные"] == "First Impressions V2"
    for r in (rows, own):
        assert "OCEAN-AI" not in r["Модальности"] and "AMLAI" not in r["Модальности"]


def test_interview_label_and_c2_follow_the_model():
    """An imported 2.0 job shown as OCEAN-AI carries the «собеседование» label of the other model: the PDF drops it
    everywhere (card, bar, note, appendix column) and does not print C2, which explains it; a job of AMLAI 1.0 keeps
    the label and gets C2 in «Как читать результаты»."""
    r = rep("B")
    r["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
    for t in r["timeline"]:
        if isinstance(t.get("scores"), dict):
            t["scores"]["interview"] = 0.4
    view, mb, ch = _parts(r)
    assert "interview" not in view
    pdf = Report()
    assert pdf._bar_rows(view["traits"], view.get("interview")) == [(k, view["traits"][k]) for k in TRAIT_KEYS]
    out = Path(tempfile.mkdtemp(dir=_TMP.name)) / "oa.pdf"
    build.build_pdf(view, out, mbti=mb, character=ch)
    text = _text(out)
    assert "собеседовани" not in text.lower() and "Собе-" not in text and "ChaLearn" not in text
    assert caveats.text("C1")[:40] in text and caveats.text("C10")[:40] in text
    own = _own("B")
    own["interview"] = {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
    for t in own["timeline"]:
        t["scores"]["interview"] = 0.4
    view, mb, ch = _parts(own)
    assert view["interview"] == {"score": 0.4011, "name_ru": "впечатление «пригласить на собеседование»"}
    out = Path(tempfile.mkdtemp(dir=_TMP.name)) / "mm.pdf"
    build.build_pdf(view, out, mbti=mb, character=ch)
    text = _text(out)
    assert "Впечатление «собеседование» 0.40" in text and caveats.text("C2")[:50] in text
    assert "Коричневая полоска — впечатление «собеседование» (метка модели AMLAI 1.0, шкала 0…1)" in text
    assert "Big Five и «собеседование», 0…1" in text


def test_short_transcript_stays_with_appendix_a():
    """A one-line transcript never opens a page of its own (review of 3.1): when appendix А ends near the foot of the
    page, the two tables are set tighter and the transcript at 8 pt so that the short appendix stays on that page;
    when even that is not enough, the appendices start on the next page together; when everything fits as it is,
    the normal spacing is kept. Swept over every start position."""
    r = rep("B")
    r["timeline"], r["segments"] = r["timeline"][:1], 1
    r["transcript"] = "Здравствуйте. Сегодня я коротко расскажу о своей работе и о том, что мне в ней нравится."
    view = scores.clean_view(r)
    seen = set()
    for y0 in range(60, 280, 2):
        pdf = Report()
        build._plan(pdf, view, None, [], {}, None)
        assert set(pdf.appx) == {"file", "transcript"}
        pdf.add_page()
        pdf.set_y(y0)
        lay = appendix._appendix_layout(pdf, view, None)
        appendix._appendices(pdf, view, None, False, None)
        kind = "tight" if lay["tight"] else "new_page" if lay["new_page"] else "normal"
        seen.add(kind)
        if kind == "tight":
            assert pdf.page_no() == 1, (y0, "tightened, so everything stays on the page")
        if pdf.page_no() == 2:
            assert pdf.get_y() > 60, (y0, "a second page holds appendix А too, never the transcript alone")
    assert seen == {"normal", "tight", "new_page"}, seen
    # a long transcript flows as before (no tightening: _transcript_size handles its tail)
    r["transcript"] = " ".join(["Это длинный транскрипт из многих предложений."] * 60)
    view = scores.clean_view(r)
    pdf = Report()
    build._plan(pdf, view, None, [], {}, None)
    pdf.add_page()
    pdf.set_y(200)
    assert appendix._appendix_layout(pdf, view, None)["tight"] is False


def test_build_pdf_computes_missing_parts():
    """A caller that passes only the raw result gets the same characterization and type."""
    r = rep("B")
    out = Path(tempfile.mkdtemp(dir=_TMP.name)) / "raw.pdf"
    build.build_pdf(r, out)
    assert out.exists()
    text = _text(out)
    assert "Характеристика личности" in text and "ENFJ" in text


def test_export_pdf_builds_without_media():
    """export_pdf (this stage owns it) opens a job folder and builds the PDF even when the job stores no media and has
    no input.* to probe, and when the stored media is an error (T6(3)): the fallback yields no media, the file appendix
    then names only the file, and the PDF stays in the folder as BS_Profiler_3_report_<stem>.pdf."""
    from bs3 import jobfiles
    from bs3.pdf import export_pdf
    for stored in (None, {"error": "ffprobe failed"}):
        d = Path(tempfile.mkdtemp(dir=_TMP.name))
        job = d / "20000101_000000_0f3a9c1e"
        job.mkdir()
        r = rep("B")
        r.pop("media", None)
        if stored is not None:
            r["media"] = stored
        jobfiles.write_json(job / jobfiles.RESULT, r)
        pdf = Path(export_pdf(job))
        assert pdf.exists() and pdf.parent == job and pdf.stat().st_size > 10_000, stored
        assert pdf.name.startswith("BS_Profiler_3_report_") and pdf.suffix == ".pdf", stored
