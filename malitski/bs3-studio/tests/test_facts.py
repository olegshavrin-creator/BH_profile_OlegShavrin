"""What the page and the PDF of BS Profiler 3.1 both show is built once (refactoring plan of 3.1, stage 14): the cards
of «Ключевые факты» and «Речь в цифрах», the head motion word and the note about the model of facial expressions
(bs3/facts.py, the renamed narrative2), the sentences about each analysis (analyses_text), the note under the key
frames (frame_captions.note_what), the caveat lists (caveats.py), the MBTI table and summary line (mbti), and the
thresholds behind the words as named constants (bands.py, mbti.CLEAR_CONFIDENCE).

Every shared builder gives exactly what the page and the PDF built before the stage. The `_old_*` functions below are
the page's and the PDF's own copies as they were, compared with the shared builder on the fixture samples and on grids
around every edge. Before the change the same outputs of the 33 real jobs and of wider grids were recorded in scratch
and compared after it (0 differences); the byte-level baseline (scripts/compare_baseline.py) covers the page and the
PDF of the 11 jobs.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import math
from contextlib import contextmanager
from pathlib import Path

from samples import english, rep

from bs3 import (analyses_text, bands, caveats, characterization, facts, frame_captions, mbti, scores)
from bs3.web import app, mbti_html, page
from bs3.analyses import speech_stats
from bs3.facts import FER_NOTE, card_item, fact_cards, head_motion_word, speech_cards
from bs3.pdf import appendix as pdf_appendix
from bs3.pdf import build as pdf_build
from bs3.pdf import frames as pdf_frames
from bs3.pdf import mbti_section
from bs3.pdf import sections as pdf_sections

ROOT = Path(__file__).resolve().parents[1]
TYPE = {"type": "ENFJ", "type_strict": "ENFJ", "type_name": "Наставник", "x_count": 0, "model": "mm",
        "source": "own_model"}


@contextmanager
def _patched(module, **names):
    old = {k: getattr(module, k) for k in names}
    for k, v in names.items():
        setattr(module, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(module, k, v)


class _Page:
    """Stands in for the PDF page in one section function of pdf/sections.py: records every call and its arguments."""

    def __init__(self, plan: dict):
        self.plan, self.appx, self.calls = plan, {}, []

    def __getattr__(self, name):
        def call(*a, **k):
            self.calls.append((name, a))
            return 1.0
        return call

    def args(self, name: str) -> list:
        return [a for n, a in self.calls if n == name]


# ------------------------------------------------------------------------ the copies as they were before stage 14

def _old_page_facts(view, mb):
    """page._facts_html: the items of the grid and whether the line under it is printed."""
    card = mbti.fact_card(mb)
    items = ([card] if card else []) + facts.key_facts(view)
    return items, any(card_item(i)[3] for i in items)


def _old_pdf_facts(view, speech_cards_follow, mb):
    """pdf_report._pdf_facts and the legend test of pdf_report._render."""
    card = mbti.fact_card(mb)
    items = facts.key_facts(view)
    if speech_cards_follow:
        items = [(lab, val, ("только время, когда человек говорит" if "минуту" in str(val) else "")
                            if lab == "Темп речи" else note, state) for lab, val, note, state in items]
    items = ([card] if card else []) + items
    return items, any(card_item(f)[3] for f in items)


def _whole(v):
    return "—" if v is None else f"{float(v):.0f}"


def _old_page_speech(sp):
    """The items of page._speech_html."""
    fillers = sp.get("fillers")
    return [("Слов всего", _whole(sp.get("words")), ""),
            ("Разных слов", _whole(sp.get("unique_words")), "без повторов"),
            ("Темп речи, слов в минуту", _whole(sp.get("words_per_min_speech")), "только время, когда человек говорит"),
            ("Темп с учётом пауз, слов в минуту", _whole(sp.get("words_per_min_wall")), "по всей длине ролика"),
            ("Доля пауз", f"{sp.get('pause_share', 0):.0%}", "паузы от 0.5 с, доля времени ролика"),
            ("Длинных пауз", _whole(sp.get("long_pauses")), "дольше 2 секунд"),
            ("Слов-заполнителей", _whole(fillers),
             f"{float(sp.get('fillers_per_100') or 0):.0f} на 100 слов" if fillers is not None else ""),
            ("Слов во фразе", _whole(sp.get("mean_sentence")), "в среднем"),
            ("Разнообразие словаря", f"{float(sp['ttr']):.0%}" if sp.get("ttr") is not None else "—",
             "доля разных слов среди всех; зависит от длины текста")]


def _old_pdf_speech(sp):
    """pdf_report._speech_cards."""
    def per_100(v) -> str:
        x = float(v or 0)
        return "меньше 1 на 100 слов" if 0 < x < 0.95 else f"{x:.0f} на 100 слов"
    out = _old_page_speech(sp)
    out[6] = ("Слов-заполнителей", _whole(sp.get("fillers")),
              per_100(sp.get("fillers_per_100")) if sp.get("fillers") is not None else "")
    return out


def _old_head(hm):                     # analyses_parts: «; голова …»
    return "почти неподвижна" if hm < 0.05 else ("двигается умеренно" if hm < 0.15 else "двигается активно")


def _old_motion(hm):                   # page._face_html, pdf_report._emotions_section: «Движение головы …»
    return "слабое" if hm < 0.05 else ("умеренное" if hm < 0.15 else "активное")


def _old_what_page(timed, tenths, described):          # page._frames_html
    return (("момент ролика (минуты:секунды" + (", после запятой — десятые доли секунды" if tenths else "") + ")")
            if timed else "его номер") + (" и коротко то, что на нём видно" if described else "")


def _old_what_pdf(timed, tenths, described):           # pdf_report._explain_section
    return ((("момент ролика (мин:с, после запятой — десятые доли секунды)" if tenths
              else "момент ролика (мин:с)") if timed else "его номер")
            + (" и коротко то, что на нём видно" if described else ""))


OLD_FER_PAGE = ("Модель выражений обучена на фотографиях FER-2013 и склонна видеть «грусть» и «страх» в спокойном "
                "лице: смотрите на изменения по ходу ролика (вкладка «Таймлайн»), а не на абсолютные доли.")
OLD_FER_PDF = ("Модель выражений обучена на фотографиях FER-2013 и склонна видеть «грусть» и «страх» в спокойном лице: "
               "смотрите на изменения по ходу ролика, а не на абсолютные доли.")


# ------------------------------------------------------------------------------------------------ fixtures

def _rep(*, emotion="neutral", arousal=0.50, wpm=120.0, interview=0.50, fillers=2.0, hm=None) -> dict:
    r = {"duration_sec": 660.0, "segments": 33,
         "analyses": {"emotions_text": {"mean": {emotion: 0.9, "joy": 0.05}},
                      "face": {"mean": {"neutral": 0.8, "happy": 0.1}},
                      "voice": {"mean": {"arousal": arousal, "dominance": 0.40, "valence": 0.49}},
                      "speech": {"words": 300, "words_per_min_speech": wpm, "pause_share": 0.21,
                                 "fillers_per_100": fillers}}}
    if interview is not None:
        r["interview"] = {"score": interview}
    if hm is not None:
        r["analyses"]["face"]["head_motion"] = hm
    return r


def _views():
    """(view, mb): the two samples and an English job (clean views, with their MBTI section and without one), and
    synthetic reports around every band of the key facts, with the type card and without it."""
    out = []
    for r in (rep("A"), rep("B"), english("B")):
        view = scores.clean_view(r)
        out += [(view, mbti.get_mbti(r, view)), (view, None)]
    for emotion in ("neutral", "joy", "sadness", "contempt"):
        for arousal in (0.35, 0.36, 0.64, 0.65):
            for wpm in (None, 0, 99.4, 99.5, 130.0, 160.4, 160.5):
                for iv in (None, 0.3, 0.65):
                    r = _rep(emotion=emotion, arousal=arousal, wpm=wpm, interview=iv)
                    out += [(r, TYPE), (r, None)]
    return out + [({}, None), ({}, TYPE)]


def _speech(rate, fillers=3) -> dict:
    return {"words": 300, "unique_words": 150, "words_per_min_speech": 123.4, "words_per_min_wall": 99.5,
            "pause_share": 0.215, "long_pauses": 2, "fillers": fillers, "fillers_per_100": rate,
            "mean_sentence": 9.5, "ttr": 0.51}


# ------------------------------------------------------------------------------------------------ the builders

def test_fact_cards_equal_what_the_page_and_the_pdf_built():
    views = _views()
    assert len(views) > 600
    for view, mb in views:
        assert fact_cards(view, mb) == _old_page_facts(view, mb)
        for follow in (False, True):
            assert fact_cards(view, mb, speech_cards_follow=follow) == _old_pdf_facts(view, follow, mb)
    # the type card first; the note of «Темп речи» when «Речь в цифрах» follows in the same document
    items, legend = fact_cards(_rep(wpm=130.0), TYPE)
    assert items[0] == mbti.fact_card(TYPE) and legend
    trimmed = {card_item(i)[0]: i for i in fact_cards(_rep(wpm=130.0), TYPE, speech_cards_follow=True)[0]}
    assert trimmed["Темп речи"][2] == "только время, когда человек говорит"
    trimmed = {card_item(i)[0]: i for i in fact_cards(_rep(wpm=None), TYPE, speech_cards_follow=True)[0]}
    assert trimmed["Темп речи"][1] == "300 слов" and trimmed["Темп речи"][2] == ""
    assert fact_cards({}, None) == ([], False)


def test_the_page_and_the_pdf_take_the_key_facts_from_fact_cards():
    r = _rep(emotion="joy", arousal=0.20, wpm=200.0)
    marker = ("Маркер", "42", "проверка", "above")
    with _patched(page, fact_cards=lambda view, mb, **k: ([marker], True)):
        html = page._facts_html(r, TYPE)
    assert ">42</div>" in html and "Маркер · выше" in html and facts.FACTS_LEGEND in html
    with _patched(page, fact_cards=lambda view, mb, **k: ([marker], False)):
        assert facts.FACTS_LEGEND not in page._facts_html(r, TYPE)
    for mod in (pdf_build, pdf_appendix, pdf_sections):
        assert "только время, когда человек говорит" not in inspect.getsource(mod), mod.__name__
        for name in ("_pdf_facts", "_speech_cards", "HOW_TO_READ"):
            assert not hasattr(mod, name), (mod.__name__, name)
    assert "fact_cards(report, mb, speech_cards_follow=" in inspect.getsource(pdf_build._render)
    assert not hasattr(app, "FOOTER_CAVEATS") and not hasattr(mbti_html, "READ_CAVEATS")


def test_speech_cards_keep_both_wordings():
    page = {c[0]: c for c in speech_cards(_speech(0.4), small_rate_words=False)}
    pdf = {c[0]: c for c in speech_cards(_speech(0.4), small_rate_words=True)}
    assert page["Слов-заполнителей"] == ("Слов-заполнителей", "3", "0 на 100 слов")
    assert pdf["Слов-заполнителей"] == ("Слов-заполнителей", "3", "меньше 1 на 100 слов")
    for flag in (False, True):
        cards = {c[0]: c for c in speech_cards(_speech(0.96), small_rate_words=flag)}
        assert cards["Слов-заполнителей"][2] == "1 на 100 слов"
        assert speech_cards(_speech(0, fillers=0), flag)[6][2] == "0 на 100 слов"
        assert speech_cards(_speech(0.4, fillers=None), flag)[6][1:] == ("—", "")
    for fillers in (None, 0, 3):
        for rate in (None, 0, 0.3, 0.4, 0.5, 0.94, 0.949, 0.95, 0.96, 1.4, 2.5, 17):
            sp = _speech(rate, fillers)
            assert speech_cards(sp, small_rate_words=False) == _old_page_speech(sp), (fillers, rate)
            assert speech_cards(sp, small_rate_words=True) == _old_pdf_speech(sp), (fillers, rate)
    for sp in ({"words": 10}, {"ttr": 0.5, "pause_share": 0.1}):
        assert speech_cards(sp, False) == _old_page_speech(sp) and speech_cards(sp, True) == _old_pdf_speech(sp)
    assert len(speech_cards({}, True)) == 9


def test_the_page_and_the_pdf_print_their_own_wording_of_the_speech_cards():
    r = {"analyses": {"speech": dict(_speech(0.4), description="Темп речи спокойный.")}, "transcript": ""}
    html = page._speech_html(r)
    assert "0 на 100 слов" in html and "меньше 1" not in html
    pdf = _Page({"voice_speech": 4})
    pdf_sections._voice_speech_section(pdf, r, {})
    assert pdf.args("cards") == [(speech_cards(r["analyses"]["speech"], small_rate_words=True),)]


def test_head_motion_word_at_the_band_edges():
    assert bands.HEAD_MOTION_WORDS == (0.05, 0.15)
    assert [head_motion_word(v, "head") for v in (0.049, 0.05, 0.15, 0.151)] == [
        "почти неподвижна", "двигается умеренно", "двигается активно", "двигается активно"]
    assert [head_motion_word(v, "motion") for v in (0.049, 0.05, 0.15, 0.151)] == [
        "слабое", "умеренное", "активное", "активное"]
    for hm in (0, 0.02, 0.049, 0.0499999, 0.05, 0.05000000000000001, 0.1, 0.1499, 0.15, 0.151, 1.0, math.nan):
        assert head_motion_word(hm, "head") == _old_head(hm), hm
        assert head_motion_word(hm, "motion") == _old_motion(hm), hm


def test_every_text_of_the_head_motion_takes_the_shared_word():
    """The sentence about the face, the face card of the page and the face caption of the PDF follow the band in
    facts: moved edges move all three."""
    def texts(hm):
        r = _rep(hm=hm)
        r["analyses"]["per_segment"] = [{"face": {"frames": 12}}, {"face": {"frames": 10}}]
        pdf = _Page({"emotions": 3})
        pdf_sections._emotions_section(pdf, r, {"face_expr": "face.png"})
        caption = next(cap for path, cap in pdf.args("chart_block") if path == "face.png")
        return analyses_text.analyses_parts(r)["face"], page._face_html(r), caption

    sentence, card, caption = texts(0.1)
    assert "; голова двигается умеренно" in sentence and ">умеренное</div>" in card
    assert caption.startswith("Движение головы умеренное: смещение между кадрами — 10% ширины лица.")
    with _patched(facts, HEAD_MOTION_WORDS=(0.2, 0.3)):
        sentence, card, caption = texts(0.1)
    assert "; голова почти неподвижна" in sentence and ">слабое</div>" in card
    assert caption.startswith("Движение головы слабое:")


def test_fer_note_with_the_page_pointer_and_without():
    assert FER_NOTE.format(where=" (вкладка «Таймлайн»)") == OLD_FER_PAGE and FER_NOTE.format(where="") == OLD_FER_PDF
    assert page.FER_NOTE is FER_NOTE and pdf_sections.FER_NOTE is FER_NOTE
    r = _rep(hm=0.1)
    assert OLD_FER_PAGE in page._face_html(r)
    pdf = _Page({"emotions": 3})
    pdf_sections._emotions_section(pdf, r, {"face_expr": "face.png"})
    assert pdf.args("chart_block")[-1][1].endswith(" " + OLD_FER_PDF)
    for mod in (page, pdf_build, pdf_appendix, pdf_sections, pdf_frames):
        assert "FER-2013" not in inspect.getsource(mod), mod.__name__


def test_note_what_gives_the_two_wordings():
    for timed in (False, True):
        for described in (False, True):
            entries = [{"moment": 12.0 if timed else None, "tail": "смотрит в камеру" if described else ""},
                       {"moment": None, "tail": ""}]
            for tenths in (False, True):
                assert frame_captions.note_what(entries, tenths, "page") == _old_what_page(timed, tenths, described)
                assert frame_captions.note_what(iter(entries), tenths, "pdf") == _old_what_pdf(timed, tenths, described)
    assert frame_captions.note_what([], True, "pdf") == "его номер"
    try:
        frame_captions.note_what([], False, "html")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown medium is refused")
    for mod in (page, pdf_build, pdf_appendix, pdf_sections, pdf_frames):
        assert "момент ролика" not in inspect.getsource(mod), mod.__name__
    # each takes its own wording (the notes of the real jobs are compared byte for byte by compare_baseline)
    assert 'frame_captions.note_what(entries, tenths, "page")' in inspect.getsource(page._frames_html)
    assert 'frame_captions.note_what(entries.values(), tenths, "pdf")' in inspect.getsource(pdf_frames._explain_section)


# ------------------------------------------------------------------------------------------------ moved names

def test_analyses_parts_moved_to_analyses_text():
    assert importlib.util.find_spec("bs3.narrative2") is None
    assert not hasattr(facts, "analyses_parts") and not hasattr(facts, "_level")
    assert pdf_sections.analyses_parts is analyses_text.analyses_parts
    assert "from ..analyses_text import analyses_parts" in inspect.getsource(mbti_html.emo_intro_html)
    parts = analyses_text.analyses_parts(_rep(hm=0.2))
    assert set(parts) == {"text_emotion", "voice", "face", "speech"}
    assert parts["face"].endswith("; голова двигается активно.")


def test_the_mbti_texts_of_the_page_and_the_pdf_live_in_mbti():
    for name in ("TABLE_ROWS", "TABLE_NOTE", "corr_cell", "summary_line"):
        assert getattr(mbti_html, name) is getattr(mbti, name), name
        assert getattr(mbti_section, name) is getattr(mbti, name), name
    for name in ("CORR_WORD", "_seg_word"):
        assert not hasattr(mbti_html, name) and hasattr(mbti, name), name
    tree = ast.parse((ROOT / "bs3" / "pdf" / "mbti_section.py").read_text(encoding="utf-8"))
    imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert "mbti_html" not in imported and "mbti" in imported
    assert mbti.corr_cell({"label": "высокая", "r": 0.74}) == "высокое, r ≈ 0.74"


def test_the_caveat_lists_equal_the_old_tuples():
    assert caveats.PAGE_FOOTER == ("C1", "C10", "C3")
    assert caveats.PDF_HOW_TO_READ == ("C1", "C2", "C10", "C11", "C14", "C15")
    assert caveats.MBTI_READ == ("C3", "C4", "C5", "C6", "C7", "C9", "C16")
    for codes in (caveats.PAGE_FOOTER, caveats.PDF_HOW_TO_READ, caveats.MBTI_READ):
        assert set(codes) <= set(caveats.CODES)
    assert "for c in caveats.PAGE_FOOTER)" in inspect.getsource(app.build_app)
    assert "for c in caveats.PDF_HOW_TO_READ if " in inspect.getsource(pdf_sections._how_to_read)
    assert "for c in caveats.MBTI_READ]" in inspect.getsource(mbti_section.mbti_section)
    assert "for c in caveats.MBTI_READ]" in inspect.getsource(mbti_html.read_html)


# ------------------------------------------------------------------------------------------------ the bands

def test_the_band_constants():
    assert bands.PAUSE_WORDS == (0.10, 0.25)
    assert bands.FILLER_WORDS == (2, 6)
    assert bands.HEAD_MOTION_WORDS == (0.05, 0.15)
    assert mbti.CLEAR_CONFIDENCE == 0.7
    # one band of the tempo for the colour of the card and for the words (owner, stage 14a; the words had 110…160)
    assert scores.TEMPO_BAND == (100, 160) and not hasattr(bands, "TEMPO_WORDS")
    # compared in percent by the callers: the products are exact
    assert [100 * x for x in bands.PAUSE_WORDS] == [10.0, 25.0]
    tree = ast.parse((ROOT / "bs3" / "bands.py").read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert [getattr(n, "module", None) for n in imports] == ["__future__"]


def _describe(wpm=None, pause=None, fillers=0.0):
    return speech_stats.describe({"words": 100, "words_per_min_speech": wpm, "pause_share": pause,
                                  "fillers_per_100": fillers})


def test_describe_at_the_band_edges_and_by_the_bands():
    # the tempo on the whole number the clause prints: 99.5 is «100», 160.5 is «160» (rounded half to even)
    assert [_describe(wpm=w).split(" (")[0] for w in (99.4, 99.5, 110, 160, 160.5, 160.6)] == [
        "Темп речи медленный", "Темп речи спокойный", "Темп речи спокойный", "Темп речи спокойный",
        "Темп речи спокойный", "Темп речи быстрый"]
    assert [_describe(pause=p).split(";")[0] for p in (0.0999, 0.1, 0.2499, 0.25)] == [
        "Пауз мало", "Паузы умеренные (10% времени)", "Паузы умеренные (25% времени)", "Много пауз (25% времени)"]
    assert [_describe(fillers=f).rstrip(".") for f in (1.99, 2, 5.99, 6)] == [
        "Слов-заполнителей почти нет", "Слова-заполнители встречаются (2 на 100 слов)",
        "Слова-заполнители встречаются (6 на 100 слов)", "Много слов-заполнителей (6 на 100 слов)"]
    with _patched(speech_stats, PAUSE_WORDS=(0.2, 0.3), FILLER_WORDS=(3, 7)), _patched(scores, TEMPO_BAND=(120, 170)):
        assert _describe(wpm=115).startswith("Темп речи медленный")
        assert _describe(wpm=165).startswith("Темп речи спокойный")
        assert _describe(pause=0.15).startswith("Пауз мало")
        assert _describe(fillers=2.5) == "Слов-заполнителей почти нет."


def test_the_behaviour_paragraph_by_the_bands():
    T = characterization.load_lexicon()["templates"]
    words = T["behavior"]["tempo_words"]

    def text(wpm, pause=None, arousal=None, extraversion=None):
        view = {"analyses": {"speech": {"words_per_min_speech": wpm, "pause_share": pause},
                             "voice": {"mean": {"arousal": arousal}}}}
        return characterization._p_behavior(view, T, {"extraversion": extraversion})

    assert words["slow"] in text(99.4) and words["calm"] in text(99.5) and words["calm"] in text(160.5)
    assert words["fast"] in text(160.6)
    few, some = T["behavior"]["pauses"]["few"], T["behavior"]["pauses"]["some"]
    assert few in text(130, 0.0999) and some in text(130, 0.1)
    # «неторопливая речь» and «быстрая речь» of the two checks are the tempo words too: the same band
    check_high, check_low = T["behavior"]["check_high"], T["behavior"]["check_low"]
    assert check_high in text(99.4, arousal=0.3, extraversion=0.9) and check_high not in text(99.5, arousal=0.3,
                                                                                              extraversion=0.9)
    assert check_low in text(160.6, arousal=0.7, extraversion=0.1) and check_low not in text(160.5, arousal=0.7,
                                                                                             extraversion=0.1)
    with _patched(characterization, PAUSE_WORDS=(0.2, 0.3)), _patched(scores, TEMPO_BAND=(120, 170)):
        assert words["slow"] in text(115) and words["calm"] in text(165) and few in text(130, 0.15)
        assert check_high in text(115, arousal=0.3, extraversion=0.9)


def test_word_for_by_clear_confidence():
    assert [mbti.word_for({"confidence": c}) for c in (0.69, 0.6999999999, 0.7, 0.71)] == [
        "умеренно", "отчётливо", "отчётливо", "отчётливо"]
    with _patched(mbti, CLEAR_CONFIDENCE=0.8):
        assert mbti.word_for({"confidence": 0.75}) == "умеренно"
