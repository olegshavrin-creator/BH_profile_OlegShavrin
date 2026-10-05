"""The page and the PDF of BS Profiler 3.1 say the same (refactoring plan of 3.1, stage 14a; owner, 2026-09-27).

- The page's table «Эмоции, голос и темп по отрезкам» and the PDF's appendix «Значения по отрезкам» print the same
  cells for a segment (facts.segment_cells) and the same segment times: the end 651.8 is «10:52» on both (the page
  cut it to «10:51»), a segment without a tempo is «—» (the page printed «0»), a segment with an empty transcript is
  «нет речи» or «нет текста» (the page printed the model's «нейтрально 100%»).
- One tempo band, scores.TEMPO_BAND (100…160 words per minute), and one classifier, scores.tempo_state, decided on the
  whole number that is printed: the colour of the card «Темп речи» and the words «медленный / спокойный / быстрый» of
  the sentence about the manner of speech and of the characterization never disagree.
- The sentence about the manner of speech is stored in result.json at analysis time; the page and the PDF show an old
  job's sentence with its tempo clause rebuilt from the stored words per minute and the rest of it as stored.
"""
from __future__ import annotations

import inspect
import re
from contextlib import contextmanager

from bs3 import analyses_text, characterization, facts, scores
from bs3.analyses import speech_stats
from bs3.facts import segment_cells
from bs3.pdf import appendix
from bs3.pdf.document import Report
from bs3.scores import ABOVE, BELOW, NEUTRAL
from bs3.web import page


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


def _seg(i: int, start: float, end: float, **kw) -> dict:
    r = {"segment": i, "start": start, "end": end, "text_en": "we talked about the new project",
         "emotions_text": {"neutral": 0.3, "joy": 0.6, "fear": 0.1},
         "voice": {"arousal": 0.41, "dominance": 0.52, "valence": 0.47},
         "face": {"frames": 12, "expressions": {"happy": 0.2, "neutral": 0.7, "sad": 0.1}},
         "speech": {"words": 40, "words_per_min_speech": 121.6, "pause_share": 0.183}}
    r.update(kw)
    return r


# one segment for each case; the model's «нейтрально 100%» on an empty transcript is what the page used to print
NEUTRAL_100 = {"neutral": 1.0, "joy": 0.0}
SEGMENTS = {
    "spoken": _seg(1, 0.0, 20.0),
    "no tempo": _seg(2, 20.0, 40.0, speech={"words": 4, "words_per_min_speech": None, "pause_share": 0.82}),
    "tempo 0": _seg(3, 40.0, 60.0, speech={"words": 0, "words_per_min_speech": 0.0, "pause_share": 0.9}),
    "no text": _seg(4, 60.0, 80.0, text_en="", emotions_text=NEUTRAL_100,
                    speech={"words": 7, "words_per_min_speech": 88.0, "pause_share": 0.4}),
    "no speech": _seg(5, 80.0, 100.0, text_en=" ", emotions_text=NEUTRAL_100,
                      speech={"words": 0, "words_per_min_speech": None, "pause_share": 1.0}),
    "no data": {"segment": 6, "start": 100.0, "end": 120.0},
    "last": _seg(7, 640.0, 651.8),
}


def _report() -> dict:
    return {"duration_sec": 651.8, "segments": len(SEGMENTS), "analyses": {"per_segment": list(SEGMENTS.values())}}


def _page_rows(rep: dict) -> list:
    """The rows of the page's table as the browser gets them: the text of every <td> of the table body."""
    body = page._segments_table(rep).split("<tbody>", 1)[1]
    trs = re.findall(r"<tr style='border:0'>(.*?)</tr>", body)
    return [re.findall(r"<td style='[^']*'>(.*?)</td>", tr) for tr in trs]


def _pdf_rows(rep: dict) -> tuple[list, list]:
    """(header, rows) of the PDF's appendix «Значения по отрезкам» as handed to Report.table."""
    pdf = Report(file_label="clip.mp4", total_pages=1)
    pdf.add_page()
    pdf.appx["segments"] = "Б"
    got = {}
    pdf.table = lambda header, rows, *a, **k: got.update(header=header, rows=rows)
    appendix._segments_table(pdf, rep, None)
    return got["header"], got["rows"]


def test_the_segment_cells_of_the_page_equal_the_pdf():
    rep = _report()
    page_rows = _page_rows(rep)
    header, pdf = _pdf_rows(rep)
    cols = len(header) - 7                   # the PDF puts the scores (and the MBTI type) between time and cells
    assert header[cols:] == ["по речи", "по лицу", "Возб.", "Увер.", "Позит.", "Темп", "Паузы"]
    assert len(page_rows) == len(pdf) == len(SEGMENTS)
    for name, p, q in zip(SEGMENTS, page_rows, pdf):
        assert p[0] == q[0], name                                      # the time of the segment
        assert p[1:] == q[cols:] == segment_cells(SEGMENTS[name]), name
    rows = dict(zip(SEGMENTS, page_rows))
    # the three cases of the owner: the end of the last segment, a segment without a tempo, an empty transcript
    assert rows["last"][0] == "10:40–10:52"
    assert rows["no tempo"][6] == rows["tempo 0"][6] == "—" and rows["spoken"][6] == "122"
    assert rows["no text"][1] == "нет текста" and rows["no speech"][1] == "нет речи"
    assert rows["no text"][6] == "88" and rows["spoken"][1] == "радость 60%" and rows["spoken"][2] == "нейтрально 70%"
    assert rows["no data"][1:] == ["—"] * 7 and rows["spoken"][3:6] == ["0.41", "0.52", "0.47"]
    assert rows["spoken"][7] == "18%" and rows["no speech"][7] == "100%"
    assert "нейтрально 100%" not in page._segments_table(rep)


def test_both_tables_take_the_cells_from_facts():
    assert "segment_cells(r)" in inspect.getsource(page._segments_table)
    assert "segment_cells(r)" in inspect.getsource(appendix._segments_table)
    assert page.segment_cells is appendix.segment_cells is facts.segment_cells
    assert not hasattr(page, "_dominant") and not hasattr(appendix, "_dominant_text")
    assert "truncate" not in inspect.getsource(page)
    # a short video keeps m:ss, an hour and more writes h:mm:ss in the whole column, rounded
    short = {"analyses": {"per_segment": [_seg(1, 0.0, 19.6), _seg(2, 19.6, 39.5)]}}
    assert [r[0] for r in _page_rows(short)] == ["0:00–0:20", "0:20–0:40"]
    long = {"analyses": {"per_segment": [_seg(1, 3580.0, 3599.6), _seg(2, 3599.6, 3619.4)]}}
    assert [r[0] for r in _page_rows(long)] == ["0:59:40–1:00:00", "1:00:00–1:00:19"]


# ------------------------------------------------------------------------------------------------ one tempo band

def _speech(wpm, description=None) -> dict:
    sp = {"words": 300, "words_per_min_speech": wpm, "pause_share": 0.17, "fillers_per_100": 0.5,
          "mean_sentence": 12.0}
    sp["description"] = speech_stats.describe(sp) if description is None else description
    return sp


def _tempo_texts(wpm, description=None) -> dict:
    """The state of the tempo card and the tempo words of every text for one words-per-minute value."""
    sp = _speech(wpm, description)
    view = {"duration_sec": 360.0, "segments": 18, "analyses": {"speech": sp, "voice": {"mean": {"arousal": 0.5}}}}
    card = next(f for f in facts.key_facts(view) if f[0] == "Темп речи")
    T = characterization.load_lexicon()["templates"]["behavior"]
    behavior = characterization._p_behavior(view, {"behavior": T}, {})
    char_word = next(k for k, w in T["tempo_words"].items() if f"Темп речи {w} (" in behavior)
    word = next(w for w in analyses_text.TEMPO_RU.values() if f"Темп речи {w} (" in sp["description"])
    page_html = page._speech_html(view)
    return {"card": card[3], "stored": word, "characterization": char_word,
            "page": next(s for s, w in analyses_text.TEMPO_RU.items() if f"Темп речи {w} (" in page_html),
            "pdf": next(s for s, w in analyses_text.TEMPO_RU.items()
                        if f"Темп речи {w} (" in analyses_text.analyses_parts(view)["speech"])}


def test_one_tempo_band_for_the_colour_and_the_words():
    words = {BELOW: "медленный", NEUTRAL: "спокойный", ABOVE: "быстрый"}
    lexicon = {BELOW: "slow", NEUTRAL: "calm", ABOVE: "fast"}
    assert analyses_text.TEMPO_RU == words and characterization.TEMPO_WORD == lexicon
    # the owner's edges: 99.4 is printed «99», 99.5 «100», 160.4 and 160.5 «160» (half to even), 160.6 «161»
    want = {99.4: BELOW, 99.5: NEUTRAL, 100: NEUTRAL, 160: NEUTRAL, 160.4: NEUTRAL, 160.5: NEUTRAL, 160.6: ABOVE}
    for wpm, state in want.items():
        assert scores.tempo_state(wpm) == state, wpm
        got = _tempo_texts(wpm)
        assert got == {"card": state, "stored": words[state], "characterization": lexicon[state], "page": state,
                       "pdf": state}, (wpm, got)
        assert f"({wpm:.0f} слов" in speech_stats.describe(_speech(wpm))
    # the band itself moves every one of them
    with _patched(scores, TEMPO_BAND=(120, 170)):
        assert _tempo_texts(115.0) == {"card": BELOW, "stored": "медленный", "characterization": "slow",
                                       "page": BELOW, "pdf": BELOW}
        assert _tempo_texts(165.0)["characterization"] == "calm" and _tempo_texts(171.0)["pdf"] == ABOVE


def test_the_stored_sentence_of_an_old_job():
    """A job analysed before the one band (the tempo words on 110…160) stored «медленный» for 99.8 words per minute,
    while its card says «100 слов в минуту» in the neutral colour; the page and the PDF rebuild the tempo clause from
    the stored number and keep the rest of the stored sentence (here in the wording of 2.0, «пауз умеренно»)."""
    old = ("Темп речи медленный (100 слов в минуту); пауз умеренно (17% времени); слов-заполнителей почти нет; "
           "средняя фраза 12 слов.")
    shown = ("Темп речи спокойный (100 слов в минуту); пауз умеренно (17% времени); слов-заполнителей почти нет; "
             "средняя фраза 12 слов.")
    sp = _speech(99.8, description=old)
    view = {"analyses": {"speech": sp}, "transcript": ""}
    assert analyses_text.speech_description(sp) == shown
    assert f">{shown}</p>" in page._speech_html(view)
    assert analyses_text.analyses_parts(view)["speech"] == shown
    assert facts.key_facts(view)[0][1:2] == ("100 слов в минуту",) and facts.key_facts(view)[0][3] == NEUTRAL
    assert sp["description"] == old                                     # result.json is not rewritten
    # a new analysis writes the same sentence as the one shown
    assert speech_stats.describe(_speech(99.8)) == shown.replace("пауз умеренно", "паузы умеренные")
    # the counts still get their form, and a sentence without a tempo clause or a job without a tempo stays as stored
    assert analyses_text.speech_description(_speech(92.4, "Темп речи медленный (92 слов в минуту).")) == \
        "Темп речи медленный (92 слова в минуту)."
    assert analyses_text.speech_description(_speech(150.2, "Речи в этом отрезке почти нет.")) == \
        "Речи в этом отрезке почти нет."
    assert analyses_text.speech_description(_speech(None, old)) == old
    assert analyses_text.speech_description({}) == ""
