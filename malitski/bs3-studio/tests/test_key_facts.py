"""«Ключевые факты» of BS Profiler 3.1: the card reads value, label, explanation, and the value of a measured card is
coloured by where it sits — green around neutral, blue below, dark orange above (change request «Ключевые факты» of
2026-09-26). Here: the three states of every card type at the edges of their bands, the cards that are never coloured
(«Тип MBTI», «Длительность ролика»), the row order on the page and in the PDF, the state word that repeats the colour
in the label line (so the card reads in a black-and-white print), and the line under the grid."""
from __future__ import annotations

import re

from bs3 import facts, palette, scores
from bs3.facts import card_item, key_facts
from bs3.web import page
from bs3.scores import ABOVE, BELOW, NEUTRAL, emotion_state, scale_state, tempo_state


def _rep(*, emotion="neutral", face="neutral", arousal=0.50, wpm=120.0, interview=0.50, words=300) -> dict:
    """A result.json-like dict with just the fields key_facts reads."""
    rep = {"duration_sec": 660.0, "segments": 33,
           "analyses": {"emotions_text": {"mean": {emotion: 0.9, "joy": 0.05}},
                        "face": {"mean": {face: 0.8, "joy": 0.1}},
                        "voice": {"mean": {"arousal": arousal, "dominance": 0.40, "valence": 0.49}},
                        "speech": {"words": words, "words_per_min_speech": wpm, "pause_share": 0.21,
                                   "fillers_per_100": 2.0}}}
    if interview is not None:
        rep["interview"] = {"score": interview}
    return rep


def _by_label(rep: dict) -> dict:
    return {card_item(f)[0]: card_item(f) for f in key_facts(rep)}


# --------------------------------------------------------------------------------- the three states of each card

def test_scale_state_edges_follow_the_level_bands():
    assert [scale_state(v) for v in (0.00, 0.20, 0.35)] == [BELOW, BELOW, BELOW]
    assert [scale_state(v) for v in (0.36, 0.50, 0.64)] == [NEUTRAL, NEUTRAL, NEUTRAL]
    assert [scale_state(v) for v in (0.65, 0.80, 1.00)] == [ABOVE, ABOVE, ABOVE]
    assert scale_state(None) is None and scale_state("нет") is None
    # the colour never contradicts the words the page prints for the same number
    for v in (0.35, 0.36, 0.64, 0.65):
        assert (scores.level(v) == "mid") == (scale_state(v) == NEUTRAL)
    # decided on the printed two decimals, as the levels are
    assert scale_state(0.6449) == NEUTRAL and scale_state(0.6451) == ABOVE


def test_tempo_state_edges():
    assert scores.TEMPO_BAND == (100, 160)
    assert [tempo_state(n) for n in (40, 99)] == [BELOW, BELOW]
    assert [tempo_state(n) for n in (100, 130, 160)] == [NEUTRAL, NEUTRAL, NEUTRAL]
    assert [tempo_state(n) for n in (161, 250)] == [ABOVE, ABOVE]
    assert tempo_state(None) is None
    assert tempo_state(99.6) == NEUTRAL and tempo_state(160.4) == NEUTRAL      # on the whole number the card prints
    assert [tempo_state(x) for x in (99.4, 99.5, 160.5, 160.6)] == [BELOW, NEUTRAL, NEUTRAL, ABOVE]   # half to even


def test_emotion_state_of_every_class_and_of_the_face_labels():
    assert emotion_state("neutral") == NEUTRAL
    for k in ("sadness", "fear", "disgust"):
        assert emotion_state(k) == BELOW, k
    for k in ("joy", "surprise", "anger"):
        assert emotion_state(k) == ABOVE, k
    # the face expressions share palette.EMO_ALIAS with the text emotions
    for face, text in palette.EMO_ALIAS.items():
        assert emotion_state(face) == emotion_state(text), face
    assert emotion_state("что-то ещё") is None


# --------------------------------------------------------------------------------- the cards themselves

def test_key_facts_states_of_every_card():
    cards = _by_label(_rep(emotion="sadness", face="joy", arousal=0.70, wpm=90.0, interview=0.30))
    assert card_item(cards["Эмоция по тексту речи"])[3] == BELOW
    assert card_item(cards["Выражение лица"])[3] == ABOVE
    assert card_item(cards["Голос: возбуждение, шкала 0…1"])[3] == ABOVE
    assert card_item(cards["Темп речи"])[3] == BELOW
    assert card_item(cards["Впечатление «собеседование»"])[3] == BELOW
    cards = _by_label(_rep(emotion="neutral", face="neutral", arousal=0.50, wpm=130.0, interview=0.50))
    for lab in ("Эмоция по тексту речи", "Выражение лица", "Голос: возбуждение, шкала 0…1", "Темп речи",
                "Впечатление «собеседование»"):
        assert cards[lab][3] == NEUTRAL, lab
    cards = _by_label(_rep(emotion="joy", face="anger", arousal=0.65, wpm=161.0, interview=0.65))
    for lab in ("Эмоция по тексту речи", "Выражение лица", "Голос: возбуждение, шкала 0…1", "Темп речи",
                "Впечатление «собеседование»"):
        assert cards[lab][3] == ABOVE, lab


def test_the_cards_that_are_never_coloured():
    from bs3 import mbti
    cards = _by_label(_rep())
    # «Длительность ролика» is not a measurement of the person
    assert cards["Длительность ролика"][3] is None
    # «Тип MBTI» comes from mbti.fact_card, which carries three fields and no state
    mb = {"type": "ENFJ", "type_strict": "ENFJ", "type_name": "Наставник", "x_count": 0, "model": "mm"}
    card = mbti.fact_card(mb)
    assert len(card) == 3 and card_item(card)[3] is None


def test_a_value_without_a_band_is_not_coloured():
    rep = _rep(wpm=None)
    cards = _by_label(rep)
    assert cards["Темп речи"][1].endswith("слов") and cards["Темп речи"][3] is None   # a word count has no band
    rep = _rep()
    rep.pop("interview")
    assert "Впечатление «собеседование»" not in _by_label(rep)


# --------------------------------------------------------------------------------- the page

def _divs(card_html: str) -> list:
    return re.findall(r"<div(?: class='[^']*')? style='(?:font-size:20px|font-size:13px)[^']*'>([^<]*)</div>",
                      card_html)


def test_page_card_reads_value_label_explanation():
    html = page._cards([("Темп речи", "90 слов в минуту", "паузы — 21% времени", BELOW)], value_first=True)
    assert _divs(html) == ["90 слов в минуту", "Темп речи · ниже", "паузы — 21% времени"]
    assert "class='bs3-fact-below'" in html
    # a card without a note is two rows, and an uncoloured card gets no word
    two = page._cards([("Длительность ролика", "11:00", "", None)], value_first=True)
    assert _divs(two) == ["11:00", "Длительность ролика"] and "bs3-fact-" not in two
    # the other card grids of the page keep label, value, note
    plain = page._cards([("Слов всего", "300", "без повторов")])
    assert _divs(plain) == ["Слов всего", "300", "без повторов"] and "bs3-fact" not in plain
    # a value longer than the card («возбуждение 0.30» in a 150 px card) wraps instead of painting over the border
    assert "overflow-wrap:anywhere" in page.CARD_VALUE


def test_the_state_is_said_in_a_word_as_well_as_in_colour():
    """The colour alone does not survive a black-and-white print or a colour-blind reader, so every coloured card
    repeats its state in the label line, in the words of the legend."""
    assert facts.FACT_STATE_RU == {NEUTRAL: "около нейтрального", BELOW: "ниже", ABOVE: "выше"}
    for state, word in facts.FACT_STATE_RU.items():
        assert facts.fact_label("Голос: возбуждение, шкала 0…1", state) == f"Голос: возбуждение, шкала 0…1 · {word}"
        assert word in facts.FACTS_LEGEND, word
    assert facts.fact_label("Тип MBTI · AMLAI 1.0", None) == "Тип MBTI · AMLAI 1.0"
    # the page prints the word of every state it shows, and the PDF prints the same label
    html = page._facts_html(_rep(emotion="joy", arousal=0.20, wpm=130.0))
    for word in facts.FACT_STATE_RU.values():
        assert f" · {word}</div>" in html, word
    drawn = [t for t, _ in _draw_cards([("Голос: возбуждение, шкала 0…1", "возбуждение 0.20", "", BELOW)], cols=1,
                                       value_first=True)]
    assert drawn == ["возбуждение 0.20", "Голос: возбуждение, шкала 0…1 · ниже"]


def test_facts_block_carries_the_colours_and_the_line_under_the_grid():
    html = page._facts_html(_rep(emotion="joy", arousal=0.20, wpm=200.0))
    for state in scores.FACT_STATES:
        assert f"div.bs3-facts .bs3-fact-{state}{{color:{palette.FACT_VALUE['light'][state]}}}" in html
        assert f".dark div.bs3-facts .bs3-fact-{state}{{color:{palette.FACT_VALUE['dark'][state]}}}" in html
    assert "class='bs3-fact-above'" in html and "class='bs3-fact-below'" in html
    # the coloured values sit inside the grid the rules are scoped to
    assert html.index("<div class='bs3-facts' style='display:grid") < html.index("class='bs3-fact-")
    assert html.count("class='bs3-facts'") == 1
    assert facts.FACTS_LEGEND in html and "зелёный — около нейтрального" in html
    # the line belongs under the grid, not above it
    assert html.index(facts.FACTS_LEGEND) > html.rindex("<div style='padding:10px")
    assert page._facts_html({}) == ""


# Gradio 5.8 paints everything inside an HTML block with the body text colour by this rule (measured with headless
# Chrome on the preview page, CSS.getMatchedStylesForNode, 2026-09-28). Before stage 14b the light rule of a key fact
# was one class, (0,1,0), and lost to it: in the light theme every value was the body text colour. The dark rule,
# «.dark .bs3-fact-<state>», (0,2,0), won only by coming later in the document.
GRADIO_TEXT_RULE = ".gradio-container-5-8-0 .prose *"
OLD_DARK_RULE = ".dark .bs3-fact-neutral"


def _specificity(selector: str) -> tuple:
    """(ids, classes, types) of a selector of compound selectors joined by spaces or «>», with no attribute
    selectors or pseudo-classes (the test checks that FACTS_CSS has none)."""
    ids = classes = types = 0
    for part in re.split(r"[\s>]+", selector.strip()):
        if not part or part == "*":
            continue
        ids += part.count("#")
        classes += part.count(".")
        types += bool(re.match(r"[A-Za-z]", part))
    return ids, classes, types


def test_the_light_colour_outranks_gradio_and_the_dark_colour_outranks_the_light():
    """Stage 14b: in the light theme the key-fact values lost their colour to Gradio's prose rule. The light rule now
    has at least the specificity of the dark rule that always won (strictly more than Gradio's rule, so the order of
    the style sheets does not matter), and the dark rule still outranks the light one and comes after it."""
    assert _specificity(GRADIO_TEXT_RULE) == (0, 2, 0) == _specificity(OLD_DARK_RULE)
    css = page.FACTS_CSS
    assert css.startswith("<style>") and css.endswith("</style>")
    rules = re.findall(r"([^{}]+)\{color:([^{}]+)\}", css[len("<style>"):-len("</style>")])
    assert len(rules) == 2 * len(scores.FACT_STATES)
    assert not any(c in sel for sel, _ in rules for c in ":[")
    for state in scores.FACT_STATES:
        light = [(i, sel) for i, (sel, col) in enumerate(rules)
                 if col == palette.FACT_VALUE["light"][state] and sel.endswith(f".bs3-fact-{state}")]
        dark = [(i, sel) for i, (sel, col) in enumerate(rules)
                if col == palette.FACT_VALUE["dark"][state] and sel.endswith(f".bs3-fact-{state}")]
        assert len(light) == len(dark) == 1, state
        (i_light, s_light), (i_dark, s_dark) = light[0], dark[0]
        assert ".dark" not in s_light and s_dark.startswith(".dark "), state
        assert _specificity(s_light) >= _specificity(OLD_DARK_RULE), (state, s_light)
        assert _specificity(s_light) > _specificity(GRADIO_TEXT_RULE), (state, s_light)
        assert _specificity(s_dark) > _specificity(s_light) and i_dark > i_light, (state, s_dark)
        # both are scoped to the grid of the key facts, which the block HTML carries itself, so the colours also
        # read right outside the app (scripts/rerender_samples.py --html-dir)
        assert f".{page.FACTS_SCOPE} " in s_light and f".{page.FACTS_SCOPE} " in s_dark, state


# --------------------------------------------------------------------------------- the PDF

def _draw_cards(items, **kw):
    """(text, ink) of every line the card grid prints, ink as the (r, g, b) the card asked for."""
    from bs3.pdf.document import Report

    class Recorder(Report):
        _depth = 0                      # fpdf re-dispatches multi_cell to itself: record the outermost call only

        def __init__(self):
            super().__init__(file_label="t", total_pages=1)
            self.drawn, self.ink = [], (0, 0, 0)

        def set_text_color(self, r, g=None, b=None):
            self.ink = (r, r, r) if g is None else (r, g, b)
            super().set_text_color(r) if g is None else super().set_text_color(r, g, b)

        def multi_cell(self, *a, **kw):
            if not self._depth and not kw.get("dry_run"):
                text = kw.get("text", kw.get("txt", a[2] if len(a) > 2 else ""))
                self.drawn.append((str(text), self.ink))
            self._depth += 1
            try:
                return super().multi_cell(*a, **kw)
            finally:
                self._depth -= 1

    pdf = Recorder()
    pdf.add_page()
    pdf.cards(items, **kw)
    return pdf.drawn


def test_pdf_card_prints_the_value_first_and_in_colour():
    from bs3.pdf.document import NOTE_GREY
    drawn = _draw_cards([("Темп речи", "200 слов в минуту", "паузы — 21% времени", ABOVE)], cols=1, value_first=True)
    assert [t for t, _ in drawn] == ["200 слов в минуту", "Темп речи · выше", "паузы — 21% времени"]
    orange = tuple(int(palette.FACT_VALUE_PDF[ABOVE].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    assert drawn[0][1] == orange and drawn[1][1] == drawn[2][1] == (NOTE_GREY,) * 3
    # a card with no state keeps black ink, and «Речь в цифрах» keeps label, value, note
    plain = _draw_cards([("Длительность ролика", "11:00", "", None)], cols=1, value_first=True)
    assert [t for t, _ in plain] == ["11:00", "Длительность ролика"] and plain[0][1] == (0, 0, 0)
    speech = _draw_cards([("Слов всего", "300", "без повторов")], cols=1)
    assert [t for t, _ in speech] == ["Слов всего", "300", "без повторов"] and speech[1][1] == (0, 0, 0)


def test_pdf_card_height_counts_the_state_word_and_not_the_order():
    from bs3.pdf.document import Report
    pdf = Report(file_label="t", total_pages=1)
    pdf.add_page()
    items = [("Темп речи", "200 слов в минуту", "паузы — 21% времени", ABOVE),
             ("Длительность ролика", "11:00", "разбит на 33 отрезка", None)]
    # the page-break reservation measures the label the card really prints, state word and all
    expanded = [(facts.fact_label(lab, st), val, note) for lab, val, note, st in items]
    assert abs(pdf.cards_height(items, 2, value_first=True) - pdf.cards_height(expanded, 2)) < 1e-9
    # with the same labels either way, the two gaps of the card are the same in both orders
    plain = [(a, b, c) for a, b, c, _ in items]
    assert abs(pdf.cards_height(plain, 2, value_first=True) - pdf.cards_height(plain, 2)) < 1e-9
