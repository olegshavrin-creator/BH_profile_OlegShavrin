"""What the page and the PDF of BS Profiler 3.1 both show, built once, so the two never say it two ways: the cards of
«Ключевые факты» (`fact_cards`, `key_facts`), the nine cards of «Речь в цифрах» (`speech_cards`), the cells of one
segment in the page's table «Эмоции, голос и темп по отрезкам» and the PDF's appendix «Значения по отрезкам»
(`segment_cells`), the word for the head motion (`head_motion_word`) and the note about the model of facial
expressions (FER_NOTE). Deterministic templates over the numbers in result.json; where the page and the PDF
deliberately differ, the difference is a parameter.

The sentences about each analysis are analyses_text.analyses_parts; the thresholds behind the words are in bands."""
from __future__ import annotations

from typing import List

from .bands import HEAD_MOTION_WORDS
from .labels import EMO_RU, EMOTION_ORDER
from .mbti import fact_card
from .scores import num
from .segments import dominant_emotion, empty_text, seg_words
from .textfmt import fmt_secs, plural_ru


# «Ключевые факты»: where the value sits is said twice — by the colour of the value and by this word in the label
# line under it. The word is not decoration. Three ink colours that all keep 4.5:1 on a light card cannot be told
# apart in a black-and-white print (the uncoloured cards are black, so the three states would have to fit between
# black and the lightest ink that still contrasts), and a colour-blind reader loses green against orange; the word
# carries the meaning in both cases. The words are the ones the legend under the grid uses.
FACT_STATE_RU = {"neutral": "около нейтрального", "below": "ниже", "above": "выше"}
# the one line under the card grid, on the page and in the PDF: what the colour and the word mean
FACTS_LEGEND = ("Цвет показателя и слово в подписи под ним говорят одно и то же: зелёный — около нейтрального, "
                "синий — ниже, оранжевый — выше.")


def fact_label(lab: str, state: str | None) -> str:
    """The label line of a key-fact card: «Голос, шкала 0…1 · ниже». Without a state («Тип MBTI», «Длительность
    ролика», the other card grids) the label is printed as it is."""
    word = FACT_STATE_RU.get(state or "")
    return f"{lab} · {word}" if word else str(lab)


def card_item(item) -> tuple:
    """(label, value, note, state) of one card: the key facts carry a state, the other card grids («Речь в цифрах»,
    «Лицо») three fields and no state."""
    lab, val, note, *rest = item
    return lab, val, note, (rest[0] if rest else None)


def key_facts(rep: dict) -> List[tuple]:
    """(label, value, note, state) cards for the overview tab. Every number carries its unit or scale; counts use the
    correct Russian plural; rounding matches the «Речь» tab (whole words per minute, whole fillers per 100 words).

    `state` is where the value sits — "neutral", "below" or "above" (scores.scale_state, tempo_state, emotion_state) —
    and the card renderers paint the value by it (palette.FACT_VALUE). It is None on a card that is not a measurement
    («Длительность ролика») and on one whose value is missing, and such a value keeps the plain text colour."""
    from .scores import emotion_state, scale_state, scored, tempo_state
    an = rep.get("analyses") or {}
    facts = []
    te = an.get("emotions_text")
    if te and te.get("mean"):
        k, v = max(te["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Эмоция по тексту речи", EMO_RU.get(k, k), f"{v:.0%} времени", emotion_state(k)))
    fa = an.get("face")
    if fa and fa.get("mean"):
        k, v = max(fa["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Выражение лица", EMO_RU.get(k, k), f"{v:.0%} кадров", emotion_state(k)))
    vo = an.get("voice")
    if vo and vo.get("mean"):
        m = vo["mean"]
        # the value of the card is the arousal, so the colour is the arousal's (the two other dimensions are the note).
        # The word «возбуждение» belongs to the label, not to the value: as part of the value it was a 11-letter word
        # in a 150 px card and the browser broke it in the middle («возбужден / ие 0.36»)
        facts.append(("Голос: возбуждение, шкала 0…1", f"{m.get('arousal', 0):.2f}",
                      f"уверенность {m.get('dominance', 0):.2f} · позитивность {m.get('valence', 0):.2f}",
                      scale_state(m.get("arousal", 0))))
    sp = an.get("speech")
    if sp and sp.get("words"):
        wpm = sp.get("words_per_min_speech")
        if wpm:
            n = int(round(wpm))
            value = f"{n} {plural_ru(n, 'слово', 'слова', 'слов')} в минуту"
        else:
            value = f"{sp['words']} {plural_ru(sp['words'], 'слово', 'слова', 'слов')}"
        fillers = int(round(sp.get("fillers_per_100", 0) or 0))
        # without a tempo the card shows the word count instead, and a word count has no band: no colour
        facts.append(("Темп речи", value, f"паузы — {sp.get('pause_share', 0):.0%} времени, "
                                          f"заполнители — {fillers} на 100 слов",
                      tempo_state(wpm) if wpm else None))
    iv = scored(rep.get("interview"))          # an entry without a numeric score is not shown
    if iv:
        facts.append(("Впечатление «собеседование»", f"{float(iv['score']):.2f}", "шкала 0…1, модель AMLAI 1.0",
                      scale_state(iv["score"])))
    dur = rep.get("duration_sec")
    if dur:
        n = int(rep.get("segments") or 1)
        facts.append(("Длительность ролика", fmt_secs(dur),
                      f"разбит на {n} {plural_ru(n, 'отрезок', 'отрезка', 'отрезков')}" if n > 1 else "один отрезок",
                      None))
    return facts


def fact_cards(view: dict, mb: dict | None, speech_cards_follow: bool = False) -> tuple[list, bool]:
    """(cards, show_legend) of «Ключевые факты» on the page and in the PDF: the MBTI type card first (mbti.fact_card,
    design 10.2), then key_facts of the clean view; show_legend — whether the line FACTS_LEGEND goes under the grid,
    which is when at least one card is coloured (carries a state).

    `speech_cards_follow`: «Речь в цифрах» follows later in the same document (the PDF with a speech section). Not
    everything belongs in the report twice: the note of «Темп речи» then no longer repeats the pauses and the fillers
    printed there in full, it says what the tempo itself is counted on."""
    card = fact_card(mb)
    facts = key_facts(view)
    if speech_cards_follow:
        facts = [(lab, val, ("только время, когда человек говорит" if "минуту" in str(val) else "")
                            if lab == "Темп речи" else note, state) for lab, val, note, state in facts]
    items = ([card] if card else []) + facts
    return items, any(card_item(i)[3] for i in items)


def speech_cards(sp: dict, small_rate_words: bool) -> list:
    """The 9 cards (label, value, note) of «Речь в цифрах» from `analyses.speech`, the same on the page and in the PDF.

    `small_rate_words`: a filler rate above 0 and below 0.95 is said in words, «меньше 1 на 100 слов» (the PDF: «0 на
    100 слов» under a value of 3 contradicts itself); without it the note is always the rounded rate (the page)."""
    def whole(v):
        return "—" if v is None else f"{float(v):.0f}"

    def per_100(v) -> str:
        x = float(v or 0)
        return "меньше 1 на 100 слов" if small_rate_words and 0 < x < 0.95 else f"{x:.0f} на 100 слов"
    fillers = sp.get("fillers")
    return [("Слов всего", whole(sp.get("words")), ""),
            ("Разных слов", whole(sp.get("unique_words")), "без повторов"),
            ("Темп речи, слов в минуту", whole(sp.get("words_per_min_speech")), "только время, когда человек говорит"),
            ("Темп с учётом пауз, слов в минуту", whole(sp.get("words_per_min_wall")), "по всей длине ролика"),
            ("Доля пауз", f"{sp.get('pause_share', 0):.0%}", "паузы от 0.5 с, доля времени ролика"),
            ("Длинных пауз", whole(sp.get("long_pauses")), "дольше 2 секунд"),
            ("Слов-заполнителей", whole(fillers),
             per_100(sp.get("fillers_per_100")) if fillers is not None else ""),
            ("Слов во фразе", whole(sp.get("mean_sentence")), "в среднем"),
            ("Разнообразие словаря", f"{float(sp['ttr']):.0%}" if sp.get("ttr") is not None else "—",
             "доля разных слов среди всех; зависит от длины текста")]


def segment_emotion(r: dict | None, source: str) -> str:
    """The dominant emotion of one segment with its share, «нейтрально 99%», by speech (`source` "text") or by the
    face ("face"); only the seven text emotions are named, a face label comes aliased to them and any other label
    stays as it is. A segment whose own transcript came out empty (segments.empty_text) has no emotion by speech:
    «нет речи» when it has no words at all, «нет текста» when the transcript of the whole video still has words in
    it, never the model's «нейтрально 100%». «—» without data."""
    if not r:
        return "—"
    if source == "text" and empty_text(r):
        return "нет речи" if seg_words(r) == 0 else "нет текста"
    d = dominant_emotion(r, source)
    return f"{EMO_RU[d[0]] if d[0] in EMOTION_ORDER else d[0]} {d[1]:.0%}" if d else "—"


def segment_cells(r: dict | None) -> list:
    """The cells of one segment (a per_segment entry of the analyses) that the page's table «Эмоции, голос и темп по
    отрезкам» and the PDF's appendix «Значения по отрезкам» both print, in this order: the dominant emotion by speech
    and by the face (segment_emotion), arousal, dominance and valence of the voice, the tempo in words per minute and
    the share of pauses. A missing value is «—»; so is the tempo of a segment that has none (less than 3 s of
    speech, the stored tempo missing or 0), not «0» (owner, 2026-09-27)."""
    r = r or {}
    vo, sp = r.get("voice") or {}, r.get("speech") or {}
    wpm = num(sp.get("words_per_min_speech"))
    return ([segment_emotion(r, "text"), segment_emotion(r, "face")]
            + [f"{float(vo[d]):.2f}" if vo.get(d) is not None else "—" for d in ("arousal", "dominance", "valence")]
            + [f"{wpm:.0f}" if wpm else "—", f"{float(sp.get('pause_share') or 0):.0%}" if sp else "—"])


# the head motion (analyses.face.head_motion, the shift between frames as a share of the face width) in words, weak,
# moderate, active by bands.HEAD_MOTION_WORDS, in the two forms the texts need
HEAD_MOTION_RU = {
    "head": ("почти неподвижна", "двигается умеренно", "двигается активно"),   # «…; голова почти неподвижна»
    "motion": ("слабое", "умеренное", "активное"),                            # «Движение головы слабое»
}


def head_motion_word(hm: float, form: str) -> str:
    """The head motion `hm` in words: form "head" completes «голова …» (the sentence about the face,
    analyses_text), form "motion" completes «Движение головы …» (the face card of the page, the caption of the PDF)."""
    low, mid, high = HEAD_MOTION_RU[form]
    lo, hi = HEAD_MOTION_WORDS
    return low if hm < lo else (mid if hm < hi else high)


# under the chart of the facial expressions, on the page and in the PDF; `where` is the page's pointer to the tab
# with the chart over time, « (вкладка «Таймлайн»)», and empty in the PDF
FER_NOTE = ("Модель выражений обучена на фотографиях FER-2013 и склонна видеть «грусть» и «страх» в спокойном лице: "
            "смотрите на изменения по ходу ролика{where}, а не на абсолютные доли.")
