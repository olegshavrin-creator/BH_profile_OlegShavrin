"""Speech analytics from the Whisper transcript with timestamps: no models, language-independent apart from
the filler-word lists. Works on the whole video and on each segment.

  words_per_min   speaking rate over speech time (pauses excluded) and over wall time
  pause_share     share of the segment spent in pauses longer than `pause_min` seconds between Whisper chunks
  fillers_per_100 filler words («ну», «вот», «как бы», «типа», «значит»; um, uh, like, you know) per 100 words
  mean_sentence   average sentence length in words
  ttr             type/token ratio (vocabulary richness, depends on length: compare only equal-length texts)
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

from ..analyses_text import tempo_clause
from ..bands import FILLER_WORDS, PAUSE_WORDS
# the words of a transcript are split the same way for the numbers here and for the frequent words the page
# shows; vocabulary is read from here by the pipeline, which stores it with the other speech numbers
from ..words import spoken_words, vocabulary  # noqa: F401

FILLERS = {
    "ru": ["ну", "вот", "как бы", "типа", "значит", "это самое", "в общем", "короче", "так сказать", "собственно",
           "э-э", "эм", "ммм", "в принципе", "получается", "то есть"],
    "en": ["um", "uh", "like", "you know", "i mean", "sort of", "kind of", "basically", "actually", "so", "well", "right"],
}
_SENT = re.compile(r"[.!?…]+")


def _count_fillers(text: str, lang: str) -> int:
    t = " " + re.sub(r"\s+", " ", (text or "").lower()) + " "
    n = 0
    for f in FILLERS.get(lang, []) + (FILLERS["en"] if lang != "en" else []):
        n += len(re.findall(rf"(?<![\w-]){re.escape(f)}(?![\w-])", t))
    return n


def stats_for(chunks: List[Tuple[float, float, str]], start: float, end: float, lang: str = "ru",
              pause_min: float = 0.5) -> Dict:
    """chunks: (start, end, text) from Whisper; the span [start, end) selects a segment (or the whole video)."""
    sel, rate_words = [], 0.0
    for s, e, t in chunks:
        if e > start and s < end and t.strip():
            a, b = max(s, start), min(e, end)
            sel.append((a, b, t))
            # a Whisper chunk that only partly overlaps the segment contributes the same share of its words to the
            # tempo; counting all its words over the overlapped seconds gave 300-500 words/min on short overlaps
            rate_words += len(spoken_words(t)) * (max(0.0, b - a) / max(1e-6, e - s))
    text = " ".join(t for _, _, t in sel)
    words = spoken_words(text)
    speech = sum(max(0.0, e - s) for s, e, _ in sel)
    wall = max(1e-6, end - start)
    pauses = [sel[i + 1][0] - sel[i][1] for i in range(len(sel) - 1)]
    pause_time = sum(p for p in pauses if p >= pause_min)
    if sel:
        pause_time += max(0.0, sel[0][0] - start) if sel[0][0] - start >= pause_min else 0.0
        pause_time += max(0.0, end - sel[-1][1]) if end - sel[-1][1] >= pause_min else 0.0
    sentences = [s for s in _SENT.split(text) if spoken_words(s)]
    n = len(words)
    fill = _count_fillers(text, lang)
    return {
        "words": n,
        "speech_sec": round(speech, 1),
        # tempo needs at least 3 s of speech, otherwise a couple of words give a meaningless rate
        "words_per_min_speech": round(60.0 * rate_words / speech, 1) if speech >= 3 else None,
        "words_per_min_wall": round(60.0 * rate_words / wall, 1),
        "pause_share": round(min(1.0, pause_time / wall), 3),
        "long_pauses": sum(1 for p in pauses if p >= 2.0),
        "fillers": fill,
        "fillers_per_100": round(100.0 * fill / n, 1) if n else 0.0,
        "mean_sentence": round(n / len(sentences), 1) if sentences else None,
        "ttr": round(len(set(words)) / n, 3) if n else None,
        "unique_words": len(set(words)),
    }


def describe(st: Dict) -> str:
    """One readable sentence about the manner of speech. The tempo clause (analyses_text.tempo_clause) takes its word
    from the one tempo band of the card colour, scores.TEMPO_BAND."""
    if not st or not st.get("words"):
        return "Речи в этом отрезке почти нет."
    parts = []
    wpm = st.get("words_per_min_speech")
    if wpm:
        parts.append(tempo_clause(wpm))
    if st.get("pause_share") is not None:
        p = st["pause_share"] * 100
        few, many = (100 * x for x in PAUSE_WORDS)          # in percent, as p: 10 and 25
        parts.append("пауз мало" if p < few else (f"паузы умеренные ({p:.0f}% времени)" if p < many
                                                  else f"много пауз ({p:.0f}% времени)"))
    f = st.get("fillers_per_100", 0)
    few, many = FILLER_WORDS
    parts.append("слов-заполнителей почти нет" if f < few else (f"слова-заполнители встречаются ({f:.0f} на 100 слов)"
                                                                if f < many
                                                                else f"много слов-заполнителей ({f:.0f} на 100 слов)"))
    if st.get("mean_sentence"):
        parts.append(f"средняя фраза {st['mean_sentence']:.0f} слов")
    return "; ".join(parts).capitalize() + "."
