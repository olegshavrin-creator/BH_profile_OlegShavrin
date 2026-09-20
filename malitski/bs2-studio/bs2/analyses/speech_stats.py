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
from collections import Counter
from typing import Dict, List, Tuple

FILLERS = {
    "ru": ["ну", "вот", "как бы", "типа", "значит", "это самое", "в общем", "короче", "так сказать", "собственно",
           "э-э", "эм", "ммм", "в принципе", "получается", "то есть"],
    "en": ["um", "uh", "like", "you know", "i mean", "sort of", "kind of", "basically", "actually", "so", "well", "right"],
}
_WORD = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё'\-]*")
_SENT = re.compile(r"[.!?…]+")


def _words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text or "")]


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
            rate_words += len(_words(t)) * (max(0.0, b - a) / max(1e-6, e - s))
    text = " ".join(t for _, _, t in sel)
    words = _words(text)
    speech = sum(max(0.0, e - s) for s, e, _ in sel)
    wall = max(1e-6, end - start)
    pauses = [sel[i + 1][0] - sel[i][1] for i in range(len(sel) - 1)]
    pause_time = sum(p for p in pauses if p >= pause_min)
    if sel:
        pause_time += max(0.0, sel[0][0] - start) if sel[0][0] - start >= pause_min else 0.0
        pause_time += max(0.0, end - sel[-1][1]) if end - sel[-1][1] >= pause_min else 0.0
    sentences = [s for s in _SENT.split(text) if _words(s)]
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


def describe(st: Dict, lang: str = "ru") -> str:
    """One readable sentence about the manner of speech."""
    if not st or not st.get("words"):
        return "Речи в этом отрезке почти нет."
    parts = []
    wpm = st.get("words_per_min_speech")
    if wpm:
        tempo = "быстрый" if wpm > 160 else ("спокойный" if wpm >= 110 else "медленный")
        parts.append(f"темп речи {tempo} ({wpm:.0f} слов в минуту)")
    if st.get("pause_share") is not None:
        p = st["pause_share"] * 100
        parts.append("пауз мало" if p < 10 else (f"паузы умеренные ({p:.0f}% времени)" if p < 25 else f"много пауз ({p:.0f}% времени)"))
    f = st.get("fillers_per_100", 0)
    parts.append("слов-заполнителей почти нет" if f < 2 else (f"слова-заполнители встречаются ({f:.0f} на 100 слов)" if f < 6
                                                              else f"много слов-заполнителей ({f:.0f} на 100 слов)"))
    if st.get("mean_sentence"):
        parts.append(f"средняя фраза {st['mean_sentence']:.0f} слов")
    return "; ".join(parts).capitalize() + "."


# English speech: function words beyond the attribution stop list (words.STOP_EN) that reach the top of a transcript
STOP_EN_SPEECH = set("""
also another around away back cause could does doing done down each else even ever every from gonna gotta have here
into itself just kinda like maybe mean might much must never okay once only other ours over quite really should since
some something sure than that them then there these they thing things this those though through till today unless
until upon very wanna were what whatever when where whether which while whom whose will with within would yeah your
yours about above after again against almost because before being below between during either enough further
""".split())


def _en_content(w: str) -> str | None:
    """English token -> content word or None: contractions (it's, don't, we're) are function words, a possessive 's
    is cut off (company's -> company)."""
    from ..words import STOP_EN
    w = w.replace("’", "'").strip("'-")
    if "n't" in w:
        return None
    if w.endswith("'s"):
        w = w[:-2]
    if "'" in w:
        return None
    return None if (len(w) <= 3 or w in STOP_EN or w in STOP_EN_SPEECH) else w


def vocabulary(text: str, top: int = 12, lang: str = "ru") -> List[Tuple[str, int]]:
    """Most frequent content words (short/function words and fillers removed)."""
    if lang == "en":
        return Counter(c for c in map(_en_content, _words(text)) if c).most_common(top)
    stop = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от меня
    еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя
    ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без будто чего раз тоже себе под будет ж тогда
    кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда
    можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
    впрочем хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между это которые который
    очень the a an and or of to in on at for with is are was were be it this that i you he she we they my your
    самый самая самое самые самого самой самому самым самом самых самыми свой своя своё свое свои своего своей своему
    своим своём своем своих своими числе включая также которая которое которого которой которому котором которым
    которых которыми которую этот этих этим этими этому такая такое такие такого таких таким такими какие каких какое
    каким какую всех всем всеми весь вся всё просто именно лишь нужно будут буду будем была были было могу может могут
    можем хотя тоже ещё чтобы между вообще сейчас потому поэтому где-то что-то кто-то наш наша наше наши нашего нашей
    наших нашим мои моих моей моего ваш ваша ваши него неё нему ними""".split())
    words = [w for w in _words(text) if len(w) > 3 and w not in stop]
    return Counter(words).most_common(top)
