"""The short phrase of the video-language model under every key frame of AMLAI 1.0: the prompt, its limits, the
validation of the answer, and the names of the saved key-frame files.

The explanation code asks for the phrase (backend_mm.describe_frame, mm/explain.key_frame_info) and the captions of
the page and the PDF read it back (frame_captions), so both sides take it from here. A phrase is kept only if it stays
a short list of visible things — no verdicts about character or mood, no digits, no sentence. The time limits of the
requests are settings (settings.PHRASE_TIMEOUT, settings.PHRASE_BUDGET).

Standard library only, no bs3 import: a text module may import it without loading the models (tests/test_layers.py).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

FRAME_PROMPT = (
    "Опиши по-русски только то, что видно на этом одном кадре: выражение лица, направление взгляда, положение "
    "головы и плеч, руки. От трёх до шести слов, одной короткой фразой, без точки в конце, без имён, без цифр. "
    "Нельзя писать о характере, настроении или впечатлении («уверенный», «дружелюбный», «нервничает»), нельзя "
    "описывать фон и обстановку — только человека."
)
PHRASE_MAX = 40                     # characters, after trimming at the last comma
PHRASE_NUM_PREDICT = 48             # the answer is a few words: a short budget keeps the five requests fast
PHRASE_TRIES = 2                    # one retry, then the caption falls back to the expression
# Verdicts about character or mood the phrase must not contain (the prompt forbids them; this is the safety net).
# The seven names of the expression classes («радость», «грусть», «страх», «злость», «отвращение», «удивление»,
# «нейтрально») are not verdicts: they are what the expression model itself reports, and the caption may repeat them.
PHRASE_VERDICTS = (
    "увере", "неувер", "дружелюб", "нервн", "тревож", "застенчив", "стеснит", "общитель", "замкнут", "харизм",
    "искренн", "агрессив", "враждеб", "доброжелат", "скучающ", "взволнован", "раздраж", "настроен", "эмоционал",
    "характер", "личност", "впечатлен", "похоже", "вероятно", "возможно",
    "спокой", "опустош", "подавлен", "угнет", "задумчив", "устал", "решительн", "мрачн",
)
PHRASE_BANNED = ("кажется", "выглядит как")
PHRASE_WORDS = (3, 6)               # the prompt asks for three to six words; anything else is not that phrase
# The report is written with ё, so a caption must be too. The model answers with a small stock of words about a
# face, a gaze and a posture: these are the ones of that stock that carry ё.
PHRASE_YO = {
    "вперед": "вперёд", "еще": "ещё", "щеки": "щёки", "щек": "щёк", "челка": "чёлка",
    "повернут": "повёрнут", "повернута": "повёрнута", "повернуто": "повёрнуто", "повернуты": "повёрнуты",
    "развернут": "развёрнут", "развернута": "развёрнута", "развернуто": "развёрнуто", "развернуты": "развёрнуты",
    "отведен": "отведён", "наклонен": "наклонён", "сведен": "сведён", "зачесан": "зачёсан", "зачесаны": "зачёсаны",
    "легкая": "лёгкая", "легкий": "лёгкий", "легкое": "лёгкое",
    "темный": "тёмный", "темная": "тёмная", "темные": "тёмные",
    "черный": "чёрный", "черная": "чёрная", "черные": "чёрные",
}
_RE_YA = re.compile(r"(?<![а-яёa-z])я(?![а-яёa-z])", re.I)
_RE_DIGIT = re.compile(r"\d")
_RE_WORD = re.compile(r"[А-Яа-яЁё]+")


def _yo(s: str) -> str:
    """«смотрит вперед» -> «смотрит вперёд»: the few words of a frame description that are written with ё."""
    def one(m: re.Match) -> str:
        w = m.group(0)
        r = PHRASE_YO.get(w.lower())
        if r is None:
            return w
        return r[:1].upper() + r[1:] if w[:1].isupper() else r
    return _RE_WORD.sub(one, s)


def clean_phrase(raw: str | None) -> Optional[str]:
    """The model's answer as a caption phrase, or None when it must be dropped.

    Quotes are stripped, a too long answer is trimmed at its last comma; the phrase is dropped when it is still
    longer than PHRASE_MAX, is not the three to six words the prompt asks for, or contains a digit, a line break,
    the pronoun «я», «кажется», «выглядит как» or any verdict about character or mood. What is kept is spelled
    with ё, like the rest of the report."""
    if raw is None:
        return None
    s = str(raw)
    if "\n" in s.strip().strip("«»\"'“”„ "):
        return None                                   # a whole answer in several lines is not a short phrase
    s = " ".join(s.split())
    s = s.strip().strip("«»\"'“”„ ").strip()
    s = s.rstrip(" .!?;:—-").strip()
    s = s.strip("«»\"'“”„ ").strip()
    if not s:
        return None
    while len(s) > PHRASE_MAX and "," in s:
        s = s.rsplit(",", 1)[0].strip().rstrip(" .!?;:—-").strip()
    if not s or len(s) > PHRASE_MAX:
        return None
    if not PHRASE_WORDS[0] <= len(s.split()) <= PHRASE_WORDS[1]:
        return None                                   # «улыбается» alone is not a description of the frame
    if _RE_DIGIT.search(s):
        return None
    low = s.lower()
    if _RE_YA.search(low) or any(b in low for b in PHRASE_BANNED) or any(v in low for v in PHRASE_VERDICTS):
        return None
    if s[:1].isupper() and s[:1].isalpha():
        s = s[:1].lower() + s[1:]
    return _yo(s)


# ---------------------------------------------------------------- the names of the saved key frames
def _frame_index(path: str) -> Optional[int]:
    """The index of the frame inside the sampled sequence, from the file name `key_<i>_frame<N>.jpg`."""
    m = re.match(r"^\w+?_(\d+)_frame\d+$", Path(path).stem)
    return int(m.group(1)) if m else None


def _decoded_index(path: str) -> Optional[int]:
    """The number of the frame in the clip, from the file name `key_<i>_frame<N>.jpg`."""
    m = re.search(r"_frame(\d+)$", Path(path).stem)
    return int(m.group(1)) if m else None
