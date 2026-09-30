"""Plain-language sentences for the BS Profiler 3.1 analyses (emotions, voice, face, speech): deterministic templates
over the numbers in result.json. The page shows the text emotion and the voice ones in «Эмоции и голос: коротко»
(mbti_html.emo_intro_html); the PDF puts every topic into its section. The sentence about the manner of speech is
written at analysis time (analyses/speech_stats.describe, which takes its tempo clause from here) and shown through
`speech_description`."""
from __future__ import annotations

import re
from typing import List

from .facts import head_motion_word
from .labels import EMO_RU
from .scores import ABOVE, BELOW, NEUTRAL, num, tempo_state
from .textfmt import fix_counts, plural_ru

# «уверенность низкая», «возбуждение низкое»: the level agrees with the gender of the voice dimension
_LEVELS = {"n": ("низкое", "среднее", "высокое"), "f": ("низкая", "средняя", "высокая")}
# emotion names inside sentences: «нейтрально» is an adverb and does not fit after «преобладает» or «как»
_TEXT_EMO = {"neutral": "нейтральный тон"}
_FACE_EMO = {"neutral": "нейтральное"}
# the speech tempo in words, by the one tempo band of the card «Темп речи» of «Ключевые факты» (scores.tempo_state,
# TEMPO_BAND 100…160 words per minute, on the whole number that is printed; owner, 2026-09-27)
TEMPO_RU = {BELOW: "медленный", NEUTRAL: "спокойный", ABOVE: "быстрый"}
# the tempo clause that opens a stored sentence about the manner of speech (speech_stats.describe)
_TEMPO_CLAUSE = re.compile(r"^Темп речи \S+ \(\d+ слов в минуту\)")


def tempo_clause(wpm: float) -> str:
    """«темп речи медленный (92 слов в минуту)», the first part of the sentence of speech_stats.describe: the word
    is decided by scores.tempo_state on the whole words per minute the clause prints, like the colour of the card."""
    return f"темп речи {TEMPO_RU[tempo_state(wpm)]} ({wpm:.0f} слов в минуту)"


def speech_description(sp: dict) -> str:
    """The sentence about the manner of speech as the page («Речь в цифрах») and the PDF (section «Голос и речь») show
    it: `analyses.speech.description`, written into result.json at analysis time, with the counts in their right
    form (fix_counts). Its tempo clause is rebuilt from the stored words per minute (tempo_clause), so a job analysed
    while the tempo words had a band of their own (110…160, before 2026-09-27) says what the colour of its tempo card
    says; the rest of the sentence stays as it was stored."""
    text = str(sp.get("description") or "")
    wpm = num(sp.get("words_per_min_speech"))
    if wpm:
        text = _TEMPO_CLAUSE.sub(lambda m: tempo_clause(wpm).capitalize(), text, count=1)
    return fix_counts(text)


def _level(v: float, gender: str = "n", low: float = 0.4, high: float = 0.6) -> str:
    lo, mid, hi = _LEVELS[gender]
    return lo if v < low else (hi if v > high else mid)


def analyses_parts(rep: dict) -> dict:
    """The sentences about each analysis, by topic: {text_emotion, voice, face, speech} ('' when the analysis is
    missing). The page shows the text emotion and the voice ones in «Эмоции и голос: коротко» (mbti_html.emo_intro_html);
    the PDF puts the topics into its sections."""
    an = rep.get("analyses") or {}
    out = dict.fromkeys(("text_emotion", "voice", "face", "speech"), "")
    te = an.get("emotions_text")
    if te and te.get("mean"):
        parts: List[str] = []
        m = te["mean"]
        top = sorted(m.items(), key=lambda kv: -kv[1])[:2]
        if top[0][0] == "neutral" and top[0][1] >= 0.6:
            s = f"По содержанию речи эмоциональная окраска в основном нейтральная ({top[0][1]:.0%})"
            if len(top) > 1 and top[1][1] >= 0.1:
                s += f", из выраженных эмоций заметнее всего {EMO_RU.get(top[1][0], top[1][0])} ({top[1][1]:.0%})"
            parts.append(s + ".")
        else:
            parts.append("По содержанию речи преобладает " + ", затем ".join(
                f"{_TEXT_EMO.get(k, EMO_RU.get(k, k))} ({v:.0%})" for k, v in top) + ".")
        dps = te.get("dominant_per_segment") or []
        changes = sum(1 for a, b in zip(dps, dps[1:]) if a and b and a != b)
        if len(dps) >= 4:
            parts.append("Эмоциональный тон речи " + (
                "ровный по всему ролику." if changes <= len(dps) // 4 else
                f"меняется по ходу ролика: преобладающая эмоция сменяется {changes} "
                f"{plural_ru(changes, 'раз', 'раза', 'раз')}."))
        out["text_emotion"] = " ".join(parts)
    vo = an.get("voice")
    if vo and vo.get("mean"):
        m = vo["mean"]
        a, d, v = m.get("arousal", 0.5), m.get("dominance", 0.5), m.get("valence", 0.5)
        out["voice"] = (f"Голос (модель эмоций в речи, шкала 0…1): возбуждение {_level(a)} ({a:.2f}), уверенность "
                        f"{_level(d, 'f')} ({d:.2f}), позитивность {_level(v, 'f')} ({v:.2f}). Модель обучена на англоязычных "
                        "записях, поэтому значения относительные: полезнее сравнивать отрезки между собой.")
    fa = an.get("face")
    if fa and fa.get("mean"):
        m = fa["mean"]
        top = sorted(m.items(), key=lambda kv: -kv[1])[:2]
        s = "Выражение лица чаще всего распознаётся как " + ", реже — как ".join(
            f"{_FACE_EMO.get(k, EMO_RU.get(k, k))} ({v:.0%} кадров)" if i == 0 else
            f"{_FACE_EMO.get(k, EMO_RU.get(k, k))} ({v:.0%})" for i, (k, v) in enumerate(top))
        if fa.get("head_motion") is not None:
            s += "; голова " + head_motion_word(fa["head_motion"], "head")
        if fa.get("face_share") is not None and fa["face_share"] < 0.9:
            s += f"; лицо видно в {fa['face_share']:.0%} кадров"
        out["face"] = s + "."
    sp = an.get("speech")
    if sp and sp.get("description"):
        out["speech"] = speech_description(sp)
    return out
