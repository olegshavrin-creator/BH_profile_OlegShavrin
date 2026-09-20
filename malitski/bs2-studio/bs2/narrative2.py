"""Plain-language sentences for the BS 2.0 analyses (emotions, voice, face, speech), appended to the Big Five
narrative. Deterministic templates over the numbers in result.json."""
from __future__ import annotations

import re
from typing import List

from .charts import EMO_RU
from .report import fmt_secs
from .ru_texts import vocabulary_shown


def plural_ru(n, one: str, few: str, many: str) -> str:
    """Russian noun form for a count: plural_ru(21, "отрезок", "отрезка", "отрезков") -> "отрезок"."""
    n = abs(int(round(float(n))))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many


# texts stored in result.json by other modules say «92 слов в минуту», «31 сегментов»: fix the noun form on display
_COUNT_NOUNS = {"слов": ("слово", "слова", "слов"), "сегментов": ("отрезок", "отрезка", "отрезков"),
                "отрезков": ("отрезок", "отрезка", "отрезков")}
_GENITIVE_BEFORE = {"из", "до", "от", "около", "без", "для", "больше", "меньше", "более", "менее", "свыше"}


def fix_counts(text: str) -> str:
    """«92 слов в минуту» -> «92 слова в минуту», «(31 сегментов)» -> «(31 сегмент)». Only where the count is in the
    nominative/accusative; after «из», «до», «больше» … the genitive plural is correct and stays."""
    def repl(m):
        prev, n, word = m.group(1) or "", int(m.group(2)), m.group(3)
        if prev.strip().lower() in _GENITIVE_BEFORE:
            return m.group(0)
        return f"{prev}{n} {plural_ru(n, *_COUNT_NOUNS[word])}"
    return re.sub(r"(\b\w+\s)?(\d+)\s(слов|сегментов|отрезков)\b", repl, text or "")


# «уверенность низкая», «возбуждение низкое»: the level agrees with the gender of the voice dimension
_LEVELS = {"n": ("низкое", "среднее", "высокое"), "f": ("низкая", "средняя", "высокая")}
# emotion names inside sentences: «нейтрально» is an adverb and does not fit after «преобладает» or «как»
_TEXT_EMO = {"neutral": "нейтральный тон"}
_FACE_EMO = {"neutral": "нейтральное"}


def _level(v: float, gender: str = "n", low: float = 0.4, high: float = 0.6) -> str:
    lo, mid, hi = _LEVELS[gender]
    return lo if v < low else (hi if v > high else mid)


def analyses_parts(rep: dict) -> dict:
    """The sentences about each analysis, by topic: {text_emotion, voice, face, face_note, speech, vocab} ('' when the
    analysis is missing). The web page joins them into one paragraph (analyses_sentences); the PDF puts each topic
    into its own section."""
    an = rep.get("analyses") or {}
    out = dict.fromkeys(("text_emotion", "voice", "face", "face_note", "speech", "vocab"), "")
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
                        f"{_level(d, 'f')} ({d:.2f}), позитивность {_level(v, 'f')} ({v:.2f}). Модель обучена на английских "
                        "записях, поэтому значения относительные: полезнее сравнивать отрезки между собой.")
    fa = an.get("face")
    if fa and fa.get("mean"):
        m = fa["mean"]
        top = sorted(m.items(), key=lambda kv: -kv[1])[:2]
        s = "Выражение лица чаще всего распознаётся как " + ", реже — как ".join(
            f"{_FACE_EMO.get(k, EMO_RU.get(k, k))} ({v:.0%} кадров)" if i == 0 else
            f"{_FACE_EMO.get(k, EMO_RU.get(k, k))} ({v:.0%})" for i, (k, v) in enumerate(top))
        if fa.get("head_motion") is not None:
            hm = fa["head_motion"]
            s += "; голова " + ("почти неподвижна" if hm < 0.05 else ("двигается умеренно" if hm < 0.15 else "двигается активно"))
        if fa.get("face_share") is not None and fa["face_share"] < 0.9:
            s += f"; лицо видно в {fa['face_share']:.0%} кадров"
        out["face"] = s + "."
        out["face_note"] = "Распознавание выражений обучено на фотографиях и склонно завышать «грусть» и «страх» у спокойного лица."
    sp = an.get("speech")
    if sp and sp.get("description"):
        out["speech"] = fix_counts(sp["description"])
        vocab = vocabulary_shown(rep)          # Russian words; for English speech their translations
        if vocab:
            out["vocab"] = "Чаще всего звучат слова: " + ", ".join(f"«{w}»" for w, _ in vocab[:6]) + "."
    return out


def analyses_sentences(rep: dict) -> str:
    return " ".join(p for p in analyses_parts(rep).values() if p)


def key_facts(rep: dict) -> List[tuple]:
    """(label, value, note) cards for the overview tab. Every number carries its unit or scale; counts use the
    correct Russian plural; rounding matches the «Речь» tab (whole words per minute, whole fillers per 100 words)."""
    an = rep.get("analyses") or {}
    facts = []
    te = an.get("emotions_text")
    if te and te.get("mean"):
        k, v = max(te["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Эмоция по тексту речи", EMO_RU.get(k, k), f"{v:.0%} времени"))
    fa = an.get("face")
    if fa and fa.get("mean"):
        k, v = max(fa["mean"].items(), key=lambda kv: kv[1])
        facts.append(("Выражение лица", EMO_RU.get(k, k), f"{v:.0%} кадров"))
    vo = an.get("voice")
    if vo and vo.get("mean"):
        m = vo["mean"]
        facts.append(("Голос, шкала 0…1", f"возбуждение {m.get('arousal', 0):.2f}",
                      f"уверенность {m.get('dominance', 0):.2f} · позитивность {m.get('valence', 0):.2f}"))
    sp = an.get("speech")
    if sp and sp.get("words"):
        wpm = sp.get("words_per_min_speech")
        if wpm:
            n = int(round(wpm))
            value = f"{n} {plural_ru(n, 'слово', 'слова', 'слов')} в минуту"
        else:
            value = f"{sp['words']} {plural_ru(sp['words'], 'слово', 'слова', 'слов')}"
        fillers = int(round(sp.get("fillers_per_100", 0) or 0))
        facts.append(("Темп речи", value, f"паузы — {sp.get('pause_share', 0):.0%} времени, "
                                          f"заполнители — {fillers} на 100 слов"))
    if rep.get("interview"):
        facts.append(("Впечатление «собеседование»", f"{rep['interview']['score']:.2f}", "шкала 0…1, своя модель (FIV2)"))
    dur = rep.get("duration_sec")
    if dur:
        n = int(rep.get("segments") or 1)
        facts.append(("Длительность ролика", fmt_secs(dur),
                      f"разбит на {n} {plural_ru(n, 'отрезок', 'отрезка', 'отрезков')}" if n > 1 else "один отрезок"))
    return facts
